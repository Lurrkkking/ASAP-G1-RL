import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.utils.helpers import pre_process_config


class PPODeltaA(PPO):
    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):
        super().__init__(env, config, log_dir, device)

        if config.policy_checkpoint is None:
            raise ValueError("algo.config.policy_checkpoint must be provided for PPODeltaA")

        self.delta_policy_checkpoint = Path(config.policy_checkpoint)
        self.loaded_policy = self._load_frozen_policy(self.delta_policy_checkpoint, env, device)

        self.delta_action_mask_mode = self._get_delta_action_mask_mode()
        self.delta_action_scale = float(getattr(config, "delta_action_scale", 1.0))
        self.ankle_delta_indices = self._get_ankle_delta_indices()
        self.non_ankle_delta_indices = [idx for idx in range(self.num_act) if idx not in self.ankle_delta_indices]
        self.lower_body_delta_indices = list(range(int(self.env.config.robot.lower_body_actions_dim)))
        self.hip_delta_indices = list(range(12))
        self.delta_action_mask = self._build_delta_action_mask()
        self.closed_loop_debug_interval = int(getattr(config, "closed_loop_debug_interval", 50))

        self.action_anchor_cfg = getattr(config, "action_anchor", None)
        self.anchor_enabled = bool(getattr(self.action_anchor_cfg, "enabled", False)) if self.action_anchor_cfg is not None else False
        self.anchor_policy = None
        self.anchor_policy_eval = None
        anchor_ckpt = None
        if self.action_anchor_cfg is not None:
            anchor_ckpt = getattr(self.action_anchor_cfg, "policy_checkpoint", None)
        if anchor_ckpt in (None, "", "null"):
            anchor_ckpt = str(self.delta_policy_checkpoint)
        if self.anchor_enabled:
            self.anchor_policy = self._load_frozen_policy(Path(anchor_ckpt), env, device)
            self.anchor_policy_eval = self.anchor_policy.eval_policy

        self._log_closed_loop_delta_config_once()
        self._log_frozen_delta_freeze_debug_once()
        self._log_action_anchor_debug_once()

    def _build_closed_loop_actor_obs(self):
        return torch.cat([
            self.env._get_obs_base_pos_z(),
            self.env._get_obs_feet_contact_force(),
            self.env._get_obs_base_lin_vel(),
            self.env._get_obs_base_ang_vel(),
            self.env._get_obs_projected_gravity(),
            self.env._get_obs_dof_pos(),
            self.env._get_obs_dof_vel(),
            self.env._get_obs_actions_closed_loop(),
            self.env._get_obs_actions_sim2real_policy(),
        ], dim=-1)

    def _ensure_closed_loop_actor_obs(self, obs_dict):
        if 'closed_loop_actor_obs' not in obs_dict:
            obs_dict['closed_loop_actor_obs'] = self._build_closed_loop_actor_obs()
        return obs_dict

    def _load_frozen_policy(self, checkpoint: Path, env, device):
        config_path_candidates = [checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"]
        policy_config = None
        for config_path in config_path_candidates:
            if config_path.exists():
                logger.info(f"Loading training config file from {config_path}")
                with open(config_path) as file:
                    policy_config = OmegaConf.load(file)
                break
        if policy_config is None:
            raise FileNotFoundError(f"Could not find config.yaml near frozen checkpoint {checkpoint}")

        if policy_config.eval_overrides is not None:
            policy_config = OmegaConf.merge(policy_config, policy_config.eval_overrides)
        pre_process_config(policy_config)

        policy: BaseAlgo = instantiate(policy_config.algo, env=env, device=device, log_dir=None)
        policy.algo_obs_dim_dict = policy_config.env.config.robot.algo_obs_dim_dict
        policy.setup()
        if hasattr(policy, "load_optimizer"):
            policy.load_optimizer = False
        policy.load(str(checkpoint))
        policy._eval_mode()
        policy.actor.eval()
        policy.critic.eval()
        policy.actor.requires_grad_(False)
        policy.critic.requires_grad_(False)
        policy.eval_policy = policy._get_inference_policy()
        return policy

    def _get_delta_action_mask_mode(self):
        return str(getattr(self.config, "delta_action_mask_mode", "full"))

    def _get_ankle_delta_indices(self):
        indices = [4, 5, 10, 11]
        if max(indices) >= self.num_act:
            raise ValueError(f"ankle delta indices {indices} exceed action dimension {self.num_act}")
        return indices

    def _build_delta_action_mask(self):
        mode = self.delta_action_mask_mode
        if mode not in {"full", "ankle_only"}:
            raise ValueError(f"Unsupported delta_action_mask_mode={mode}. Expected one of ['full', 'ankle_only'].")
        mask = torch.ones(self.num_act, dtype=torch.float32, device=self.device)
        if mode == "ankle_only":
            mask.zero_()
            mask[self.ankle_delta_indices] = 1.0
        return mask

    def _compose_closed_loop_actions(self, trainable_policy_action, frozen_delta_action_raw):
        masked_delta = frozen_delta_action_raw * self.delta_action_mask
        scaled_masked_delta = masked_delta * self.delta_action_scale
        final_action = trainable_policy_action + scaled_masked_delta

        if self.non_ankle_delta_indices:
            non_ankle_delta = masked_delta[:, self.non_ankle_delta_indices]
            assert non_ankle_delta.abs().max().item() < 1e-6
            non_ankle_final_diff = final_action[:, self.non_ankle_delta_indices] - trainable_policy_action[:, self.non_ankle_delta_indices]
            assert non_ankle_final_diff.abs().max().item() < 1e-6
        else:
            non_ankle_delta = masked_delta.new_zeros(masked_delta.shape[0], 0)

        final_equals_main = torch.allclose(final_action, trainable_policy_action, atol=1e-7, rtol=0.0)
        effective_delta_mean_abs = scaled_masked_delta.abs().mean()

        return {
            "trainable_policy_action": trainable_policy_action,
            "frozen_delta_action_raw": frozen_delta_action_raw,
            "frozen_delta_action_masked": masked_delta,
            "frozen_delta_action_scaled": scaled_masked_delta,
            "final_action": final_action,
            "non_ankle_delta_after_mask": non_ankle_delta,
            "final_action_equals_main_policy_action": final_equals_main,
            "effective_delta_mean_abs": effective_delta_mean_abs,
        }

    def _count_trainable_params(self, module):
        return sum(param.numel() for param in module.parameters() if param.requires_grad)

    def _count_optimizer_params(self):
        total = 0
        optimizers = []
        if hasattr(self, "actor_optimizer"):
            optimizers.append(self.actor_optimizer)
        if hasattr(self, "critic_optimizer"):
            optimizers.append(self.critic_optimizer)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                total += sum(param.numel() for param in group["params"])
        return total

    def _count_main_policy_trainable_params(self):
        if not hasattr(self, "actor") or not hasattr(self, "critic"):
            return None
        return self._count_trainable_params(self.actor) + self._count_trainable_params(self.critic)

    def _format_debug_value(self, value):
        return "NA" if value is None else str(value)

    def _log_closed_loop_delta_config_once(self):
        joint_order = list(self.env.config.robot.dof_names)
        ankle_joint_names = [joint_order[idx] for idx in self.ankle_delta_indices]
        logger.info(
            "[closed-loop delta config] "
            f"closed_loop_delta_action_mask_mode={self.delta_action_mask_mode} "
            f"closed_loop_delta_action_scale={self.delta_action_scale:.6f} "
            f"ankle_delta_indices={self.ankle_delta_indices} "
            f"ankle_joint_names={ankle_joint_names} "
            f"joint_order={joint_order}"
        )

    def _log_frozen_delta_freeze_debug_once(self):
        loaded_policy_exists = hasattr(self, "loaded_policy") and self.loaded_policy is not None
        loaded_actor_eval = loaded_policy_exists and hasattr(self.loaded_policy, "actor") and (not self.loaded_policy.actor.training)
        loaded_critic_eval = loaded_policy_exists and hasattr(self.loaded_policy, "critic") and (not self.loaded_policy.critic.training)
        delta_model_requires_grad_count = self._count_trainable_params(self.loaded_policy.actor) + self._count_trainable_params(self.loaded_policy.critic)
        loaded_policy_all_frozen = delta_model_requires_grad_count == 0
        logger.info(
            "[closed-loop freeze debug] "
            f"loaded_policy_exists={loaded_policy_exists} "
            f"loaded_policy_actor_eval={loaded_actor_eval} "
            f"loaded_policy_critic_eval={loaded_critic_eval} "
            f"loaded_policy_all_params_frozen={loaded_policy_all_frozen} "
            f"delta_model_requires_grad_count={delta_model_requires_grad_count}"
        )

    def _log_action_anchor_debug_once(self):
        logger.info(
            "[action anchor config] "
            f"enabled={self.anchor_enabled} "
            f"weight={float(getattr(self.action_anchor_cfg, 'weight', 0.0) if self.action_anchor_cfg is not None else 0.0):.6f} "
            f"lower_body_weight={float(getattr(self.action_anchor_cfg, 'lower_body_weight', 0.0) if self.action_anchor_cfg is not None else 0.0):.6f} "
            f"ankle_hip_weight={float(getattr(self.action_anchor_cfg, 'ankle_hip_weight', 0.0) if self.action_anchor_cfg is not None else 0.0):.6f}"
        )

    def _action_anchor_loss(self, actor_obs, current_action):
        if not self.anchor_enabled or self.anchor_policy_eval is None:
            zero = current_action.new_tensor(0.0)
            return zero, {
                "action_anchor_diff_mean_abs": zero,
                "lower_body_action_diff_mean_abs": zero,
                "ankle_action_diff_mean_abs": zero,
                "hip_action_diff_mean_abs": zero,
                "action_anchor_loss": zero,
            }
        with torch.no_grad():
            baseline_action = self.anchor_policy_eval(actor_obs)
        diff = current_action - baseline_action
        diff_abs = diff.abs()
        full_loss = diff.pow(2).mean(dim=-1)
        lower_body_loss = diff[:, self.lower_body_delta_indices].pow(2).mean(dim=-1) if self.lower_body_delta_indices else full_loss
        ankle_loss = diff[:, self.ankle_delta_indices].pow(2).mean(dim=-1)
        hip_loss = diff[:, self.hip_delta_indices].pow(2).mean(dim=-1)
        weight = float(getattr(self.action_anchor_cfg, "weight", 0.0))
        lower_weight = float(getattr(self.action_anchor_cfg, "lower_body_weight", 0.0))
        ankle_hip_weight = float(getattr(self.action_anchor_cfg, "ankle_hip_weight", 0.0))
        anchor_loss = weight * full_loss + lower_weight * lower_body_loss + ankle_hip_weight * 0.5 * (ankle_loss + hip_loss)
        metrics = {
            "action_anchor_diff_mean_abs": diff_abs.mean(),
            "lower_body_action_diff_mean_abs": diff_abs[:, self.lower_body_delta_indices].mean() if self.lower_body_delta_indices else diff_abs.mean(),
            "ankle_action_diff_mean_abs": diff_abs[:, self.ankle_delta_indices].mean(),
            "hip_action_diff_mean_abs": diff_abs[:, self.hip_delta_indices].mean(),
            "action_anchor_loss": anchor_loss.mean(),
        }
        return anchor_loss.mean(), metrics

    def _tensor_log_scalar(self, key, default=0.0):
        value = self.env.log_dict.get(key, default)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _log_closed_loop_action_debug(self, debug_iter, action_pack, anchor_metrics=None):
        clip_action_limit = float(self.env.config.robot.control.action_clip_value)
        final_action_clipped = torch.clamp(action_pack["final_action"].detach(), -clip_action_limit, clip_action_limit)
        delta_model_requires_grad_count = self._count_trainable_params(self.loaded_policy.actor) + self._count_trainable_params(self.loaded_policy.critic)
        policy_trainable_param_count = self._count_main_policy_trainable_params()
        optimizer_param_count = self._count_optimizer_params() if (hasattr(self, "actor_optimizer") or hasattr(self, "critic_optimizer")) else None

        non_ankle_delta_after_mask = action_pack["non_ankle_delta_after_mask"].detach()
        non_ankle_delta_after_mask_mean_abs = non_ankle_delta_after_mask.abs().mean().item() if non_ankle_delta_after_mask.numel() > 0 else 0.0

        current_policy_action = action_pack["trainable_policy_action"].detach()
        baseline_mean_abs = 0.0
        action_anchor_diff_mean_abs = 0.0
        lower_body_action_diff_mean_abs = 0.0
        ankle_action_diff_mean_abs = 0.0
        hip_action_diff_mean_abs = 0.0
        action_anchor_loss = 0.0
        if anchor_metrics is not None:
            action_anchor_diff_mean_abs = float(anchor_metrics["action_anchor_diff_mean_abs"].item())
            lower_body_action_diff_mean_abs = float(anchor_metrics["lower_body_action_diff_mean_abs"].item())
            ankle_action_diff_mean_abs = float(anchor_metrics["ankle_action_diff_mean_abs"].item())
            hip_action_diff_mean_abs = float(anchor_metrics["hip_action_diff_mean_abs"].item())
            action_anchor_loss = float(anchor_metrics["action_anchor_loss"].item())
        if self.anchor_enabled and self.anchor_policy_eval is not None:
            with torch.no_grad():
                baseline_action = self.anchor_policy_eval(self.last_obs_dict["actor_obs"]).detach() if hasattr(self, "last_obs_dict") else self.anchor_policy_eval(current_policy_action)
            baseline_mean_abs = baseline_action.abs().mean().item()
            if anchor_metrics is None:
                diff = current_policy_action - baseline_action
                action_anchor_diff_mean_abs = diff.abs().mean().item()
                lower_body_action_diff_mean_abs = diff[:, self.lower_body_delta_indices].abs().mean().item() if self.lower_body_delta_indices else action_anchor_diff_mean_abs
                ankle_action_diff_mean_abs = diff[:, self.ankle_delta_indices].abs().mean().item()
                hip_action_diff_mean_abs = diff[:, self.hip_delta_indices].abs().mean().item()

        root_height = self._tensor_log_scalar("root_height", self.env.simulator.robot_root_states[:, 2].mean())
        upper_body_diff_norm = self._tensor_log_scalar("upper_body_diff_norm", 0.0)
        lower_body_diff_norm = self._tensor_log_scalar("lower_body_diff_norm", 0.0)
        joint_pos_diff_norm = self._tensor_log_scalar("joint_pos_diff_norm", 0.0)

        logger.info(
            f"[closed-loop action debug][iter {debug_iter}] "
            f"delta_action_scale={self.delta_action_scale:.6f} "
            f"delta_action_mask_mode={self.delta_action_mask_mode} "
            f"final_action_equals_main_policy_action={action_pack['final_action_equals_main_policy_action']} "
            f"effective_delta_mean_abs={float(action_pack['effective_delta_mean_abs'].item()):.6f} "
            f"frozen_delta_masked_mean_abs={action_pack['frozen_delta_action_masked'].detach().abs().mean().item():.6f} "
            f"non_ankle_delta_after_mask_mean_abs={non_ankle_delta_after_mask_mean_abs:.6f} "
            f"main_policy_action_mean_abs={current_policy_action.abs().mean().item():.6f} "
            f"baseline_policy_action_mean_abs={baseline_mean_abs:.6f} "
            f"action_anchor_diff_mean_abs={action_anchor_diff_mean_abs:.6f} "
            f"lower_body_action_diff_mean_abs={lower_body_action_diff_mean_abs:.6f} "
            f"ankle_action_diff_mean_abs={ankle_action_diff_mean_abs:.6f} "
            f"hip_action_diff_mean_abs={hip_action_diff_mean_abs:.6f} "
            f"final_action_mean_abs={action_pack['final_action'].detach().abs().mean().item():.6f} "
            f"final_action_max_abs={action_pack['final_action'].detach().abs().max().item():.6f} "
            f"action_clip_frac={(final_action_clipped.abs() == clip_action_limit).float().mean().item():.6f} "
            f"delta_over_policy_mean={(action_pack['frozen_delta_action_scaled'].detach().abs() / (current_policy_action.abs() + 1e-6)).mean().item():.6f} "
            f"action_anchor_loss={action_anchor_loss:.6f} "
            f"root_height={root_height:.6f} "
            f"upper_body_diff_norm={upper_body_diff_norm:.6f} "
            f"lower_body_diff_norm={lower_body_diff_norm:.6f} "
            f"joint_pos_diff_norm={joint_pos_diff_norm:.6f} "
            f"episode_length={int(self.cur_episode_length.mean().item())}"
        )

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            obs_dict = self._ensure_closed_loop_actor_obs(obs_dict)
            self.last_obs_dict = obs_dict
            debug_iter = getattr(self, "_debug_action_stats_iteration", self.current_learning_iteration)
            should_log_action_debug = debug_iter % self.closed_loop_debug_interval == 0
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                trainable_policy_action = policy_state_dict["actions"]
                with torch.no_grad():
                    frozen_delta_action_raw = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()
                action_pack = self._compose_closed_loop_actions(trainable_policy_action, frozen_delta_action_raw)

                actor_state = {
                    "actions": action_pack["final_action"],
                    "actions_closed_loop": action_pack["frozen_delta_action_scaled"],
                    "actions_main_policy": trainable_policy_action,
                }

                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                obs_dict = self._ensure_closed_loop_actor_obs(obs_dict)
                if should_log_action_debug and i == 0:
                    anchor_metrics = None
                    if self.anchor_enabled:
                        _, anchor_metrics = self._action_anchor_loss(obs_dict["actor_obs"], trainable_policy_action)
                    self._log_closed_loop_action_debug(debug_iter, action_pack, anchor_metrics)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), dones=self.storage.query_key('dones'), rewards=self.storage.query_key('rewards')),
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _pre_eval_env_step(self, actor_state: dict):
        actor_state['obs'] = self._ensure_closed_loop_actor_obs(actor_state['obs'])
        trainable_policy_action = self.eval_policy(actor_state["obs"]['actor_obs'])
        with torch.no_grad():
            frozen_delta_action_raw = self.loaded_policy.eval_policy(actor_state['obs']['closed_loop_actor_obs']).detach()
        action_pack = self._compose_closed_loop_actions(trainable_policy_action, frozen_delta_action_raw)

        actor_state.update({
            "actions": action_pack["final_action"],
            "actions_closed_loop": action_pack["frozen_delta_action_scaled"],
            "actions_main_policy": trainable_policy_action,
        })
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        anchor_loss = torch.tensor(0.0, device=self.device)
        anchor_metrics = None
        if self.anchor_enabled and self.anchor_policy_eval is not None:
            anchor_loss, anchor_metrics = self._action_anchor_loss(policy_state_dict['actor_obs'], actions_batch)
            actor_loss = actor_loss + anchor_loss

        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict.setdefault('Anchor', 0.0)
        loss_dict['Anchor'] += float(anchor_loss.item())
        if anchor_metrics is not None:
            for key, value in anchor_metrics.items():
                loss_dict.setdefault(key, 0.0)
                loss_dict[key] += float(value.item())
        return loss_dict

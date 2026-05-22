import torch
from loguru import logger

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.torch_utils import torch_rand_float


class DeltaA_ClosedLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.closed_loop_actions = torch.zeros(
            self.num_envs, self.dim_actions, device=self.device, requires_grad=False
        )
        self.trainable_policy_actions = torch.zeros(
            self.num_envs, self.dim_actions, device=self.device, requires_grad=False
        )
        self.executed_actions_total = torch.zeros(
            self.num_envs, self.dim_actions, device=self.device, requires_grad=False
        )

    def _init_buffers(self):
        super()._init_buffers()
        if self.config.domain_rand.cotrain_with_without_delta_a:
            without_delta_a_ratio = self.config.domain_rand.without_delta_a_ratio
            self.with_delta_a_or_not = torch.rand(self.num_envs, device=self.device) > without_delta_a_ratio
            self.with_delta_a_or_not = self.with_delta_a_or_not.unsqueeze(1)

        self.delta_a_scale = torch.ones(self.num_envs, device=self.device, requires_grad=False)
        if self.config.domain_rand.rescale_delta_a:
            self.delta_a_scale = torch_rand_float(
                self.config.domain_rand.delta_a_scale_range[0],
                self.config.domain_rand.delta_a_scale_range[1],
                (self.num_envs, self.num_dofs),
                device=self.device,
            )

    def _episodic_domain_randomization(self, env_ids):
        super()._episodic_domain_randomization(env_ids)

        if self.config.domain_rand.rescale_delta_a:
            self.delta_a_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.delta_a_scale_range[0],
                self.config.domain_rand.delta_a_scale_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )

    def _compute_torques(self, actions):
        delta_action = self.get_closed_loop_action_at_current_timestep() if self.config['add_extra_action'] else torch.zeros_like(actions)

        if self.config.domain_rand.cotrain_with_without_delta_a:
            delta_action = delta_action * self.with_delta_a_or_not
        if self.config.domain_rand.rescale_delta_a:
            delta_action = delta_action * self.delta_a_scale

        if hasattr(self.config.domain_rand, 'action_noise') and self.config.domain_rand.action_noise:
            logger.info("adding action noise")
            logger.info(f"noise percentage {self.config.domain_rand.action_noise_percentage}")
            action_noise = (torch.rand_like(actions) * 2. - 1.) * self.config.domain_rand.action_noise_percentage
            actions = actions + action_noise

        final_action = actions + delta_action

        if hasattr(self.config, 'anklePR') and self.config.anklePR:
            final_action[:, [i for i in range(final_action.shape[1]) if i not in [4, 5, 10, 11]]] *= 0

        clip_action_limit = self.config.robot.control.action_clip_value
        final_action = torch.clip(final_action, -clip_action_limit, clip_action_limit)
        self.executed_actions_total = final_action * self.config.robot.control.action_scale
        self.log_dict["closed_loop/final_action_clip_frac"] = (
            final_action.abs() == clip_action_limit
        ).float().mean()

        actions_scaled = final_action * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type

        if hasattr(self.config, 'delta_a_gradient_search') and self.config.delta_a_gradient_search:
            if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                iter_for_gradient_descent = 2000
                torch.set_grad_enabled(True)

                current_closed_loop_actor_obs = self.current_closed_loop_actor_obs
                fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()
                norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                for _ in range(iter_for_gradient_descent):
                    new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                    loss_fn = new_best_action + self.loaded_extra_policy.eval_policy(new_input_for_deltaA) - norminal_policy_action
                    loss_fn = loss_fn.norm(dim=1)
                    loss_fn.backward()
                    with torch.no_grad():
                        new_best_action -= 0.0002 * new_best_action.grad
                    new_best_action.grad.zero_()

                torch.set_grad_enabled(False)
                actions_scaled = new_best_action * self.config.robot.control.action_scale

        if hasattr(self.config, 'delta_a_fixed_point_iteration') and self.config.delta_a_fixed_point_iteration:
            if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                iter_for_gradient_descent = 10

                current_closed_loop_actor_obs = self.current_closed_loop_actor_obs
                fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()
                norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                for _ in range(iter_for_gradient_descent):
                    new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                    new_best_action = norminal_policy_action - self.loaded_extra_policy.eval_policy(new_input_for_deltaA)

                actions_scaled = new_best_action * self.config.robot.control.action_scale

        if control_type == "P":
            torques = self._kp_scale * self.p_gains * (
                actions_scaled + self.default_dof_pos - self.simulator.dof_pos
            ) - self._kd_scale * self.d_gains * self.simulator.dof_vel
        elif control_type == "V":
            torques = self._kp_scale * self.p_gains * (actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains * (self.simulator.dof_vel - self.last_dof_vel) / self.sim_dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques) * 2. - 1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques

    def get_closed_loop_action_at_current_timestep(self):
        return self.closed_loop_actions.clone()

    def _get_obs_actions_closed_loop(self):
        return self.closed_loop_actions.clone()

    def _get_obs_actions_sim2real_policy(self):
        return self.trainable_policy_actions.clone()

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        return self.simulator.dof_pos[:, [4, 5, 10, 11]]

    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, [4, 5, 10, 11]]

    def _get_obs_actions_closed_loop_ankle_pitch_roll(self):
        return self.closed_loop_actions.clone()[:, [4, 5, 10, 11]]

    def _get_obs_actions_sim2real_policy_ankle_pitch_roll(self):
        return self.trainable_policy_actions[:, [4, 5, 10, 11]]

    def step(self, actor_state):
        actions = actor_state["actions"]
        self.closed_loop_actions = actor_state["actions_closed_loop"].to(self.device)
        clip_action_limit = self.config.robot.control.action_clip_value
        self.trainable_policy_actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        self._pre_physics_step(self.trainable_policy_actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def reset_all(self):
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actions_closed_loop = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {
            "actions": actions,
            "actions_closed_loop": actions_closed_loop,
        }
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float

import torch.nn as nn

class ResidualDynamicsMLP(nn.Module):
    """Simple MLP: input = [s_t, a_t], output = 46-dim (delta_a + alpha)."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeltaActionPatchMLP(nn.Module):
    """Offline delta-action patch: input = [s_t, a_base], output = 23-dim delta_a."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        d = state_dim + action_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s_t, a_base):
        return self.net(torch.cat([s_t, a_base], dim=-1))


class DeltaA_ClosedLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)

        # 闭环基础策略动作维度固定为 num_dof（23），与 46 维 Attention-Delta 输出解耦
        self.closed_loop_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # 每步缓存 attention 权重，供 reward 计算
        self.alpha_t = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # 缓存 raw_delta_a，供观测使用（保持 obs.actions 维度为 23）
        self.raw_delta_a = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # Distinguish standard PPO-in-patched-env from PPODeltaA patch-training mode.
        self.use_policy_action_as_base = False
        # Cache the final normalized action actually executed by the PD controller.
        self.executed_actions_total = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_patch_cache_valid = False
        self._frozen_patch_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_raw_delta_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_alpha_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._debug_delta_a_once = os.environ.get("DELTA_A_DEBUG_ONCE", "0").lower() in {"1", "true", "yes"}
        self._debug_delta_a_printed = False
        # 可配置残差动作最大幅值；默认继承 action_scale
        self.max_delta_scale = float(getattr(self.config, "max_delta_scale", self.config.robot.control.action_scale))

        # ---------------- 新增：挂载 Attention-Delta 补丁 ----------------
        # 优先兼容 self.cfg.env.config.delta_ckpt_path；不存在时回退 self.config.delta_ckpt_path
        delta_ckpt_path = None
        if (
            hasattr(self, "cfg")
            and hasattr(self.cfg, "env")
            and hasattr(self.cfg.env, "config")
            and hasattr(self.cfg.env.config, "delta_ckpt_path")
        ):
            delta_ckpt_path = self.cfg.env.config.delta_ckpt_path
        elif hasattr(self.config, "delta_ckpt_path"):
            delta_ckpt_path = self.config.delta_ckpt_path

        if isinstance(delta_ckpt_path, str) and delta_ckpt_path.lower() in {"", "none", "null"}:
            delta_ckpt_path = None

        self.use_frozen_patch = delta_ckpt_path is not None
        self.frozen_patch_type = None
        self.frozen_patch_max_delta_scale = self.max_delta_scale

        if self.use_frozen_patch:
            patch_path = delta_ckpt_path
            print(f"========== 挂载冻结的物理补丁: {patch_path} ==========")
            patch_state_dict = torch.load(patch_path, map_location=self.device)

            # 兼容 state_dict 嵌套
            if 'model_state_dict' in patch_state_dict:
                model_state_dict = patch_state_dict['model_state_dict']
            else:
                model_state_dict = patch_state_dict

            if "state_dim" in patch_state_dict and "action_dim" in patch_state_dict:
                state_dim = int(patch_state_dict["state_dim"])
                action_dim = int(patch_state_dict["action_dim"])
                hidden_dim = int(patch_state_dict.get("hidden_dim", 256))
                depth = int(patch_state_dict.get("depth", 3))
                self.frozen_patch_max_delta_scale = float(
                    patch_state_dict.get("max_delta_scale", self.max_delta_scale)
                )
                if action_dim != self.num_dof:
                    raise ValueError(
                        f"Frozen delta-action patch action_dim={action_dim} does not match num_dof={self.num_dof}."
                    )
                self.delta_model = DeltaActionPatchMLP(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                )
                self.frozen_patch_type = "delta_action_patch"
            else:
                in_dim = int(patch_state_dict.get("in_dim", 76))
                out_dim = int(
                    patch_state_dict.get(
                        "out_dim",
                        next(iter(model_state_dict.values())).shape[0],
                    )
                )
                hidden_dim = int(patch_state_dict.get("hidden_dim", 512))
                depth = int(patch_state_dict.get("depth", 3))
                self.delta_model = ResidualDynamicsMLP(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                )
                self.frozen_patch_type = "attention_delta" if out_dim >= self.num_dof * 2 else "delta_action_patch_raw"

            self.delta_model.load_state_dict(model_state_dict)
            self.delta_model.eval()  # 必须强制评估模式
            self.delta_model.to(self.device)

            # 冻结所有参数，防止被误更新
            for param in self.delta_model.parameters():
                param.requires_grad = False
            print(
                "========== 冻结补丁类型: "
                f"{self.frozen_patch_type}, max_delta_scale={self.frozen_patch_max_delta_scale:.4f} =========="
            )
        else:
            self.delta_model = None
            print("========== 未提供物理补丁，执行标准 23 维环境 ==========")
        # ----------------------------------------------------------------

        gap_model_path = getattr(self.config, "gap_model_path", None)
        self.use_gap_reward = bool(getattr(self.config, "use_gap_reward", False)) and gap_model_path is not None
        self.gap_reward_scale = float(getattr(self.config, "gap_reward_scale", 0.25))
        self.gap_reward_sign = float(getattr(self.config, "gap_reward_sign", 1.0))
        self.gap_model = None

        if self.use_gap_reward:
            print(f"========== Loading reward gap model: {gap_model_path} ==========")
            gap_ckpt = torch.load(gap_model_path, map_location=self.device)
            in_dim = int(gap_ckpt.get("in_dim", 76))
            out_dim = int(gap_ckpt.get("out_dim", 53))
            hidden_dim = int(gap_ckpt.get("hidden_dim", 512))
            depth = int(gap_ckpt.get("depth", 3))
            if out_dim < 7 + 2 * self.num_dof:
                raise ValueError(f"gap model out_dim={out_dim} is smaller than expected {7 + 2 * self.num_dof}")
            self.gap_model = ResidualDynamicsMLP(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, depth=depth)
            gap_state_dict = gap_ckpt["model_state_dict"] if "model_state_dict" in gap_ckpt else gap_ckpt
            self.gap_model.load_state_dict(gap_state_dict)
            self.gap_model.eval()
            self.gap_model.to(self.device)
            for param in self.gap_model.parameters():
                param.requires_grad = False

    def _init_buffers(self):
        super()._init_buffers()

        # Support both 23-dim direct-delta and 46-dim attention-delta training.
        if self.dim_actions not in {self.num_dof, self.num_dof * 2}:
            logger.warning(
                f"DeltaA_ClosedLoop expects actions_dim in {{{self.num_dof}, {self.num_dof * 2}}}, "
                f"got {self.dim_actions}."
            )

        # PD/力矩张量保持关节维度（23）
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = self.p_gains[: self.num_dof].clone()
        self.d_gains = self.d_gains[: self.num_dof].clone()

        # 控制延迟队列跟 actor 动作维度（46）保持一致
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs,
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                self.dim_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

        self.closed_loop_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.alpha_t = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.raw_delta_a = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.use_policy_action_as_base = False
        self.executed_actions_total = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_patch_cache_valid = False
        self._frozen_patch_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_raw_delta_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self._frozen_alpha_cache = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)

    def _apply_gap_reward_correction(self, ref_joint_pos, ref_joint_vel):
        if not self.use_gap_reward or self.gap_model is None:
            return ref_joint_pos, ref_joint_vel

        with torch.no_grad():
            s_t = torch.cat([self.simulator.robot_root_states[:, :7], self.simulator.dof_pos, self.simulator.dof_vel], dim=-1)
            gap_input = torch.cat([s_t, self.actions_closed_loop], dim=-1)
            gap = self.gap_model(gap_input)

        dof_start = 7
        dof_end = dof_start + self.num_dof
        vel_end = dof_end + self.num_dof
        gap_dof_pos = gap[:, dof_start:dof_end]
        gap_dof_vel = gap[:, dof_end:vel_end]

        ref_joint_pos = ref_joint_pos + self.gap_reward_sign * self.gap_reward_scale * gap_dof_pos
        ref_joint_vel = ref_joint_vel + self.gap_reward_sign * self.gap_reward_scale * gap_dof_vel
        return ref_joint_pos, ref_joint_vel

    def _episodic_domain_randomization(self, env_ids):
        super()._episodic_domain_randomization(env_ids)

    def _compute_attention_delta_action(self, actions):
        """46 维动作拆分并门控融合。

        输入:
            actions[..., :23]   -> raw_delta_a
            actions[..., 23:46] -> raw_alpha
        输出:
            actions_scaled = tanh(raw_delta_a) * max_delta_scale * sigmoid(raw_alpha)
        """
        required_dim = self.num_dof * 2
        if actions.shape[1] < required_dim:
            raise ValueError(
                f"actions dim {actions.shape[1]} is smaller than required {required_dim} "
                f"(num_dof={self.num_dof})."
            )

        # 1) 切片剥离
        raw_delta_a = actions[:, :self.num_dof]
        raw_alpha = actions[:, self.num_dof: required_dim]

        # 2) 激活限制
        delta_a = torch.tanh(raw_delta_a) * self.max_delta_scale
        alpha_t = torch.sigmoid(raw_alpha)

        # 3) Attention gating
        return delta_a * alpha_t, raw_delta_a, alpha_t

    def _compute_direct_delta_action(self, actions, max_delta_scale):
        raw_delta_a = actions[:, :self.num_dof]
        delta_a = torch.tanh(raw_delta_a) * max_delta_scale
        alpha_t = torch.zeros_like(delta_a)
        return delta_a, raw_delta_a, alpha_t

    def _target_pos_from_normalized_action(self, actions):
        return actions[:, : self.num_dof] * self.config.robot.control.action_scale + self.default_dof_pos

    def _compute_frozen_patch_action(self, base_actions):
        with torch.no_grad():
            root_states = getattr(self, "root_states", self.simulator.robot_root_states)
            dof_pos = getattr(self, "dof_pos", self.simulator.dof_pos)
            dof_vel = getattr(self, "dof_vel", self.simulator.dof_vel)
            s_t = torch.cat([root_states[:, :7], dof_pos, dof_vel], dim=-1)

            if self.frozen_patch_type == "delta_action_patch":
                # Offline oracle patches were trained on absolute PD target joint
                # positions: target = normalized_action * action_scale + default_dof_pos.
                base_target_pos = self._target_pos_from_normalized_action(base_actions)
                raw_delta_a = self.delta_model(s_t, base_target_pos)
                delta_a = torch.tanh(raw_delta_a) * self.frozen_patch_max_delta_scale
                alpha_t = torch.zeros_like(delta_a)
                return delta_a, raw_delta_a, alpha_t

            patch_input = torch.cat([s_t, base_actions], dim=-1)
            patch_output = self.delta_model(patch_input)

        if patch_output.shape[1] >= self.num_dof * 2:
            raw_delta_a = patch_output[:, :self.num_dof]
            raw_alpha = patch_output[:, self.num_dof: self.num_dof * 2]
            delta_a = torch.tanh(raw_delta_a) * self.frozen_patch_max_delta_scale
            alpha_t = torch.sigmoid(raw_alpha)
            return delta_a * alpha_t, raw_delta_a, alpha_t

        if patch_output.shape[1] >= self.num_dof:
            raw_delta_a = patch_output[:, :self.num_dof]
            delta_a = torch.tanh(raw_delta_a) * self.frozen_patch_max_delta_scale
            alpha_t = torch.zeros_like(delta_a)
            return delta_a, raw_delta_a, alpha_t

        raise ValueError(
            f"Frozen patch output dim={patch_output.shape[1]} is smaller than required {self.num_dof}."
        )

    def _debug_log_delta_a_once(
        self,
        base_actions,
        motion_action,
        frozen_patch,
        frozen_raw_delta,
        train_patch,
        train_raw_delta,
        actions_patched,
        actions_total,
        action_scale,
    ):
        if (not self._debug_delta_a_once) or self._debug_delta_a_printed:
            return

        # Skip reset-time bootstrap calls that use all-zero policy/base actions.
        # We only want the first sample where the trainable policy path actually participates.
        signal_tensors = (base_actions, motion_action, train_patch)
        has_policy_signal = any(tensor.abs().max().item() > 1e-6 for tensor in signal_tensors)
        if not has_policy_signal:
            return

        self._debug_delta_a_printed = True

        def _sample_stats(tensor):
            v = tensor[0].detach().float().cpu()
            return (
                f"mean_abs={v.abs().mean().item():.9f} "
                f"min={v.min().item():.9f} "
                f"max={v.max().item():.9f} "
                f"vec={v.numpy().tolist()}"
            )

        def _sample_clip_excess(tensor, limit):
            v = tensor[0].detach().float().cpu()
            excess = torch.clamp(v.abs() - float(limit), min=0.0)
            return (
                f"limit={float(limit):.9f} "
                f"mean_abs_excess={excess.mean().item():.9f} "
                f"max_abs_excess={excess.max().item():.9f} "
                f"vec={excess.numpy().tolist()}"
            )

        clip_action_limit = float(self.config.robot.control.action_clip_value)
        frozen_patch_norm = frozen_patch / action_scale if action_scale != 0 else frozen_patch
        train_patch_norm = train_patch if self.frozen_patch_type != "delta_action_patch" else train_patch
        actions_patched_norm = actions_patched / action_scale if action_scale != 0 else actions_patched
        actions_total_norm = actions_total / action_scale if action_scale != 0 else actions_total

        print(
            "[DELTA_A_DEBUG] "
            f"use_policy_action_as_base={self.use_policy_action_as_base} "
            f"frozen_patch_type={self.frozen_patch_type} "
            f"action_scale={float(action_scale):.9f} "
            f"action_clip={clip_action_limit:.9f} "
            f"train_max_delta_scale={float(self.max_delta_scale):.9f} "
            f"frozen_patch_max_delta_scale={float(self.frozen_patch_max_delta_scale):.9f}"
        )
        print(f"[DELTA_A_DEBUG] base_actions_norm sample0: {_sample_stats(base_actions)}")
        print(
            f"[DELTA_A_DEBUG] base_actions_norm sample0 clip_excess: "
            f"{_sample_clip_excess(base_actions, clip_action_limit)}"
        )
        print(f"[DELTA_A_DEBUG] motion_action_rad_offset sample0: {_sample_stats(motion_action)}")
        print(f"[DELTA_A_DEBUG] frozen_raw_delta sample0: {_sample_stats(frozen_raw_delta)}")
        if self.frozen_patch_type == "delta_action_patch":
            print(f"[DELTA_A_DEBUG] frozen_patch_rad sample0: {_sample_stats(frozen_patch)}")
            print(
                f"[DELTA_A_DEBUG] frozen_patch_rad sample0 max_delta_excess: "
                f"{_sample_clip_excess(frozen_patch, self.frozen_patch_max_delta_scale)}"
            )
            print(f"[DELTA_A_DEBUG] frozen_patch_norm_equiv sample0: {_sample_stats(frozen_patch_norm)}")
        else:
            print(f"[DELTA_A_DEBUG] frozen_patch_norm sample0: {_sample_stats(frozen_patch_norm)}")
            print(
                f"[DELTA_A_DEBUG] frozen_patch_norm sample0 max_delta_excess: "
                f"{_sample_clip_excess(frozen_patch_norm, self.frozen_patch_max_delta_scale)}"
            )
        print(f"[DELTA_A_DEBUG] train_raw_delta sample0: {_sample_stats(train_raw_delta)}")
        print(f"[DELTA_A_DEBUG] train_patch_norm sample0: {_sample_stats(train_patch_norm)}")
        print(
            f"[DELTA_A_DEBUG] train_patch_norm sample0 max_delta_excess: "
            f"{_sample_clip_excess(train_patch_norm, self.max_delta_scale)}"
        )
        print(f"[DELTA_A_DEBUG] actions_patched_norm sample0: {_sample_stats(actions_patched_norm)}")
        print(f"[DELTA_A_DEBUG] actions_total_norm sample0: {_sample_stats(actions_total_norm)}")
        print(
            f"[DELTA_A_DEBUG] actions_total_norm sample0 clip_excess: "
            f"{_sample_clip_excess(actions_total_norm, clip_action_limit)}"
        )
        print(f"[DELTA_A_DEBUG] target_pos_rad sample0: {_sample_stats(actions_total + self.default_dof_pos)}")

    def _pre_physics_step(self, actions):
        self._frozen_patch_cache_valid = False
        super()._pre_physics_step(actions)

    def _compute_torques(self, actions):
        """Compute torques from actions.

        Attention-Delta 版本:
            a_patch = motion_action + alpha_t ⊙ delta_a
        """
        # 此时 actions 是主策略给出的动作；兼容 [N,23] 与 [N,>=23]
        base_actions = actions[:, : self.num_dof]
        frozen_patch = torch.zeros_like(base_actions)
        frozen_raw_delta = torch.zeros_like(base_actions)
        if self.use_frozen_patch:
            if not self._frozen_patch_cache_valid:
                frozen_patch, frozen_raw_delta, frozen_alpha = self._compute_frozen_patch_action(base_actions)
                self._frozen_patch_cache = frozen_patch.clone()
                self._frozen_raw_delta_cache = frozen_raw_delta.clone()
                self._frozen_alpha_cache = frozen_alpha.clone()
                self._frozen_patch_cache_valid = True
            else:
                frozen_patch = self._frozen_patch_cache
                frozen_raw_delta = self._frozen_raw_delta_cache

        train_patch = torch.zeros_like(base_actions)
        train_raw_delta = torch.zeros_like(base_actions)
        self.alpha_t = torch.zeros_like(base_actions)

        if (not self.use_policy_action_as_base) and actions.shape[1] >= self.num_dof * 2:
            train_patch, train_raw_delta, self.alpha_t = self._compute_attention_delta_action(actions)
        elif (not self.use_policy_action_as_base) and actions.shape[1] >= self.num_dof:
            train_patch, train_raw_delta, self.alpha_t = self._compute_direct_delta_action(
                actions,
                self.max_delta_scale,
            )

        action_scale = self.config.robot.control.action_scale
        if self.frozen_patch_type == "delta_action_patch":
            # Frozen offline patch is already in joint-radian target-position units.
            actions_patched = frozen_patch + train_patch * action_scale
        else:
            actions_patched = (frozen_patch + train_patch) * action_scale
        self.raw_delta_a = frozen_raw_delta + train_raw_delta
            
        control_type = self.config.robot.control.control_type
        if self.use_policy_action_as_base:
            motion_action = base_actions * action_scale
        elif self.config['add_extra_action']:
            motion_action = self.get_closed_loop_action_at_current_timestep() * action_scale
        else:
            motion_action = torch.zeros_like(actions_patched)

        if hasattr(self.config.domain_rand, 'action_noise'):
            if self.config.domain_rand.action_noise:
                action_noise = (torch.rand_like(actions_patched) * 2.0 - 1.0) * self.config.domain_rand.action_noise_percentage
                actions_patched += action_noise * action_scale

        if hasattr(self.config, 'delta_a_gradient_search'):
            if self.config.delta_a_gradient_search:
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 2000
                    torch.set_grad_enabled(True)

                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs
                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-self.num_dof].detach().clone()

                    norminal_policy_action = current_closed_loop_actor_obs[:, -self.num_dof:].detach().clone()
                    new_best_action = current_closed_loop_actor_obs[:, -self.num_dof:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent):
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        loss_fn = new_best_action + self.loaded_extra_policy.eval_policy(new_input_for_deltaA) - norminal_policy_action

                        loss_fn = loss_fn.norm(dim=1)
                        loss_fn.backward()
                        with torch.no_grad():
                            new_best_action -= 0.0002 * new_best_action.grad
                        new_best_action.grad.zero_()
                        print("iter", i)
                        print("loss_fn", loss_fn)
                        print("new_best_action", new_best_action)
                        print("-----------------------------------------")

                    torch.set_grad_enabled(False)
                    actions_patched = new_best_action * self.config.robot.control.action_scale

        if hasattr(self.config, 'delta_a_fixed_point_iteration'):
            if self.config.delta_a_fixed_point_iteration:
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 10

                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs
                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-self.num_dof].detach().clone()

                    norminal_policy_action = current_closed_loop_actor_obs[:, -self.num_dof:].detach().clone()
                    new_best_action = current_closed_loop_actor_obs[:, -self.num_dof:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent):
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        new_best_action = norminal_policy_action - self.loaded_extra_policy.eval_policy(new_input_for_deltaA)

                        print("iter", i)
                        print("new_best_action", new_best_action)
                        print("-----------------------------------------")

                    actions_patched = new_best_action * self.config.robot.control.action_scale

        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR:
                motion_action[:, [i for i in range(actions_patched.shape[1]) if i not in [4, 5, 10, 11]]] *= 0
                print("zeroing out non-anklePR actions")

        actions_total = actions_patched + motion_action
        self.executed_actions_total = actions_total.clone()
        self._debug_log_delta_a_once(
            base_actions=base_actions,
            motion_action=motion_action,
            frozen_patch=frozen_patch,
            frozen_raw_delta=frozen_raw_delta,
            train_patch=train_patch,
            train_raw_delta=train_raw_delta,
            actions_patched=actions_patched,
            actions_total=actions_total,
            action_scale=action_scale,
        )

        if control_type == "P":
            # Keep torque semantics aligned with LeggedRobotBase: scale actions before PD.
            torques = self._kp_scale * self.p_gains * (
                actions_total + self.default_dof_pos - self.simulator.dof_pos
            ) - self._kd_scale * self.d_gains * self.simulator.dof_vel
        elif control_type == "V":
            torques = self._kp_scale * self.p_gains * (actions_total - self.simulator.dof_vel) - self._kd_scale * self.d_gains * (self.simulator.dof_vel - self.last_dof_vel) / self.sim_dt
        elif control_type == "T":
            torques = actions_total
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques) * 2.0 - 1.0) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques

    def get_closed_loop_action_at_current_timestep(self):
        return self.actions_closed_loop.clone()

    def _get_obs_actions_closed_loop(self):
        return self.raw_delta_a

    # Keep actor_obs.actions compatible with the frozen 23-dim base policy.
    def _get_obs_actions(self):
        return self.actions_closed_loop.clone()

    # In closed_loop_actor_obs, this field is intended to carry base-policy actions.
    def _get_obs_actions_sim2real_policy(self):
        return self.actions_closed_loop.clone()

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        return self.simulator.dof_pos[:, [4, 5, 10, 11]]

    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, [4, 5, 10, 11]]

    def _get_obs_actions_closed_loop_ankle_pitch_roll(self):
        return self.raw_delta_a[:, [4, 5, 10, 11]]

    def _get_obs_actions_sim2real_policy_ankle_pitch_roll(self):
        return self.actions_closed_loop.clone()[:, [4, 5, 10, 11]]

    def step(self, actor_state):
        """Apply actions, simulate, call post_physics_step()."""
        actions = actor_state["actions"]
        if "actions_closed_loop" in actor_state and actor_state["actions_closed_loop"] is not None:
            # DeltaA / PPODeltaA path: the frozen base policy is supplied explicitly.
            self.use_policy_action_as_base = False
            self.actions_closed_loop = actor_state["actions_closed_loop"][:, :self.num_dof]
        else:
            # Standard PPO path: treat the current policy action itself as the base action,
            # then execute base_action + frozen_patch(base_action) inside the environment.
            self.use_policy_action_as_base = True
            self.actions_closed_loop = actions[:, :self.num_dof]
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions_closed_loop = torch.clip(self.actions_closed_loop, -clip_action_limit, clip_action_limit).to(self.device)

        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def reset_all(self):
        """Reset all robots."""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)

        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actions_closed_loop = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        actor_state = {"actions": actions, "actions_closed_loop": actions_closed_loop}
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict

    def _reward_attention_sparsity(self):
        """attention 稀疏惩罚：sum(alpha_t)。"""
        return torch.sum(self.alpha_t, dim=-1)

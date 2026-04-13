import torch
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


class DeltaA_ClosedLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)

        # 闭环基础策略动作维度固定为 num_dof（23），与 46 维 Attention-Delta 输出解耦
        self.closed_loop_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # 每步缓存 attention 权重，供 reward 计算
        self.alpha_t = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # 缓存 raw_delta_a，供观测使用（保持 obs.actions 维度为 23）
        self.raw_delta_a = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # 可配置残差动作最大幅值；默认继承 action_scale
        self.max_delta_scale = float(getattr(self.config, "max_delta_scale", self.config.robot.control.action_scale))

    def _init_buffers(self):
        super()._init_buffers()

        # 训练动作空间：23 raw_delta_a + 23 raw_alpha
        if self.dim_actions != self.num_dof * 2:
            logger.warning(
                f"DeltaA_ClosedLoop expects actions_dim={self.num_dof * 2}, got {self.dim_actions}. "
                "Please set robot.actions_dim=46 for G1-23DoF."
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

        # 缓存给 obs 使用
        self.raw_delta_a = raw_delta_a

        # 2) 激活限制
        delta_a = torch.tanh(raw_delta_a) * self.max_delta_scale
        self.alpha_t = torch.sigmoid(raw_alpha)

        # 3) Attention gating
        return delta_a * self.alpha_t

    def _compute_torques(self, actions):
        """Compute torques from actions.

        Attention-Delta 版本:
            a_patch = motion_action + alpha_t ⊙ delta_a
        """
        # 核心替换：46 维动作 -> 23 维 gated delta
        actions_scaled = self._compute_attention_delta_action(actions)

        control_type = self.config.robot.control.control_type
        if self.config['add_extra_action']:
            motion_action = self.get_closed_loop_action_at_current_timestep()
        else:
            motion_action = torch.zeros_like(actions_scaled)

        if hasattr(self.config.domain_rand, 'action_noise'):
            if self.config.domain_rand.action_noise:
                action_noise = (torch.rand_like(actions_scaled) * 2.0 - 1.0) * self.config.domain_rand.action_noise_percentage
                actions_scaled += action_noise

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
                    actions_scaled = new_best_action * self.config.robot.control.action_scale

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

                    actions_scaled = new_best_action * self.config.robot.control.action_scale

        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR:
                motion_action[:, [i for i in range(actions_scaled.shape[1]) if i not in [4, 5, 10, 11]]] *= 0
                print("zeroing out non-anklePR actions")

        if control_type == "P":
            # a_patch = motion_action + alpha_t ⊙ delta_a
            torques = self._kp_scale * self.p_gains * (
                actions_scaled + motion_action + self.default_dof_pos - self.simulator.dof_pos
            ) - self._kd_scale * self.d_gains * self.simulator.dof_vel
        elif control_type == "V":
            torques = self._kp_scale * self.p_gains * (actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains * (self.simulator.dof_vel - self.last_dof_vel) / self.sim_dt
        elif control_type == "T":
            torques = actions_scaled
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
        return self.actions_closed_loop.clone()

    # 覆盖基类：obs.actions 保持 23 维，避免因 actions_dim=46 造成观测维度不匹配
    def _get_obs_actions(self):
        return self.raw_delta_a

    def _get_obs_actions_sim2real_policy(self):
        return self.raw_delta_a

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        return self.simulator.dof_pos[:, [4, 5, 10, 11]]

    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, [4, 5, 10, 11]]

    def _get_obs_actions_closed_loop_ankle_pitch_roll(self):
        return self.actions_closed_loop.clone()[:, [4, 5, 10, 11]]

    def _get_obs_actions_sim2real_policy_ankle_pitch_roll(self):
        return self.raw_delta_a[:, [4, 5, 10, 11]]

    def step(self, actor_state):
        """Apply actions, simulate, call post_physics_step()."""
        actions = actor_state["actions"]
        # 闭环基础动作只取前 num_dof 维
        self.actions_closed_loop = actor_state["actions_closed_loop"][:, :self.num_dof]
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

import torch

from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_rotate, quat_unit


class BallControlRewardMixin:
    def _init_ball_control_rewards(self):
        self.task_mode = str(getattr(self.config, "task_mode", "ball_control")).lower()
        self.is_kickup_task_mode = self.task_mode == "kickup"
        rewards_cfg = getattr(self.config, "rewards", self.config)
        self.reposition_zone_reward_target = torch.tensor(
            getattr(rewards_cfg, "reposition_zone_reward_target", [0.45, 0.0, 0.90]),
            dtype=torch.float,
            device=self.device,
        )
        self.reposition_zone_reward_weight = torch.tensor(
            getattr(rewards_cfg, "reposition_zone_reward_weight", [4.0, 4.0, 2.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.recover_reposition_zone_scale = float(
            getattr(rewards_cfg, "recover_reposition_zone_scale", 0.0)
        )
        self.control_zone_x_range = torch.tensor(
            getattr(rewards_cfg, "control_zone_x_range", [0.25, 0.65]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_y_range = torch.tensor(
            getattr(rewards_cfg, "control_zone_y_range", [-0.25, 0.25]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_z_range = torch.tensor(
            getattr(rewards_cfg, "control_zone_z_range", [0.65, 1.15]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_vz_min = float(getattr(rewards_cfg, "control_zone_vz_min", 0.0))
        self.control_zone_z_max = float(self.control_zone_z_range[1].item())
        self.control_touch_contact_bonus = float(
            getattr(rewards_cfg, "control_touch_contact_bonus", 1.0)
        )
        self.wrong_body_contact_penalty_value = float(
            getattr(rewards_cfg, "wrong_body_contact_penalty", 1.0)
        )
        self.reposition_contact_violation_penalty_value = float(
            getattr(rewards_cfg, "reposition_contact_violation_penalty", 1.0)
        )
        self.invalid_stance_penalty_value = float(
            getattr(rewards_cfg, "invalid_stance_penalty", 1.0)
        )
        self.ball_horizontal_vel_penalty_weight = float(
            getattr(rewards_cfg, "ball_horizontal_vel_penalty", 0.25)
        )
        self.ball_horizontal_speed_penalty_tol = float(
            getattr(rewards_cfg, "ball_horizontal_speed_penalty_tol", 0.0)
        )
        self.missed_target_state_cycle_penalty_value = float(
            getattr(rewards_cfg, "missed_target_state_cycle_penalty", 1.0)
        )
        self.post_contact_vz_min = float(getattr(rewards_cfg, "post_contact_vz_min", 0.0))
        self.post_contact_vz_max = float(getattr(rewards_cfg, "post_contact_vz_max", 1.0e9))
        self.post_contact_height_min = float(
            getattr(rewards_cfg, "post_contact_height_min", getattr(self.config.ball, "radius", 0.10) + 0.15)
        )
        self.post_contact_height_max = float(
            getattr(rewards_cfg, "post_contact_height_max", 1.0e9)
        )
        self.post_contact_recover_reward_scale = float(
            getattr(rewards_cfg, "post_contact_recover_reward_scale", 1.0)
        )
        self.recover_base_upright_scale = float(
            getattr(rewards_cfg, "recover_base_upright_scale", 1.0)
        )
        self.recover_base_height_scale = float(
            getattr(rewards_cfg, "recover_base_height_scale", 1.0)
        )
        self.recover_base_vel_scale = float(getattr(rewards_cfg, "recover_base_vel_scale", 1.0))
        self.recover_dof_ready_scale = float(getattr(rewards_cfg, "recover_dof_ready_scale", 1.0))
        self.recover_foot_ready_scale = float(
            getattr(rewards_cfg, "recover_foot_ready_scale", 1.0)
        )
        self.recover_ball_body_dist_scale = float(
            getattr(rewards_cfg, "recover_ball_body_dist_scale", 0.0)
        )
        self.recover_base_upright_k = float(getattr(rewards_cfg, "recover_base_upright_k", 6.0))
        self.recover_base_height_k = float(getattr(rewards_cfg, "recover_base_height_k", 10.0))
        self.recover_base_vel_k = float(getattr(rewards_cfg, "recover_base_vel_k", 2.0))
        self.recover_dof_ready_k = float(getattr(rewards_cfg, "recover_dof_ready_k", 8.0))
        self.recover_foot_ready_k = float(getattr(rewards_cfg, "recover_foot_ready_k", 10.0))
        self.recover_ball_body_dist_k = float(
            getattr(rewards_cfg, "recover_ball_body_dist_k", 30.0)
        )
        self.recover_ball_body_dist_min = float(
            getattr(rewards_cfg, "recover_ball_body_dist_min", 0.30)
        )
        self.recover_ball_body_dist_max = float(
            getattr(rewards_cfg, "recover_ball_body_dist_max", 0.50)
        )
        self.recover_ball_front_x_min = float(
            getattr(rewards_cfg, "recover_ball_front_x_min", 0.10)
        )
        self.default_root_height = float(
            getattr(rewards_cfg, "recover_default_root_height", self.config.robot.init_state.pos[2])
        )

        joint_weights_cfg = getattr(rewards_cfg, "recover_joint_weights", {})
        self.recover_joint_weights = torch.ones(
            self.num_dof, dtype=torch.float, device=self.device
        )
        if joint_weights_cfg:
            for i, joint_name in enumerate(self.dof_names):
                if joint_name in joint_weights_cfg:
                    self.recover_joint_weights[i] = float(joint_weights_cfg[joint_name])
        self.recover_joint_weight_sum = torch.clamp(self.recover_joint_weights.sum(), min=1.0)

        self.recover_foot_ready_center = torch.tensor(
            getattr(rewards_cfg, "recover_foot_ready_center", [0.0, 0.0, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.recover_foot_ready_weight = torch.tensor(
            getattr(rewards_cfg, "recover_foot_ready_weight", [6.0, 6.0, 4.0]),
            dtype=torch.float,
            device=self.device,
        )
        right_toe_pos, _ = self._get_right_toe_state()
        default_base_pos = self.simulator.robot_root_states[:, 0:3]
        self.default_right_toe_rel_heading = quat_rotate(
            self._get_base_heading_quat_inv(),
            right_toe_pos - default_base_pos,
        ).mean(dim=0)
        if not bool(torch.any(self.recover_foot_ready_center).item()):
            self.recover_foot_ready_center = self.default_right_toe_rel_heading.clone()

        self._last_reposition_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._last_pre_touch_reposition_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_post_recover_reposition_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_target_state_zone_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_contact_bonus = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._last_wrong_body_contact_penalty = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_reposition_contact_violation_penalty = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_invalid_stance_penalty = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_control_touch_foot_ball_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_foot_to_ball_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_horizontal_vel_penalty = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_overshoot_penalty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._last_missed_target_state_cycle_penalty = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_post_contact_upward_vel_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_post_contact_height_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_post_contact_recover_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_base_upright = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_base_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_base_vel_xy = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_dof_ready = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_foot_ready = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_recover_ball_body_dist = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _get_base_heading_quat_inv(self):
        base_quat = quat_unit(self.base_quat.clone())
        ref_dir = torch.zeros_like(base_quat[:, 0:3])
        ref_dir[:, 0] = 1.0
        rot_dir = quat_rotate(base_quat, ref_dir)
        heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
        axis = torch.zeros_like(base_quat[:, 0:3])
        axis[:, 2] = 1.0
        return quat_from_angle_axis(-heading, axis)

    def _get_ball_rel_base_heading(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        heading_inv = self._get_base_heading_quat_inv()
        return quat_rotate(heading_inv, self.ball_pos - base_pos)

    def _get_ball_rel_base_heading_xy(self):
        return self._get_ball_rel_base_heading()[:, :2]

    def _get_ball_vel_rel_base_heading(self):
        heading_inv = self._get_base_heading_quat_inv()
        return quat_rotate(heading_inv, self.ball_lin_vel)

    def _get_ball_control_position_zone_mask(self):
        ball_rel_xy = self._get_ball_rel_base_heading_xy()
        ball_z_world = self.ball_pos[:, 2]
        return (
            (ball_rel_xy[:, 0] >= self.control_zone_x_range[0])
            & (ball_rel_xy[:, 0] <= self.control_zone_x_range[1])
            & (ball_rel_xy[:, 1] >= self.control_zone_y_range[0])
            & (ball_rel_xy[:, 1] <= self.control_zone_y_range[1])
            & (ball_z_world >= self.control_zone_z_range[0])
            & (ball_z_world <= self.control_zone_z_range[1])
        )

    def _get_ball_target_state_zone_mask(self):
        return self._get_ball_control_position_zone_mask() & (
            self.ball_lin_vel[:, 2] >= self.control_zone_vz_min
        )

    def _get_ball_control_zone_mask(self):
        return self._get_ball_control_position_zone_mask()

    def _get_ball_target_state_coords(self):
        ball_rel_xy = self._get_ball_rel_base_heading_xy()
        return torch.cat(
            [
                ball_rel_xy,
                self.ball_pos[:, 2:3],
            ],
            dim=-1,
        )

    def _compute_ball_control_reward_logs(self):
        ball_vel_rel = self._get_ball_vel_rel_base_heading()
        ball_horizontal_speed = torch.norm(ball_vel_rel[:, :2], dim=-1)
        ball_vz = self.ball_lin_vel[:, 2]
        ball_z_world = self.ball_pos[:, 2]
        position_zone = self._get_ball_control_position_zone_mask()
        target_state_zone = self._get_ball_target_state_zone_mask()
        overshoot_height = torch.relu(ball_z_world - self.control_zone_z_max)
        self.log_dict["ball_control/ball_horizontal_speed_mean"] = ball_horizontal_speed.mean()
        self.log_dict["ball_control/ball_in_control_position_zone_rate"] = position_zone.float().mean()
        self.log_dict["ball_control/ball_in_target_state_zone_rate"] = target_state_zone.float().mean()
        self.log_dict["ball_control/ball_vz_mean"] = ball_vz.mean()
        self.log_dict["ball_control/ball_vz_ok_rate"] = (ball_vz >= self.control_zone_vz_min).float().mean()
        self.log_dict["ball_control/overshoot_rate"] = (overshoot_height > 0.0).float().mean()
        self.log_dict["ball_control/overshoot_height_mean"] = overshoot_height.mean()
        self.log_dict["ball_control/missed_target_state_cycle_rate"] = (
            self.just_started_control_touch & (~self.entered_target_state_zone_since_contact)
        ).float().mean()
        active_post_contact = self.post_contact_steps > 0
        active_post_contact_count = active_post_contact.float().sum().clamp(min=1.0)
        self.log_dict["ball_control/post_contact_max_height_mean"] = torch.where(
            active_post_contact,
            self.post_contact_max_height,
            torch.zeros_like(self.post_contact_max_height),
        ).sum() / active_post_contact_count
        self.log_dict["ball_control/post_contact_vz_mean"] = torch.where(
            active_post_contact,
            self.post_contact_max_vz,
            torch.zeros_like(self.post_contact_max_vz),
        ).sum() / active_post_contact_count
        self.log_dict["ball_control/post_contact_horizontal_speed_mean"] = torch.where(
            active_post_contact,
            ball_horizontal_speed,
            torch.zeros_like(ball_horizontal_speed),
        ).sum() / active_post_contact_count
        self.log_dict["ball_control/post_contact_height_hit_rate"] = (
            active_post_contact & (self.post_contact_max_height >= self.post_contact_height_min)
        ).float().sum() / active_post_contact_count

    def _apply_valid_stance_gate(self, reward):
        return reward * self.valid_stance_mask.float()

    def _reward_reposition_zone_reward(self):
        ball_target_state_coords = self._get_ball_target_state_coords()
        weighted_error = (ball_target_state_coords - self.reposition_zone_reward_target.unsqueeze(0)) ** 2
        base_reward = torch.exp(-torch.sum(weighted_error * self.reposition_zone_reward_weight, dim=-1))
        pre_touch_reward = base_reward * self.pre_touch_reposition_mask.float()
        post_recover_reward = (
            base_reward
            * self.post_touch_recover_mask.float()
            * self.recover_reposition_zone_scale
        )
        reward = pre_touch_reward + post_recover_reward
        reward = self._apply_valid_stance_gate(reward)
        self._last_reposition_reward = reward.detach()
        self._last_pre_touch_reposition_reward = self._apply_valid_stance_gate(
            pre_touch_reward
        ).detach()
        self._last_post_recover_reposition_reward = self._apply_valid_stance_gate(
            post_recover_reward
        ).detach()
        self.log_dict["ball_control/reposition_reward_mean"] = self._last_reposition_reward.mean()
        self.log_dict["ball_control/pre_touch_reposition_zone_reward"] = (
            self._last_pre_touch_reposition_reward.mean()
        )
        self.log_dict["ball_control/post_recover_reposition_zone_reward"] = (
            self._last_post_recover_reposition_reward.mean()
        )
        self.log_dict["ball_control/recover_reposition_zone_scale"] = torch.tensor(
            self.recover_reposition_zone_scale,
            dtype=torch.float,
            device=self.device,
        )
        return reward

    def _reward_ball_in_target_state_zone_reward(self):
        reward = self._get_ball_target_state_zone_mask().float()
        reward = self._apply_valid_stance_gate(reward)
        self._last_target_state_zone_reward = reward.detach()
        return reward

    def _reward_control_touch_contact_bonus(self):
        reward = self.just_target_contact.float() * self.control_touch_contact_bonus
        reward = self._apply_valid_stance_gate(reward)
        self._last_contact_bonus = reward.detach()
        return reward

    def _reward_wrong_body_contact_penalty(self):
        penalty = self.just_wrong_body_contact.float() * self.wrong_body_contact_penalty_value
        self._last_wrong_body_contact_penalty = penalty.detach()
        return penalty

    def _reward_reposition_contact_violation_penalty(self):
        penalty = (
            self.just_reposition_contact_violation.float()
            * self.reposition_contact_violation_penalty_value
        )
        self._last_reposition_contact_violation_penalty = penalty.detach()
        return penalty

    def _reward_invalid_stance_penalty(self):
        penalty = (~self.valid_stance_mask).float() * self.invalid_stance_penalty_value
        self._last_invalid_stance_penalty = penalty.detach()
        return penalty

    def _reward_control_touch_foot_to_ball_reward(self):
        right_toe_pos, _ = self._get_right_toe_state()
        foot_to_ball_dist = torch.norm(self.ball_pos - right_toe_pos, dim=-1)
        entered_control_touch = self.just_started_control_touch
        if torch.any(entered_control_touch):
            self.prev_control_touch_foot_ball_dist[entered_control_touch] = foot_to_ball_dist[
                entered_control_touch
            ]

        active_pre_contact_mask = (
            (self.control_mode == self.CONTROL_TOUCH)
            & (~self.just_target_contact)
            & (~entered_control_touch)
        )
        progress = torch.relu(self.prev_control_touch_foot_ball_dist - foot_to_ball_dist)
        reward = progress * active_pre_contact_mask.float()
        reward = self._apply_valid_stance_gate(reward)

        update_prev_mask = (self.control_mode == self.CONTROL_TOUCH) & (~self.just_target_contact)
        self.prev_control_touch_foot_ball_dist[update_prev_mask] = foot_to_ball_dist[update_prev_mask]
        self._last_foot_to_ball_reward = reward.detach()
        self.log_dict["ball_control/foot_ball_dist_mean"] = foot_to_ball_dist.mean()
        self.log_dict["ball_control/foot_ball_approach_progress_mean"] = progress.mean()
        return reward

    def _reward_ball_horizontal_vel_penalty(self):
        ball_vel_rel = self._get_ball_vel_rel_base_heading()
        ball_horizontal_speed = torch.norm(ball_vel_rel[:, :2], dim=-1)
        penalty = torch.relu(ball_horizontal_speed - self.ball_horizontal_speed_penalty_tol)
        penalty = penalty * self.ball_horizontal_vel_penalty_weight
        if self.is_kickup_task_mode:
            penalty = penalty * self.has_target_contact.float()
        self._last_horizontal_vel_penalty = penalty.detach()
        return penalty

    def _reward_ball_overshoot_penalty(self):
        penalty = torch.relu(self.ball_pos[:, 2] - self.control_zone_z_max)
        if self.is_kickup_task_mode:
            penalty = penalty * self.has_target_contact.float()
        self._last_overshoot_penalty = penalty.detach()
        return penalty

    def _reward_missed_target_state_cycle_penalty(self):
        penalty = (
            self.just_started_control_touch & (~self.entered_target_state_zone_since_contact)
        ).float() * self.missed_target_state_cycle_penalty_value
        self._last_missed_target_state_cycle_penalty = penalty.detach()
        return penalty

    def _reward_post_contact_upward_vel_reward(self):
        capped_vz = torch.clamp(self.post_contact_max_vz, max=self.post_contact_vz_max)
        upward_vz = torch.relu(capped_vz - self.post_contact_vz_min)
        reward = upward_vz * self.has_target_contact.float()
        reward = self._apply_valid_stance_gate(reward)
        self._last_post_contact_upward_vel_reward = reward.detach()
        return reward

    def _reward_post_contact_height_reward(self):
        capped_height = torch.clamp(self.post_contact_max_height, max=self.post_contact_height_max)
        reward = torch.relu(capped_height - self.post_contact_height_min)
        reward = reward * self.has_target_contact.float()
        reward = self._apply_valid_stance_gate(reward)
        self._last_post_contact_height_reward = reward.detach()
        return reward

    def _reward_post_contact_recover_reward(self):
        recover_mask = self.post_touch_recover_mask.float()
        gravity_xy_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        base_upright = torch.exp(-self.recover_base_upright_k * gravity_xy_sq)

        root_height = self.simulator.robot_root_states[:, 2]
        base_height = torch.exp(
            -self.recover_base_height_k * torch.abs(root_height - self.default_root_height)
        )

        base_vel_xy_sq = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=-1)
        base_vel_xy = torch.exp(-self.recover_base_vel_k * base_vel_xy_sq)

        dof_err_sq = torch.square(self.simulator.dof_pos - self.default_dof_pos)
        weighted_dof_err = torch.sum(dof_err_sq * self.recover_joint_weights.unsqueeze(0), dim=-1)
        weighted_dof_err = weighted_dof_err / self.recover_joint_weight_sum
        dof_ready = torch.exp(-self.recover_dof_ready_k * weighted_dof_err)

        right_toe_pos, _ = self._get_right_toe_state()
        foot_rel_heading = quat_rotate(
            self._get_base_heading_quat_inv(),
            right_toe_pos - self.simulator.robot_root_states[:, 0:3],
        )
        foot_err_sq = torch.square(foot_rel_heading - self.recover_foot_ready_center.unsqueeze(0))
        foot_ready = torch.exp(
            -self.recover_foot_ready_k
            * torch.sum(foot_err_sq * self.recover_foot_ready_weight.unsqueeze(0), dim=-1)
        )
        ball_rel_xy = self._get_ball_rel_base_heading_xy()
        ball_dist_xy = torch.norm(ball_rel_xy, dim=-1)
        ball_dist_err = torch.square(
            torch.relu(self.recover_ball_body_dist_min - ball_dist_xy)
        ) + torch.square(torch.relu(ball_dist_xy - self.recover_ball_body_dist_max))
        ball_front_mask = (ball_rel_xy[:, 0] >= self.recover_ball_front_x_min).float()
        ball_body_dist = torch.exp(-self.recover_ball_body_dist_k * ball_dist_err) * ball_front_mask

        reward = (
            self.recover_base_upright_scale * base_upright
            + self.recover_base_height_scale * base_height
            + self.recover_base_vel_scale * base_vel_xy
            + self.recover_dof_ready_scale * dof_ready
            + self.recover_foot_ready_scale * foot_ready
            + self.recover_ball_body_dist_scale * ball_body_dist
        )
        reward = reward * recover_mask * self.post_contact_recover_reward_scale
        reward = self._apply_valid_stance_gate(reward)

        self._last_recover_base_upright = (base_upright * recover_mask).detach()
        self._last_recover_base_height = (base_height * recover_mask).detach()
        self._last_recover_base_vel_xy = (base_vel_xy * recover_mask).detach()
        self._last_recover_dof_ready = (dof_ready * recover_mask).detach()
        self._last_recover_foot_ready = (foot_ready * recover_mask).detach()
        self._last_recover_ball_body_dist = (ball_body_dist * recover_mask).detach()
        self._last_post_contact_recover_reward = reward.detach()

        self.log_dict["ball_control/post_contact_recover_reward"] = (
            self._last_post_contact_recover_reward.mean()
        )
        self.log_dict["ball_control/recover_base_upright"] = (
            self._last_recover_base_upright.mean()
        )
        self.log_dict["ball_control/recover_base_height"] = self._last_recover_base_height.mean()
        self.log_dict["ball_control/recover_base_vel_xy"] = (
            self._last_recover_base_vel_xy.mean()
        )
        self.log_dict["ball_control/recover_dof_ready"] = self._last_recover_dof_ready.mean()
        self.log_dict["ball_control/recover_foot_ready"] = self._last_recover_foot_ready.mean()
        self.log_dict["ball_control/recover_ball_body_dist"] = (
            self._last_recover_ball_body_dist.mean()
        )
        return reward

    def _reward_penalty_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

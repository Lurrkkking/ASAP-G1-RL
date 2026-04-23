import torch

from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_rotate, quat_unit


class BallControlRewardMixin:
    def _init_ball_control_rewards(self):
        self.reposition_zone_reward_target = torch.tensor(
            getattr(self.config.rewards, "reposition_zone_reward_target", [0.45, 0.0, 0.90]),
            dtype=torch.float,
            device=self.device,
        )
        self.reposition_zone_reward_weight = torch.tensor(
            getattr(self.config.rewards, "reposition_zone_reward_weight", [4.0, 4.0, 2.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_x_range = torch.tensor(
            getattr(self.config.rewards, "control_zone_x_range", [0.25, 0.65]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_y_range = torch.tensor(
            getattr(self.config.rewards, "control_zone_y_range", [-0.25, 0.25]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_z_range = torch.tensor(
            getattr(self.config.rewards, "control_zone_z_range", [0.65, 1.15]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_vz_min = float(getattr(self.config.rewards, "control_zone_vz_min", 0.0))
        self.control_zone_z_max = float(self.control_zone_z_range[1].item())
        self.control_touch_contact_bonus = float(
            getattr(self.config.rewards, "control_touch_contact_bonus", 1.0)
        )
        self.wrong_body_contact_penalty_value = float(
            getattr(self.config.rewards, "wrong_body_contact_penalty", 1.0)
        )
        self.control_touch_foot_to_ball_alpha = float(
            getattr(self.config.rewards, "control_touch_foot_to_ball_alpha", 4.0)
        )
        self.ball_horizontal_vel_penalty_weight = float(
            getattr(self.config.rewards, "ball_horizontal_vel_penalty", 0.25)
        )
        self.missed_target_state_cycle_penalty_value = float(
            getattr(self.config.rewards, "missed_target_state_cycle_penalty", 1.0)
        )

        self._last_reposition_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._last_target_state_zone_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._last_contact_bonus = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._last_wrong_body_contact_penalty = torch.zeros(
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

    def _reward_reposition_zone_reward(self):
        reposition_mask = (self.control_mode == self.REPOSITION).float()
        ball_target_state_coords = self._get_ball_target_state_coords()
        weighted_error = (ball_target_state_coords - self.reposition_zone_reward_target.unsqueeze(0)) ** 2
        reward = torch.exp(-torch.sum(weighted_error * self.reposition_zone_reward_weight, dim=-1))
        reward = reward * reposition_mask
        self._last_reposition_reward = reward.detach()
        self.log_dict["ball_control/reposition_reward_mean"] = self._last_reposition_reward.mean()
        return reward

    def _reward_ball_in_target_state_zone_reward(self):
        reward = self._get_ball_target_state_zone_mask().float()
        self._last_target_state_zone_reward = reward.detach()
        return reward

    def _reward_control_touch_contact_bonus(self):
        reward = self.just_target_contact.float() * self.control_touch_contact_bonus
        self._last_contact_bonus = reward.detach()
        return reward

    def _reward_wrong_body_contact_penalty(self):
        penalty = self.just_wrong_body_contact.float() * self.wrong_body_contact_penalty_value
        self._last_wrong_body_contact_penalty = penalty.detach()
        return penalty

    def _reward_control_touch_foot_to_ball_reward(self):
        control_touch_mask = (self.control_mode == self.CONTROL_TOUCH).float()
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        foot_to_ball_dist = torch.norm(self.ball_pos - right_foot_pos, dim=-1)
        reward = torch.exp(-self.control_touch_foot_to_ball_alpha * foot_to_ball_dist)
        reward = reward * control_touch_mask
        self._last_foot_to_ball_reward = reward.detach()
        self.log_dict["ball_control/control_touch_foot_to_ball_dist_mean"] = foot_to_ball_dist.mean()
        return reward

    def _reward_ball_horizontal_vel_penalty(self):
        ball_vel_rel = self._get_ball_vel_rel_base_heading()
        penalty = torch.norm(ball_vel_rel[:, :2], dim=-1) * self.ball_horizontal_vel_penalty_weight
        self._last_horizontal_vel_penalty = penalty.detach()
        return penalty

    def _reward_ball_overshoot_penalty(self):
        penalty = torch.relu(self.ball_pos[:, 2] - self.control_zone_z_max)
        self._last_overshoot_penalty = penalty.detach()
        return penalty

    def _reward_missed_target_state_cycle_penalty(self):
        penalty = (
            self.just_started_control_touch & (~self.entered_target_state_zone_since_contact)
        ).float() * self.missed_target_state_cycle_penalty_value
        self._last_missed_target_state_cycle_penalty = penalty.detach()
        return penalty

    def _reward_penalty_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

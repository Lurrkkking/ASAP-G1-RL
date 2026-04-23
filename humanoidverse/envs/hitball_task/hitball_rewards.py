import torch

from humanoidverse.utils.torch_utils import quat_rotate
from isaac_utils.rotations import (
    calc_heading_quat_inv,
    quat_conjugate,
    quat_mul,
    quat_to_angle_axis,
)


class HitBallRewardMixin:
    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _get_ball_pos_rel_base_heading(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        ball_rel_world = self.ball_pos - base_pos
        heading_inv = calc_heading_quat_inv(self.base_quat.clone(), w_last=True)
        return quat_rotate(heading_inv, ball_rel_world)

    def _get_prep_to_swing_ready_mask(self):
        ball_rel_base = self._get_ball_pos_rel_base_heading()
        xy_error = torch.abs(ball_rel_base[:, :2] - self.prep_target_xy.unsqueeze(0))
        xy_ready = torch.all(xy_error <= self.prep_xy_tolerance.unsqueeze(0), dim=-1)
        if not self.prep_to_swing_require_z_valid:
            return xy_ready
        z_valid = (
            (self.ball_pos[:, 2] >= self.prep_min_ball_height)
            & (self.ball_pos[:, 2] <= self.prep_max_ball_height)
        )
        return xy_ready & z_valid

    def _reward_pre_contact_prep_xy_alignment(self):
        ball_rel_base = self._get_ball_pos_rel_base_heading()
        xy_error = ball_rel_base[:, :2] - self.prep_target_xy.unsqueeze(0)
        weighted_sq_error = torch.sum(self.pre_contact_prep_xy_weight.unsqueeze(0) * (xy_error ** 2), dim=-1)
        rew = torch.exp(-self.pre_contact_prep_xy_alpha * weighted_sq_error)
        return rew * (self.task_phase == self.PRE_CONTACT_PREP).float()

    def _reward_pre_contact_swing_alignment(self):
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        target_point = right_foot_pos + self.pre_contact_swing_target_offset.unsqueeze(0)
        dist = torch.norm(self.ball_pos - target_point, dim=-1)
        rew = torch.exp(-4.0 * dist)
        return rew * (self.task_phase == self.PRE_CONTACT_SWING).float()

    def _reward_first_right_foot_contact_bonus(self):
        return self.just_target_contact.float()

    def _reward_wrong_body_contact_penalty(self):
        wrong_contact = self.has_first_contact & (~self.first_contact_is_target)
        return -wrong_contact.float()

    def _reward_post_contact_upward_vel(self):
        vz = self.ball_lin_vel[:, 2]
        center = 0.5 * (self.ball_target_upward_vel_min + self.ball_target_upward_vel_max)
        half_width = 0.5 * (self.ball_target_upward_vel_max - self.ball_target_upward_vel_min)
        score = 1.0 - torch.abs(vz - center) / max(half_width, 1e-6)
        score = torch.clamp(score, min=0.0)
        active = self.has_first_contact & self.first_contact_is_target
        return score * active.float()

    def _reward_ball_horizontal_vel_penalty(self):
        xy_speed = torch.norm(self.ball_lin_vel[:, :2], dim=-1)
        penalty = torch.clamp(xy_speed - self.ball_horizontal_vel_limit, min=0.0)
        active = self.has_first_contact & self.first_contact_is_target
        return -(penalty * active.float())

    def _reward_post_contact_ball_height(self):
        apex = self.post_contact_ball_max_height
        target_min = self.success_ball_height_min
        target_max = self.success_ball_height_max
        target_center = float(getattr(self.config.rewards, "ball_target_height", 0.5 * (target_min + target_max)))
        half_width = max(0.5 * (target_max - target_min), 1e-6)
        score = 1.0 - torch.abs(apex - target_center) / half_width
        score = torch.clamp(score, min=0.0, max=1.0)
        active = self.has_first_contact & self.first_contact_is_target
        return score * active.float()

    def _reward_success_bonus(self):
        return self.just_became_success.float()

    def _reward_tracking_joint_position_lower_body(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_lower_body_dof_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        joint_diff = (
            self.ref_dof_pos[:, self.tracking_lower_body_dof_indices]
            - self.simulator.dof_pos[:, self.tracking_lower_body_dof_indices]
        )
        joint_err = torch.mean(joint_diff ** 2, dim=-1)
        rew = torch.exp(-joint_err / max(self.tracking_joint_pos_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

    def _reward_tracking_joint_position_right_ankle(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_right_ankle_dof_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        ankle_diff = (
            self.ref_dof_pos[:, self.tracking_right_ankle_dof_indices]
            - self.simulator.dof_pos[:, self.tracking_right_ankle_dof_indices]
        )
        ankle_err = torch.mean(ankle_diff ** 2, dim=-1)
        rew = torch.exp(-ankle_err / max(self.tracking_right_ankle_pos_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

    def _reward_tracking_joint_position_upper_body(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_upper_body_dof_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        joint_diff = (
            self.ref_dof_pos[:, self.tracking_upper_body_dof_indices]
            - self.simulator.dof_pos[:, self.tracking_upper_body_dof_indices]
        )
        joint_err = torch.mean(joint_diff ** 2, dim=-1)
        rew = torch.exp(-joint_err / max(self.tracking_upper_body_joint_pos_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

    def _reward_tracking_body_position_feet(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_feet_body_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        body_ids = self.tracking_feet_body_indices
        curr_local_pos = self._heading_local_body_pos(
            self.simulator._rigid_body_pos[:, body_ids],
            self.simulator.robot_root_states[:, 0:3],
            self.base_quat,
        )
        ref_local_pos = self._heading_local_body_pos(
            self.ref_body_pos[:, body_ids],
            self.ref_root_pos,
            self.ref_root_rot,
        )
        feet_err = torch.mean((ref_local_pos - curr_local_pos) ** 2, dim=(-1, -2))
        rew = torch.exp(-feet_err / max(self.tracking_feet_pos_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

    def _reward_tracking_body_position_upper_body(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_upper_body_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        body_ids = self.tracking_upper_body_indices
        curr_local_pos = self._heading_local_body_pos(
            self.simulator._rigid_body_pos[:, body_ids],
            self.simulator.robot_root_states[:, 0:3],
            self.base_quat,
        )
        ref_local_pos = self._heading_local_body_pos(
            self.ref_body_pos[:, body_ids],
            self.ref_root_pos,
            self.ref_root_rot,
        )
        upper_err = torch.mean((ref_local_pos - curr_local_pos) ** 2, dim=(-1, -2))
        rew = torch.exp(-upper_err / max(self.tracking_upper_body_pos_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

    def _reward_tracking_body_rotation_trunk(self):
        if (not self.hitball_motion_prior_enabled) or len(self.tracking_trunk_body_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        body_ids = self.tracking_trunk_body_indices
        curr_local_rot = self._heading_local_body_rot(
            self.simulator._rigid_body_rot[:, body_ids],
            self.base_quat,
        )
        ref_local_rot = self._heading_local_body_rot(
            self.ref_body_rot[:, body_ids],
            self.ref_root_rot,
        )
        rot_diff = quat_mul(ref_local_rot, quat_conjugate(curr_local_rot, w_last=True), w_last=True)
        rot_diff = rot_diff / torch.norm(rot_diff, dim=-1, keepdim=True).clamp(min=1e-6)
        angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = torch.mean(angle ** 2, dim=-1)
        rew = torch.exp(-rot_err / max(self.tracking_trunk_rot_sigma, 1e-6))
        return rew * self.hitball_tracking_phase_gate

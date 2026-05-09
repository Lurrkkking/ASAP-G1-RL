import torch
from humanoidverse.utils.torch_utils import quat_rotate

from humanoidverse.envs.ball_control_task.ball_control_reset import (
    BallControlResetMixin,
)
from humanoidverse.envs.ball_control_task.ball_control_rewards import (
    BallControlRewardMixin,
)
from humanoidverse.envs.ball_control_task.ball_control_state_machine import (
    BallControlStateMachineMixin,
)
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class BallControlTask(
    BallControlResetMixin,
    BallControlRewardMixin,
    BallControlStateMachineMixin,
    LeggedRobotBase,
):
    """Minimal ball-control task shell.

    This shell intentionally keeps the task policy-free: two high-level modes,
    no motion prior, and a zero task reward.
    """

    REPOSITION = 0
    CONTROL_TOUCH = 1

    TERM_NONE = 0
    TERM_BALL_GROUND = 1
    TERM_BASE_ENV = 2

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _init_buffers(self):
        super()._init_buffers()

        if not hasattr(self.simulator, "cubeA_state"):
            raise AttributeError(
                "BallControlTask expects a multi-actor simulator with `cubeA_state`. "
                "Use the same HOI-style IsaacGym simulator setup as hitball."
            )

        self.ball_root_states = self.simulator.cubeA_state
        self.ball_pos = self.ball_root_states[:, 0:3]
        self.ball_quat = self.ball_root_states[:, 3:7]
        self.ball_lin_vel = self.ball_root_states[:, 7:10]
        self.ball_ang_vel = self.ball_root_states[:, 10:13]
        self.task_mode = str(getattr(self.config, "task_mode", "ball_control")).lower()
        self.is_kickup_task_mode = self.task_mode == "kickup"

        self.right_foot_body_name = getattr(
            self.config,
            "target_contact_body",
            getattr(self.config.robot, "right_foot_name", None),
        )
        if self.right_foot_body_name is None:
            raise AttributeError(
                "BallControlTask requires `target_contact_body` or `config.robot.right_foot_name`."
            )
        self.right_foot_body_idx = self.simulator.find_rigid_body_indice(self.right_foot_body_name)
        self.left_foot_body_name = getattr(self.config.robot, "left_foot_name", None)
        if self.left_foot_body_name is None:
            raise AttributeError("BallControlTask requires `config.robot.left_foot_name`.")
        self.left_foot_body_idx = self.simulator.find_rigid_body_indice(self.left_foot_body_name)
        self.right_toe_local_offset = torch.tensor(
            getattr(self.config, "right_toe_local_offset", [0.12, 0.0, -0.02]),
            dtype=torch.float,
            device=self.device,
        )
        self.left_toe_local_offset = torch.tensor(
            getattr(self.config, "left_toe_local_offset", [0.12, 0.0, -0.02]),
            dtype=torch.float,
            device=self.device,
        )

        self.ball_init_offset = torch.tensor(
            getattr(self.config, "ball_init_offset", [0.48, 0.0, 0.75]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_pos_noise = torch.tensor(
            getattr(self.config, "ball_init_pos_noise", [0.005, 0.005, 0.005]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel = torch.tensor(
            getattr(self.config, "ball_init_lin_vel", [0.0, 0.0, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel_noise = torch.tensor(
            getattr(self.config, "ball_init_lin_vel_noise", [0.0, 0.0, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_ground_height = float(
            getattr(
                self.config.termination,
                "ball_ground_height",
                getattr(self.config.ball, "radius", 0.10) + 0.02,
            )
        )
        terminate_on_ball_ground = bool(
            getattr(self.config.termination, "terminate_on_ball_ground", False)
        )
        self.enable_ball_ground_termination = bool(
            getattr(
                self.config.termination,
                "enable_ball_ground_termination",
                terminate_on_ball_ground,
            )
        )
        self.terminate_on_ball_ground = self.enable_ball_ground_termination
        self.reset_pose_right_foot_offset = (
            self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
            - self.simulator.robot_root_states[:, 0:3]
        ).mean(dim=0)
        self.term_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.has_target_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.post_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.post_contact_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.post_contact_max_vz = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.control_target_center = torch.tensor(
            getattr(self.config, "control_target_center", [0.45, 0.0, 0.90]),
            dtype=torch.float,
            device=self.device,
        )
        self.valid_stance_base_height_min = float(
            getattr(self.config, "valid_stance_base_height_min", 0.55)
        )
        self.valid_stance_roll_pitch_max = float(
            getattr(self.config, "valid_stance_roll_pitch_max", 0.50)
        )
        self.valid_stance_nonfoot_contact_check = bool(
            getattr(self.config, "valid_stance_nonfoot_contact_check", False)
        )
        self.nonfoot_contact_force_threshold = float(
            getattr(
                self.config,
                "nonfoot_contact_force_threshold",
                getattr(self.config, "contact_force_threshold", 1.0),
            )
        )
        self.valid_stance_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.nonfoot_contact_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._init_ball_control_state_machine()
        self._init_ball_control_rewards()

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self._reset_ball_control_state_machine(env_ids)
        self.prev_control_touch_foot_ball_dist[env_ids] = 0.0
        self.term_reason[env_ids] = self.TERM_NONE
        self.has_target_contact[env_ids] = False
        self.post_contact_steps[env_ids] = 0
        self.post_contact_max_height[env_ids] = 0.0
        self.post_contact_max_vz[env_ids] = 0.0

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._update_ball_control_state_machine()
        self._publish_ball_control_extras()

    def _check_termination(self):
        super()._check_termination()

        base_reset = self.reset_buf.bool().clone()
        base_reset &= self.term_reason == self.TERM_NONE
        self.term_reason[base_reset] = self.TERM_BASE_ENV

        if self.terminate_on_ball_ground:
            ball_ground = self.ball_pos[:, 2] <= self.ball_ground_height
            self._mark_termination(ball_ground, self.TERM_BALL_GROUND)

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

    def _get_nonfoot_contact_mask(self):
        if not self.valid_stance_nonfoot_contact_check:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot_body_forces = torch.norm(self.simulator.contact_forces[:, : self.num_bodies], dim=-1)
        nonfoot_mask = torch.ones_like(robot_body_forces, dtype=torch.bool)
        nonfoot_mask[:, self.feet_indices] = False
        body_force_max = robot_body_forces.masked_fill(~nonfoot_mask, 0.0).max(dim=1).values
        return body_force_max > self.nonfoot_contact_force_threshold

    def _get_valid_stance_mask(self):
        base_height = self.simulator.robot_root_states[:, 2]
        base_roll = self._wrap_to_pi(self.rpy[:, 0])
        base_pitch = self._wrap_to_pi(self.rpy[:, 1])

        height_ok = base_height >= self.valid_stance_base_height_min
        roll_ok = torch.abs(base_roll) <= self.valid_stance_roll_pitch_max
        pitch_ok = torch.abs(base_pitch) <= self.valid_stance_roll_pitch_max

        nonfoot_contact = self._get_nonfoot_contact_mask()
        valid_stance = height_ok & roll_ok & pitch_ok & (~nonfoot_contact)

        self.valid_stance_mask[:] = valid_stance
        self.nonfoot_contact_mask[:] = nonfoot_contact

        self.log_dict["ball_control/valid_stance_rate"] = valid_stance.float().mean()
        self.log_dict["ball_control/invalid_stance_rate"] = (~valid_stance).float().mean()
        self.log_dict["ball_control/base_height_mean"] = base_height.mean()
        self.log_dict["ball_control/base_roll_abs_mean"] = torch.abs(base_roll).mean()
        self.log_dict["ball_control/base_pitch_abs_mean"] = torch.abs(base_pitch).mean()
        if self.valid_stance_nonfoot_contact_check:
            self.log_dict["ball_control/nonfoot_contact_rate"] = nonfoot_contact.float().mean()

        return valid_stance

    def _get_foot_point_state(self, body_idx, local_offset):
        body_pos = self.simulator._rigid_body_pos[:, body_idx]
        body_rot = self.simulator._rigid_body_rot[:, body_idx]
        body_vel = self.simulator._rigid_body_vel[:, body_idx]
        body_ang_vel = self.simulator._rigid_body_ang_vel[:, body_idx]

        offset_world = quat_rotate(body_rot, local_offset.unsqueeze(0).expand(self.num_envs, -1))
        point_pos = body_pos + offset_world
        point_vel = body_vel + torch.cross(body_ang_vel, offset_world, dim=-1)
        return point_pos, point_vel

    def _get_right_toe_state(self):
        return self._get_foot_point_state(self.right_foot_body_idx, self.right_toe_local_offset)

    def _get_left_toe_state(self):
        return self._get_foot_point_state(self.left_foot_body_idx, self.left_toe_local_offset)

    def _publish_ball_control_extras(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        ball_rel_base = self.ball_pos - base_pos
        ball_rel_heading = self._get_ball_rel_base_heading()
        ball_target_state_coords = self._get_ball_target_state_coords()
        target_error = ball_target_state_coords - self.control_target_center.unsqueeze(0)
        valid_stance = self._get_valid_stance_mask()

        self._log_ball_control_state_machine()
        self.log_dict["ball_control/task_mode_is_kickup"] = torch.full(
            (),
            1.0 if self.is_kickup_task_mode else 0.0,
            dtype=torch.float,
            device=self.device,
        )
        self.log_dict["ball_control/term_ball_ground_rate"] = (
            self.term_reason == self.TERM_BALL_GROUND
        ).float().mean()
        self.log_dict["ball_control/term_base_env_rate"] = (
            self.term_reason == self.TERM_BASE_ENV
        ).float().mean()
        self.log_dict["ball_control/ball_height"] = self.ball_pos[:, 2].mean()
        self.log_dict["ball_control/target_error_norm"] = torch.norm(target_error, dim=-1).mean()
        self._compute_ball_control_reward_logs()
        self.extras["ball_control"] = {
            "mode": self.control_mode.clone(),
            "ball_rel_base": ball_rel_base.clone(),
            "ball_rel_heading": ball_rel_heading.clone(),
            "ball_target_state_coords": ball_target_state_coords.clone(),
            "target_error": target_error.clone(),
            "valid_stance": valid_stance.clone(),
            "target_contact": self.just_target_contact.clone(),
            "reposition_contact_violation": self.just_reposition_contact_violation.clone(),
            "wrong_body_contact": self.just_wrong_body_contact.clone(),
            "term_reason": self.term_reason.clone(),
            "has_target_contact": self.has_target_contact.clone(),
            "has_recent_valid_contact": self.has_recent_valid_contact.clone(),
            "time_since_last_target_contact": self.time_since_last_target_contact.clone(),
            "post_contact_steps": self.post_contact_steps.clone(),
            "post_contact_max_height": self.post_contact_max_height.clone(),
            "post_contact_max_vz": self.post_contact_max_vz.clone(),
            "pre_touch_reposition": self.pre_touch_reposition_mask.clone(),
            "post_touch_recover": self.post_touch_recover_mask.clone(),
            "entered_target_state_zone_since_contact": (
                self.entered_target_state_zone_since_contact.clone()
            ),
            "missed_target_state_cycle": self.missed_target_state_cycle_buf.clone(),
            "ball_in_control_position_zone": self._get_ball_control_position_zone_mask().clone(),
            "ball_in_target_state_zone": self._get_ball_target_state_zone_mask().clone(),
        }

    def _get_obs_ball_control_task(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        ball_rel_base = self.ball_pos - base_pos
        ball_vel_rel_heading = self._get_ball_vel_rel_base_heading()
        right_toe_pos, right_toe_vel = self._get_right_toe_state()
        left_toe_pos, left_toe_vel = self._get_left_toe_state()
        ball_rel_right_foot = self.ball_pos - right_toe_pos
        ball_vel_rel_right_foot = self.ball_lin_vel - right_toe_vel
        ball_rel_left_foot = self.ball_pos - left_toe_pos
        ball_vel_rel_left_foot = self.ball_lin_vel - left_toe_vel
        ball_target_state_coords = self._get_ball_target_state_coords()
        target_error = ball_target_state_coords - self.control_target_center.unsqueeze(0)
        mode = self.control_mode.float().unsqueeze(-1)
        valid_stance = self.valid_stance_mask.float().unsqueeze(-1)
        touch_window_progress = torch.clamp(
            self.mode_step_buf.float() / max(float(self.control_touch_window_steps), 1.0),
            0.0,
            1.0,
        ).unsqueeze(-1)

        return torch.cat(
            [
                ball_rel_base,
                ball_vel_rel_heading,
                ball_rel_right_foot,
                ball_vel_rel_right_foot,
                ball_rel_left_foot,
                ball_vel_rel_left_foot,
                target_error,
                mode,
                valid_stance,
                touch_window_progress,
            ],
            dim=-1,
        )

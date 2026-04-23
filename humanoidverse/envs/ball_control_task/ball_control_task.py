import torch

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
        self.reset_pose_right_foot_offset = (
            self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
            - self.simulator.robot_root_states[:, 0:3]
        ).mean(dim=0)
        self.term_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.control_target_center = torch.tensor(
            getattr(self.config, "control_target_center", [0.45, 0.0, 0.90]),
            dtype=torch.float,
            device=self.device,
        )
        self._init_ball_control_state_machine()
        self._init_ball_control_rewards()

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self._reset_ball_control_state_machine(env_ids)
        self.term_reason[env_ids] = self.TERM_NONE

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._update_ball_control_state_machine()
        self._publish_ball_control_extras()

    def _check_termination(self):
        super()._check_termination()

        base_reset = self.reset_buf.bool().clone()
        base_reset &= self.term_reason == self.TERM_NONE
        self.term_reason[base_reset] = self.TERM_BASE_ENV

        ball_ground = self.ball_pos[:, 2] <= self.ball_ground_height
        self._mark_termination(ball_ground, self.TERM_BALL_GROUND)

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

    def _publish_ball_control_extras(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        ball_rel_base = self.ball_pos - base_pos
        ball_rel_heading = self._get_ball_rel_base_heading()
        ball_target_state_coords = self._get_ball_target_state_coords()
        target_error = ball_target_state_coords - self.control_target_center.unsqueeze(0)

        self._log_ball_control_state_machine()
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
            "target_contact": self.just_target_contact.clone(),
            "wrong_body_contact": self.just_wrong_body_contact.clone(),
            "term_reason": self.term_reason.clone(),
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
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        right_foot_vel = self.simulator._rigid_body_vel[:, self.right_foot_body_idx]
        ball_rel_right_foot = self.ball_pos - right_foot_pos
        ball_vel_rel_right_foot = self.ball_lin_vel - right_foot_vel
        ball_target_state_coords = self._get_ball_target_state_coords()
        target_error = ball_target_state_coords - self.control_target_center.unsqueeze(0)
        mode = self.control_mode.float().unsqueeze(-1)

        return torch.cat(
            [
                ball_rel_base,
                ball_vel_rel_heading,
                ball_rel_right_foot,
                ball_vel_rel_right_foot,
                target_error,
                mode,
            ],
            dim=-1,
        )

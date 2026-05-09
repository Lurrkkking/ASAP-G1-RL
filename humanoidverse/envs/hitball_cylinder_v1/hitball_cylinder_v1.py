from isaacgym import gymapi
from pathlib import Path
import torch

from humanoidverse.envs.hitball_cylinder_v1.hitball_cylinder_events import update_cylinder_events
from humanoidverse.envs.hitball_cylinder_v1.hitball_cylinder_obs import compute_cylinder_obs
from humanoidverse.envs.hitball_cylinder_v1.hitball_cylinder_reset import (
    HitBallCylinderResetMixin,
    reset_cylinder_actors,
)
from humanoidverse.envs.hitball_cylinder_v1.hitball_cylinder_rewards import (
    HitBallCylinderRewardMixin,
    compute_cylinder_rewards,
)
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_rotate, quat_unit


class HitBallCylinderV1(
    HitBallCylinderResetMixin,
    HitBallCylinderRewardMixin,
    LeggedRobotBase,
):
    PHASE_WAIT = 0
    PHASE_PREPARE = 1
    PHASE_STRIKE = 2
    PHASE_RECOVER = 3

    TERM_NONE = 0
    TERM_BALL_GROUND = 1
    TERM_BALL_TOO_FAR = 2
    TERM_BASE_ENV = 3
    TERM_NON_TARGET_CONTACT = 4

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _init_buffers(self):
        super()._init_buffers()

        if not hasattr(self.simulator, "cubeA_state"):
            raise AttributeError(
                "HitBallCylinderV1 expects a multi-actor simulator with `cubeA_state`. "
                "Use the same HOI-style IsaacGym simulator setup as ball_control/hitball."
            )

        self.ball_root_states = self.simulator.cubeA_state
        self.ball_pos = self.ball_root_states[:, 0:3]
        self.ball_quat = self.ball_root_states[:, 3:7]
        self.ball_lin_vel = self.ball_root_states[:, 7:10]
        self.ball_ang_vel = self.ball_root_states[:, 10:13]

        self.target_foot_body_name = getattr(
            self.config,
            "target_contact_body",
            getattr(self.config.robot, "right_foot_name", None),
        )
        if self.target_foot_body_name is None:
            raise AttributeError("HitBallCylinderV1 requires target_contact_body or robot.right_foot_name.")
        self.target_foot_body_idx = self.simulator.find_rigid_body_indice(self.target_foot_body_name)
        left_foot_name = getattr(self.config.robot, "left_foot_name", None)
        self.left_foot_body_idx = (
            self.simulator.find_rigid_body_indice(left_foot_name)
            if left_foot_name is not None
            else None
        )

        candidate_body_names = getattr(self.config, "target_contact_candidate_bodies", None)
        if candidate_body_names is None:
            candidate_body_names = [
                name
                for name in self.body_names
                if name == self.target_foot_body_name or name.startswith("right_ankle")
            ]
        if self.target_foot_body_name not in candidate_body_names:
            candidate_body_names = list(candidate_body_names) + [self.target_foot_body_name]
        self.target_foot_candidate_body_names = list(dict.fromkeys(candidate_body_names))
        self.target_foot_candidate_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.target_foot_candidate_body_names],
            dtype=torch.long,
            device=self.device,
        )
        upper_body_terminate_names = getattr(self.config, "wrong_contact_terminate_body_names", None)
        if upper_body_terminate_names is None:
            upper_body_terminate_names = [
                "pelvis",
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ]
        self.wrong_contact_terminate_body_names = [
            name for name in upper_body_terminate_names if name in self.body_names
        ]
        self.wrong_contact_terminate_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.wrong_contact_terminate_body_names],
            dtype=torch.long,
            device=self.device,
        ) if self.wrong_contact_terminate_body_names else torch.zeros(0, dtype=torch.long, device=self.device)
        debug_body_names = getattr(self.config, "debug_non_target_ball_contact_body_names", None)
        if debug_body_names is None:
            debug_body_names = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ]
        self.debug_non_target_ball_contact_body_names = [
            name for name in debug_body_names if name in self.body_names
        ]
        self.debug_non_target_ball_contact_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.debug_non_target_ball_contact_body_names],
            dtype=torch.long,
            device=self.device,
        ) if self.debug_non_target_ball_contact_body_names else torch.zeros(0, dtype=torch.long, device=self.device)
        elbow_proxy_body_names = []
        elbow_proxy_body_indices = []
        elbow_proxy_local_offsets = []
        for proxy_name, body_name in (
            ("left_hand_proxy", "left_elbow_link"),
            ("right_hand_proxy", "right_elbow_link"),
        ):
            if body_name in self.body_names:
                elbow_proxy_body_names.append(proxy_name)
                elbow_proxy_body_indices.append(self.simulator.find_rigid_body_indice(body_name))
                elbow_proxy_local_offsets.append([0.20, 0.0, 0.0])
        self.elbow_proxy_body_names = elbow_proxy_body_names
        self.elbow_proxy_body_indices = torch.tensor(
            elbow_proxy_body_indices,
            dtype=torch.long,
            device=self.device,
        ) if elbow_proxy_body_indices else torch.zeros(0, dtype=torch.long, device=self.device)
        self.elbow_proxy_local_offsets = torch.tensor(
            elbow_proxy_local_offsets,
            dtype=torch.float,
            device=self.device,
        ) if elbow_proxy_local_offsets else torch.zeros(0, 3, dtype=torch.float, device=self.device)
        try:
            self.ball_body_env_idx = self.simulator.gym.get_actor_rigid_body_index(
                self.simulator.envs[0],
                self.simulator._cubeA_id,
                0,
                gymapi.DOMAIN_ENV,
            )
        except Exception:
            self.ball_body_env_idx = self.num_bodies

        self.target_foot_local_offset = torch.tensor(
            getattr(self.config, "target_foot_local_offset", [0.12, 0.0, -0.02]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_offset = torch.tensor(
            getattr(self.config, "ball_init_offset", [0.24, 0.0, 0.30]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_pos_noise = torch.tensor(
            getattr(self.config, "ball_init_pos_noise", [0.02, 0.01, 0.02]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel = torch.tensor(
            getattr(self.config, "ball_init_lin_vel", [0.0, 0.0, -0.6]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel_noise = torch.tensor(
            getattr(self.config, "ball_init_lin_vel_noise", [0.05, 0.03, 0.08]),
            dtype=torch.float,
            device=self.device,
        )
        self.reset_pose_target_foot_offset = (
            self.simulator._rigid_body_pos[:, self.target_foot_body_idx]
            - self.simulator.robot_root_states[:, 0:3]
        ).mean(dim=0)

        self.prediction_horizons = torch.tensor(
            getattr(self.config, "prediction_horizons", [0.25, 0.40, 0.55]),
            dtype=torch.float,
            device=self.device,
        )
        self.strike_gate_horizon_index = int(getattr(self.config, "strike_gate_horizon_index", 1))
        self.strike_gate_horizon_index = max(0, min(self.strike_gate_horizon_index, len(self.prediction_horizons) - 1))

        self.control_zone_center = torch.tensor(
            getattr(self.config, "control_zone_center", [0.38, 0.0, 0.75]),
            dtype=torch.float,
            device=self.device,
        )
        self.control_zone_radius = float(getattr(self.config, "control_zone_radius", 0.22))
        self.control_zone_z_range = torch.tensor(
            getattr(self.config, "control_zone_z_range", [0.45, 1.15]),
            dtype=torch.float,
            device=self.device,
        )
        self.strike_window = torch.tensor(
            getattr(self.config, "strike_window", [[0.02, 0.24], [-0.10, 0.10], [0.02, 0.28]]),
            dtype=torch.float,
            device=self.device,
        )
        self.impact_window = torch.tensor(
            getattr(self.config, "impact_window", [[0.00, 0.26], [-0.13, 0.13], [0.00, 0.32]]),
            dtype=torch.float,
            device=self.device,
        )
        self.strike_gate_vz_max = float(getattr(self.config, "strike_gate_vz_max", -0.05))
        self.strike_cooldown_steps = int(getattr(self.config, "strike_cooldown_steps", 8))
        self.max_contact_steps = int(getattr(self.config, "max_contact_steps", 5))
        self.prepare_min_steps = int(getattr(self.config, "prepare_min_steps", 3))
        self.prepare_max_steps = int(getattr(self.config, "prepare_max_steps", 18))
        self.strike_max_steps = int(getattr(self.config, "strike_max_steps", 10))
        self.recover_min_steps = int(getattr(self.config, "recover_min_steps", 10))
        self.phase_gravity_xy_max = float(getattr(self.config, "phase_gravity_xy_max", 0.45))
        self.phase_root_height_min = float(getattr(self.config, "phase_root_height_min", 0.55))
        self.phase_base_vel_xy_max = float(getattr(self.config, "phase_base_vel_xy_max", 1.0))
        self.near_foot_thresh = float(getattr(self.config, "near_foot_thresh", 0.16))
        self.carry_rel_speed_thresh = float(getattr(self.config, "carry_rel_speed_thresh", 0.35))
        self.carry_foot_up_thresh = float(getattr(self.config, "carry_foot_up_thresh", 0.20))
        self.carry_ball_up_thresh = float(getattr(self.config, "carry_ball_up_thresh", 0.15))
        self.max_near_foot_steps = int(getattr(self.config, "max_near_foot_steps", 8))
        self.release_dist_thresh = float(getattr(self.config, "release_dist_thresh", 0.16))
        self.release_rel_speed_thresh = float(getattr(self.config, "release_rel_speed_thresh", 0.45))
        self.release_vz_min = float(getattr(self.config, "release_vz_min", 0.25))
        self.release_vz_max = float(getattr(self.config, "release_vz_max", 3.5))
        self.release_horizontal_speed_max = float(getattr(self.config, "release_horizontal_speed_max", 2.0))
        self.release_toward_zone_speed_min = float(getattr(self.config, "release_toward_zone_speed_min", -0.05))
        self.contact_force_threshold = float(getattr(self.config, "contact_force_threshold", 0.5))
        self.ball_contact_force_threshold = float(
            getattr(self.config, "ball_contact_force_threshold", self.contact_force_threshold)
        )
        self.contact_dist_threshold = float(getattr(self.config, "contact_dist_threshold", 0.18))
        self.debug_print_non_target_ball_contact = bool(
            getattr(self.config, "debug_print_non_target_ball_contact", False)
        )
        self.debug_print_non_target_ball_contact_topk = int(
            getattr(self.config, "debug_print_non_target_ball_contact_topk", 3)
        )
        self.debug_print_ball_contact_nearest_bodies = bool(
            getattr(self.config, "debug_print_ball_contact_nearest_bodies", False)
        )
        self.debug_non_target_ball_contact_dist_thresh = float(
            getattr(self.config, "debug_non_target_ball_contact_dist_thresh", self.contact_dist_threshold * 1.5)
        )
        self.ball_contact_min_height = float(getattr(self.config, "ball_contact_min_height", 0.12))
        self.ball_ground_height = float(getattr(self.config.termination, "ball_ground_height", 0.12))
        self.ball_max_dist_to_base = float(getattr(self.config.termination, "ball_max_dist_to_base", 2.0))

        self.zone_reward_weight = torch.tensor(
            getattr(self.config.rewards, "zone_reward_weight", [8.0, 4.0, 4.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.zone_sticky_disable_steps = int(getattr(self.config.rewards, "zone_sticky_disable_steps", 5))
        self.zone_sticky_scale = torch.tensor(
            float(getattr(self.config.rewards, "zone_sticky_scale", 0.1)),
            dtype=torch.float,
            device=self.device,
        )
        self.style_prior_sigma = float(getattr(self.config.rewards, "style_prior_sigma", 0.25))

        self._init_cylinder_event_buffers()
        self._init_assisted_kick()
        self._init_motion_ref_reset()

    def _init_cylinder_event_buffers(self):
        self.strike_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_duration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_since_last_contact = torch.full(
            (self.num_envs,), self.strike_cooldown_steps + 1, dtype=torch.long, device=self.device
        )
        self.ball_near_foot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_near_foot_duration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.carry_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.valid_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_valid_kick_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.phase = torch.full(
            (self.num_envs,), self.PHASE_WAIT, dtype=torch.long, device=self.device
        )
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_at_contact_start = torch.full(
            (self.num_envs,), self.PHASE_WAIT, dtype=torch.long, device=self.device
        )
        self.saved_strike_gate_at_contact_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.saved_ball_pos_foot_at_contact_start = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.saved_foot_vel_at_contact_start = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.non_target_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_non_target_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.upper_body_ball_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_non_target_ball_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_debug_non_target_ball_contact_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_ball_has_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_post_release_horizontal_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_start_base_pos = self.simulator.robot_root_states[:, 0:3].clone()
        self.term_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        obs_dim = 12 + 7 * len(self.prediction_horizons)
        self.ball_pos_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_pos_foot = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_foot = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_pred_pos_base_tau = torch.zeros(self.num_envs, 3 * len(self.prediction_horizons), dtype=torch.float, device=self.device)
        self.ball_pred_vz_tau = torch.zeros(self.num_envs, len(self.prediction_horizons), dtype=torch.float, device=self.device)
        self.ball_pred_pos_foot_tau = torch.zeros(self.num_envs, 3 * len(self.prediction_horizons), dtype=torch.float, device=self.device)
        self._hitball_cylinder_obs_dim = obs_dim
        self.reset_ball_pos_foot_mean = torch.zeros(3, dtype=torch.float, device=self.device)

    def _init_assisted_kick(self):
        assisted_cfg = getattr(self.config, "assisted_kick", None)
        self.assisted_kick_enabled = bool(getattr(assisted_cfg, "enable", False))
        self.assisted_kick_alpha_start = float(getattr(assisted_cfg, "alpha_start", 0.0))
        self.assisted_kick_alpha_end = float(getattr(assisted_cfg, "alpha_end", 0.0))
        self.assisted_kick_decay_steps = int(getattr(assisted_cfg, "decay_steps", 0))
        self.assisted_kick_phase_steps = {
            "lift": max(1, int(getattr(assisted_cfg, "lift_steps", 4))),
            "swing": max(1, int(getattr(assisted_cfg, "swing_steps", 5))),
            "recover": max(1, int(getattr(assisted_cfg, "recover_steps", 4))),
        }

        self.assisted_kick_joint_names = [
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        self.assisted_kick_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.assisted_kick_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.assisted_kick_phase_targets = {
            "lift": self._build_assisted_kick_phase_target(
                getattr(
                    assisted_cfg,
                    "lift_pose",
                    {
                        "right_hip_pitch_joint": -0.55,
                        "right_hip_roll_joint": -0.03,
                        "right_hip_yaw_joint": 0.02,
                        "right_knee_joint": 0.70,
                        "right_ankle_pitch_joint": -0.40,
                        "right_ankle_roll_joint": 0.0,
                    },
                )
            ),
            "swing": self._build_assisted_kick_phase_target(
                getattr(
                    assisted_cfg,
                    "swing_pose",
                    {
                        "right_hip_pitch_joint": -0.15,
                        "right_hip_roll_joint": -0.02,
                        "right_hip_yaw_joint": 0.02,
                        "right_knee_joint": 0.18,
                        "right_ankle_pitch_joint": -0.10,
                        "right_ankle_roll_joint": 0.0,
                    },
                )
            ),
            "recover": self._build_assisted_kick_phase_target(
                getattr(assisted_cfg, "recover_pose", {})
            ),
        }
        self.assisted_kick_gate_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.assisted_kick_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.assisted_kick_applied_on_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.saved_assisted_kick_at_contact_start = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.assisted_kick_alpha_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _build_assisted_kick_phase_target(self, pose_override):
        target = self.default_dof_pos[0, self.assisted_kick_joint_indices].clone()
        if pose_override is None:
            return target
        for name, value in pose_override.items():
            if name in self.assisted_kick_joint_names:
                local_idx = self.assisted_kick_joint_names.index(name)
                target[local_idx] = float(value)
        return target

    def _get_assisted_kick_alpha(self):
        if (not self.assisted_kick_enabled) or self.is_evaluating:
            return 0.0
        if self.assisted_kick_decay_steps <= 0:
            return self.assisted_kick_alpha_end
        progress = min(max(float(self.common_step_counter), 0.0), float(self.assisted_kick_decay_steps))
        mix = progress / float(self.assisted_kick_decay_steps)
        return (1.0 - mix) * self.assisted_kick_alpha_start + mix * self.assisted_kick_alpha_end

    def _get_assisted_kick_script_action(self, policy_action):
        script_action = policy_action.clone()
        phase_step = self.assisted_kick_gate_steps
        lift_end = self.assisted_kick_phase_steps["lift"]
        swing_end = lift_end + self.assisted_kick_phase_steps["swing"]
        lift_mask = phase_step < lift_end
        swing_mask = (phase_step >= lift_end) & (phase_step < swing_end)
        recover_mask = phase_step >= swing_end

        joint_target = self.assisted_kick_phase_targets["recover"].unsqueeze(0).repeat(self.num_envs, 1)
        joint_target[lift_mask] = self.assisted_kick_phase_targets["lift"]
        joint_target[swing_mask] = self.assisted_kick_phase_targets["swing"]
        joint_target[recover_mask] = self.assisted_kick_phase_targets["recover"]

        action_scale = float(self.config.robot.control.action_scale)
        joint_action = (joint_target - self.default_dof_pos[:, self.assisted_kick_joint_indices]) / max(action_scale, 1e-6)
        script_action[:, self.assisted_kick_joint_indices] = joint_action
        return script_action

    def _apply_assisted_kick(self, policy_action):
        action_total = policy_action.clone()
        alpha = self._get_assisted_kick_alpha()
        self.assisted_kick_alpha_buf.fill_(alpha)
        self.assisted_kick_active.zero_()
        self.assisted_kick_applied_on_contact.zero_()
        if alpha <= 0.0:
            self.assisted_kick_gate_steps.zero_()
            return action_total

        assist_mask = (
            self.strike_gate
            & (~self.contact_now)
            & (~self.carry_flag)
            & (~self.contact_end)
        )
        if not torch.any(assist_mask):
            self.assisted_kick_gate_steps.zero_()
            return action_total

        script_action = self._get_assisted_kick_script_action(policy_action)
        action_total[assist_mask] = (
            (1.0 - alpha) * policy_action[assist_mask] + alpha * script_action[assist_mask]
        )
        self.assisted_kick_active[assist_mask] = True
        self.assisted_kick_applied_on_contact[:] = self.assisted_kick_active
        self.assisted_kick_gate_steps[assist_mask] += 1
        self.assisted_kick_gate_steps[~assist_mask] = 0
        return action_total

    def _motion_ref_file_is_available(self, motion_file):
        motion_path = Path(str(motion_file))
        if motion_path.is_file():
            return True
        if motion_path.is_dir():
            return any(motion_path.glob("*.pkl"))
        return False

    def _init_motion_ref_reset(self):
        motion_cfg = getattr(self.config.robot, "motion", None)
        motion_file = getattr(motion_cfg, "motion_file", None) if motion_cfg is not None else None
        self.motion_ref_reset_enabled = (
            bool(getattr(self.config, "motion_ref_reset_enabled", True))
            and motion_cfg is not None
            and motion_file is not None
            and self._motion_ref_file_is_available(motion_file)
        )
        self.motion_ref_reset_random_sample = bool(
            getattr(self.config, "motion_ref_reset_random_sample", False)
        )
        self.motion_ref_reset_time_s = float(getattr(self.config, "motion_ref_reset_time_s", 0.0))
        self.motion_ref_reset_time_window_s = float(
            getattr(self.config, "motion_ref_reset_time_window_s", 0.0)
        )
        self.motion_ref_reset_dof_vel_scale = float(
            getattr(self.config, "motion_ref_reset_dof_vel_scale", 1.0)
        )
        self.motion_ref_reset_root_vel_scale = float(
            getattr(self.config, "motion_ref_reset_root_vel_scale", 1.0)
        )
        self.motion_ref_reset_root_ang_vel_scale = float(
            getattr(self.config, "motion_ref_reset_root_ang_vel_scale", 1.0)
        )

        if not self.motion_ref_reset_enabled:
            self.motion_ref_motion_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.motion_ref_motion_len = torch.ones(
                self.num_envs, dtype=torch.float, device=self.device
            )
            return

        self.config.robot.motion.step_dt = self.dt
        self._motion_ref_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self._motion_ref_lib.load_motions(random_sample=self.motion_ref_reset_random_sample)
        self.motion_ref_num_motions = self._motion_ref_lib._num_unique_motions
        self.motion_ref_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.motion_ref_reset_random_sample and self.motion_ref_num_motions > 1:
            self.motion_ref_motion_ids = torch.randint(
                low=0,
                high=self.motion_ref_num_motions,
                size=(self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
        self.motion_ref_motion_len = self._motion_ref_lib.get_motion_length(self.motion_ref_motion_ids)

    def _compute_observations(self):
        super()._compute_observations()

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        policy_action = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)
        blended_action = self._apply_assisted_kick(policy_action)
        self.actions = torch.clip(blended_action, -clip_action_limit, clip_action_limit)

        self.log_dict["action_clip_frac"] = (
            self.actions.abs() == clip_action_limit
        ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[
                torch.arange(self.num_envs), self.action_delay_idx
            ].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    def _update_event_buffers(self):
        update_cylinder_events(self)

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._update_event_buffers()

    def _compute_reward(self):
        compute_cylinder_rewards(self)

    def _compute_reset(self):
        self._check_termination()

    def _reset_actors(self, env_ids):
        reset_cylinder_actors(self, env_ids)

    def _check_termination(self):
        super()._check_termination()
        base_reset = self.reset_buf.bool().clone()
        base_reset &= self.term_reason == self.TERM_NONE
        self.term_reason[base_reset] = self.TERM_BASE_ENV

        self._mark_termination(self.upper_body_ball_contact_now, self.TERM_NON_TARGET_CONTACT)
        ball_ground = self.ball_pos[:, 2] <= self.ball_ground_height
        ball_too_far = (
            torch.norm(self.ball_pos - self.simulator.robot_root_states[:, 0:3], dim=-1)
            >= self.ball_max_dist_to_base
        )
        self._mark_termination(ball_ground, self.TERM_BALL_GROUND)
        self._mark_termination(ball_too_far, self.TERM_BALL_TOO_FAR)

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

    def _get_base_heading_quat_inv(self):
        base_quat = quat_unit(self.base_quat.clone())
        ref_dir = torch.zeros_like(base_quat[:, 0:3])
        ref_dir[:, 0] = 1.0
        rot_dir = quat_rotate(base_quat, ref_dir)
        heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
        axis = torch.zeros_like(base_quat[:, 0:3])
        axis[:, 2] = 1.0
        return quat_from_angle_axis(-heading, axis)

    def _get_foot_point_state(self, body_idx, local_offset):
        body_pos = self.simulator._rigid_body_pos[:, body_idx]
        body_rot = self.simulator._rigid_body_rot[:, body_idx]
        body_vel = self.simulator._rigid_body_vel[:, body_idx]
        body_ang_vel = self.simulator._rigid_body_ang_vel[:, body_idx]
        offset_world = quat_rotate(body_rot, local_offset.unsqueeze(0).expand(self.num_envs, -1))
        point_pos = body_pos + offset_world
        point_vel = body_vel + torch.cross(body_ang_vel, offset_world, dim=-1)
        return point_pos, point_vel

    def _get_target_foot_point_state(self):
        self.target_foot_pos, self.target_foot_vel = self._get_foot_point_state(
            self.target_foot_body_idx,
            self.target_foot_local_offset,
        )
        return self.target_foot_pos, self.target_foot_vel

    def _get_control_zone_mask(self):
        ball_pos_base = self.ball_pos_base
        radial = torch.norm(ball_pos_base[:, :2] - self.control_zone_center[:2].unsqueeze(0), dim=-1)
        z = ball_pos_base[:, 2]
        return (
            (radial <= self.control_zone_radius)
            & (z >= self.control_zone_z_range[0])
            & (z <= self.control_zone_z_range[1])
            & (~self.carry_flag)
        )

    def _get_obs_hitball_cylinder_task(self):
        return compute_cylinder_obs(self)

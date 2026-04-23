from isaacgym import gymapi
import torch

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.envs.hitball_task.hitball_logging import HitBallLoggingMixin
from humanoidverse.envs.hitball_task.hitball_motion_prior import HitBallMotionPriorMixin
from humanoidverse.envs.hitball_task.hitball_reset import HitBallResetMixin
from humanoidverse.envs.hitball_task.hitball_rewards import HitBallRewardMixin
from humanoidverse.envs.hitball_task.hitball_state_machine import HitBallStateMachineMixin

# 这是一个基于状态机的单次击球 env：config 决定 reset 分布、接触阈值、终止阈值和 reward scale
# env 代码负责在每步里根据真实刚体接触事件驱动 phase、termination 和 reward 计算。

class HitBallTask(
    HitBallResetMixin,
    HitBallStateMachineMixin,
    HitBallRewardMixin,
    HitBallMotionPriorMixin,
    HitBallLoggingMixin,
    LeggedRobotBase,
):
    PRE_CONTACT_PREP = 0
    PRE_CONTACT_SWING = 1
    CONTACTED = 2
    POST_CONTACT_EVAL = 3
    DONE_PHASE = 4

    TERM_NONE = 0
    TERM_TIMEOUT_NO_CONTACT = 1
    TERM_WRONG_BODY_CONTACT = 2
    TERM_BALL_GROUND = 3
    TERM_POST_CONTACT_WINDOW = 4
    TERM_BALL_TOO_HIGH = 5
    TERM_BALL_TOO_FAR = 6
    TERM_BASE_ENV = 7

    def __init__(self, config, device):
        super().__init__(config, device)
        self._init_hitball_motion_prior()

    def _init_buffers(self):
        super()._init_buffers()

        if not hasattr(self.simulator, "cubeA_state"):
            raise AttributeError(
                "HitBallTask expects a multi-actor simulator with `cubeA_state`. "
                "Configure this env with the HOI-style IsaacGym simulator."
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
                "HitBallTask requires `target_contact_body` or `config.robot.right_foot_name`."
            )
        self.right_foot_body_idx = self.simulator.find_rigid_body_indice(self.right_foot_body_name)

        left_foot_name = getattr(self.config.robot, "left_foot_name", None)
        self.left_foot_body_idx = None
        if left_foot_name is not None:
            self.left_foot_body_idx = self.simulator.find_rigid_body_indice(left_foot_name)

        self.contact_force_threshold = float(getattr(self.config, "contact_force_threshold", 1.0))
        self.ball_contact_force_threshold = float(
            getattr(self.config, "ball_contact_force_threshold", self.contact_force_threshold)
        )
        self.contact_dist_threshold = float(getattr(self.config, "contact_dist_threshold", 0.12))
        self.ball_body_env_idx = self.simulator.gym.get_actor_rigid_body_index(
            self.simulator.envs[0],
            self.simulator._cubeA_id,
            0,
            gymapi.DOMAIN_ENV,
        )
        self.ball_contact_min_height = float(
            getattr(
                self.config,
                "ball_contact_min_height",
                getattr(self.config.ball, "radius", 0.10) * 0.5 + float(getattr(self.config.termination, "ball_ground_height", 0.12)),
            )
        )
        candidate_body_names = getattr(self.config, "target_contact_candidate_bodies", None)
        if candidate_body_names is None:
            candidate_body_names = [
                name
                for name in self.body_names
                if name == self.right_foot_body_name or name.startswith("right_ankle")
            ]
        if self.right_foot_body_name not in candidate_body_names:
            candidate_body_names = list(candidate_body_names) + [self.right_foot_body_name]
        self.right_foot_candidate_body_names = list(dict.fromkeys(candidate_body_names))
        self.right_foot_candidate_body_indices = torch.tensor(
            [
                self.simulator.find_rigid_body_indice(name)
                for name in self.right_foot_candidate_body_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.pre_contact_timeout_steps = int(getattr(self.config, "pre_contact_timeout_steps", 30))
        self.post_contact_eval_steps_max = int(getattr(self.config, "post_contact_eval_steps", 12))
        self.ball_ground_height = float(getattr(self.config.termination, "ball_ground_height", 0.12))
        self.ball_max_height = float(getattr(self.config.termination, "ball_max_height", 2.0))
        self.ball_max_dist_to_base = float(getattr(self.config.termination, "ball_max_dist_to_base", 1.5))

        self.ball_target_height = float(getattr(self.config.rewards, "ball_target_height", 1.2))
        self.ball_target_upward_vel_min = float(getattr(self.config.rewards, "ball_target_upward_vel_min", 1.5))
        self.ball_target_upward_vel_max = float(getattr(self.config.rewards, "ball_target_upward_vel_max", 4.0))
        self.ball_horizontal_vel_limit = float(getattr(self.config.rewards, "ball_horizontal_vel_limit", 1.0))
        self.prep_target_xy = torch.tensor(
            getattr(self.config, "prep_target_xy", [0.35, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.prep_xy_tolerance = torch.tensor(
            getattr(self.config, "prep_xy_tolerance", [0.08, 0.08]),
            dtype=torch.float,
            device=self.device,
        )
        self.prep_to_swing_require_z_valid = bool(
            getattr(self.config, "prep_to_swing_require_z_valid", False)
        )
        self.prep_min_ball_height = float(
            getattr(self.config, "prep_min_ball_height", self.ball_ground_height)
        )
        self.prep_max_ball_height = float(
            getattr(self.config, "prep_max_ball_height", self.ball_max_height)
        )
        self.pre_contact_prep_xy_weight = torch.tensor(
            getattr(self.config.rewards, "pre_contact_prep_xy_weight", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        )
        self.pre_contact_prep_xy_alpha = float(
            getattr(self.config.rewards, "pre_contact_prep_xy_alpha", 12.0)
        )
        self.pre_contact_swing_target_offset = torch.tensor(
            getattr(self.config.rewards, "pre_contact_swing_target_offset", [0.10, 0.0, 0.10]),
            dtype=torch.float,
            device=self.device,
        )

        self.prep_root_pos_noise = torch.tensor(
            getattr(self.config, "prep_root_pos_noise", [0.01, 0.01, 0.005]),
            dtype=torch.float,
            device=self.device,
        )
        self.prep_root_lin_vel_noise = torch.tensor(
            getattr(self.config, "prep_root_lin_vel_noise", [0.03, 0.03, 0.02]),
            dtype=torch.float,
            device=self.device,
        )
        self.prep_root_ang_vel_noise = torch.tensor(
            getattr(self.config, "prep_root_ang_vel_noise", [0.03, 0.03, 0.03]),
            dtype=torch.float,
            device=self.device,
        )

        self.ball_init_offset = torch.tensor(
            getattr(self.config, "ball_init_offset", [0.16, -0.02, 0.16]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_pos_noise = torch.tensor(
            getattr(self.config, "ball_init_pos_noise", [0.02, 0.02, 0.03]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel = torch.tensor(
            getattr(self.config, "ball_init_lin_vel", [0.0, 0.0, -0.15]),
            dtype=torch.float,
            device=self.device,
        )
        self.ball_init_lin_vel_noise = torch.tensor(
            getattr(self.config, "ball_init_lin_vel_noise", [0.03, 0.03, 0.05]),
            dtype=torch.float,
            device=self.device,
        )
        # Cache the reset-pose right-foot offset in world coordinates relative
        # to the base root. During env reset, rigid-body tensors can lag one
        # step behind the freshly written root/dof states; anchoring ball spawn
        # to this cached offset avoids reusing a stale "foot flew away" pose.
        self.reset_pose_right_foot_offset = (
            self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
            - self.simulator.robot_root_states[:, 0:3]
        ).mean(dim=0)

        self.prep_dof_pos = self.default_dof_pos.clone()
        prep_pose_override = getattr(self.config, "prep_dof_pos", None)
        if prep_pose_override is not None:
            if isinstance(prep_pose_override, dict):
                for name, value in prep_pose_override.items():
                    self.prep_dof_pos[0, self.dof_names.index(name)] = float(value)
            else:
                prep_tensor = torch.tensor(prep_pose_override, dtype=torch.float, device=self.device)
                self.prep_dof_pos[0] = prep_tensor

        self.task_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.has_first_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_contact_is_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_contact_body_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.first_contact_time = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.post_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contacted_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.just_target_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_became_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_ball_height_min = float(getattr(self.config.rewards, "success_ball_height_min", 1.0))
        self.success_ball_height_max = float(getattr(self.config.rewards, "success_ball_height_max", 1.6))
        self.success_upward_vel_min = float(getattr(self.config.rewards, "success_upward_vel_min", self.ball_target_upward_vel_min))
        self.success_upward_vel_max = float(getattr(self.config.rewards, "success_upward_vel_max", self.ball_target_upward_vel_max))
        self.post_contact_ball_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.upward_vel_in_range = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        self._update_single_hit_state_machine()
        self._check_termination()
        self._update_hitball_motion_prior_reference()
        self._compute_reward()
        self._debug_single_hit_log()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations()
        self._post_compute_observations_callback()

        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

        self._publish_hitball_extras()

        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()

    def _get_obs_hitball_task(self):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        base_vel = self.simulator.robot_root_states[:, 7:10]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        right_foot_vel = self.simulator._rigid_body_vel[:, self.right_foot_body_idx]

        obs_parts = [
            self.ball_pos - base_pos,
            self.ball_lin_vel - base_vel,
            self.ball_pos - right_foot_pos,
            self.ball_lin_vel - right_foot_vel,
        ]

        if self.left_foot_body_idx is not None:
            left_foot_pos = self.simulator._rigid_body_pos[:, self.left_foot_body_idx]
            left_foot_vel = self.simulator._rigid_body_vel[:, self.left_foot_body_idx]
            obs_parts.extend(
                [
                    self.ball_pos - left_foot_pos,
                    self.ball_lin_vel - left_foot_vel,
                ]
            )

        phase_encoding = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device)
        phase_encoding.scatter_(1, self.task_phase.unsqueeze(1), 1.0)
        obs_parts.extend(
            [
                self.has_first_contact.float().unsqueeze(-1),
                phase_encoding,
            ]
        )
        return torch.cat(obs_parts, dim=-1)

    def _check_termination(self):
        super()._check_termination()

        base_reset = self.reset_buf.bool().clone()
        base_reset &= self.term_reason == self.TERM_NONE
        self.term_reason[base_reset] = self.TERM_BASE_ENV

        no_contact_timeout = (~self.has_first_contact) & (self.episode_length_buf >= self.pre_contact_timeout_steps)
        wrong_contact = self.has_first_contact & (~self.first_contact_is_target)
        ball_ground = self.ball_pos[:, 2] <= self.ball_ground_height
        ball_too_high = self.ball_pos[:, 2] >= self.ball_max_height
        ball_too_far = (
            torch.norm(self.ball_pos - self.simulator.robot_root_states[:, 0:3], dim=-1)
            >= self.ball_max_dist_to_base
        )
        post_contact_done = self.has_first_contact & self.first_contact_is_target & (
            self.post_contact_steps >= self.post_contact_eval_steps_max
        )

        self._mark_termination(no_contact_timeout, self.TERM_TIMEOUT_NO_CONTACT)
        self._mark_termination(wrong_contact, self.TERM_WRONG_BODY_CONTACT)
        self._mark_termination(ball_ground, self.TERM_BALL_GROUND)
        self._mark_termination(post_contact_done, self.TERM_POST_CONTACT_WINDOW)
        self._mark_termination(ball_too_high, self.TERM_BALL_TOO_HIGH)
        self._mark_termination(ball_too_far, self.TERM_BALL_TOO_FAR)

        self._log_hitball_termination_stats()

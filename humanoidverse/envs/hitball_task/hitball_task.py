import torch
from isaacgym import gymapi

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.torch_utils import torch_rand_float

# 这是一个基于状态机的单次击球 env：config 决定 reset 分布、接触阈值、终止阈值和 reward scale
# env 代码负责在每步里根据真实刚体接触事件驱动 phase、termination 和 reward 计算。

class HitBallTask(LeggedRobotBase):
    PRE_CONTACT = 0
    CONTACTED = 1
    POST_CONTACT_EVAL = 2
    DONE_PHASE = 3

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
        self.pre_contact_target_offset = torch.tensor(
            getattr(self.config.rewards, "pre_contact_target_offset", [0.10, 0.0, 0.10]),
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

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
        self.task_phase[env_ids] = self.PRE_CONTACT
        self.has_first_contact[env_ids] = False
        self.first_contact_is_target[env_ids] = False
        self.first_contact_body_id[env_ids] = -1
        self.first_contact_time[env_ids] = -1
        self.post_contact_steps[env_ids] = 0
        self.contacted_steps[env_ids] = 0
        self.just_target_contact[env_ids] = False
        self.just_became_success[env_ids] = False
        self.term_reason[env_ids] = self.TERM_NONE
        self.post_contact_ball_max_height[env_ids] = 0.0
        self.success_buf[env_ids] = False
        self.upward_vel_in_range[env_ids] = False

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        if target_states is not None:
            super()._reset_robot_states_callback(env_ids, target_states=target_states)
            self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
            self._refresh_sim_tensors()
            self._reset_ball_states(env_ids)
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self._refresh_sim_tensors()
        self._reset_ball_states(env_ids)

    def _reset_dofs(self, env_ids, target_state=None):
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
            return

        dof_noise_scale = float(getattr(self.config, "prep_dof_pos_noise_scale", 0.03))
        dof_noise = torch_rand_float(
            -dof_noise_scale,
            dof_noise_scale,
            (len(env_ids), self.num_dof),
            device=str(self.device),
        )
        self.simulator.dof_pos[env_ids] = self.prep_dof_pos + dof_noise
        self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            return

        pos_noise = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=str(self.device))
            * self.prep_root_pos_noise.unsqueeze(0)
        )
        lin_vel_noise = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=str(self.device))
            * self.prep_root_lin_vel_noise.unsqueeze(0)
        )
        ang_vel_noise = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=str(self.device))
            * self.prep_root_ang_vel_noise.unsqueeze(0)
        )

        self.simulator.robot_root_states[env_ids] = self.base_init_state
        self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids] + pos_noise
        self.simulator.robot_root_states[env_ids, 7:10] = lin_vel_noise
        self.simulator.robot_root_states[env_ids, 10:13] = ang_vel_noise

    def _reset_ball_states(self, env_ids):
        if len(env_ids) == 0:
            return

        pos_noise = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=str(self.device))
            * self.ball_init_pos_noise.unsqueeze(0)
        )
        lin_vel_noise = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=str(self.device))
            * self.ball_init_lin_vel_noise.unsqueeze(0)
        )

        right_foot_reset_anchor = (
            self.simulator.robot_root_states[env_ids, 0:3]
            + self.reset_pose_right_foot_offset.unsqueeze(0)
        )
        self.ball_root_states[env_ids, 0:3] = (
            right_foot_reset_anchor + self.ball_init_offset.unsqueeze(0) + pos_noise
        )
        self.ball_root_states[env_ids, 3:7] = 0.0
        self.ball_root_states[env_ids, 6] = 1.0
        self.ball_root_states[env_ids, 7:10] = self.ball_init_lin_vel.unsqueeze(0) + lin_vel_noise
        self.ball_root_states[env_ids, 10:13] = 0.0

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        self._update_single_hit_state_machine()
        self._check_termination()
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

        self.extras["to_log"] = self.log_dict
        self.extras["hitball"] = {
            "task_phase": self.task_phase.clone(),
            "has_first_contact": self.has_first_contact.clone(),
            "first_contact_is_target": self.first_contact_is_target.clone(),
            "first_contact_body_id": self.first_contact_body_id.clone(),
            "first_contact_time": self.first_contact_time.clone(),
            "post_contact_steps": self.post_contact_steps.clone(),
            "success": self.success_buf.clone(),
            "just_became_success": self.just_became_success.clone(),
            "upward_vel_in_range": self.upward_vel_in_range.clone(),
            "post_contact_ball_max_height": self.post_contact_ball_max_height.clone(),
            "term_reason": self.term_reason.clone(),
        }

        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()

    def _update_single_hit_state_machine(self):
        self.just_target_contact[:] = False
        self.just_became_success[:] = False

        robot_body_forces = torch.norm(self.simulator.contact_forces[:, : self.num_bodies], dim=-1)
        ball_contact_force_norm = torch.norm(
            self.simulator.contact_forces[:, self.ball_body_env_idx], dim=-1
        )
        ball_above_ground_contact_band = self.ball_pos[:, 2] > self.ball_contact_min_height
        ball_has_contact = (ball_contact_force_norm > self.ball_contact_force_threshold) & ball_above_ground_contact_band
        robot_body_pos = self.simulator._rigid_body_pos[:, : self.num_bodies]
        body_dist_to_ball = torch.norm(robot_body_pos - self.ball_pos.unsqueeze(1), dim=-1)
        near_ball = body_dist_to_ball < self.contact_dist_threshold
        contact_like = (robot_body_forces > self.contact_force_threshold) & near_ball
        candidate_body_dist = body_dist_to_ball[:, self.right_foot_candidate_body_indices]
        min_candidate_body_dist, min_candidate_body_idx = torch.min(candidate_body_dist, dim=1)
        candidate_near_ball = min_candidate_body_dist < self.contact_dist_threshold

        new_target_contact = (
            (~self.has_first_contact)
            & ball_has_contact
            & candidate_near_ball
        )

        non_target_contact = contact_like.clone()
        non_target_contact[:, self.right_foot_candidate_body_indices] = False
        wrong_contact_force = robot_body_forces.masked_fill(~non_target_contact, -1.0)
        wrong_force_max, wrong_contact_body_id = torch.max(wrong_contact_force, dim=1)
        new_wrong_contact = (
            (~self.has_first_contact)
            & (~new_target_contact)
            & (wrong_force_max > self.contact_force_threshold)
        )
        max_contact_lambda = torch.max(robot_body_forces.masked_fill(~contact_like, 0.0), dim=1).values

        self.has_first_contact[new_target_contact | new_wrong_contact] = True
        self.first_contact_time[new_target_contact | new_wrong_contact] = self.episode_length_buf[
            new_target_contact | new_wrong_contact
        ]
        self.first_contact_is_target[new_target_contact] = True
        self.first_contact_is_target[new_wrong_contact] = False
        if torch.any(new_target_contact):
            target_candidate_ids = self.right_foot_candidate_body_indices[min_candidate_body_idx]
            self.first_contact_body_id[new_target_contact] = target_candidate_ids[new_target_contact]
        self.just_target_contact[new_target_contact] = True

        if torch.any(new_wrong_contact):
            self.first_contact_body_id[new_wrong_contact] = wrong_contact_body_id[new_wrong_contact]

        new_contacted = (self.task_phase == self.PRE_CONTACT) & new_target_contact
        self.task_phase[new_contacted] = self.CONTACTED
        self.contacted_steps[new_contacted] = 0

        contacted_mask = self.task_phase == self.CONTACTED
        self.contacted_steps[contacted_mask] += 1
        advance_to_post = contacted_mask & (self.contacted_steps >= 2)
        self.task_phase[advance_to_post] = self.POST_CONTACT_EVAL

        post_mask = self.task_phase == self.POST_CONTACT_EVAL
        self.post_contact_steps[post_mask] += 1
        self.post_contact_steps[~post_mask] = 0
        self.contacted_steps[~contacted_mask] = 0
        self.post_contact_ball_max_height[post_mask] = torch.maximum(
            self.post_contact_ball_max_height[post_mask],
            self.ball_pos[post_mask, 2],
        )
        in_upward_range = (
            (self.ball_lin_vel[:, 2] >= self.success_upward_vel_min)
            & (self.ball_lin_vel[:, 2] <= self.success_upward_vel_max)
        )
        self.upward_vel_in_range |= post_mask & in_upward_range
        success_height = (
            (self.post_contact_ball_max_height >= self.success_ball_height_min)
            & (self.post_contact_ball_max_height <= self.success_ball_height_max)
        )
        prev_success = self.success_buf.clone()
        self.success_buf = self.has_first_contact & self.first_contact_is_target & self.upward_vel_in_range & success_height
        self.just_became_success = self.success_buf & (~prev_success)

        right_foot_ball_dist = torch.norm(
            robot_body_pos[:, self.right_foot_body_idx] - self.ball_pos, dim=-1
        )

        contact_count = self.has_first_contact.float().sum().clamp(min=1.0)
        target_contact_count = (self.has_first_contact & self.first_contact_is_target).float().sum()

        self.log_dict["first_contact_rate"] = self.has_first_contact.float().mean()
        self.log_dict["target_contact_given_contact"] = target_contact_count / contact_count
        self.log_dict["first_contact_step_mean"] = torch.where(
            self.first_contact_time >= 0,
            self.first_contact_time.float(),
            torch.zeros_like(self.first_contact_time, dtype=torch.float),
        ).mean()
        self.log_dict["success_rate"] = self.success_buf.float().mean()
        self.log_dict["upward_vel_hit_rate_after_target_contact"] = (
            self.upward_vel_in_range & self.has_first_contact & self.first_contact_is_target
        ).float().sum() / target_contact_count.clamp(min=1.0)
        self.log_dict["success_given_target_contact"] = self.success_buf.float().sum() / target_contact_count.clamp(min=1.0)
        self.log_dict["ball_contact_lambda_mean"] = ball_contact_force_norm.mean()
        self.log_dict["right_foot_ball_dist_mean"] = right_foot_ball_dist.mean()
        self.log_dict["ball_has_contact_rate"] = ball_has_contact.float().mean()
        self.log_dict["candidate_near_ball_rate"] = candidate_near_ball.float().mean()
        self.log_dict["target_contact_gate_open_rate"] = (ball_has_contact & candidate_near_ball).float().mean()
        active_post_contact = self.has_first_contact & self.first_contact_is_target
        self.log_dict["post_contact_ball_max_height_mean"] = torch.where(
            active_post_contact,
            self.post_contact_ball_max_height,
            torch.zeros_like(self.post_contact_ball_max_height),
        ).sum() / active_post_contact.float().sum().clamp(min=1.0)
        self.log_dict["apex_height_hit_rate"] = (
            active_post_contact
            & (self.post_contact_ball_max_height >= self.success_ball_height_min)
        ).float().sum() / active_post_contact.float().sum().clamp(min=1.0)

    def _debug_single_hit_log(self):
        if self.num_envs == 0:
            return

        self.log_dict["phase_env0"] = self.task_phase[0].float()
        self.log_dict["first_contact_body_env0"] = self.first_contact_body_id[0].float()
        self.log_dict["first_contact_time_env0"] = self.first_contact_time[0].float()
        self.log_dict["first_contact_target_env0"] = self.first_contact_is_target[0].float()
        self.log_dict["success_env0"] = self.success_buf[0].float()
        self.log_dict["term_reason_env0"] = self.term_reason[0].float()

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

        phase_encoding = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
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

        self.log_dict["term_timeout_no_contact"] = (self.term_reason == self.TERM_TIMEOUT_NO_CONTACT).float().mean()
        self.log_dict["term_wrong_body"] = (self.term_reason == self.TERM_WRONG_BODY_CONTACT).float().mean()
        self.log_dict["term_ball_ground"] = (self.term_reason == self.TERM_BALL_GROUND).float().mean()
        self.log_dict["term_post_window"] = (self.term_reason == self.TERM_POST_CONTACT_WINDOW).float().mean()

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_pre_contact_alignment(self):
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        target_point = right_foot_pos + self.pre_contact_target_offset.unsqueeze(0)
        dist = torch.norm(self.ball_pos - target_point, dim=-1)
        rew = torch.exp(-4.0 * dist)
        return rew * (~self.has_first_contact).float()

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

"""Goalkeeper task — robot stands in goal, blocks incoming balls.

Architecture:
- No motion prior (unlike hitball)
- No locomotion commands (unlike locomotion)
- Config-driven body indices (no hardcoded G1/Q1 names)
- Ball accessed through multi-actor simulator (cubeA_state)
- Prepared-pose reset (captures simulator equilibrium, same as stand_smoke)
"""

from isaacgym import gymapi
import torch
import numpy as np
from loguru import logger

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.envs.goalkeeper_task.goalkeeper_reset import GoalkeeperResetMixin
from humanoidverse.envs.goalkeeper_task.goalkeeper_rewards import GoalkeeperRewardMixin


class GoalkeeperTask(
    GoalkeeperResetMixin,
    GoalkeeperRewardMixin,
    LeggedRobotBase,
):
    """Simple goalkeeper: ball flies toward goal from the front, robot blocks."""

    TERM_NONE = 0
    TERM_BALL_IN_GOAL = 1
    TERM_BALL_PAST_ROBOT = 2
    TERM_BASE_ENV = 3

    def __init__(self, config, device):
        super().__init__(config, device)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()

        # --- Ball access (same pattern as hitball / ball_control) ---
        if not hasattr(self.simulator, "cubeA_state"):
            raise AttributeError(
                "GoalkeeperTask expects a multi-actor simulator with `cubeA_state`. "
                "Configure this env with the HOI-style IsaacGym simulator."
            )

        self.ball_root_states = self.simulator.cubeA_state
        self.ball_pos = self.ball_root_states[:, 0:3]
        self.ball_quat = self.ball_root_states[:, 3:7]
        self.ball_lin_vel = self.ball_root_states[:, 7:10]
        self.ball_ang_vel = self.ball_root_states[:, 10:13]

        # --- Ball actor body index (for contact force queries) ---
        self.ball_body_env_idx = self.simulator.gym.get_actor_rigid_body_index(
            self.simulator.envs[0],
            self.simulator._cubeA_id,
            0,
            gymapi.DOMAIN_ENV,
        )

        # --- Config-driven body indices (NOT hardcoded) ---
        # Foot indices (for observations and contact)
        self.right_foot_body_name = getattr(self.config.robot, "right_foot_name", None)
        if self.right_foot_body_name is None:
            raise AttributeError("GoalkeeperTask requires config.robot.right_foot_name.")
        self.right_foot_body_idx = self.simulator.find_rigid_body_indice(self.right_foot_body_name)

        self.left_foot_body_name = getattr(self.config.robot, "left_foot_name", None)
        if self.left_foot_body_name is None:
            raise AttributeError("GoalkeeperTask requires config.robot.left_foot_name.")
        self.left_foot_body_idx = self.simulator.find_rigid_body_indice(self.left_foot_body_name)

        # Block body indices — all robot bodies (goalkeeper can block with any body part)
        self.block_body_names = getattr(self.config, "block_body_names", None)
        if self.block_body_names is None:
            self.block_body_names = list(self.body_names)
        self.block_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(n) for n in self.block_body_names],
            dtype=torch.long, device=self.device,
        )

        # --- Goal definition ---
        self.goal_x = float(getattr(self.config, "goal_x", -4.0))  # goal line x in world frame (behind robot)
        self.goal_width = float(getattr(self.config, "goal_width", 2.0))
        self.goal_height = float(getattr(self.config, "goal_height", 1.5))
        self.goal_y_min = float(getattr(self.config, "goal_y_min", -1.0))
        self.goal_y_max = float(getattr(self.config, "goal_y_max", 1.0))
        self.goal_z_max = float(getattr(self.config, "goal_z_max", self.goal_height))

        # --- Contact thresholds ---
        self.contact_force_threshold = float(
            getattr(self.config, "contact_force_threshold", 1.0)
        )
        self.contact_dist_threshold = float(
            getattr(self.config, "contact_dist_threshold", 0.22)
        )
        self.ball_ground_height = float(
            getattr(self.config.termination, "ball_ground_height",
                    getattr(self.config.ball, "radius", 0.10) + 0.02)
        )

        # --- Episode-level state ---
        self.ball_contact_this_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ball_blocked_this_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_conceded_this_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_got_ball_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.term_reason = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Track which body made contact
        self.ball_contact_body_name_buf = ["" for _ in range(self.num_envs)]
        self.ball_contact_body_idx_buf = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.ball_lin_vel_before_contact = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # --- Init sub-mixins ---
        self._init_goalkeeper_reset()
        self._init_goalkeeper_rewards()

    # ------------------------------------------------------------------
    # Post-physics step (main control flow)
    # ------------------------------------------------------------------

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        self._update_goalkeeper_contact()
        self._check_goalkeeper_goal()
        self._check_termination()
        self._compute_reward()
        self._debug_goalkeeper_log()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(
                refresh_env_ids, self.simulator.all_root_states
            )
            self.simulator.set_dof_state_tensor(
                refresh_env_ids, self.simulator.dof_state
            )
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations()
        self._post_compute_observations_callback()

        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        self._publish_goalkeeper_extras()

        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()

    # ------------------------------------------------------------------
    # Contact detection
    # ------------------------------------------------------------------

    def _update_goalkeeper_contact(self):
        """Detect robot-ball contact using config-driven body indices."""
        self.just_got_ball_contact[:] = False

        # Ball contact force
        ball_contact_force = torch.norm(
            self.simulator.contact_forces[:, self.ball_body_env_idx], dim=-1
        )
        ball_has_contact = ball_contact_force > self.contact_force_threshold

        # Which robot body is nearest to the ball?
        robot_body_pos = self.simulator._rigid_body_pos[:, : self.num_bodies]
        body_dist_to_ball = torch.norm(
            robot_body_pos - self.ball_pos.unsqueeze(1), dim=-1
        )
        # Only consider block_body_indices
        block_dists = body_dist_to_ball[:, self.block_body_indices]
        min_block_dist, min_block_local_idx = torch.min(block_dists, dim=1)

        # Robot body contact forces
        robot_body_forces = torch.norm(
            self.simulator.contact_forces[:, : self.num_bodies], dim=-1
        )
        block_forces = robot_body_forces[:, self.block_body_indices]
        max_block_force, max_force_local_idx = torch.max(block_forces, dim=1)

        # Contact = ball contact force > threshold AND robot body near ball
        new_contact = (
            ball_has_contact
            & (min_block_dist < self.contact_dist_threshold)
            & (~self.ball_contact_this_episode)
        )

        # Record first contact
        self.ball_contact_this_episode[new_contact] = True
        self.just_got_ball_contact[new_contact] = True
        self.ball_lin_vel_before_contact[new_contact] = (
            self.ball_lin_vel[new_contact].clone()
        )

        # Track which body made contact
        global_idx = self.block_body_indices[min_block_local_idx]
        for i in new_contact.nonzero(as_tuple=False).flatten():
            idx = int(global_idx[i].item())
            self.ball_contact_body_idx_buf[i] = idx
            self.ball_contact_body_name_buf[i] = (
                self.body_names[idx] if 0 <= idx < len(self.body_names) else "unknown"
            )

    # ------------------------------------------------------------------
    # Goal detection
    # ------------------------------------------------------------------

    def _check_goalkeeper_goal(self):
        """Check if ball has entered the goal or passed the robot."""
        # Goal conceded: ball passes goal line (x < goal_x) AND within goal width/height
        ball_past_goal_line = self.ball_pos[:, 0] < self.goal_x
        ball_in_goal_width = (
            (self.ball_pos[:, 1] > self.goal_y_min)
            & (self.ball_pos[:, 1] < self.goal_y_max)
        )
        ball_in_goal_height = self.ball_pos[:, 2] < self.goal_z_max
        ball_in_goal = ball_past_goal_line & ball_in_goal_width & ball_in_goal_height
        self.goal_conceded_this_episode[ball_in_goal] = True

        # Blocked: ball velocity direction reversed away from goal after contact
        post_contact = self.ball_contact_this_episode
        vel_toward_goal_before = -self.ball_lin_vel_before_contact[:, 0]
        vel_toward_goal_now = -self.ball_lin_vel[:, 0]
        vel_changed_away = (vel_toward_goal_now - vel_toward_goal_before) > 0.5
        self.ball_blocked_this_episode[post_contact & vel_changed_away] = True

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self):
        """Base termination + goalkeeper-specific."""
        super()._check_termination()

        # Mark base-env terminations
        base_reset = self.reset_buf.bool().clone()
        base_reset &= self.term_reason == self.TERM_NONE
        self.term_reason[base_reset] = self.TERM_BASE_ENV

        # Ball in goal
        self._mark_termination(
            self.goal_conceded_this_episode & (~self.reset_buf.bool()),
            self.TERM_BALL_IN_GOAL,
        )

        # Ball past robot (passed the robot + behind it, but not necessarily in goal)
        # This catches balls that go wide or over
        ball_behind_robot = self.ball_pos[:, 0] < (
            self.simulator.robot_root_states[:, 0] - 1.0
        )
        self._mark_termination(
            ball_behind_robot & (~self.reset_buf.bool()),
            self.TERM_BALL_PAST_ROBOT,
        )

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

    # ------------------------------------------------------------------
    # Reset overrides
    # ------------------------------------------------------------------

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
        self._reset_goalkeeper_buffers(env_ids)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        if target_states is not None:
            super()._reset_robot_states_callback(env_ids, target_states=target_states)
            self.simulator.set_actor_root_state_tensor(
                env_ids, self.simulator.all_root_states
            )
            self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
            self._refresh_sim_tensors()
            self._reset_ball_states(env_ids)
            self.simulator.set_actor_root_state_tensor(
                env_ids, self.simulator.all_root_states
            )
            return

        # Standard reset: use prepared pose
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.set_actor_root_state_tensor(
            env_ids, self.simulator.all_root_states
        )
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self._refresh_sim_tensors()
        self._reset_ball_states(env_ids)
        self.simulator.set_actor_root_state_tensor(
            env_ids, self.simulator.all_root_states
        )

        # Store initial ball velocity for reward computation
        self.ball_init_vel_dir[env_ids] = self.ball_lin_vel[env_ids].clone()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs_goalkeeper_task(self):
        """Goalkeeper-specific observations: ball state, contact flags."""
        base_pos = self.simulator.robot_root_states[:, 0:3]
        base_vel = self.simulator.robot_root_states[:, 7:10]

        # Ball relative to robot base
        ball_rel_base = self.ball_pos - base_pos
        ball_vel_rel_base = self.ball_lin_vel - base_vel

        # Ball relative to feet (config-driven)
        right_foot_pos = self.simulator._rigid_body_pos[:, self.right_foot_body_idx]
        left_foot_pos = self.simulator._rigid_body_pos[:, self.left_foot_body_idx]

        # Contact history
        has_contact = self.ball_contact_this_episode.float().unsqueeze(-1)

        return torch.cat(
            [
                ball_rel_base,                      # 3
                ball_vel_rel_base,                  # 3
                self.ball_pos - right_foot_pos,     # 3
                self.ball_pos - left_foot_pos,      # 3
                has_contact,                        # 1
            ],
            dim=-1,
        )  # total: 13

    # ------------------------------------------------------------------
    # Logging & extras
    # ------------------------------------------------------------------

    def _publish_goalkeeper_extras(self):
        """Publish goalkeeper-specific extras for logging."""
        self.extras["goalkeeper"] = {
            "ball_pos": self.ball_pos.clone(),
            "ball_lin_vel": self.ball_lin_vel.clone(),
            "ball_contact": self.ball_contact_this_episode.clone(),
            "ball_blocked": self.ball_blocked_this_episode.clone(),
            "goal_conceded": self.goal_conceded_this_episode.clone(),
            "term_reason": self.term_reason.clone(),
        }

    def _debug_goalkeeper_log(self):
        """Minimal debug logging for smoke test."""
        if self.num_envs == 0:
            return

        self.log_dict["gk/ball_contact_rate"] = (
            self.ball_contact_this_episode.float().mean()
        )
        self.log_dict["gk/ball_blocked_rate"] = (
            self.ball_blocked_this_episode.float().mean()
        )
        self.log_dict["gk/goal_conceded_rate"] = (
            self.goal_conceded_this_episode.float().mean()
        )
        self.log_dict["gk/ball_x_mean"] = self.ball_pos[:, 0].mean()
        self.log_dict["gk/ball_z_mean"] = self.ball_pos[:, 2].mean()
        self.log_dict["gk/ball_init_z_mean"] = self.ball_pos[:, 2].mean()  # approximate
        if hasattr(self, "ball_shot_level"):
            self.log_dict["gk/shot_level_low"] = (self.ball_shot_level == 0).float().mean()
            self.log_dict["gk/shot_level_mid"] = (self.ball_shot_level == 1).float().mean()
            self.log_dict["gk/shot_level_high"] = (self.ball_shot_level == 2).float().mean()
        self.log_dict["gk/term_ball_in_goal"] = (
            self.term_reason == self.TERM_BALL_IN_GOAL
        ).float().mean()
        self.log_dict["gk/term_ball_past_robot"] = (
            self.term_reason == self.TERM_BALL_PAST_ROBOT
        ).float().mean()
        self.log_dict["gk/term_base_env"] = (
            self.term_reason == self.TERM_BASE_ENV
        ).float().mean()

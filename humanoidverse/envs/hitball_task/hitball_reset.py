from humanoidverse.utils.torch_utils import torch_rand_float


class HitBallResetMixin:
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
        self.task_phase[env_ids] = self.PRE_CONTACT_PREP
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
        if self.hitball_motion_prior_enabled:
            self._reset_hitball_motion_phase_clock(env_ids)

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

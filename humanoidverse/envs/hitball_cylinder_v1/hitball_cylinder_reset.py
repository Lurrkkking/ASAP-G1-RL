from humanoidverse.utils.torch_utils import quat_rotate_inverse, torch_rand_float


def reset_cylinder_actors(env, env_ids):
    env._reset_cylinder_ball_states(env_ids)


class HitBallCylinderResetMixin:
    def _sample_motion_ref_state(self, env_ids):
        if self.motion_ref_reset_random_sample and self.motion_ref_num_motions > 1:
            self.motion_ref_motion_ids[env_ids] = torch.randint(
                low=0,
                high=self.motion_ref_num_motions,
                size=(len(env_ids),),
                device=self.device,
                dtype=torch.long,
            )

        motion_ids = self.motion_ref_motion_ids[env_ids]
        motion_len = self._motion_ref_lib.get_motion_length(motion_ids).clamp(min=1e-6)
        motion_times = torch.full(
            (len(env_ids),),
            self.motion_ref_reset_time_s,
            dtype=torch.float,
            device=self.device,
        )
        if self.motion_ref_reset_time_window_s > 0.0:
            motion_times += torch_rand_float(
                0.0,
                self.motion_ref_reset_time_window_s,
                (len(env_ids), 1),
                device=str(self.device),
            ).squeeze(-1)
        motion_times = torch.clamp(motion_times, min=0.0, max=motion_len - 1e-4)
        return self._motion_ref_lib.get_motion_state(
            motion_ids,
            motion_times,
            offset=self.env_origins[env_ids],
        )

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
        self.strike_gate[env_ids] = False
        self.contact_now[env_ids] = False
        self.prev_contact_now[env_ids] = False
        self.contact_start[env_ids] = False
        self.contact_end[env_ids] = False
        self.contact_duration[env_ids] = 0
        self.time_since_last_contact[env_ids] = self.strike_cooldown_steps + 1
        self.ball_near_foot[env_ids] = False
        self.ball_near_foot_duration[env_ids] = 0
        self.carry_flag[env_ids] = False
        self.valid_kick[env_ids] = False
        self.last_valid_kick_step[env_ids] = -1
        self.phase[env_ids] = self.PHASE_WAIT
        self.phase_step[env_ids] = 0
        self.phase_at_contact_start[env_ids] = self.PHASE_WAIT
        self.saved_strike_gate_at_contact_start[env_ids] = False
        self.saved_assisted_kick_at_contact_start[env_ids] = False
        self.saved_ball_pos_foot_at_contact_start[env_ids] = 0.0
        self.saved_foot_vel_at_contact_start[env_ids] = 0.0
        self.non_target_contact_now[env_ids] = False
        self.prev_non_target_contact_now[env_ids] = False
        self.upper_body_ball_contact_now[env_ids] = False
        self.debug_non_target_ball_contact_now[env_ids] = False
        self.prev_debug_non_target_ball_contact_now[env_ids] = False
        self.prev_ball_has_contact[env_ids] = False
        self.last_post_release_horizontal_speed[env_ids] = 0.0
        self.assisted_kick_gate_steps[env_ids] = 0
        self.assisted_kick_active[env_ids] = False
        self.assisted_kick_applied_on_contact[env_ids] = False
        self.assisted_kick_alpha_buf[env_ids] = 0.0
        self.episode_start_base_pos[env_ids] = self.simulator.robot_root_states[env_ids, 0:3]
        self.term_reason[env_ids] = self.TERM_NONE

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        if target_states is not None:
            super()._reset_robot_states_callback(env_ids, target_states=target_states)
            self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
            self._refresh_sim_tensors()
            reset_cylinder_actors(self, env_ids)
            self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
            return

        if self.motion_ref_reset_enabled:
            motion_res = self._sample_motion_ref_state(env_ids)
            self.simulator.dof_pos[env_ids] = motion_res["dof_pos"][:, : self.num_dof]
            self.simulator.dof_vel[env_ids] = (
                motion_res["dof_vel"][:, : self.num_dof] * self.motion_ref_reset_dof_vel_scale
            )
            self.simulator.robot_root_states[env_ids, :3] = motion_res["root_pos"]
            self.simulator.robot_root_states[env_ids, 3:7] = motion_res["root_rot"]
            self.simulator.robot_root_states[env_ids, 7:10] = (
                motion_res["root_vel"] * self.motion_ref_reset_root_vel_scale
            )
            self.simulator.robot_root_states[env_ids, 10:13] = (
                motion_res["root_ang_vel"] * self.motion_ref_reset_root_ang_vel_scale
            )
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos
            self.simulator.dof_vel[env_ids] = 0.0
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.simulator.robot_root_states[env_ids, 7:13] = 0.0
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self._refresh_sim_tensors()
        reset_cylinder_actors(self, env_ids)
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)

    def _reset_cylinder_ball_states(self, env_ids):
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
        root_anchor = self.simulator.robot_root_states[env_ids, 0:3]
        self.ball_root_states[env_ids, 0:3] = (
            root_anchor + self.ball_init_offset.unsqueeze(0) + pos_noise
        )
        self.ball_root_states[env_ids, 3:7] = 0.0
        self.ball_root_states[env_ids, 6] = 1.0
        self.ball_root_states[env_ids, 7:10] = self.ball_init_lin_vel.unsqueeze(0) + lin_vel_noise
        self.ball_root_states[env_ids, 10:13] = 0.0
        target_foot_pos, _ = self._get_target_foot_point_state()
        target_foot_rot = self.simulator._rigid_body_rot[env_ids, self.target_foot_body_idx]
        ball_pos_foot = quat_rotate_inverse(
            target_foot_rot,
            self.ball_root_states[env_ids, 0:3] - target_foot_pos[env_ids],
        )
        self.reset_ball_pos_foot_mean[:] = ball_pos_foot.mean(dim=0)

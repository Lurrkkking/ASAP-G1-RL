import torch


class HitBallStateMachineMixin:
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

        non_target_near_ball = near_ball.clone()
        non_target_near_ball[:, self.right_foot_candidate_body_indices] = False
        non_target_contact = contact_like & non_target_near_ball
        wrong_contact_force = robot_body_forces.masked_fill(~non_target_near_ball, -1.0)
        wrong_force_max, wrong_contact_body_id = torch.max(wrong_contact_force, dim=1)
        non_target_body_dist = body_dist_to_ball.masked_fill(~non_target_near_ball, float("inf"))
        min_non_target_body_dist, nearest_non_target_body_id = torch.min(non_target_body_dist, dim=1)
        wrong_body_near_ball = min_non_target_body_dist < self.contact_dist_threshold
        wrong_contact_body_id = torch.where(
            wrong_force_max > self.contact_force_threshold,
            wrong_contact_body_id,
            nearest_non_target_body_id,
        )
        new_wrong_contact = (
            (~self.has_first_contact)
            & (~new_target_contact)
            & (
                (wrong_force_max > self.contact_force_threshold)
                | (ball_has_contact & wrong_body_near_ball)
            )
        )

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

        prep_ready_for_swing = self._get_prep_to_swing_ready_mask()
        new_swing = (self.task_phase == self.PRE_CONTACT_PREP) & prep_ready_for_swing & (~self.has_first_contact)
        self.task_phase[new_swing] = self.PRE_CONTACT_SWING

        new_contacted = (
            (self.task_phase == self.PRE_CONTACT_PREP) | (self.task_phase == self.PRE_CONTACT_SWING)
        ) & new_target_contact
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
        self._log_single_hit_state_machine_stats(
            contact_count,
            target_contact_count,
            ball_contact_force_norm,
            right_foot_ball_dist,
            ball_has_contact,
            candidate_near_ball,
            prep_ready_for_swing,
        )

    def _debug_single_hit_log(self):
        if self.num_envs == 0:
            return

        self.log_dict["phase_env0"] = self.task_phase[0].float()
        self.log_dict["first_contact_body_env0"] = self.first_contact_body_id[0].float()
        self.log_dict["first_contact_time_env0"] = self.first_contact_time[0].float()
        self.log_dict["first_contact_target_env0"] = self.first_contact_is_target[0].float()
        self.log_dict["success_env0"] = self.success_buf[0].float()
        self.log_dict["term_reason_env0"] = self.term_reason[0].float()

    def _mark_termination(self, mask, reason):
        reset_mask = self.reset_buf.bool()
        new_mask = mask & (~reset_mask)
        self.reset_buf |= mask
        self.term_reason[new_mask] = reason

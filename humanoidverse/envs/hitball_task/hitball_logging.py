import torch


class HitBallLoggingMixin:
    def _log_single_hit_state_machine_stats(
        self,
        contact_count,
        target_contact_count,
        ball_contact_force_norm,
        right_foot_ball_dist,
        ball_has_contact,
        candidate_near_ball,
        prep_ready_for_swing,
    ):
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
        self.log_dict["prep_to_swing_ready_rate"] = prep_ready_for_swing.float().mean()
        self.log_dict["prep_phase_rate"] = (self.task_phase == self.PRE_CONTACT_PREP).float().mean()
        self.log_dict["swing_phase_rate"] = (self.task_phase == self.PRE_CONTACT_SWING).float().mean()
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

    def _log_hitball_termination_stats(self):
        self.log_dict["term_timeout_no_contact"] = (self.term_reason == self.TERM_TIMEOUT_NO_CONTACT).float().mean()
        self.log_dict["term_wrong_body"] = (self.term_reason == self.TERM_WRONG_BODY_CONTACT).float().mean()
        self.log_dict["term_ball_ground"] = (self.term_reason == self.TERM_BALL_GROUND).float().mean()
        self.log_dict["term_post_window"] = (self.term_reason == self.TERM_POST_CONTACT_WINDOW).float().mean()

    def _publish_hitball_extras(self):
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
        if self.hitball_motion_prior_enabled:
            self.extras["hitball"].update(
                {
                    "ref_motion_phase": self.ref_motion_phase.clone(),
                    "ref_motion_time": self.ref_motion_time.clone(),
                    "ref_motion_frame": self.ref_motion_frame.clone(),
                    "tracking_phase_gate": self.hitball_tracking_phase_gate.clone(),
                    "motion_phase_stage": self.motion_phase_stage.clone(),
                    "motion_phase_start_step": self.motion_phase_start_step.clone(),
                    "motion_phase_local_time": self.motion_phase_local_time.clone(),
                    "motion_phase_anchor_time": self.motion_phase_anchor_time.clone(),
                }
            )

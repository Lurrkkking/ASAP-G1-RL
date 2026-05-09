import torch


class BallControlStateMachineMixin:
    def _init_ball_control_state_machine(self):
        self.task_mode = str(getattr(self.config, "task_mode", "ball_control")).lower()
        self.is_kickup_task_mode = self.task_mode == "kickup"
        self.control_touch_window_steps = int(getattr(self.config, "control_touch_window_steps", 40))
        self.reposition_contact_ignore_steps = int(
            getattr(self.config, "reposition_contact_ignore_steps", 4)
        )
        self.control_touch_vz_trigger_max = float(
            getattr(self.config, "control_touch_vz_trigger_max", 0.0)
        )
        self.contact_force_threshold = float(getattr(self.config, "contact_force_threshold", 1.0))
        self.ball_contact_force_threshold = float(
            getattr(self.config, "ball_contact_force_threshold", self.contact_force_threshold)
        )
        self.contact_dist_threshold = float(getattr(self.config, "contact_dist_threshold", 0.18))
        self.ball_contact_min_height = float(
            getattr(
                self.config,
                "ball_contact_min_height",
                getattr(self.config.ball, "radius", 0.10) * 0.5,
            )
        )
        self.kickoff_ball_z_max = float(
            getattr(self.config, "kickoff_ball_z_max", getattr(self.config.ball, "radius", 0.10) + 0.20)
        )
        self.kickoff_x_range = torch.tensor(
            getattr(self.config, "kickoff_x_range", [0.10, 0.55]),
            dtype=torch.float,
            device=self.device,
        )
        self.kickoff_y_range = torch.tensor(
            getattr(self.config, "kickoff_y_range", [-0.30, 0.10]),
            dtype=torch.float,
            device=self.device,
        )

        self.right_foot_body_name = getattr(
            self.config,
            "target_contact_body",
            getattr(self.config.robot, "right_foot_name", None),
        )
        if self.right_foot_body_name is None:
            raise AttributeError(
                "BallControlTask requires `target_contact_body` or `config.robot.right_foot_name`."
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
        self.ball_body_env_idx = self.num_bodies

        self.control_mode = torch.full(
            (self.num_envs,),
            self.REPOSITION,
            dtype=torch.long,
            device=self.device,
        )
        self.task_phase = self.control_mode
        self.mode_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reposition_contact_ignore_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.just_target_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_wrong_body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_reposition_contact_violation = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_started_control_touch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.entered_target_state_zone_since_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.missed_target_state_cycle_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.kickup_ready_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_ball_control_state_machine(self, env_ids):
        if len(env_ids) == 0:
            return
        self.control_mode[env_ids] = self.REPOSITION
        self.mode_step_buf[env_ids] = 0
        self.reposition_contact_ignore_buf[env_ids] = 0
        self.just_target_contact[env_ids] = False
        self.just_wrong_body_contact[env_ids] = False
        self.just_reposition_contact_violation[env_ids] = False
        self.just_started_control_touch[env_ids] = False
        self.entered_target_state_zone_since_contact[env_ids] = False
        self.missed_target_state_cycle_buf[env_ids] = False
        self.kickup_ready_mask[env_ids] = False

    def _update_ball_control_state_machine(self):
        self._get_valid_stance_mask()
        self.mode_step_buf += 1
        self.reposition_contact_ignore_buf[:] = torch.clamp_min(
            self.reposition_contact_ignore_buf - 1, 0
        )
        prev_control_mode = self.control_mode.clone()
        self.just_target_contact[:] = False
        self.just_wrong_body_contact[:] = False
        self.just_reposition_contact_violation[:] = False
        self.just_started_control_touch[:] = False
        self.missed_target_state_cycle_buf[:] = False
        target_contact = self._get_target_contact_mask()
        self.just_target_contact[target_contact] = True
        self.has_target_contact |= target_contact
        wrong_body_contact = self._get_wrong_body_contact_mask() & (~target_contact)
        self.just_wrong_body_contact[wrong_body_contact] = True
        self.entered_target_state_zone_since_contact |= (
            (self.control_mode == self.REPOSITION) & self._get_ball_target_state_zone_mask()
        )

        active_post_contact = self.has_target_contact
        self.post_contact_steps[active_post_contact] += 1
        self.post_contact_steps[~active_post_contact] = 0
        self.post_contact_max_height[active_post_contact] = torch.maximum(
            self.post_contact_max_height[active_post_contact],
            self.ball_pos[active_post_contact, 2],
        )
        self.post_contact_max_vz[active_post_contact] = torch.maximum(
            self.post_contact_max_vz[active_post_contact],
            self.ball_lin_vel[active_post_contact, 2],
        )

        in_reposition = self.control_mode == self.REPOSITION
        if self.is_kickup_task_mode:
            kickup_ready = self._get_kickup_ready_mask()
            self.kickup_ready_mask[:] = kickup_ready
            start_touch = in_reposition & kickup_ready
            self.missed_target_state_cycle_buf[:] = False
        else:
            ball_in_target_state_zone = self._get_ball_target_state_zone_mask()
            ball_vz = self.ball_lin_vel[:, 2]
            start_touch = (
                in_reposition
                & (~ball_in_target_state_zone)
                & (ball_vz < self.control_touch_vz_trigger_max)
            )
            self.missed_target_state_cycle_buf[:] = (
                start_touch & (~self.entered_target_state_zone_since_contact)
            )

        in_control_touch = self.control_mode == self.CONTROL_TOUCH
        touch_window_done = self.mode_step_buf >= self.control_touch_window_steps
        end_touch = in_control_touch & (target_contact | touch_window_done)

        self.control_mode[end_touch] = self.REPOSITION
        self.mode_step_buf[end_touch] = 0
        self.reposition_contact_ignore_buf[end_touch] = self.reposition_contact_ignore_steps
        self.entered_target_state_zone_since_contact[target_contact] = False

        self.control_mode[start_touch] = self.CONTROL_TOUCH
        self.mode_step_buf[start_touch] = 0
        self.just_started_control_touch[start_touch] = True
        self.has_target_contact[start_touch] = False
        self.post_contact_steps[start_touch] = 0
        self.post_contact_max_height[start_touch] = 0.0
        self.post_contact_max_vz[start_touch] = 0.0

        reposition_contact_violation = (
            (prev_control_mode == self.REPOSITION)
            & target_contact
            & (self.reposition_contact_ignore_buf == 0)
        )
        self.just_reposition_contact_violation[reposition_contact_violation] = True

    def _get_kickup_ready_mask(self):
        ball_rel_base = self._get_ball_rel_base_heading()
        ball_z_world = self.ball_pos[:, 2]
        ball_xy_ready = (
            (ball_rel_base[:, 0] >= self.kickoff_x_range[0])
            & (ball_rel_base[:, 0] <= self.kickoff_x_range[1])
            & (ball_rel_base[:, 1] >= self.kickoff_y_range[0])
            & (ball_rel_base[:, 1] <= self.kickoff_y_range[1])
        )
        return (
            self.valid_stance_mask
            & ball_xy_ready
            & (ball_z_world <= self.kickoff_ball_z_max)
            & (torch.norm(ball_rel_base[:, :2], dim=-1) <= self.contact_dist_threshold * 3.0)
        )

    def _get_target_contact_mask(self):
        if self.ball_body_env_idx >= self.simulator.contact_forces.shape[1]:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        ball_contact_force = torch.norm(
            self.simulator.contact_forces[:, self.ball_body_env_idx], dim=-1
        )
        ball_has_contact = (
            (ball_contact_force > self.ball_contact_force_threshold)
            & (self.ball_pos[:, 2] > self.ball_contact_min_height)
        )
        candidate_pos = self.simulator._rigid_body_pos[:, self.right_foot_candidate_body_indices]
        candidate_dist = torch.norm(candidate_pos - self.ball_pos.unsqueeze(1), dim=-1)
        candidate_near_ball = torch.min(candidate_dist, dim=1).values < self.contact_dist_threshold
        return ball_has_contact & candidate_near_ball

    def _get_wrong_body_contact_mask(self):
        if self.ball_body_env_idx >= self.simulator.contact_forces.shape[1]:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot_body_forces = torch.norm(self.simulator.contact_forces[:, : self.num_bodies], dim=-1)
        ball_contact_force = torch.norm(
            self.simulator.contact_forces[:, self.ball_body_env_idx], dim=-1
        )
        ball_has_contact = (
            (ball_contact_force > self.ball_contact_force_threshold)
            & (self.ball_pos[:, 2] > self.ball_contact_min_height)
        )
        robot_body_pos = self.simulator._rigid_body_pos[:, : self.num_bodies]
        body_dist_to_ball = torch.norm(robot_body_pos - self.ball_pos.unsqueeze(1), dim=-1)
        near_ball = body_dist_to_ball < self.contact_dist_threshold
        non_target_near_ball = near_ball.clone()
        non_target_near_ball[:, self.right_foot_candidate_body_indices] = False
        wrong_contact_force = robot_body_forces.masked_fill(~non_target_near_ball, -1.0)
        wrong_force_max = torch.max(wrong_contact_force, dim=1).values
        wrong_body_near_ball = torch.any(non_target_near_ball, dim=1)
        return ball_has_contact & (
            (wrong_force_max > self.contact_force_threshold) | wrong_body_near_ball
        )

    def _log_ball_control_state_machine(self):
        reposition_mask = self.control_mode == self.REPOSITION
        control_touch_mask = self.control_mode == self.CONTROL_TOUCH

        self.log_dict["ball_control/mode"] = self.control_mode.float().mean()
        self.log_dict["ball_control/reposition_phase_rate"] = reposition_mask.float().mean()
        self.log_dict["ball_control/control_touch_phase_rate"] = control_touch_mask.float().mean()
        self.log_dict["ball_control/target_contact_rate"] = self.just_target_contact.float().mean()
        self.log_dict["ball_control/reposition_contact_violation_rate"] = (
            self.just_reposition_contact_violation.float().mean()
        )
        self.log_dict["ball_control/wrong_body_contact_rate"] = self.just_wrong_body_contact.float().mean()
        self.log_dict["ball_control/kickup_ready_rate"] = self.kickup_ready_mask.float().mean()

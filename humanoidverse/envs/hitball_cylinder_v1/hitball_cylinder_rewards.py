import torch

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


def compute_cylinder_rewards(env):
    LeggedRobotBase._compute_reward(env)
    env.log_dict["hitball_cylinder/reward_mean"] = env.rew_buf.mean()


class HitBallCylinderRewardMixin:
    def _phase_mask(self, phase_id):
        return self.phase == phase_id

    def _get_phase_robot_stable_mask(self):
        gravity_xy = torch.norm(self.projected_gravity[:, :2], dim=-1)
        root_height = self.simulator.robot_root_states[:, 2]
        base_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        return (
            (gravity_xy <= self.phase_gravity_xy_max)
            & (root_height >= self.phase_root_height_min)
            & (base_vel_xy <= self.phase_base_vel_xy_max)
        )

    def _get_pre_impact_foot_to_ball_stats(self):
        tau = self.prediction_horizons[self.strike_gate_horizon_index]
        rewards_cfg = getattr(self.config, "rewards", self.config)
        prediction_gravity_z = float(getattr(rewards_cfg, "prediction_gravity_z", -9.81))
        pre_impact_pos_sigma = float(getattr(rewards_cfg, "pre_impact_pos_sigma", 0.05))
        pre_impact_vel_norm = float(getattr(rewards_cfg, "pre_impact_vel_norm", 1.0))

        g_vec = torch.zeros(self.num_envs, 3, device=self.device)
        g_vec[:, 2] = prediction_gravity_z
        pred_ball_world = self.ball_pos + self.ball_lin_vel * tau + 0.5 * g_vec * (tau * tau)

        foot_pos, foot_vel = self._get_target_foot_point_state()
        to_ball = pred_ball_world - foot_pos
        dist = torch.norm(to_ball, dim=-1)
        pos_reward = torch.exp(-(dist * dist) / max(pre_impact_pos_sigma, 1e-6))

        to_ball_dir = to_ball / dist.unsqueeze(-1).clamp(min=1e-6)
        speed_toward_ball = torch.sum(foot_vel * to_ball_dir, dim=-1)
        vel_reward = torch.relu(speed_toward_ball) / max(pre_impact_vel_norm, 1e-6)
        vel_reward = torch.clamp(vel_reward, 0.0, 1.0)

        reward = pos_reward * (0.7 + 0.3 * vel_reward)
        active = (
            self.strike_gate
            & (~self.contact_now)
            & (~self.carry_flag)
            & (self.ball_near_foot_duration <= self.zone_sticky_disable_steps)
        )
        return reward, active, dist

    def _reward_zone_reward(self):
        error = env_error_to_zone(self)
        reward = torch.exp(-torch.sum(error * error * self.zone_reward_weight.unsqueeze(0), dim=-1))
        sticky = self.carry_flag | (self.ball_near_foot_duration > self.zone_sticky_disable_steps)
        reward = reward * torch.where(sticky, self.zone_sticky_scale, torch.ones_like(reward))
        return reward

    def _reward_return_progress_reward(self):
        ball_pos_base = self.ball_pos_base
        ball_vel_base = self.ball_vel_base
        error_xy = self.control_zone_center[:2].unsqueeze(0) - ball_pos_base[:, :2]
        dist_xy = torch.norm(error_xy, dim=-1)
        direction = error_xy / dist_xy.unsqueeze(-1).clamp(min=1e-6)
        speed_to_zone = torch.sum(ball_vel_base[:, :2] * direction, dim=-1)
        outside = (~self._get_control_zone_mask()).float()
        sticky = self.carry_flag | (self.ball_near_foot_duration > self.zone_sticky_disable_steps)
        active = (~sticky).float()
        return torch.relu(speed_to_zone) * outside * active

    def _reward_valid_kick_reward(self):
        return self.valid_kick.float()

    def _reward_pre_impact_foot_to_ball_reward(self):
        reward, active, _ = self._get_pre_impact_foot_to_ball_stats()
        return reward * active.float()

    def _reward_phase_wait_stability_reward(self):
        active = self._phase_mask(self.PHASE_WAIT)
        return self._get_phase_robot_stable_mask().float() * active.float()

    def _reward_phase_prepare_foot_to_ball_reward(self):
        reward, _, _ = self._get_pre_impact_foot_to_ball_stats()
        active = (
            self._phase_mask(self.PHASE_PREPARE)
            & (~self.contact_now)
            & (~self.carry_flag)
            & (self.ball_near_foot_duration <= self.zone_sticky_disable_steps)
        )
        return reward * active.float()

    def _reward_phase_recover_stability_reward(self):
        active = self._phase_mask(self.PHASE_RECOVER)
        return self._get_phase_robot_stable_mask().float() * active.float()

    def _reward_carry_penalty(self):
        return self.carry_flag.float()

    def _reward_non_target_contact_penalty(self):
        return (self.non_target_contact_now & (~self.prev_non_target_contact_now)).float()

    def _reward_base_drift_penalty(self):
        drift = torch.norm(
            self.simulator.robot_root_states[:, 0:2] - self.episode_start_base_pos[:, 0:2],
            dim=-1,
        )
        return drift

    def _reward_style_prior(self):
        dof_err = torch.mean((self.simulator.dof_pos - self.default_dof_pos) ** 2, dim=-1)
        return torch.exp(-dof_err / max(self.style_prior_sigma, 1e-6))

    def _reward_penalty_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)


def env_error_to_zone(env):
    pos = env.ball_pos_base
    center = env.control_zone_center.unsqueeze(0)
    radial = torch.norm(pos[:, :2] - center[:, :2], dim=-1) - env.control_zone_radius
    z_low = env.control_zone_z_range[0] - pos[:, 2]
    z_high = pos[:, 2] - env.control_zone_z_range[1]
    return torch.stack([torch.relu(radial), torch.relu(z_low), torch.relu(z_high)], dim=-1)

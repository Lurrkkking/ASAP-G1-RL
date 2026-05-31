"""Goalkeeper reward functions — simple block-orientated shaping for smoke test."""
import torch


class GoalkeeperRewardMixin:
    """Mixin providing goalkeeper-specific reward terms."""

    def _init_goalkeeper_rewards(self):
        """Called from _init_buffers — initialise reward-related tensors."""
        self.goal_center = torch.tensor(
            getattr(self.config, "goal_center", [-0.5, 0.0, 0.0]),
            dtype=torch.float, device=self.device,
        )
        self.goal_width = float(getattr(self.config, "goal_width", 2.0))
        self.goal_height = float(getattr(self.config, "goal_height", 1.5))
        self.block_vel_change_threshold = float(
            getattr(self.config, "block_vel_change_threshold", 0.5)
        )
        self.block_contact_force_threshold = float(
            getattr(self.config, "block_contact_force_threshold", 1.0)
        )
        # Store ball velocity at episode start for reference
        self.ball_init_vel_dir = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    def _reward_alive(self):
        """Survival reward."""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_ball_blocked(self):
        """Reward for making contact with the ball (any body).

        Scaled by the component of ball velocity change away from goal.
        """
        # Ball velocity component toward goal (negative x in robot frame = toward goal)
        # "Toward goal" means velocity in -x direction in world frame
        vel_toward_goal_before = -self.ball_init_vel_dir[:, 0]  # how fast toward goal initially
        vel_toward_goal_now = -self.ball_lin_vel[:, 0]  # how fast toward goal now

        # Positive reward when ball is slowed down / reversed
        vel_change_away = vel_toward_goal_before - vel_toward_goal_now
        # Clamp to avoid huge rewards
        vel_change_away = torch.clamp(vel_change_away, min=-1.0, max=5.0)

        # Only reward if there was contact this episode
        contact_bonus = self.ball_contact_this_episode.float()
        return vel_change_away * contact_bonus

    def _reward_ball_contact_bonus(self):
        """One-time bonus for first contact with the ball."""
        return self.just_got_ball_contact.float()

    def _reward_goal_conceded_penalty(self):
        """Penalty when ball enters the goal (passes robot toward -x past goal line)."""
        return -self.goal_conceded_this_episode.float()

    def _reward_ball_near_robot(self):
        """Small shaping reward: ball being close to robot (encourages positioning)."""
        base_pos = self.simulator.robot_root_states[:, 0:3]
        ball_dist = torch.norm(self.ball_pos - base_pos, dim=-1)
        # Reward when ball is within ~1.5m (interaction range)
        # But only before contact (after contact ball may fly away which is fine)
        near = torch.exp(-ball_dist / 0.5)
        pre_contact = (~self.ball_contact_this_episode).float()
        return near * pre_contact

    def _reward_ball_vel_away_from_goal(self):
        """Continuous reward: ball velocity component AWAY from goal (positive x)."""
        # Positive x velocity = away from goal
        vel_away = self.ball_lin_vel[:, 0]
        # Only reward after contact
        post_contact = self.ball_contact_this_episode.float()
        return torch.clamp(vel_away, min=-2.0, max=5.0) * post_contact

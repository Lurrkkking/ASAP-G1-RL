"""Goalkeeper reset logic — ball init in front of robot, moving toward goal."""
import torch
from humanoidverse.utils.torch_utils import torch_rand_float


class GoalkeeperResetMixin:
    """Mixin providing goalkeeper-specific reset: prepared pose + ball spawn."""

    def _init_goalkeeper_reset(self):
        """Called from _init_buffers — initialise ball/reset tensors."""
        # Ball init ranges (config-driven, Q1 smoke defaults)
        self.ball_init_x_range = torch.tensor(
            getattr(self.config, "ball_init_x_range", [2.0, 3.0]),
            dtype=torch.float, device=self.device,
        )
        self.ball_init_y_range = torch.tensor(
            getattr(self.config, "ball_init_y_range", [-0.2, 0.2]),
            dtype=torch.float, device=self.device,
        )
        # Three discrete shot heights: low / mid / high (config-driven)
        shot_heights = getattr(self.config, "ball_shot_heights", None)
        if shot_heights is not None and len(shot_heights) == 3:
            self.ball_shot_heights = torch.tensor(shot_heights, dtype=torch.float, device=self.device)
            self.ball_shot_height_noise = float(getattr(self.config, "ball_shot_height_noise", 0.03))
            self.use_three_heights = True
        else:
            # Fallback: uniform z range
            self.ball_init_z_range = torch.tensor(
                getattr(self.config, "ball_init_z_range", [0.12, 0.25]),
                dtype=torch.float, device=self.device,
            )
            self.use_three_heights = False

        self.ball_init_speed_range = torch.tensor(
            getattr(self.config, "ball_init_speed_range", [1.0, 2.0]),
            dtype=torch.float, device=self.device,
        )
        self.ball_init_yaw_deviation = float(
            getattr(self.config, "ball_init_yaw_deviation", 0.1)
        )
        self.ball_init_angle_noise = float(
            getattr(self.config, "ball_init_angle_noise", 0.05)
        )

        # Track which shot height was selected this episode
        self.ball_shot_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Snapshot prepared pose after simulator equilibrium (like stand_smoke)
        self._prepared_dof_pos = self.simulator.dof_pos[0].clone()
        self.default_dof_pos[:] = self._prepared_dof_pos.unsqueeze(0)

    def _reset_goalkeeper_buffers(self, env_ids):
        """Reset per-episode goalkeeper state."""
        self.ball_contact_this_episode[env_ids] = False
        self.ball_blocked_this_episode[env_ids] = False
        self.goal_conceded_this_episode[env_ids] = False
        for i in env_ids.cpu().tolist():
            self.ball_contact_body_name_buf[i] = ""
        self.ball_contact_body_idx_buf[env_ids] = -1

    def _reset_dofs(self, env_ids, target_state=None):
        """Use prepared pose (simulator equilibrium), not yaml default."""
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = (
                self._prepared_dof_pos.unsqueeze(0).expand(len(env_ids), -1)
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Reset root to origin + env offset, zero velocity."""
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.simulator.robot_root_states[env_ids, 7:13] = 0.0

    def _reset_ball_states(self, env_ids):
        """Spawn ball in front of robot with velocity toward goal (robot)."""
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        # Random ball position in front of robot
        ball_x = torch_rand_float(
            self.ball_init_x_range[0], self.ball_init_x_range[1],
            (n, 1), device=str(self.device)
        ).squeeze(-1)
        ball_y = torch_rand_float(
            self.ball_init_y_range[0], self.ball_init_y_range[1],
            (n, 1), device=str(self.device)
        ).squeeze(-1)
        if self.use_three_heights:
            # Select one of three discrete heights per env, then add small noise
            level_idx = torch.randint(0, 3, (n,), device=self.device)
            self.ball_shot_level[env_ids] = level_idx
            base_z = self.ball_shot_heights[level_idx]
            z_noise = torch_rand_float(
                -self.ball_shot_height_noise, self.ball_shot_height_noise,
                (n, 1), device=str(self.device)
            ).squeeze(-1)
            ball_z = base_z + z_noise
            # Clamp to prevent ball spawning below ground
            ball_z = torch.clamp(ball_z, min=0.13)
        else:
            ball_z = torch_rand_float(
                self.ball_init_z_range[0], self.ball_init_z_range[1],
                (n, 1), device=str(self.device)
            ).squeeze(-1)

        # Ball position in world frame: robot root + offset
        robot_pos = self.simulator.robot_root_states[env_ids, 0:3]
        self.ball_root_states[env_ids, 0] = robot_pos[:, 0] + ball_x
        self.ball_root_states[env_ids, 1] = robot_pos[:, 1] + ball_y
        self.ball_root_states[env_ids, 2] = ball_z

        # Ball orientation (identity)
        self.ball_root_states[env_ids, 3:7] = 0.0
        self.ball_root_states[env_ids, 6] = 1.0

        # Ball velocity: toward robot (negative x), with slight angle noise
        speed = torch_rand_float(
            self.ball_init_speed_range[0], self.ball_init_speed_range[1],
            (n, 1), device=str(self.device)
        ).squeeze(-1)

        # Direction: toward robot (-x) with yaw deviation
        yaw_noise = torch_rand_float(
            -self.ball_init_yaw_deviation, self.ball_init_yaw_deviation,
            (n, 1), device=str(self.device)
        ).squeeze(-1)

        # Also add angle noise for z component (slight lob)
        angle_noise = torch_rand_float(
            -self.ball_init_angle_noise, self.ball_init_angle_noise,
            (n, 1), device=str(self.device)
        ).squeeze(-1)

        self.ball_root_states[env_ids, 7] = -speed * torch.cos(yaw_noise) * torch.cos(angle_noise)
        self.ball_root_states[env_ids, 8] = speed * torch.sin(yaw_noise)
        self.ball_root_states[env_ids, 9] = speed * torch.sin(angle_noise)

        # Zero angular velocity
        self.ball_root_states[env_ids, 10:13] = 0.0

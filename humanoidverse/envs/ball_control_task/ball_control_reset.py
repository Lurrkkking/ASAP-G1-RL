from humanoidverse.utils.torch_utils import torch_rand_float


class BallControlResetMixin:
    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states=target_states)
        self._reset_ball_states(env_ids)

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

import torch

from humanoidverse.utils.torch_utils import quat_rotate, quat_rotate_inverse


def compute_cylinder_obs(env):
    base_pos = env.simulator.robot_root_states[:, 0:3]
    heading_inv = env._get_base_heading_quat_inv()
    target_foot_pos, target_foot_vel = env._get_target_foot_point_state()
    target_foot_rot = env.simulator._rigid_body_rot[:, env.target_foot_body_idx]

    ball_pos_base = quat_rotate(heading_inv, env.ball_pos - base_pos)
    ball_vel_base = quat_rotate(heading_inv, env.ball_lin_vel)
    ball_pos_foot = quat_rotate_inverse(target_foot_rot, env.ball_pos - target_foot_pos)
    ball_vel_foot = quat_rotate_inverse(target_foot_rot, env.ball_lin_vel - target_foot_vel)

    g_vec = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)
    g_vec[:, 2] = float(getattr(env.config, "prediction_gravity_z", -9.81))

    pred_pos_base_parts = []
    pred_vz_parts = []
    pred_pos_foot_parts = []
    for tau in env.prediction_horizons:
        pred_pos = env.ball_pos + env.ball_lin_vel * tau + 0.5 * g_vec * tau * tau
        pred_vel = env.ball_lin_vel + g_vec * tau
        pred_pos_base_parts.append(quat_rotate(heading_inv, pred_pos - base_pos))
        pred_vz_parts.append(pred_vel[:, 2:3])
        pred_pos_foot_parts.append(quat_rotate_inverse(target_foot_rot, pred_pos - target_foot_pos))

    env.ball_pos_base = ball_pos_base
    env.ball_vel_base = ball_vel_base
    env.ball_pos_foot = ball_pos_foot
    env.ball_vel_foot = ball_vel_foot
    env.ball_pred_pos_base_tau = torch.cat(pred_pos_base_parts, dim=-1)
    env.ball_pred_vz_tau = torch.cat(pred_vz_parts, dim=-1)
    env.ball_pred_pos_foot_tau = torch.cat(pred_pos_foot_parts, dim=-1)

    return torch.cat(
        [
            ball_pos_base,
            ball_vel_base,
            ball_pos_foot,
            ball_vel_foot,
            env.ball_pred_pos_base_tau,
            env.ball_pred_vz_tau,
            env.ball_pred_pos_foot_tau,
        ],
        dim=-1,
    )

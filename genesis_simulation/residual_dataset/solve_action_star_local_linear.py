import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genesis_simulation.residual_dataset.build_paired_delta_from_isaac_anchors import (  # noqa: E402
    _get_state,
    _set_state_from_s,
)
import genesis_simulation.run_onnx_motiontracking as mt  # noqa: E402


def _load_isaac_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        for key in ("s", "a", "s_next"):
            if key not in data:
                raise KeyError(f"Missing key in isaac npz: {key}")

        s = data["s"].astype(np.float32)
        a = data["a"].astype(np.float32)
        s_next = data["s_next"].astype(np.float32)

        if s.ndim != 3:
            raise ValueError(f"Expected padded 3D fixed-state npz, got s.shape={s.shape}")

        if "mask" in data:
            mask = data["mask"].astype(bool)
        elif "episode_lengths" in data:
            lengths = data["episode_lengths"].astype(np.int32)
            mask = np.zeros(s.shape[:2], dtype=bool)
            for idx, length in enumerate(lengths):
                mask[idx, : int(length)] = True
        else:
            raise KeyError("Padded npz needs mask or episode_lengths")

    return s, a, s_next, mask


def _make_weight_vector(num_dof: int, root_weight: float, dof_pos_weight: float, dof_vel_weight: float):
    weights = np.concatenate(
        [
            np.full(7, root_weight, dtype=np.float32),
            np.full(num_dof, dof_pos_weight, dtype=np.float32),
            np.full(num_dof, dof_vel_weight, dtype=np.float32),
        ]
    )
    return weights


def _weighted_mae(delta: np.ndarray, weights: np.ndarray):
    return float(np.mean(np.abs(delta) * weights))


def _weighted_mse(delta: np.ndarray, weights: np.ndarray):
    return float(np.mean((delta * weights) ** 2))


def _rollout_one_step(scene, robot, motor_dofs, state, target_pos):
    _set_state_from_s(robot, motor_dofs, state)
    target_pos_t = torch.tensor(target_pos.astype(np.float32), dtype=torch.float32, device=mt.SIM_DEVICE)
    for _ in range(mt.CONTROL_DECIMATION):
        if mt.USE_IMPLICIT_PD:
            robot.control_dofs_position(position=target_pos_t, dofs_idx_local=motor_dofs)
        else:
            dof_pos = mt.as_tensor_2d(robot.get_dofs_position(dofs_idx_local=motor_dofs))[0].detach().cpu().numpy()
            dof_vel = mt.as_tensor_2d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))[0].detach().cpu().numpy()
            torques_req = mt.KP * (target_pos - dof_pos) - mt.KD * dof_vel
            torques = np.clip(torques_req, -mt.TORQUE_LIMITS, mt.TORQUE_LIMITS)
            robot.control_dofs_force(
                torch.tensor(torques, dtype=torch.float32, device=mt.SIM_DEVICE),
                dofs_idx_local=motor_dofs,
            )
        scene.step()
    return _get_state(robot, motor_dofs)


def _build_scene():
    mt.maybe_autofill_time_offset()
    if not Path(mt.URDF_PATH).is_file():
        raise FileNotFoundError(f"URDF not found: {mt.URDF_PATH}")

    mt.gs.init(backend=mt.gs.gpu if "cuda" in mt.SIM_DEVICE else mt.gs.cpu)
    scene = mt.gs.Scene(
        sim_options=mt.gs.options.SimOptions(dt=mt.SIM_DT, substeps=1),
        rigid_options=mt.gs.options.RigidOptions(enable_self_collision=mt.USE_SELF_COLLISION),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(mt.gs.morphs.Plane(), material=mt.gs.materials.Rigid(friction=mt.FLOOR_FRICTION))
    robot = scene.add_entity(
        mt.gs.morphs.URDF(
            file=mt.URDF_PATH,
            merge_fixed_links=True,
            links_to_keep=mt.BODY_NAMES,
            pos=(0.0, 0.0, 0.8),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )
    scene.build()

    motor_dofs = [robot.get_joint(name).dof_idx_local for name in mt.RL_JOINT_NAMES]
    if mt.USE_IMPLICIT_PD:
        robot.set_dofs_kp(kp=torch.tensor(mt.KP, dtype=torch.float32, device=mt.SIM_DEVICE), dofs_idx_local=motor_dofs)
        robot.set_dofs_kv(kv=torch.tensor(mt.KD, dtype=torch.float32, device=mt.SIM_DEVICE), dofs_idx_local=motor_dofs)
    return scene, robot, motor_dofs


def main():
    parser = argparse.ArgumentParser(description="Solve local-linear action targets a_star with Genesis finite differences")
    parser.add_argument("--isaac-npz", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--max-delta", type=float, default=0.02)
    parser.add_argument("--root-weight", type=float, default=0.0)
    parser.add_argument("--dof-pos-weight", type=float, default=1.0)
    parser.add_argument("--dof-vel-weight", type=float, default=0.1)
    args = parser.parse_args()

    isaac_npz = Path(args.isaac_npz)
    out_npz = Path(args.out_npz)
    s, a_base, s_next_isaac, mask = _load_isaac_npz(isaac_npz)

    scene, robot, motor_dofs = _build_scene()
    action_dim = a_base.shape[-1]
    state_dim = s.shape[-1]
    weights = _make_weight_vector(action_dim, args.root_weight, args.dof_pos_weight, args.dof_vel_weight)
    sqrt_weights = np.sqrt(weights).astype(np.float32)
    regularizer = float(args.ridge_lambda) * np.eye(action_dim, dtype=np.float32)

    valid_indices = list(zip(*np.nonzero(mask)))
    if args.max_samples > 0:
        valid_indices = valid_indices[: args.max_samples]

    s_out = np.zeros((len(valid_indices), state_dim), dtype=np.float32)
    a_base_out = np.zeros((len(valid_indices), action_dim), dtype=np.float32)
    a_star_out = np.zeros((len(valid_indices), action_dim), dtype=np.float32)
    delta_star_out = np.zeros((len(valid_indices), action_dim), dtype=np.float32)
    s_next_isaac_out = np.zeros((len(valid_indices), state_dim), dtype=np.float32)
    s_next_base_out = np.zeros((len(valid_indices), state_dim), dtype=np.float32)
    s_next_star_out = np.zeros((len(valid_indices), state_dim), dtype=np.float32)
    weighted_jacobian_out = np.zeros((len(valid_indices), state_dim, action_dim), dtype=np.float32)
    base_mae = np.zeros((len(valid_indices),), dtype=np.float32)
    star_mae = np.zeros((len(valid_indices),), dtype=np.float32)
    base_all_mae = np.zeros((len(valid_indices),), dtype=np.float32)
    star_all_mae = np.zeros((len(valid_indices),), dtype=np.float32)
    improvement = np.zeros((len(valid_indices),), dtype=np.float32)
    eta = np.zeros((len(valid_indices),), dtype=np.float32)
    base_mse = np.zeros((len(valid_indices),), dtype=np.float32)
    pred_mse = np.zeros((len(valid_indices),), dtype=np.float32)
    delta_norm = np.zeros((len(valid_indices),), dtype=np.float32)

    for out_idx, (episode_idx, timestep_idx) in enumerate(valid_indices):
        state = s[episode_idx, timestep_idx]
        action = a_base[episode_idx, timestep_idx]
        target_next = s_next_isaac[episode_idx, timestep_idx]

        base_next = _rollout_one_step(scene, robot, motor_dofs, state, action)
        residual = target_next - base_next

        jacobian = np.zeros((state_dim, action_dim), dtype=np.float32)
        for action_idx in range(action_dim):
            action_plus = action.copy()
            action_minus = action.copy()
            action_plus[action_idx] += args.eps
            action_minus[action_idx] -= args.eps
            next_plus = _rollout_one_step(scene, robot, motor_dofs, state, action_plus)
            next_minus = _rollout_one_step(scene, robot, motor_dofs, state, action_minus)
            jacobian[:, action_idx] = (next_plus - next_minus) / (2.0 * args.eps)

        weighted_jacobian = jacobian * sqrt_weights[:, None]
        weighted_residual = residual * sqrt_weights
        lhs = weighted_jacobian.T @ weighted_jacobian + regularizer
        rhs = weighted_jacobian.T @ weighted_residual
        delta_star = np.linalg.solve(lhs, rhs).astype(np.float32)
        delta_star = np.clip(delta_star, -args.max_delta, args.max_delta)
        action_star = (action + delta_star).astype(np.float32)
        star_next = _rollout_one_step(scene, robot, motor_dofs, state, action_star)

        s_out[out_idx] = state
        a_base_out[out_idx] = action
        a_star_out[out_idx] = action_star
        delta_star_out[out_idx] = delta_star
        s_next_isaac_out[out_idx] = target_next
        s_next_base_out[out_idx] = base_next
        s_next_star_out[out_idx] = star_next
        weighted_jacobian_out[out_idx] = weighted_jacobian
        base_delta = target_next - base_next
        star_delta = target_next - star_next
        base_mae[out_idx] = _weighted_mae(base_delta, weights)
        star_mae[out_idx] = _weighted_mae(star_delta, weights)
        base_all_mae[out_idx] = float(np.mean(np.abs(base_delta)))
        star_all_mae[out_idx] = float(np.mean(np.abs(star_delta)))
        improvement[out_idx] = base_mae[out_idx] - star_mae[out_idx]
        eta[out_idx] = improvement[out_idx] / max(base_mae[out_idx], 1e-8)
        base_mse[out_idx] = _weighted_mse(base_delta, weights)
        pred_mse[out_idx] = float(np.mean((weighted_jacobian @ delta_star - weighted_residual) ** 2))
        delta_norm[out_idx] = float(np.linalg.norm(delta_star))

        print(
            f"[{out_idx + 1:04d}/{len(valid_indices):04d}] "
            f"base_mae={base_mae[out_idx]:.6e} star_mae={star_mae[out_idx]:.6e} "
            f"delta_norm={delta_norm[out_idx]:.6e}"
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s_out,
        a_base=a_base_out,
        a_star=a_star_out,
        delta_a_star=delta_star_out,
        s_next_isaac=s_next_isaac_out,
        s_next_base_genesis=s_next_base_out,
        s_next_star_genesis=s_next_star_out,
        weighted_jacobian=weighted_jacobian_out,
        base_mae=base_mae,
        star_mae=star_mae,
        base_all_mae=base_all_mae,
        star_all_mae=star_all_mae,
        improvement=improvement,
        eta=eta,
        base_mse=base_mse,
        pred_mse=pred_mse,
        delta_norm=delta_norm,
        eps=np.asarray(args.eps, dtype=np.float32),
        ridge_lambda=np.asarray(args.ridge_lambda, dtype=np.float32),
        max_delta=np.asarray(args.max_delta, dtype=np.float32),
        weights=weights,
    )

    print(f"[DONE] saved: {out_npz}")
    print(f"[INFO] samples={len(valid_indices)}")
    print(f"[INFO] base_mae_mean={float(base_mae.mean()):.6e}")
    print(f"[INFO] star_mae_mean={float(star_mae.mean()):.6e}")
    print(f"[INFO] improvement_mean={float(improvement.mean()):.6e}")
    print(f"[INFO] improvement_frac={float(np.mean(improvement > 0.0)):.6f}")


if __name__ == "__main__":
    main()

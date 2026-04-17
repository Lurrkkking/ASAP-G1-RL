import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import genesis_simulation.run_onnx_motiontracking as mt


def _to_numpy_1d(x):
    x = mt.as_tensor_2d(x)
    return x[0].detach().cpu().numpy().astype(np.float32)


def _get_state(robot, motor_dofs):
    base_pos = _to_numpy_1d(robot.get_pos())
    quat_wxyz = mt.as_tensor_2d(robot.get_quat())
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]][0].detach().cpu().numpy().astype(np.float32)
    dof_pos = _to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
    dof_vel = _to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
    return np.concatenate([base_pos, quat_xyzw, dof_pos, dof_vel], axis=0).astype(np.float32)


def _set_state_from_s(robot, motor_dofs, s):
    # s = [x,y,z,qx,qy,qz,qw,dof_pos...,dof_vel...]
    n = len(motor_dofs)
    base_pos = s[:3]
    quat_xyzw = s[3:7]
    dof_pos = s[7 : 7 + n]
    dof_vel = s[7 + n : 7 + 2 * n]

    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    robot.set_pos(torch.tensor(base_pos, dtype=torch.float32, device=mt.SIM_DEVICE))
    robot.set_quat(torch.tensor(quat_wxyz, dtype=torch.float32, device=mt.SIM_DEVICE))
    robot.set_dofs_position(
        position=torch.tensor(dof_pos, dtype=torch.float32, device=mt.SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )
    robot.set_dofs_velocity(
        velocity=torch.tensor(dof_vel, dtype=torch.float32, device=mt.SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build paired delta using Isaac anchors: delta_target = s_next_isaac - s_next_genesis(s_t,a_t)"
    )
    parser.add_argument("--isaac-npz", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=-1, help="for quick debug; -1 means all")
    args = parser.parse_args()

    isaac_npz = Path(args.isaac_npz)
    out_npz = Path(args.out_npz)

    if not isaac_npz.is_file():
        raise FileNotFoundError(f"isaac npz not found: {isaac_npz}")

    with np.load(isaac_npz, allow_pickle=False) as d:
        for k in ("s", "a", "s_next"):
            if k not in d:
                raise KeyError(f"Missing key in isaac npz: {k}")

        s = d["s"].astype(np.float32)
        a = d["a"].astype(np.float32)
        s_next_isaac = d["s_next"].astype(np.float32)

        if s.ndim == 3:
            if "mask" in d:
                mask = d["mask"].astype(bool)
            elif "episode_lengths" in d:
                lengths = d["episode_lengths"].astype(np.int32)
                mask = np.zeros(s.shape[:2], dtype=bool)
                for i, l in enumerate(lengths):
                    mask[i, : int(l)] = True
            else:
                raise KeyError("Padded npz needs mask or episode_lengths")
        elif s.ndim == 2:
            mask = np.ones((s.shape[0],), dtype=bool)
        else:
            raise ValueError(f"Unsupported shape for s: {s.shape}")

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

    s_next_genesis = np.zeros_like(s_next_isaac, dtype=np.float32)

    count = 0
    if s.ndim == 3:
        E, T = s.shape[:2]
        for ei in range(E):
            for ti in range(T):
                if not mask[ei, ti]:
                    continue
                if args.max_samples > 0 and count >= args.max_samples:
                    break

                _set_state_from_s(robot, motor_dofs, s[ei, ti])

                target_pos = a[ei, ti].astype(np.float32)
                for _ in range(mt.CONTROL_DECIMATION):
                    if mt.USE_IMPLICIT_PD:
                        robot.control_dofs_position(
                            position=torch.tensor(target_pos, dtype=torch.float32, device=mt.SIM_DEVICE),
                            dofs_idx_local=motor_dofs,
                        )
                    else:
                        dof_pos_sub = _to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
                        dof_vel_sub = _to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
                        torques_req = mt.KP * (target_pos - dof_pos_sub) - mt.KD * dof_vel_sub
                        torques = np.clip(torques_req, -mt.TORQUE_LIMITS, mt.TORQUE_LIMITS)
                        robot.control_dofs_force(
                            torch.tensor(torques, dtype=torch.float32, device=mt.SIM_DEVICE),
                            dofs_idx_local=motor_dofs,
                        )
                    scene.step()

                s_next_genesis[ei, ti] = _get_state(robot, motor_dofs)
                count += 1
            if args.max_samples > 0 and count >= args.max_samples:
                break
    else:
        N = s.shape[0]
        for i in range(N):
            if args.max_samples > 0 and count >= args.max_samples:
                break
            _set_state_from_s(robot, motor_dofs, s[i])
            target_pos = a[i].astype(np.float32)
            for _ in range(mt.CONTROL_DECIMATION):
                if mt.USE_IMPLICIT_PD:
                    robot.control_dofs_position(
                        position=torch.tensor(target_pos, dtype=torch.float32, device=mt.SIM_DEVICE),
                        dofs_idx_local=motor_dofs,
                    )
                else:
                    dof_pos_sub = _to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
                    dof_vel_sub = _to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
                    torques_req = mt.KP * (target_pos - dof_pos_sub) - mt.KD * dof_vel_sub
                    torques = np.clip(torques_req, -mt.TORQUE_LIMITS, mt.TORQUE_LIMITS)
                    robot.control_dofs_force(
                        torch.tensor(torques, dtype=torch.float32, device=mt.SIM_DEVICE),
                        dofs_idx_local=motor_dofs,
                    )
                scene.step()
            s_next_genesis[i] = _get_state(robot, motor_dofs)
            count += 1

    delta_target = (s_next_isaac - s_next_genesis).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if s.ndim == 3:
        ep_len = np.sum(mask, axis=1).astype(np.int32)
        np.savez_compressed(
            out_npz,
            s=s,
            a=a,
            delta_target=delta_target,
            s_next_isaac=s_next_isaac,
            s_next_genesis=s_next_genesis,
            mask=mask,
            episode_lengths=ep_len,
        )
    else:
        np.savez_compressed(
            out_npz,
            s=s,
            a=a,
            delta_target=delta_target,
            s_next_isaac=s_next_isaac,
            s_next_genesis=s_next_genesis,
        )

    abs_delta = np.abs(delta_target)
    mean_abs_all = float(np.mean(abs_delta))
    if s.ndim == 3:
        mean_abs_valid = float(np.mean(abs_delta[mask]))
    else:
        mean_abs_valid = mean_abs_all

    print(f"[DONE] saved: {out_npz}")
    print(f"[INFO] used_samples={count}")
    print(f"[INFO] mean_abs_delta_all={mean_abs_all:.6e}")
    print(f"[INFO] mean_abs_delta_valid={mean_abs_valid:.6e}")


if __name__ == "__main__":
    main()

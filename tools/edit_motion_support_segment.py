#!/usr/bin/env python3
import argparse
import copy
import math
from pathlib import Path

import joblib
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl"
)
DEFAULT_ROBOT_CONFIG = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"
)


def load_robot_dof_names(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["robot"]["dof_names"]]


def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_weight(num_frames, fps, plateau_start, plateau_end, ramp_in, ramp_out):
    times = np.arange(num_frames, dtype=np.float32) / float(fps)
    w = np.zeros(num_frames, dtype=np.float32)

    if ramp_in > 0.0:
        start = plateau_start - ramp_in
        mask = (times >= start) & (times < plateau_start)
        w[mask] = smoothstep01((times[mask] - start) / ramp_in)

    mask = (times >= plateau_start) & (times <= plateau_end)
    w[mask] = 1.0

    if ramp_out > 0.0:
        end = plateau_end + ramp_out
        mask = (times > plateau_end) & (times <= end)
        w[mask] = 1.0 - smoothstep01((times[mask] - plateau_end) / ramp_out)

    return w


def update_pose_aa_from_dof_array(pose_aa, dof_idx, dof_values):
    pose_joint_idx = dof_idx + 1
    pose_aa[:, pose_joint_idx, :] = 0.0
    pose_aa[:, pose_joint_idx, 0] = dof_values


def parse_joint_delta(items):
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --joint-delta '{item}', expected joint_name=value")
        name, value = item.split("=", 1)
        out.append((name.strip(), float(value)))
    return out


def recompute_linear_velocity(values, dt, sigma):
    vel = np.gradient(values, dt, axis=0, edge_order=1).astype(np.float32)
    if sigma > 0.0:
        vel = gaussian_filter1d(vel, sigma=sigma, axis=0, mode="nearest")
    return vel.astype(np.float32)


def recompute_root_ang_vel(root_quat_xyzw, dt, sigma):
    rot = R.from_quat(root_quat_xyzw.astype(np.float64))
    rel = rot[1:] * rot[:-1].inv()
    rotvec = rel.as_rotvec() / dt
    ang_vel = np.zeros((root_quat_xyzw.shape[0], 3), dtype=np.float32)
    if rotvec.shape[0] > 0:
        ang_vel[:-1] = rotvec.astype(np.float32)
        ang_vel[-1] = ang_vel[-2] if ang_vel.shape[0] > 1 else ang_vel[-1]
    if sigma > 0.0:
        ang_vel = gaussian_filter1d(ang_vel, sigma=sigma, axis=0, mode="nearest")
    return ang_vel.astype(np.float32)


def recompute_motion_velocity_fields(motion, root_trans, root_quat, dof, sigma_frames):
    fps = int(motion["fps"])
    dt = 1.0 / float(fps)
    root_lin_vel = recompute_linear_velocity(root_trans, dt, sigma_frames)
    root_ang_vel = recompute_root_ang_vel(root_quat, dt, sigma_frames)
    dof_vel = recompute_linear_velocity(dof, dt, sigma_frames)

    motion["root_lin_vel"] = root_lin_vel.astype(np.float32)
    motion["root_vel"] = root_lin_vel.astype(np.float32)
    motion["root_ang_vel"] = root_ang_vel.astype(np.float32)
    motion["dof_vel"] = dof_vel.astype(np.float32)
    motion["dof_vels"] = dof_vel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Apply a smooth support-phase pose fix to an ASAP motion PKL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plateau-start", type=float, default=0.5)
    parser.add_argument("--plateau-end", type=float, default=0.9)
    parser.add_argument("--ramp-in", type=float, default=0.1)
    parser.add_argument("--ramp-out", type=float, default=0.1)
    parser.add_argument("--root-height-delta", type=float, default=0.0)
    parser.add_argument("--root-roll-delta-deg", type=float, default=0.0)
    parser.add_argument("--root-pitch-delta-deg", type=float, default=0.0)
    parser.add_argument("--root-yaw-delta-deg", type=float, default=0.0)
    parser.add_argument(
        "--joint-delta",
        action="append",
        default=[],
        help="Repeatable joint_name=value pair applied over the weighted segment.",
    )
    parser.add_argument("--velocity-sigma-frames", type=float, default=1.0)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    args = parser.parse_args()

    data = joblib.load(args.input)
    key = next(iter(data))
    motion = copy.deepcopy(data[key])
    fps = int(motion["fps"])
    dof_names = motion.get("dof_names") or motion.get("joint_names") or load_robot_dof_names(args.robot_config)
    dof_names = [str(x) for x in dof_names]
    dof_index = {name: i for i, name in enumerate(dof_names)}

    weight = build_weight(
        num_frames=np.asarray(motion["dof"]).shape[0],
        fps=fps,
        plateau_start=args.plateau_start,
        plateau_end=args.plateau_end,
        ramp_in=args.ramp_in,
        ramp_out=args.ramp_out,
    )

    root_trans_dtype = np.asarray(motion["root_trans_offset"]).dtype
    root_rot_dtype = np.asarray(motion["root_rot"]).dtype
    pose_dtype = np.asarray(motion["pose_aa"]).dtype
    dof_dtype = np.asarray(motion["dof"]).dtype

    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32).copy()
    root_quat = np.asarray(motion["root_rot"], dtype=np.float64).copy()
    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float32).copy()
    dof = np.asarray(motion["dof"], dtype=np.float32).copy()

    root_trans[:, 2] += args.root_height_delta * weight
    root_delta_deg = np.array(
        [args.root_roll_delta_deg, args.root_pitch_delta_deg, args.root_yaw_delta_deg],
        dtype=np.float64,
    )
    for i in range(root_quat.shape[0]):
        if weight[i] == 0.0:
            continue
        delta = R.from_euler("xyz", np.deg2rad(root_delta_deg * float(weight[i])), degrees=False)
        new_rot = R.from_quat(root_quat[i]) * delta
        root_quat[i] = new_rot.as_quat()
        pose_aa[i, 0] = new_rot.as_rotvec().astype(np.float32)

    joint_deltas = parse_joint_delta(args.joint_delta)
    for joint_name, delta_value in joint_deltas:
        if joint_name not in dof_index:
            raise KeyError(f"Joint '{joint_name}' not found in motion dof_names")
        idx = dof_index[joint_name]
        dof[:, idx] += delta_value * weight
        update_pose_aa_from_dof_array(pose_aa, idx, dof[:, idx])

    motion["root_trans_offset"] = root_trans.astype(root_trans_dtype, copy=False)
    motion["root_rot"] = root_quat.astype(root_rot_dtype, copy=False)
    motion["pose_aa"] = pose_aa.astype(pose_dtype, copy=False)
    motion["dof"] = dof.astype(dof_dtype, copy=False)
    recompute_motion_velocity_fields(motion, root_trans, root_quat, dof, args.velocity_sigma_frames)

    out = copy.deepcopy(data)
    out[key] = motion
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, args.output)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"plateau={args.plateau_start:.3f}-{args.plateau_end:.3f}s ramp_in={args.ramp_in:.3f}s ramp_out={args.ramp_out:.3f}s")
    print(
        "root_delta="
        f"height:{args.root_height_delta:.4f}m "
        f"roll:{args.root_roll_delta_deg:.4f}deg "
        f"pitch:{args.root_pitch_delta_deg:.4f}deg "
        f"yaw:{args.root_yaw_delta_deg:.4f}deg"
    )
    print(f"velocity_sigma_frames={args.velocity_sigma_frames:.3f}")
    print("joint_deltas=" + (" ".join(f"{name}:{value:.4f}" for name, value in joint_deltas) if joint_deltas else "none"))


if __name__ == "__main__":
    main()

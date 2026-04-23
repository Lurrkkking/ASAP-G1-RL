#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import yaml


DEFAULT_INPUT = Path("/root/autodl-tmp/ASAP/outputs/kickball_example_asap_motion.pkl")
DEFAULT_ROBOT_CONFIG = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"
)


def load_robot_dof_names(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["robot"]["dof_names"]]


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_anklefix{input_path.suffix}")


def normalize_name(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def score_candidate(name, required_tokens, bonus_tokens=()):
    norm = normalize_name(name)
    score = 0
    for token in required_tokens:
        token = token.lower()
        if token in norm:
            score += 10
        else:
            return -1
    for token in bonus_tokens:
        if token.lower() in norm:
            score += 3
    if "left" in norm or norm.startswith("l_"):
        score -= 20
    return score


def find_dof_index(names, kind):
    if kind == "right_ankle_pitch":
        candidate_specs = [
            (("right", "ankle", "pitch"), ("joint",)),
            (("r", "ankle", "pitch"), ("joint",)),
            (("right", "ankle"), ("pitch", "joint")),
            (("r", "ankle"), ("pitch", "joint")),
        ]
    elif kind == "right_knee":
        candidate_specs = [
            (("right", "knee"), ("joint",)),
            (("r", "knee"), ("joint",)),
        ]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    best = (-1, None)
    for idx, name in enumerate(names):
        for required, bonus in candidate_specs:
            score = score_candidate(name, required, bonus)
            if score > best[0]:
                best = (score, idx)
    if best[1] is None:
        return None
    return best[1]


def smooth_bump(length):
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.sin(np.pi * x).astype(np.float32)


def smooth_ramp(length):
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32)


def recompute_velocity(pos, fps):
    pos = np.asarray(pos, dtype=np.float32)
    if pos.shape[0] == 1:
        return np.zeros_like(pos, dtype=np.float32)
    dt = 1.0 / float(fps)
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[:-1] = (pos[1:] - pos[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def update_pose_aa_from_dof(motion, dof_idx, dof_values):
    pose_aa = motion.get("pose_aa")
    if pose_aa is None:
        return False
    pose_aa = np.asarray(pose_aa)
    pose_joint_idx = dof_idx + 1
    if pose_aa.ndim != 3 or pose_joint_idx >= pose_aa.shape[1]:
        return False
    pose_aa[:, pose_joint_idx, :] = 0.0
    pose_aa[:, pose_joint_idx, 0] = dof_values
    motion["pose_aa"] = pose_aa
    return True


def update_velocity_fields(motion, fps):
    updated = []
    dof = motion.get("dof")
    if dof is None:
        return updated

    for key in ("dof_vel", "dof_vels", "joint_vel", "joint_vels", "dof_velocity", "joint_velocity"):
        if key not in motion:
            continue
        arr = np.asarray(motion[key])
        if arr.shape == np.asarray(dof).shape:
            motion[key] = recompute_velocity(dof, fps).astype(arr.dtype, copy=False)
            updated.append(key)
    return updated


def describe_motion(motion):
    lines = []
    for key, value in motion.items():
        if hasattr(value, "shape"):
            lines.append(f"{key}: shape={value.shape}, dtype={getattr(value, 'dtype', None)}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key}: {type(value).__name__}, len={len(value)}")
        else:
            lines.append(f"{key}: {type(value).__name__}, value={value}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Locally smooth-edit right ankle pitch in an ASAP motion PKL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--ankle-delta", type=float, required=True, help="Additive radians at bump peak.")
    parser.add_argument("--knee-delta", type=float, default=0.0, help="Optional additive radians at bump peak.")
    parser.add_argument(
        "--prep-start-frame",
        type=int,
        default=None,
        help="Optional frame to start ankle preparation earlier with a smooth ramp.",
    )
    parser.add_argument(
        "--prep-target",
        type=float,
        default=None,
        help="Optional absolute right ankle pitch target during the preparation segment.",
    )
    parser.add_argument(
        "--prep-end-frame",
        type=int,
        default=None,
        help="Optional end frame for the preparation ramp. Defaults to start-frame.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    args = parser.parse_args()

    output = args.output or default_output_path(args.input)
    data = joblib.load(args.input)
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level dict in {args.input}, got {type(data)}")

    top_key = next(iter(data))
    if not isinstance(data[top_key], dict):
        raise TypeError(f"Expected outer dict values to be motion dicts, got {type(data[top_key])}")
    motion = copy.deepcopy(data[top_key])

    print(f"input={args.input}")
    print(f"output={output}")
    print(f"motion_key={top_key}")
    print("fields:")
    for line in describe_motion(motion):
        print(f"  {line}")

    dof_names = motion.get("dof_names") or motion.get("joint_names")
    if dof_names is None:
        dof_names = load_robot_dof_names(args.robot_config)
        print(f"dof_names_source=robot_config:{args.robot_config}")
    else:
        dof_names = [str(x) for x in dof_names]
        print("dof_names_source=pkl")

    ankle_idx = find_dof_index(dof_names, "right_ankle_pitch")
    knee_idx = find_dof_index(dof_names, "right_knee")
    if ankle_idx is None:
        raise ValueError(f"Could not identify right ankle pitch from names: {dof_names}")

    print(f"right_ankle_match index={ankle_idx} name={dof_names[ankle_idx]}")
    if knee_idx is None:
        print("right_knee_match index=None name=None")
    else:
        print(f"right_knee_match index={knee_idx} name={dof_names[knee_idx]}")

    dof = np.asarray(motion.get("dof"), dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"Expected motion['dof'] shape (T,D), got {dof.shape}")
    if ankle_idx >= dof.shape[1]:
        raise ValueError(f"Matched ankle index {ankle_idx} outside dof shape {dof.shape}")
    if knee_idx is not None and knee_idx >= dof.shape[1]:
        raise ValueError(f"Matched knee index {knee_idx} outside dof shape {dof.shape}")

    num_frames = dof.shape[0]
    start = max(0, int(args.start_frame))
    end = min(num_frames - 1, int(args.end_frame))
    if start > end:
        raise ValueError(f"Invalid frame range after clipping: start={start}, end={end}, num_frames={num_frames}")

    window = smooth_bump(end - start + 1)
    print(f"frame_range={start}:{end} inclusive, num_frames={num_frames}")
    print(f"ankle_delta_peak={args.ankle_delta:.6f} rad")
    print(f"knee_delta_peak={args.knee_delta:.6f} rad")
    print(f"ankle_before_minmax={dof[start:end+1, ankle_idx].min():.6f},{dof[start:end+1, ankle_idx].max():.6f}")
    if knee_idx is not None:
        print(f"knee_before_minmax={dof[start:end+1, knee_idx].min():.6f},{dof[start:end+1, knee_idx].max():.6f}")

    prep_start = None
    prep_end = None
    if args.prep_start_frame is not None:
        prep_start = max(0, int(args.prep_start_frame))
        prep_end = start if args.prep_end_frame is None else min(num_frames - 1, int(args.prep_end_frame))
        if prep_start > prep_end:
            raise ValueError(
                f"Invalid prep frame range after clipping: prep_start={prep_start}, prep_end={prep_end}, num_frames={num_frames}"
            )
        if args.prep_target is None:
            raise ValueError("--prep-start-frame requires --prep-target")

        prep_window = smooth_ramp(prep_end - prep_start + 1)
        prep_target = float(args.prep_target)
        prep_before = dof[prep_start : prep_end + 1, ankle_idx].copy()
        prep_delta = prep_target - prep_before
        dof[prep_start : prep_end + 1, ankle_idx] = prep_before + prep_delta * prep_window
        print(
            f"prep_frame_range={prep_start}:{prep_end} inclusive, prep_target={prep_target:.6f} rad"
        )
        print(
            f"prep_after_minmax={dof[prep_start:prep_end+1, ankle_idx].min():.6f},{dof[prep_start:prep_end+1, ankle_idx].max():.6f}"
        )

    if args.dry_run:
        print("dry_run=True, no file written")
        return

    dof[start : end + 1, ankle_idx] += args.ankle_delta * window
    pose_updated = update_pose_aa_from_dof(motion, ankle_idx, dof[:, ankle_idx])

    if abs(args.knee_delta) > 0.0:
        if knee_idx is None:
            raise ValueError("--knee-delta was nonzero but right knee could not be identified")
        dof[start : end + 1, knee_idx] += args.knee_delta * window
        update_pose_aa_from_dof(motion, knee_idx, dof[:, knee_idx])

    motion["dof"] = dof.astype(np.asarray(motion["dof"]).dtype, copy=False)
    fps = int(motion.get("fps", 30))
    velocity_fields = update_velocity_fields(motion, fps)

    out_data = copy.deepcopy(data)
    out_data[top_key] = motion
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out_data, output)

    print(f"ankle_after_minmax={dof[start:end+1, ankle_idx].min():.6f},{dof[start:end+1, ankle_idx].max():.6f}")
    if knee_idx is not None:
        print(f"knee_after_minmax={dof[start:end+1, knee_idx].min():.6f},{dof[start:end+1, knee_idx].max():.6f}")
    print(f"pose_aa_updated={pose_updated}")
    print(f"velocity_fields_recomputed={velocity_fields if velocity_fields else 'none'}")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()

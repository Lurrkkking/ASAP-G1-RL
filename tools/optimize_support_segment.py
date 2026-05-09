#!/usr/bin/env python3
import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

import sys

sys.path.insert(0, "/root/autodl-tmp/ASAP")
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


DEFAULT_INPUT = Path("/root/autodl-tmp/ASAP/tmp/support_fix/base_waist035_rooth03.pkl")
DEFAULT_MJCF = Path("/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml")
DEFAULT_ROBOT_CONFIG = Path("/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml")


def build_robot_cfg():
    robot_base_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/robot/robot_base.yaml")
    cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml")
    robot_cfg = OmegaConf.merge(robot_base_cfg.robot, cfg.robot)
    robot_cfg.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    robot_cfg.asset.assetFileName = "g1_29dof_anneal_23dof_fitmotionONLY.xml"
    robot_cfg.motion.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    robot_cfg.motion.asset.assetFileName = "g1_29dof_anneal_23dof_fitmotionONLY.xml"
    if "extend_config" not in robot_cfg.motion:
        robot_cfg.motion.extend_config = []
    return robot_cfg.motion


def load_robot_dof_names(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["robot"]["dof_names"]]


def parse_vec(text, size):
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape[0] != size:
        raise ValueError(f"Expected {size} values, got {values}")
    return values


def parse_mjcf_metadata(mjcf_path):
    tree = ET.parse(mjcf_path)
    worldbody = tree.getroot().find("worldbody")
    body_mass = {}
    body_inertial_pos = {}
    foot_geom_points = []

    def walk(body_elem):
        name = body_elem.attrib["name"]
        inertial = body_elem.find("inertial")
        if inertial is not None:
            body_mass[name] = float(inertial.attrib["mass"])
            body_inertial_pos[name] = parse_vec(inertial.attrib["pos"], 3)
        for geom in body_elem.findall("geom"):
            if name == "left_ankle_roll_link" and "pos" in geom.attrib and "mesh" not in geom.attrib:
                foot_geom_points.append(parse_vec(geom.attrib["pos"], 3))
        for child in body_elem.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return body_mass, body_inertial_pos, np.asarray(foot_geom_points, dtype=np.float64)


def world_points_from_local(body_pos, body_quat, local_points):
    return body_pos[None, :] + R.from_quat(body_quat).apply(local_points)


def compute_com(body_pos, body_quat, body_names, body_mass, body_inertial_pos):
    total_mass = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    for name, pos, quat in zip(body_names, body_pos, body_quat):
        if name not in body_mass:
            continue
        com_world = pos + R.from_quat(quat).apply(body_inertial_pos[name])
        weighted += body_mass[name] * com_world
        total_mass += body_mass[name]
    return weighted / max(total_mass, 1e-8)


def local_y_of_point(world_xy, foot_xy, foot_yaw):
    dx, dy = world_xy - foot_xy
    c = math.cos(foot_yaw)
    s = math.sin(foot_yaw)
    return -s * dx + c * dy


def quat_angle_rad(q_a, q_b):
    rel = R.from_quat(q_a) * R.from_quat(q_b).inv()
    return float(rel.magnitude())


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


def apply_frame_delta(frame_root_quat, frame_pose_aa, frame_dof, dof_index, delta):
    delta_root_roll, delta_lhip_roll, delta_lankle_roll, delta_waist_roll = delta
    root_rot = R.from_quat(frame_root_quat)
    new_root = root_rot * R.from_euler("x", delta_root_roll, degrees=False)
    out_root_quat = new_root.as_quat().astype(np.float32)
    out_pose = frame_pose_aa.copy()
    out_pose[0] = new_root.as_rotvec().astype(np.float32)
    out_dof = frame_dof.copy()
    for joint_name, joint_delta in (
        ("left_hip_roll_joint", delta_lhip_roll),
        ("left_ankle_roll_joint", delta_lankle_roll),
        ("waist_roll_joint", delta_waist_roll),
    ):
        idx = dof_index[joint_name]
        out_dof[idx] += float(joint_delta)
        out_pose[idx + 1, :] = 0.0
        out_pose[idx + 1, 0] = out_dof[idx]
    return out_root_quat, out_pose, out_dof


def frame_fk(mesh, pose_frame, trans_frame):
    fk = mesh.fk_batch(
        torch.from_numpy(pose_frame[None, None].astype(np.float32)),
        torch.from_numpy(trans_frame[None, None].astype(np.float32)),
        return_full=False,
        dt=1.0 / 50.0,
    )
    body_pos = fk.global_translation[0, 0].detach().cpu().numpy()
    body_rot = fk.global_rotation[0, 0].detach().cpu().numpy()
    return body_pos, body_rot


def main():
    parser = argparse.ArgumentParser(description="Optimize a support segment while keeping the left foot anchored.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--end", type=float, default=0.9)
    parser.add_argument("--target-com-y", type=float, default=-0.035)
    parser.add_argument("--target-patch-z", type=float, default=0.0)
    parser.add_argument("--smooth-sigma-frames", type=float, default=1.0)
    parser.add_argument("--velocity-sigma-frames", type=float, default=1.0)
    parser.add_argument("--w-foot-xy", type=float, default=600.0)
    parser.add_argument("--w-foot-z", type=float, default=120.0)
    parser.add_argument("--w-foot-rot", type=float, default=30.0)
    parser.add_argument("--w-patch-z", type=float, default=10.0)
    parser.add_argument("--w-com-y", type=float, default=8.0)
    parser.add_argument("--w-reg", type=float, default=2.0)
    parser.add_argument("--root-roll-bound-deg", type=float, default=6.0)
    parser.add_argument("--left-hip-roll-bound", type=float, default=0.16)
    parser.add_argument("--left-ankle-roll-bound", type=float, default=0.16)
    parser.add_argument("--waist-roll-bound", type=float, default=0.20)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    args = parser.parse_args()

    data = joblib.load(args.input)
    key = next(iter(data))
    motion = copy.deepcopy(data[key])
    fps = int(motion["fps"])
    dt = 1.0 / float(fps)
    dof_names = motion.get("dof_names") or motion.get("joint_names") or load_robot_dof_names(args.robot_config)
    dof_names = [str(x) for x in dof_names]
    dof_index = {name: i for i, name in enumerate(dof_names)}

    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float32).copy()
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32).copy()
    root_quat = np.asarray(motion["root_rot"], dtype=np.float32).copy()
    dof = np.asarray(motion["dof"], dtype=np.float32).copy()

    mesh = Humanoid_Batch(build_robot_cfg())
    body_names = list(mesh.body_names)
    left_idx = body_names.index("left_ankle_roll_link")
    body_mass, body_inertial_pos, foot_geom_points = parse_mjcf_metadata(args.mjcf)

    times = np.arange(dof.shape[0], dtype=np.float64) / float(fps)
    frame_ids = np.nonzero((times >= args.start) & (times <= args.end))[0]
    if frame_ids.size == 0:
        raise ValueError("No frames in target window")

    base_deltas = np.zeros((dof.shape[0], 4), dtype=np.float32)
    summary = []
    for frame_id in frame_ids:
        ref_body_pos, ref_body_rot = frame_fk(mesh, pose_aa[frame_id], root_trans[frame_id])
        ref_left_pos = ref_body_pos[left_idx]
        ref_left_quat = ref_body_rot[left_idx]
        ref_patch_min_z = float(world_points_from_local(ref_left_pos, ref_left_quat, foot_geom_points)[:, 2].min())

        def objective(x):
            out_root_quat, out_pose, _ = apply_frame_delta(root_quat[frame_id], pose_aa[frame_id], dof[frame_id], dof_index, x)
            body_pos, body_rot = frame_fk(mesh, out_pose, root_trans[frame_id])
            left_pos = body_pos[left_idx]
            left_quat = body_rot[left_idx]
            left_yaw = R.from_quat(left_quat).as_euler("xyz", degrees=False)[2]
            patch_min_z = float(world_points_from_local(left_pos, left_quat, foot_geom_points)[:, 2].min())
            com = compute_com(body_pos, body_rot, body_names, body_mass, body_inertial_pos)
            com_local_y = local_y_of_point(com[:2], left_pos[:2], left_yaw)
            foot_xy_err = np.linalg.norm(left_pos[:2] - ref_left_pos[:2])
            foot_z_err = float(left_pos[2] - ref_left_pos[2])
            foot_rot_err = quat_angle_rad(left_quat, ref_left_quat)
            patch_z_err = patch_min_z - args.target_patch_z
            reg = np.linalg.norm(x)
            return (
                args.w_foot_xy * foot_xy_err ** 2
                + args.w_foot_z * foot_z_err ** 2
                + args.w_foot_rot * foot_rot_err ** 2
                + args.w_patch_z * patch_z_err ** 2
                + args.w_com_y * (com_local_y - args.target_com_y) ** 2
                + args.w_reg * reg ** 2
            )

        res = minimize(
            objective,
            x0=np.zeros(4, dtype=np.float64),
            method="Powell",
            bounds=[
                (-math.radians(args.root_roll_bound_deg), math.radians(args.root_roll_bound_deg)),
                (-args.left_hip_roll_bound, args.left_hip_roll_bound),
                (-args.left_ankle_roll_bound, args.left_ankle_roll_bound),
                (-args.waist_roll_bound, args.waist_roll_bound),
            ],
            options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-4},
        )
        base_deltas[frame_id] = res.x.astype(np.float32)
        summary.append({"frame": int(frame_id), "time_s": float(times[frame_id]), "delta": res.x.tolist(), "fun": float(res.fun)})

    if args.smooth_sigma_frames > 0.0:
        for i in range(base_deltas.shape[1]):
            base_deltas[:, i] = gaussian_filter1d(base_deltas[:, i], sigma=args.smooth_sigma_frames, mode="nearest")
        outside = np.ones(base_deltas.shape[0], dtype=bool)
        outside[frame_ids] = False
        base_deltas[outside] = 0.0

    for frame_id in frame_ids:
        out_root_quat, out_pose, out_dof = apply_frame_delta(
            root_quat[frame_id], pose_aa[frame_id], dof[frame_id], dof_index, base_deltas[frame_id]
        )
        root_quat[frame_id] = out_root_quat
        pose_aa[frame_id] = out_pose
        dof[frame_id] = out_dof

    motion["root_rot"] = root_quat.astype(np.asarray(motion["root_rot"]).dtype, copy=False)
    motion["pose_aa"] = pose_aa.astype(np.asarray(motion["pose_aa"]).dtype, copy=False)
    motion["dof"] = dof.astype(np.asarray(motion["dof"]).dtype, copy=False)
    motion["root_lin_vel"] = recompute_linear_velocity(root_trans, dt, args.velocity_sigma_frames)
    motion["root_vel"] = motion["root_lin_vel"].copy()
    motion["root_ang_vel"] = recompute_root_ang_vel(root_quat, dt, args.velocity_sigma_frames)
    motion["dof_vel"] = recompute_linear_velocity(dof, dt, args.velocity_sigma_frames)
    motion["dof_vels"] = motion["dof_vel"].copy()

    out = copy.deepcopy(data)
    out[key] = motion
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, args.output)

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "window": [args.start, args.end],
        "target_com_y": args.target_com_y,
        "smooth_sigma_frames": args.smooth_sigma_frames,
        "velocity_sigma_frames": args.velocity_sigma_frames,
        "weights": {
            "foot_xy": args.w_foot_xy,
            "foot_z": args.w_foot_z,
            "foot_rot": args.w_foot_rot,
            "patch_z": args.w_patch_z,
            "com_y": args.w_com_y,
            "reg": args.w_reg,
        },
        "optimized_frames": len(frame_ids),
        "mean_abs_delta": np.abs(base_deltas[frame_ids]).mean(axis=0).tolist(),
        "max_abs_delta": np.abs(base_deltas[frame_ids]).max(axis=0).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()

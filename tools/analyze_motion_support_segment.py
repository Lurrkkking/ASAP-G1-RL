#!/usr/bin/env python3
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

import sys

sys.path.insert(0, "/root/autodl-tmp/ASAP")
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


DEFAULT_MOTION = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/"
    "TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl"
)
DEFAULT_MJCF = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml"
)


def build_robot_cfg():
    robot_base_cfg = OmegaConf.load(
        "/root/autodl-tmp/ASAP/humanoidverse/config/robot/robot_base.yaml"
    )
    cfg = OmegaConf.load(
        "/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"
    )
    robot_cfg = OmegaConf.merge(robot_base_cfg.robot, cfg.robot)
    robot_cfg.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    robot_cfg.asset.assetFileName = "g1_29dof_anneal_23dof_fitmotionONLY.xml"
    robot_cfg.motion.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    robot_cfg.motion.asset.assetFileName = "g1_29dof_anneal_23dof_fitmotionONLY.xml"
    if "extend_config" not in robot_cfg.motion:
        robot_cfg.motion.extend_config = []
    return robot_cfg.motion


def load_motion(motion_path: Path, motion_name: Optional[str]):
    data = joblib.load(motion_path)
    if isinstance(data, dict) and "pose_aa" in data:
        return motion_name or motion_path.stem, data
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported motion container type: {type(data)}")
    if motion_name is None:
        motion_name = next(iter(data))
    motion = data[motion_name]
    return motion_name, motion


def parse_vec(text: str, size: int):
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape[0] != size:
        raise ValueError(f"Expected {size} values, got {values}")
    return values


def parse_mjcf_metadata(mjcf_path: Path):
    tree = ET.parse(mjcf_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF missing worldbody")

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


def quat_xyzw_to_euler_xyz(quat_xyzw):
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def quat_angle_delta_deg(q_ref, q_cur):
    rel = R.from_quat(q_cur) * R.from_quat(q_ref).inv()
    return np.rad2deg(rel.magnitude())


def world_points_from_local(body_pos, body_quat, local_points):
    rot = R.from_quat(body_quat)
    return body_pos[None, :] + rot.apply(local_points)


def compute_com(body_pos, body_quat, body_names, body_mass, body_inertial_pos):
    total_mass = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    for name, pos, quat in zip(body_names, body_pos, body_quat):
        mass = body_mass.get(name)
        inertial_local = body_inertial_pos.get(name)
        if mass is None or inertial_local is None:
            continue
        com_world = pos + R.from_quat(quat).apply(inertial_local)
        weighted += mass * com_world
        total_mass += mass
    if total_mass <= 0.0:
        raise ValueError("No body masses found for COM computation")
    return weighted / total_mass


def foot_yaw_local_xy(world_xy, foot_xy, foot_yaw):
    c = math.cos(foot_yaw)
    s = math.sin(foot_yaw)
    dx, dy = world_xy - foot_xy
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)


def summarize(series):
    return {
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze support quality of a motion reference segment.")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-name", default=None)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--end", type=float, default=0.9)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    motion_name, motion = load_motion(args.motion, args.motion_name)
    fps = int(motion["fps"])
    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float32)
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)

    mesh = Humanoid_Batch(build_robot_cfg())
    fk = mesh.fk_batch(
        torch.from_numpy(pose_aa[None]),
        torch.from_numpy(root_trans[None]),
        return_full=True,
        dt=1.0 / fps,
    )

    body_names = list(mesh.body_names)
    left_idx = body_names.index(args.left_foot_body)
    right_idx = body_names.index(args.right_foot_body)
    body_pos = fk.global_translation[0].detach().cpu().numpy()
    body_rot = fk.global_rotation[0].detach().cpu().numpy()
    body_vel = fk.global_velocity[0].detach().cpu().numpy()
    root_vel = fk.global_root_velocity[0].detach().cpu().numpy()

    body_mass, body_inertial_pos, foot_geom_points = parse_mjcf_metadata(args.mjcf)

    frame_times = np.arange(body_pos.shape[0], dtype=np.float64) / float(fps)
    mask = (frame_times >= args.start) & (frame_times <= args.end)
    frame_ids = np.nonzero(mask)[0]
    if frame_ids.size == 0:
        raise ValueError("No frames inside the requested time window")

    rows = []
    left_pos0 = body_pos[frame_ids[0], left_idx]
    left_rot0 = body_rot[frame_ids[0], left_idx]

    for frame_id in frame_ids:
        t = frame_times[frame_id]
        root_pos = body_pos[frame_id, 0]
        root_quat = body_rot[frame_id, 0]
        root_rpy = quat_xyzw_to_euler_xyz(root_quat)
        left_pos = body_pos[frame_id, left_idx]
        left_quat = body_rot[frame_id, left_idx]
        left_rpy = quat_xyzw_to_euler_xyz(left_quat)
        left_vel = body_vel[frame_id, left_idx]
        right_pos = body_pos[frame_id, right_idx]
        right_quat = body_rot[frame_id, right_idx]
        com_pos = compute_com(
            body_pos[frame_id],
            body_rot[frame_id],
            body_names,
            body_mass,
            body_inertial_pos,
        )
        foot_points_world = world_points_from_local(left_pos, left_quat, foot_geom_points)
        right_foot_points_world = world_points_from_local(right_pos, right_quat, foot_geom_points)
        foot_min_z = float(foot_points_world[:, 2].min()) if foot_points_world.size > 0 else float(left_pos[2])
        right_foot_min_z = (
            float(right_foot_points_world[:, 2].min())
            if right_foot_points_world.size > 0
            else float(right_pos[2])
        )
        root_rel_xy_local = foot_yaw_local_xy(root_pos[:2], left_pos[:2], left_rpy[2])
        com_rel_xy_local = foot_yaw_local_xy(com_pos[:2], left_pos[:2], left_rpy[2])
        row = {
            "frame": int(frame_id),
            "time_s": float(t),
            "left_foot_x": float(left_pos[0]),
            "left_foot_y": float(left_pos[1]),
            "left_foot_z": float(left_pos[2]),
            "left_foot_xy_drift_from_0p5": float(np.linalg.norm(left_pos[:2] - left_pos0[:2])),
            "left_foot_z_delta_from_0p5": float(left_pos[2] - left_pos0[2]),
            "left_foot_speed": float(np.linalg.norm(left_vel)),
            "left_foot_speed_xy": float(np.linalg.norm(left_vel[:2])),
            "left_foot_speed_z": float(left_vel[2]),
            "left_foot_roll": float(left_rpy[0]),
            "left_foot_pitch": float(left_rpy[1]),
            "left_foot_yaw": float(left_rpy[2]),
            "left_foot_rot_drift_deg": float(quat_angle_delta_deg(left_rot0, left_quat)),
            "left_foot_contact_patch_min_z": foot_min_z,
            "right_foot_z": float(right_pos[2]),
            "right_foot_contact_patch_min_z": right_foot_min_z,
            "root_vx": float(root_vel[frame_id, 0]),
            "root_vy": float(root_vel[frame_id, 1]),
            "root_vz": float(root_vel[frame_id, 2]),
            "root_speed": float(np.linalg.norm(root_vel[frame_id])),
            "root_roll": float(root_rpy[0]),
            "root_pitch": float(root_rpy[1]),
            "root_yaw": float(root_rpy[2]),
            "root_height": float(root_pos[2]),
            "root_rel_left_foot_x_world": float(root_pos[0] - left_pos[0]),
            "root_rel_left_foot_y_world": float(root_pos[1] - left_pos[1]),
            "root_rel_left_foot_x_local": float(root_rel_xy_local[0]),
            "root_rel_left_foot_y_local": float(root_rel_xy_local[1]),
            "com_x": float(com_pos[0]),
            "com_y": float(com_pos[1]),
            "com_z": float(com_pos[2]),
            "com_rel_left_foot_x_world": float(com_pos[0] - left_pos[0]),
            "com_rel_left_foot_y_world": float(com_pos[1] - left_pos[1]),
            "com_rel_left_foot_x_local": float(com_rel_xy_local[0]),
            "com_rel_left_foot_y_local": float(com_rel_xy_local[1]),
        }
        rows.append(row)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        header = list(rows[0].keys())
        with args.output_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in header) + "\n")

    def arr(key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    summary = {
        "motion": str(args.motion),
        "motion_name": motion_name,
        "fps": fps,
        "window_s": [float(args.start), float(args.end)],
        "num_frames": int(len(rows)),
        "left_foot_xy_drift_from_0p5_m": summarize(arr("left_foot_xy_drift_from_0p5")),
        "left_foot_height_m": summarize(arr("left_foot_z")),
        "left_foot_contact_patch_min_z_m": summarize(arr("left_foot_contact_patch_min_z")),
        "left_foot_speed_mps": summarize(arr("left_foot_speed")),
        "left_foot_speed_xy_mps": summarize(arr("left_foot_speed_xy")),
        "left_foot_rot_drift_deg": summarize(arr("left_foot_rot_drift_deg")),
        "right_foot_height_m": summarize(arr("right_foot_z")),
        "right_foot_contact_patch_min_z_m": summarize(arr("right_foot_contact_patch_min_z")),
        "root_speed_mps": summarize(arr("root_speed")),
        "root_vx_mps": summarize(arr("root_vx")),
        "root_vy_mps": summarize(arr("root_vy")),
        "root_vz_mps": summarize(arr("root_vz")),
        "root_roll_deg": summarize(np.rad2deg(arr("root_roll"))),
        "root_pitch_deg": summarize(np.rad2deg(arr("root_pitch"))),
        "root_height_m": summarize(arr("root_height")),
        "root_rel_left_foot_local_x_m": summarize(arr("root_rel_left_foot_x_local")),
        "root_rel_left_foot_local_y_m": summarize(arr("root_rel_left_foot_y_local")),
        "com_rel_left_foot_local_x_m": summarize(arr("com_rel_left_foot_x_local")),
        "com_rel_left_foot_local_y_m": summarize(arr("com_rel_left_foot_y_local")),
        "support_polygon_hint_local": {
            "x_min": -0.05,
            "x_max": 0.12,
            "y_min": -0.03,
            "y_max": 0.03,
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

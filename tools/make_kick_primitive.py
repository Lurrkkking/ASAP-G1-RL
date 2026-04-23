#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import torch

import sys

sys.path.insert(0, "/root/autodl-tmp/ASAP")
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


DEFAULT_TEMPLATE = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/"
    "TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/custom/kick_primitive.pkl"
)


# pose_aa uses the fitmotion skeleton order:
# 0 pelvis/root, 1..23 map to the 23 actuated joints in robot config order.
DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
DOF_INDEX = {name: idx for idx, name in enumerate(DOF_NAMES)}


def quat_xyzw_from_rotvec(rotvec):
    return R.from_rotvec(rotvec).as_quat().astype(np.float64)


def interpolate_keyframes(key_times, key_values, num_frames):
    frame_times = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    out = np.empty((num_frames,) + key_values.shape[1:], dtype=np.float32)
    flat = key_values.reshape(key_values.shape[0], -1)
    flat_out = np.empty((num_frames, flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        flat_out[:, i] = np.interp(frame_times, key_times, flat[:, i])
    out[...] = flat_out.reshape((num_frames,) + key_values.shape[1:])
    return out


def build_key_dofs(base_dof):
    keys = np.repeat(base_dof[None, :], 5, axis=0).astype(np.float32)

    r_hip_pitch = DOF_INDEX["right_hip_pitch_joint"]
    r_knee = DOF_INDEX["right_knee_joint"]
    r_ankle_pitch = DOF_INDEX["right_ankle_pitch_joint"]
    l_hip_pitch = DOF_INDEX["left_hip_pitch_joint"]
    l_knee = DOF_INDEX["left_knee_joint"]
    l_ankle_pitch = DOF_INDEX["left_ankle_pitch_joint"]
    waist_pitch = DOF_INDEX["waist_pitch_joint"]
    waist_roll = DOF_INDEX["waist_roll_joint"]

    # frame 1: more visible thigh lift, with the shin folded under naturally.
    keys[1, r_hip_pitch] = base_dof[r_hip_pitch] - 0.42
    keys[1, r_knee] = base_dof[r_knee] + 0.40
    keys[1, r_ankle_pitch] = 0.34
    keys[1, l_hip_pitch] = base_dof[l_hip_pitch] + 0.03
    keys[1, l_knee] = base_dof[l_knee] + 0.03
    keys[1, l_ankle_pitch] = base_dof[l_ankle_pitch] - 0.02
    keys[1, waist_roll] = base_dof[waist_roll] - 0.02
    keys[1, waist_pitch] = base_dof[waist_pitch] + 0.015

    # frame 2: peak tap, hip still lifted, knee snaps forward, ankle pointed and aligned with the shin.
    keys[2, r_hip_pitch] = base_dof[r_hip_pitch] - 0.34
    keys[2, r_knee] = base_dof[r_knee] - 0.42
    keys[2, r_ankle_pitch] = 0.50
    keys[2, l_hip_pitch] = base_dof[l_hip_pitch] + 0.05
    keys[2, l_knee] = base_dof[l_knee] + 0.06
    keys[2, l_ankle_pitch] = base_dof[l_ankle_pitch] - 0.04
    keys[2, waist_roll] = base_dof[waist_roll] - 0.03
    keys[2, waist_pitch] = base_dof[waist_pitch] + 0.025

    # frame 3: retract after contact while keeping the foot pointed rather than hooked.
    keys[3, r_hip_pitch] = base_dof[r_hip_pitch] - 0.16
    keys[3, r_knee] = base_dof[r_knee] + 0.18
    keys[3, r_ankle_pitch] = 0.30
    keys[3, l_hip_pitch] = base_dof[l_hip_pitch] + 0.02
    keys[3, l_knee] = base_dof[l_knee] + 0.02
    keys[3, waist_roll] = base_dof[waist_roll] - 0.01
    keys[3, waist_pitch] = base_dof[waist_pitch] + 0.01

    return keys


def build_mesh_parser():
    robot_base_cfg = OmegaConf.load(
        "/root/autodl-tmp/ASAP/humanoidverse/config/robot/robot_base.yaml"
    )
    cfg = OmegaConf.load(
        "/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"
    )
    robot_cfg = OmegaConf.merge(robot_base_cfg.robot, cfg.robot)
    robot_cfg.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    robot_cfg.asset.assetFileName = "g1_29dof_anneal_23dof_fitmotionONLY.xml"
    if "extend_config" not in robot_cfg:
        robot_cfg.extend_config = []
    return Humanoid_Batch(robot_cfg)


def ground_root_height(pose_aa, root_trans, fps, target_clearance=0.005):
    mesh = build_mesh_parser()
    pose_t = torch.from_numpy(pose_aa[None].astype(np.float32))
    trans_t = torch.from_numpy(root_trans[None].astype(np.float32))
    fk = mesh.fk_batch(pose_t, trans_t, return_full=True, dt=1.0 / fps)
    joint_min_per_frame = fk.global_translation[0, :, :, 2].amin(dim=-1)
    joint_min_frame = int(torch.argmin(joint_min_per_frame).item())
    mesh_obj = mesh.mesh_fk(
        pose_t[:, joint_min_frame : joint_min_frame + 1],
        trans_t[:, joint_min_frame : joint_min_frame + 1],
    )
    mesh_min_z = float(np.asarray(mesh_obj.vertices)[..., 2].min())
    root_trans[:, 2] -= mesh_min_z
    root_trans[:, 2] += target_clearance
    return root_trans, mesh_min_z, joint_min_frame


def main():
    parser = argparse.ArgumentParser(description="Build a short in-place G1 kick primitive motion PKL.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=28, help="Output frame count, suggested 20-40.")
    parser.add_argument("--name", default="kick_primitive")
    args = parser.parse_args()

    if args.frames < 5:
        raise ValueError("--frames must be at least 5")

    template = joblib.load(args.template)
    template_key = next(iter(template))
    motion = template[template_key]

    base_root = motion["root_trans_offset"][0].astype(np.float32)
    base_pose = motion["pose_aa"][0].astype(np.float32)
    base_dof = motion["dof"][0].astype(np.float32)
    base_root_rot = motion["root_rot"][0].astype(np.float64)
    base_smpl = motion["smpl_joints"][0].astype(np.float32)
    fps = int(motion["fps"])

    key_times = np.array([0.0, 0.25, 0.5, 0.72, 1.0], dtype=np.float32)

    key_dofs = build_key_dofs(base_dof)
    dof = interpolate_keyframes(key_times, key_dofs, args.frames)

    pose_aa = np.repeat(base_pose[None, :, :], args.frames, axis=0).astype(np.float32)
    pose_aa[:, 1:24, :] = 0.0
    pose_aa[:, 1:24, 0] = dof

    # Keep root orientation nearly fixed with a tiny forward settle.
    root_rotvecs = np.repeat(base_pose[None, 0, :], args.frames, axis=0).astype(np.float32)
    root_rotvecs[:, 1] += np.linspace(0.0, 0.015, args.frames, dtype=np.float32)
    pose_aa[:, 0, :] = root_rotvecs

    root_trans = np.repeat(base_root[None, :], args.frames, axis=0).astype(np.float32)
    phase = np.linspace(0.0, 1.0, args.frames, dtype=np.float32)
    root_trans[:, 0] += 0.004 * np.sin(np.pi * phase) ** 2
    root_trans[:, 1] += 0.003 * np.sin(np.pi * phase)
    root_trans[:, 2] += 0.002 * np.sin(np.pi * phase) ** 2
    root_trans, ground_shift, grounded_frame = ground_root_height(
        pose_aa, root_trans, fps, target_clearance=0.005
    )

    root_rot = np.repeat(base_root_rot[None, :], args.frames, axis=0).astype(np.float64)
    for i in range(args.frames):
        root_rot[i] = quat_xyzw_from_rotvec(pose_aa[i, 0])

    smpl_joints = np.repeat(base_smpl[None, :, :], args.frames, axis=0).astype(np.float32)

    out_motion = {}
    for key, value in motion.items():
        if key == "root_trans_offset":
            out_motion[key] = root_trans
        elif key == "pose_aa":
            out_motion[key] = pose_aa
        elif key == "dof":
            out_motion[key] = dof
        elif key == "root_rot":
            out_motion[key] = root_rot
        elif key == "smpl_joints":
            out_motion[key] = smpl_joints
        elif key == "fps":
            out_motion[key] = fps
        else:
            out_motion[key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({args.name: out_motion}, args.output)

    print(f"template={args.template}")
    print(f"template_key={template_key}")
    print(f"output={args.output}")
    print(f"output_key={args.name}")
    print(f"frames={args.frames}")
    print(f"fps={fps}")
    print(f"ground_shift={ground_shift:.6f}")
    print(f"grounded_frame={grounded_frame}")
    print("edited_dofs=right_hip_pitch,right_knee,right_ankle_pitch,left_leg_small_support,waist_small_comp")


if __name__ == "__main__":
    main()

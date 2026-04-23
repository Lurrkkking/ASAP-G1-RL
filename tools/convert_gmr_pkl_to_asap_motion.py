import argparse
from pathlib import Path

import joblib
import numpy as np
import yaml
from lxml.etree import XMLParser, parse
from scipy.spatial.transform import Rotation as R


def load_joint_order(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["robot"]["dof_names"])


def load_mjcf_dof_axes_by_name(mjcf_path: Path):
    parser = XMLParser(remove_blank_text=True)
    tree = parse(str(mjcf_path), parser=parser)
    worldbody = tree.getroot().find("worldbody")
    joint_nodes = worldbody.findall(".//joint")
    if "type" in joint_nodes[0].attrib and joint_nodes[0].attrib["type"] == "free":
        iter_joints = joint_nodes[1:]
    elif "type" not in joint_nodes[0].attrib:
        iter_joints = joint_nodes
    else:
        iter_joints = joint_nodes[6:]

    axis_by_name = {}
    for joint in iter_joints:
        if "axis" not in joint.attrib or "name" not in joint.attrib:
            continue
        axis_by_name[joint.attrib["name"]] = np.asarray(
            [int(x) for x in joint.attrib["axis"].split(" ")],
            dtype=np.float32,
        )
    return axis_by_name


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return quat / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--motion-name",
        default="converted_motion",
        help="Outer dict key expected by ASAP motion loader",
    )
    parser.add_argument(
        "--robot-config",
        default="/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml",
    )
    parser.add_argument(
        "--mjcf",
        default="/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    robot_config = Path(args.robot_config)
    mjcf_path = Path(args.mjcf)

    data = joblib.load(input_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in input pkl, got {type(data)}")

    required = ["fps", "root_pos", "root_rot", "joint_names", "dof_pos"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    expected_joint_names = load_joint_order(robot_config)
    actual_joint_names = [str(x) for x in data["joint_names"]]
    if actual_joint_names == expected_joint_names:
        joint_indices = np.arange(len(expected_joint_names), dtype=np.int64)
    else:
        actual_index_by_name = {name: idx for idx, name in enumerate(actual_joint_names)}
        missing_expected = [name for name in expected_joint_names if name not in actual_index_by_name]
        if missing_expected:
            raise ValueError(
                "joint_names do not match ASAP robot config order and missing expected joints.\n"
                f"missing={missing_expected}\nactual={actual_joint_names}\nexpected={expected_joint_names}"
            )
        joint_indices = np.asarray([actual_index_by_name[name] for name in expected_joint_names], dtype=np.int64)

    axis_by_name = load_mjcf_dof_axes_by_name(mjcf_path)
    missing_axes = [name for name in expected_joint_names if name not in axis_by_name]
    if missing_axes:
        raise ValueError(f"Missing joint axes in MJCF for: {missing_axes}")
    dof_axes = np.stack([axis_by_name[name] for name in expected_joint_names], axis=0)

    fps = int(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = normalize_quat_xyzw(np.asarray(data["root_rot"], dtype=np.float32))
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be (T,D), got {dof_pos.shape}")
    if dof_pos.shape[1] < len(expected_joint_names):
        raise ValueError(
            f"dof_pos has fewer joints than expected: got {dof_pos.shape[1]}, expected at least {len(expected_joint_names)}"
        )
    dof_pos = dof_pos[:, joint_indices]

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root_pos.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {root_rot_xyzw.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(expected_joint_names):
        raise ValueError(f"dof_pos must be (T,{len(expected_joint_names)}), got {dof_pos.shape}")
    if not (len(root_pos) == len(root_rot_xyzw) == len(dof_pos)):
        raise ValueError("Frame count mismatch among root_pos/root_rot/dof_pos")

    root_rotvec = R.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    joint_pose_aa = (dof_pos[..., None] * dof_axes[None, :, :]).astype(np.float32)

    # ASAP G1 fitmotionONLY config adds 3 extend markers (left hand, right hand, head).
    extend_pose_aa = np.zeros((dof_pos.shape[0], 3, 3), dtype=np.float32)
    pose_aa = np.concatenate(
        [root_rotvec[:, None, :], joint_pose_aa, extend_pose_aa],
        axis=1,
    ).astype(np.float32)

    out_motion = {
        "root_trans_offset": root_pos.astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "fps": fps,
    }
    out_data = {args.motion_name: out_motion}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out_data, output_path)

    print(f"saved: {output_path}")
    print(f"motion_name: {args.motion_name}")
    print(f"frames: {len(root_pos)} fps: {fps}")
    print(f"pose_aa shape: {pose_aa.shape}")
    print(f"dof shape: {dof_pos.shape}")


if __name__ == "__main__":
    main()

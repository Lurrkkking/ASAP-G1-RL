import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import rclpy
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "sim2real" / "rl_policy") not in sys.path:
    sys.path.append(str(REPO_ROOT / "sim2real" / "rl_policy"))

from sim2real.rl_policy.deepmimic_dec_loco_height import MotionTrackingDecLocoHeightPolicy
from sim2real.sim_env.base_sim import BaseSimulator
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


DEFAULT_TEMPLATE_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_gym_rollout_with_action.pkl"
)
DEFAULT_OUTPUT_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_mujoco_rollout_with_action.pkl"
)

FITMOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]
FITMOTION_EXTEND_CONFIG = [
    {"joint_name": "left_hand_link", "parent_name": "left_elbow_link", "pos": np.array([0.25, 0.0, 0.0], dtype=np.float32)},
    {"joint_name": "right_hand_link", "parent_name": "right_elbow_link", "pos": np.array([0.25, 0.0, 0.0], dtype=np.float32)},
    {"joint_name": "head_link", "parent_name": "torso_link", "pos": np.array([0.0, 0.0, 0.42], dtype=np.float32)},
]


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_dir = config_path.parent
    for key in ("ROBOT_SCENE", "ROBOT", "ASSET_ROOT"):
        if key in config:
            config[key] = _resolve_path(config_dir, config[key])
    return config


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _shape_of(value: Any):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return quat_wxyz[[1, 2, 3, 0]].copy()


def _load_pickle(path: str):
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as file:
        return pickle.load(file)


def _dump_pickle(data: Any, path: str):
    if joblib is not None:
        joblib.dump(data, path)
        return
    with open(path, "wb") as file:
        pickle.dump(data, file)


def _load_motion_template(path: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    data = _load_pickle(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Expected non-empty dict motion pkl, got {type(data).__name__}")
    motion_key = next(iter(data.keys()))
    motion_dict = data[motion_key]
    if not isinstance(motion_dict, dict):
        raise ValueError(f"Expected dict motion entry, got {type(motion_dict).__name__}")
    return data, motion_key, motion_dict


def _parse_dof_axis_from_model(mj_model: mujoco.MjModel) -> np.ndarray:
    dof_axis: List[np.ndarray] = []
    for joint_id in range(mj_model.njnt):
        if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_axis.append(np.asarray(mj_model.jnt_axis[joint_id], dtype=np.float32))
    return np.stack(dof_axis, axis=0)


class LockedBaseSimulator(BaseSimulator):
    def __init__(self, config, node):
        self.mj_lock = threading.Lock()
        super().__init__(config, node)

    def sim_step(self):
        with self.mj_lock:
            super().sim_step()

    def capture_rollout_frame(self, num_dof: int) -> Dict[str, np.ndarray]:
        with self.mj_lock:
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()
            body_pos, body_quat, body_lin_vel, body_ang_vel = self._capture_body_diagnostics_locked()

        return {
            "dof": qpos[7 : 7 + num_dof].astype(np.float32, copy=True),
            "root_trans_offset": qpos[:3].astype(np.float32, copy=True),
            "root_rot": _wxyz_to_xyzw(qpos[3:7]),
            "dof_vel": qvel[6 : 6 + num_dof].astype(np.float32, copy=True),
            "root_lin_vel": qvel[:3].astype(np.float32, copy=True),
            "root_ang_vel": qvel[3:6].astype(np.float32, copy=True),
            "body_pos": body_pos,
            "body_pos_w": body_pos.copy(),
            "body_quat": body_quat,
            "body_quat_w": body_quat.copy(),
            "body_lin_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
        }

    def _capture_body_diagnostics_locked(self):
        body_pos = []
        body_quat = []
        body_lin_vel = []
        body_ang_vel = []
        parent_cache = {}

        for body_name in FITMOTION_BODY_NAMES:
            body_id = self.mj_model.body(body_name).id
            vel6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.mj_model,
                self.mj_data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                vel6,
                0,
            )
            pos = self.mj_data.xpos[body_id].astype(np.float32, copy=True)
            quat_xyzw = _wxyz_to_xyzw(self.mj_data.xquat[body_id])
            lin_vel = vel6[3:6].astype(np.float32, copy=True)
            ang_vel = vel6[0:3].astype(np.float32, copy=True)

            parent_cache[body_name] = {
                "pos": pos,
                "quat_xyzw": quat_xyzw,
                "lin_vel": lin_vel,
                "ang_vel": ang_vel,
            }
            body_pos.append(pos)
            body_quat.append(quat_xyzw)
            body_lin_vel.append(lin_vel)
            body_ang_vel.append(ang_vel)

        for extend_cfg in FITMOTION_EXTEND_CONFIG:
            parent = parent_cache[extend_cfg["parent_name"]]
            parent_rot = sRot.from_quat(parent["quat_xyzw"])
            offset_world = parent_rot.apply(extend_cfg["pos"])
            pos = (parent["pos"] + offset_world).astype(np.float32, copy=True)
            quat_xyzw = parent["quat_xyzw"].copy()
            lin_vel = (parent["lin_vel"] + np.cross(parent["ang_vel"], offset_world)).astype(np.float32, copy=True)
            ang_vel = parent["ang_vel"].copy()

            body_pos.append(pos)
            body_quat.append(quat_xyzw)
            body_lin_vel.append(lin_vel)
            body_ang_vel.append(ang_vel)

        return (
            np.stack(body_pos, axis=0),
            np.stack(body_quat, axis=0),
            np.stack(body_lin_vel, axis=0),
            np.stack(body_ang_vel, axis=0),
        )


class MotionTrackingDecLocoHeightCollector(MotionTrackingDecLocoHeightPolicy):
    def __init__(
        self,
        config,
        node,
        simulator: LockedBaseSimulator,
        loco_model_path,
        mimic_model_paths,
        use_jit,
        template_motion_path: str,
        output_path: str,
        num_steps: int,
        mimic_policy_name: Optional[str],
    ):
        self.simulator = simulator
        self.template_motion_path = template_motion_path
        self.output_path = output_path
        self.num_steps = int(num_steps)
        self.mimic_policy_name = mimic_policy_name
        self.motion_template_data, self.motion_key, self.motion_template = _load_motion_template(template_motion_path)
        super().__init__(
            config=config,
            node=node,
            loco_model_path=loco_model_path,
            mimic_model_paths=mimic_model_paths,
            use_jit=use_jit,
            rl_rate=50,
            decimation=4,
        )
        self.default_dof_angles_np = np.asarray(self.default_dof_angles, dtype=np.float32)
        self.dof_axis = _parse_dof_axis_from_model(self.simulator.mj_model)
        self.num_augment_joint = len(FITMOTION_EXTEND_CONFIG)

        if self.history_handler is not None:
            self.history_handler.reset([0])
        self.use_policy_action = True
        self.get_ready_state = False
        self.stand_command[:] = 1
        self.lin_vel_command[:] = 0
        self.ang_vel_command[:] = 0

        if self.mimic_policy_name:
            self._enable_mimic_mode(self.mimic_policy_name)

    def start_key_listener(self):
        return

    def setup_mimic_policies(self):
        if self.mimic_model_paths is None:
            self.policies_mimic = []
            self.policy_mimic_names = []
            self.start_upper_dof_pos = []
            self.motion_length_s = []
            self.policy_mimic_robot_types = []
            self.policy_mimic_robot_dofs = []
            self.num_mimic_policies = 0
            logger.warning("mimic_model_paths is None, mimic policies are disabled for collect mode.")
            return
        super().setup_mimic_policies()

    def _enable_mimic_mode(self, policy_name: str):
        if policy_name not in self.policy_mimic_names:
            raise ValueError(
                f"mimic_policy_name={policy_name} not found in config mimic_models: {self.policy_mimic_names}"
            )
        self.policy_mimic_idx = self.policy_mimic_names.index(policy_name)
        self.policy_locomotion_mimic_flag = 1
        self.interpolation_done = True
        self.interpolation_active = False
        self.interpolation_emergency = False
        self.policy = self.policies_mimic[self.policy_mimic_idx]
        self.phase = 0.0
        self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
        if self.history_handler is not None:
            self.history_handler.reset([0])
        logger.info(f"Collect mode starts directly in mimic policy: {policy_name}")

    def collect_one_step(self):
        robot_state_data = self.state_processor._prepare_low_state()
        if robot_state_data is None:
            return None

        frame = self.simulator.capture_rollout_frame(self.num_dofs)
        scaled_policy_action = self.get_policy_action(robot_state_data)

        if self.get_ready_state:
            q_target = self.get_init_target(robot_state_data)
            if self.init_count > 500:
                self.init_count = 500
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
        else:
            if scaled_policy_action.shape[1] != self.num_dofs:
                scaled_policy_action = np.concatenate(
                    [scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))],
                    axis=1,
                )
            q_target = scaled_policy_action + self.default_dof_angles_np[None, :]

        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        cmd_q = q_target[0].astype(np.float32, copy=True)
        cmd_dq = np.zeros(self.num_dofs, dtype=np.float32)
        cmd_tau = np.zeros(self.num_dofs, dtype=np.float32)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

        effective_action = ((cmd_q - self.default_dof_angles_np) / float(self.policy_action_scale)).astype(
            np.float32,
            copy=True,
        )
        frame["action"] = effective_action
        return frame

    def build_output_motion(self, rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
        output_motion = dict(self.motion_template)
        num_steps = int(rollout["action"].shape[0])

        output_motion["dof"] = rollout["dof"]
        output_motion["dof_vel"] = rollout["dof_vel"]
        output_motion["root_trans_offset"] = rollout["root_trans_offset"]
        output_motion["root_rot"] = rollout["root_rot"]
        output_motion["root_lin_vel"] = rollout["root_lin_vel"]
        output_motion["root_ang_vel"] = rollout["root_ang_vel"]
        output_motion["action"] = rollout["action"]
        output_motion["body_pos"] = rollout["body_pos"]
        output_motion["body_pos_w"] = rollout["body_pos_w"]
        output_motion["body_quat"] = rollout["body_quat"]
        output_motion["body_quat_w"] = rollout["body_quat_w"]
        output_motion["body_lin_vel"] = rollout["body_lin_vel"]
        output_motion["body_ang_vel"] = rollout["body_ang_vel"]
        output_motion["fps"] = 50
        output_motion["dt"] = float(1.0 / 50.0)
        output_motion["motion_length"] = num_steps
        output_motion["pose_aa"] = np.stack(
            [self._build_pose_aa(output_motion["root_rot"][i], output_motion["dof"][i]) for i in range(num_steps)],
            axis=0,
        ).astype(np.float32)
        return output_motion

    def _build_pose_aa(self, root_rot_xyzw: np.ndarray, dof: np.ndarray) -> np.ndarray:
        root_rot_vec = sRot.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
        return np.concatenate(
            [
                root_rot_vec[None, :],
                self.dof_axis * dof[:, None],
                np.zeros((self.num_augment_joint, 3), dtype=np.float32),
            ],
            axis=0,
        )


def _stack_rollout_buffer(buffer: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack(value, axis=0).astype(np.float32) for key, value in buffer.items()}


def _check_array(name: str, value: Any):
    arr = _to_numpy(value)
    if not np.issubdtype(arr.dtype, np.number):
        return
    logger.info(f"check[{name}] has_nan={bool(np.isnan(arr).any())} has_inf={bool(np.isinf(arr).any())}")


def _print_summary(output_path: str, motion_key: str, motion: Dict[str, Any]):
    logger.info(f"output_path={output_path}")
    logger.info(f"motion_key={motion_key}")
    for key, value in motion.items():
        logger.info(f"output[{key}] shape={_shape_of(value)} type={type(value).__name__}")

    action = motion["action"]
    logger.info(
        "action_stats "
        f"mean={float(action.mean()):.6f} "
        f"std={float(action.std()):.6f} "
        f"min={float(action.min()):.6f} "
        f"max={float(action.max()):.6f}"
    )
    for key, value in motion.items():
        _check_array(key, value)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Sim2Sim deltaA rollout collector")
    parser.add_argument("--config", type=str, default="sim2real/config/g1_29dof_hist.yaml", help="config file")
    parser.add_argument("--loco_model_path", type=str, required=True, help="locomotion model path")
    parser.add_argument("--mimic_model_paths", type=str, default=None, help="mimic model root dir")
    parser.add_argument("--mimic_policy_name", type=str, default=None, help="optional mimic policy name from config")
    parser.add_argument("--use_jit", action="store_true", default=False, help="use jit")
    parser.add_argument("--use_mocap", action="store_true", default=False, help="use mocap")
    parser.add_argument("--num_steps", type=int, default=134, help="number of policy steps to collect")
    parser.add_argument(
        "--template_motion_path",
        type=str,
        default=DEFAULT_TEMPLATE_MOTION_PATH,
        help="existing gym rollout pkl used as output template",
    )
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="output pkl path")
    parser.add_argument("--wait_for_state_timeout_s", type=float, default=10.0, help="timeout for first lowstate")
    args = parser.parse_args()

    config = _load_config(args.config)

    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    rclpy.init(args=None)
    node = rclpy.create_node("mujoco_rollout_collector")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    simulator = LockedBaseSimulator(config, node)
    simulator.sim_thread.start()

    collector = MotionTrackingDecLocoHeightCollector(
        config=config,
        node=node,
        simulator=simulator,
        loco_model_path=args.loco_model_path,
        mimic_model_paths=args.mimic_model_paths,
        use_jit=args.use_jit,
        template_motion_path=args.template_motion_path,
        output_path=args.output_path,
        num_steps=args.num_steps,
        mimic_policy_name=args.mimic_policy_name,
    )

    start_wait = time.time()
    while time.time() - start_wait < args.wait_for_state_timeout_s:
        if collector.state_processor._prepare_low_state() is not None:
            break
        time.sleep(0.05)
    else:
        raise TimeoutError(f"No low state received within {args.wait_for_state_timeout_s:.2f}s")

    rollout_buffer = {
        "dof": [],
        "dof_vel": [],
        "root_trans_offset": [],
        "root_rot": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "body_pos": [],
        "body_pos_w": [],
        "body_quat": [],
        "body_quat_w": [],
        "body_lin_vel": [],
        "body_ang_vel": [],
        "action": [],
    }

    try:
        step = 0
        while len(rollout_buffer["action"]) < args.num_steps:
            frame = collector.collect_one_step()
            if frame is None:
                time.sleep(0.02)
                continue
            for key, value in frame.items():
                rollout_buffer[key].append(value)
            if step == 0 or step == args.num_steps - 1:
                logger.info(
                    f"step={step} action_mean={float(frame['action'].mean()):.6f} "
                    f"action_std={float(frame['action'].std()):.6f} "
                    f"action_min={float(frame['action'].min()):.6f} "
                    f"action_max={float(frame['action'].max()):.6f}"
                )
            step += 1
            collector.rate.sleep()
    finally:
        if hasattr(simulator.viewer, "close"):
            simulator.viewer.close()
        if rclpy.ok():
            rclpy.shutdown()

    rollout = _stack_rollout_buffer(rollout_buffer)
    output_motion = collector.build_output_motion(rollout)
    output_data = {collector.motion_key: output_motion}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_pickle(output_data, str(output_path))
    _print_summary(str(output_path), collector.motion_key, output_motion)


if __name__ == "__main__":
    main()

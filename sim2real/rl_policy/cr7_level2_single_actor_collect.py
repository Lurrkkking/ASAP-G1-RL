import argparse
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import onnxruntime
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import rclpy
except ImportError:  # pragma: no cover
    rclpy = None

try:
    import mujoco.viewer as mujoco_viewer
except ImportError:  # pragma: no cover
    mujoco_viewer = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sim2real.utils.robot import Robot
try:
    from sim2real.utils.command_sender import CommandSender
    from sim2real.utils.state_processor import StateProcessor
    from sim2real.utils.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    HAS_UNITREE_SDK = True
except ImportError:  # pragma: no cover
    CommandSender = None
    StateProcessor = None
    ElasticBand = None
    UnitreeSdk2Bridge = None
    ChannelFactoryInitialize = None
    HAS_UNITREE_SDK = False


POLICY_PATH = "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/baseline13000group/exported/model_13000.onnx"
TRAIN_CONFIG_PATH = "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/baseline13000group/config.yaml"
SOURCE_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
)
TEMPLATE_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_gym_rollout_with_action.pkl"
)
OUTPUT_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_mujoco_rollout_with_action.pkl"
)

EXPECTED_ACTOR_OBS_ORDER = [
    "base_ang_vel",
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "actions",
    "ref_motion_phase",
    "history_actor",
]
EXPECTED_HISTORY_ACTOR = {
    "base_ang_vel": 4,
    "projected_gravity": 4,
    "dof_pos": 4,
    "dof_vel": 4,
    "actions": 4,
    "ref_motion_phase": 4,
}

BODY_NAMES_24 = [
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
EXTEND_OFFSETS = [
    ("left_elbow_link", np.array([0.25, 0.0, 0.0], dtype=np.float32)),
    ("right_elbow_link", np.array([0.25, 0.0, 0.0], dtype=np.float32)),
    ("torso_link", np.array([0.0, 0.0, 0.42], dtype=np.float32)),
]


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = [
        (base_dir / path).resolve(),
        (base_dir.parent / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


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


def _stats_dict(array: Any) -> Dict[str, float]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _stats_text(name: str, array: Any) -> str:
    stats = _stats_dict(array)
    return (
        f"{name}[mean={stats['mean']:.6f} std={stats['std']:.6f} "
        f"min={stats['min']:.6f} max={stats['max']:.6f}]"
    )


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


def _load_sim_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config_dir = config_path.parent
    for key in ("ROBOT_SCENE", "ROBOT", "ASSET_ROOT"):
        if key in config:
            config[key] = _resolve_path(config_dir, config[key])
    return config


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def _load_motion_dict(path: str) -> Tuple[str, Dict[str, Any]]:
    data = _load_pickle(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Expected non-empty dict motion pkl, got {type(data).__name__}")
    key = next(iter(data.keys()))
    motion = data[key]
    if not isinstance(motion, dict):
        raise ValueError(f"Expected dict motion entry, got {type(motion).__name__}")
    return key, motion


def _validate_actor_obs_config(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    robot_cfg = train_cfg["robot"]
    obs_cfg = train_cfg["obs"]
    actor_obs_order = obs_cfg["obs_dict"]["actor_obs"]
    history_actor_cfg = obs_cfg["obs_auxiliary"]["history_actor"]

    if actor_obs_order != EXPECTED_ACTOR_OBS_ORDER:
        raise ValueError(f"actor_obs order mismatch: {actor_obs_order}")
    if history_actor_cfg != EXPECTED_HISTORY_ACTOR:
        raise ValueError(f"history_actor config mismatch: {history_actor_cfg}")

    obs_dims_cfg = {}
    for item in obs_cfg["obs_dims"]:
        obs_dims_cfg.update(item)

    def _resolve_dim(value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value == "${robot.dof_obs_size}":
                return int(robot_cfg["dof_obs_size"])
            raise ValueError(f"Unsupported obs dim expression: {value}")
        return int(value)

    expected_dims = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "dof_pos": int(robot_cfg["dof_obs_size"]),
        "dof_vel": int(robot_cfg["dof_obs_size"]),
        "actions": int(robot_cfg["dof_obs_size"]),
        "ref_motion_phase": 1,
    }
    for key, expected in expected_dims.items():
        actual = _resolve_dim(obs_dims_cfg[key])
        if actual != expected:
            raise ValueError(f"obs_dims[{key}] mismatch: actual={actual} expected={expected}")

    history_single_dim = (
        expected_dims["base_ang_vel"]
        + expected_dims["projected_gravity"]
        + expected_dims["dof_pos"]
        + expected_dims["dof_vel"]
        + expected_dims["actions"]
        + expected_dims["ref_motion_phase"]
    )
    history_dim = history_single_dim * sum(history_actor_cfg.values()) // len(history_actor_cfg)
    actor_obs_dim = (
        expected_dims["base_ang_vel"]
        + expected_dims["projected_gravity"]
        + expected_dims["dof_pos"]
        + expected_dims["dof_vel"]
        + expected_dims["actions"]
        + expected_dims["ref_motion_phase"]
        + history_dim
    )
    if actor_obs_dim != 380:
        raise ValueError(f"actor_obs_dim mismatch: actual={actor_obs_dim} expected=380")

    return {
        "num_actions": int(robot_cfg["actions_dim"]),
        "dof_names": list(robot_cfg["dof_names"]),
        "default_dof_pos": np.array(
            [robot_cfg["init_state"]["default_joint_angles"][name] for name in robot_cfg["dof_names"]],
            dtype=np.float32,
        ),
        "action_scale": float(robot_cfg["control"]["action_scale"]),
        "action_clip_value": float(robot_cfg["control"]["action_clip_value"]),
        "obs_scale_base_ang_vel": float(obs_cfg["obs_scales"]["base_ang_vel"]),
        "obs_scale_projected_gravity": float(obs_cfg["obs_scales"]["projected_gravity"]),
        "obs_scale_dof_pos": float(obs_cfg["obs_scales"]["dof_pos"]),
        "obs_scale_dof_vel": float(obs_cfg["obs_scales"]["dof_vel"]),
        "obs_scale_actions": float(obs_cfg["obs_scales"]["actions"]),
        "obs_scale_ref_motion_phase": float(obs_cfg["obs_scales"]["ref_motion_phase"]),
        "obs_scale_history_actor": float(obs_cfg["obs_scales"]["history_actor"]),
        "clip_observations": float(train_cfg["env"]["config"]["normalization"]["clip_observations"]),
        "history_len": int(next(iter(history_actor_cfg.values()))),
    }


def _infer_cycle_time(source_motion: Dict[str, Any]) -> float:
    fps = source_motion.get("fps", None)
    if fps is None:
        raise ValueError("source motion missing fps, cannot infer cycle_time")
    num_frames = int(source_motion["dof"].shape[0])
    if num_frames < 2:
        raise ValueError(f"source motion has too few frames: {num_frames}")
    return float((num_frames - 1) / float(fps))


def _load_model_signature(policy_path: str) -> Tuple[onnxruntime.InferenceSession, str, str]:
    session = onnxruntime.InferenceSession(policy_path)
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or len(output_meta) != 1:
        raise ValueError("Expected exactly one ONNX input and one ONNX output")
    if list(input_meta[0].shape) != [1, 380]:
        raise ValueError(f"Unexpected ONNX input shape: {input_meta[0].shape}, expected [1, 380]")
    if list(output_meta[0].shape) != [1, 23]:
        raise ValueError(f"Unexpected ONNX output shape: {output_meta[0].shape}, expected [1, 23]")
    return session, input_meta[0].name, output_meta[0].name

class _FallbackRate:
    def __init__(self, hz: float):
        self.period = 1.0 / float(hz)

    def sleep(self):
        time.sleep(self.period)


class _FallbackClockNow:
    @property
    def nanoseconds(self):
        return int(time.time() * 1e9)


class _FallbackClock:
    def now(self):
        return _FallbackClockNow()


class _FallbackLogger:
    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


class _FallbackNode:
    def create_rate(self, hz: float):
        return _FallbackRate(hz)

    def get_clock(self):
        return _FallbackClock()

    def get_logger(self):
        return _FallbackLogger()


class _HeadlessViewer:
    def __init__(self):
        self._running = True

    def is_running(self):
        return self._running

    def sync(self):
        return

    def close(self):
        self._running = False


class LockedBaseSimulator:
    def __init__(self, config, node):
        self.mj_lock = threading.Lock()
        self.config = config
        self.node = node
        self.rate = self.node.create_rate(1 / self.config["SIMULATE_DT"])
        self.viewer_rate = self.node.create_rate(1 / self.config["VIEWER_DT"])
        self.init_config()
        self.init_scene()
        self.init_unitree_bridge()
        self.sim_thread = threading.Thread(target=self.SimulationThread)

    def init_config(self):
        self.robot = Robot(self.config)
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)

    def init_scene(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        ctrlrange = np.asarray(self.mj_model.actuator_ctrlrange, dtype=np.float32)
        ctrllimited = np.asarray(self.mj_model.actuator_ctrllimited, dtype=bool)
        ctrl_limit = np.full((self.robot.NUM_JOINTS,), np.inf, dtype=np.float32)
        usable = min(self.robot.NUM_JOINTS, ctrlrange.shape[0])
        if usable > 0:
            ctrl_limit[:usable] = np.where(
                ctrllimited[:usable],
                np.max(np.abs(ctrlrange[:usable]), axis=1),
                np.inf,
            ).astype(np.float32)
        self.actuator_ctrl_limit = ctrl_limit
        self._viewer_alive = True
        self.viewer = _HeadlessViewer()
        self.elastic_band = None
        if mujoco_viewer is None or not os.environ.get("DISPLAY"):
            return
        if self.config["ENABLE_ELASTIC_BAND"] and ElasticBand is not None:
            self.elastic_band = ElasticBand()
            if "h1" in self.config["ROBOT_TYPE"] or "g1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            self.viewer = mujoco_viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco_viewer.launch_passive(self.mj_model, self.mj_data)

    def init_unitree_bridge(self):
        self.unitree_bridge = None
        if HAS_UNITREE_SDK:
            self.unitree_bridge = UnitreeSdk2Bridge(self.mj_model, self.mj_data, self.config)
        config_limit = np.asarray(self.config["motor_effort_limit_list"], dtype=np.float32)
        bridge_limit = None
        if self.unitree_bridge is not None and hasattr(self.unitree_bridge, "torque_limit"):
            bridge_limit = np.asarray(self.unitree_bridge.torque_limit, dtype=np.float32)
        self.effective_torque_limit = config_limit.copy()
        self.effective_torque_limit = np.minimum(self.effective_torque_limit, self.actuator_ctrl_limit)
        if bridge_limit is not None:
            self.effective_torque_limit = np.minimum(self.effective_torque_limit, bridge_limit)

    def compute_torques(self):
        if self.unitree_bridge is None:
            return
        if self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_motor):
                if self.unitree_bridge.use_sensor:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].dq - self.mj_data.sensordata[i + self.num_dof])
                    )
                else:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.qpos[7 + i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                    )
        self.torques = np.clip(
            self.torques,
            -self.effective_torque_limit,
            self.effective_torque_limit,
        )

    def sim_step(self):
        with self.mj_lock:
            if self.unitree_bridge is not None:
                self.unitree_bridge.PublishLowState()
            if self.config["ENABLE_ELASTIC_BAND"] and self.elastic_band is not None and self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
            self.compute_torques()
            if self.unitree_bridge is not None and self.unitree_bridge.free_base:
                self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
            else:
                self.mj_data.ctrl = self.torques
            mujoco.mj_step(self.mj_model, self.mj_data)

    def SimulationThread(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running() and self._viewer_alive:
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.node.get_logger().info(f"FPS: {100 / (end_time - start_time)}")
                start_time = end_time
            self.rate.sleep()

    def close(self):
        self._viewer_alive = False
        if hasattr(self.viewer, "close"):
            self.viewer.close()

    def get_frame_data(self, selected_qpos_indices: np.ndarray) -> Dict[str, np.ndarray]:
        with self.mj_lock:
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()
            body_pos, body_quat, body_lin_vel, body_ang_vel = self._capture_body_diagnostics_locked()

        return {
            "root_trans_offset": qpos[:3].astype(np.float32, copy=True),
            "root_rot": _wxyz_to_xyzw(qpos[3:7]),
            "dof": qpos[7:][selected_qpos_indices].astype(np.float32, copy=True),
            "dof_vel": qvel[6:][selected_qpos_indices].astype(np.float32, copy=True),
            "root_lin_vel": qvel[:3].astype(np.float32, copy=True),
            "root_ang_vel": qvel[3:6].astype(np.float32, copy=True),
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_lin_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
        }

    def get_obs_state(self, selected_qpos_indices: np.ndarray) -> Dict[str, np.ndarray]:
        with self.mj_lock:
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()

        root_quat_xyzw = _wxyz_to_xyzw(qpos[3:7])
        root_rot = sRot.from_quat(root_quat_xyzw)
        base_ang_vel_local = root_rot.apply(qvel[3:6], inverse=True).astype(np.float32)
        projected_gravity = root_rot.apply(np.array([0.0, 0.0, -1.0], dtype=np.float32), inverse=True).astype(np.float32)
        dof_pos = qpos[7:][selected_qpos_indices].astype(np.float32, copy=True)
        dof_vel = qvel[6:][selected_qpos_indices].astype(np.float32, copy=True)
        return {
            "base_ang_vel_local": base_ang_vel_local,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }

    def get_diagnostics(self, selected_qpos_indices: np.ndarray) -> Dict[str, np.ndarray]:
        with self.mj_lock:
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()
            qacc = self.mj_data.qacc.copy()
            torques = self.torques.copy()
        selected_q = qpos[7:][selected_qpos_indices].astype(np.float32, copy=True)
        selected_qvel = qvel[6:][selected_qpos_indices].astype(np.float32, copy=True)
        qacc_abs = np.abs(qacc.astype(np.float32, copy=False))
        qacc_nonfinite = bool((~np.isfinite(qacc_abs)).any())
        qacc_max_abs = float(np.nanmax(qacc_abs)) if qacc_abs.size > 0 else 0.0
        return {
            "dof_q": selected_q,
            "dof_vel": selected_qvel,
            "tau": torques.astype(np.float32, copy=True),
            "root_height": np.float32(qpos[2]),
            "qacc_max_abs": np.float32(qacc_max_abs),
            "qacc_nonfinite": np.bool_(qacc_nonfinite),
        }

    def _capture_body_diagnostics_locked(self):
        body_pos = []
        body_quat = []
        body_lin_vel = []
        body_ang_vel = []
        for body_name in BODY_NAMES_24:
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
            body_pos.append(self.mj_data.xpos[body_id].astype(np.float32, copy=True))
            body_quat.append(_wxyz_to_xyzw(self.mj_data.xquat[body_id]))
            body_lin_vel.append(vel6[3:6].astype(np.float32, copy=True))
            body_ang_vel.append(vel6[0:3].astype(np.float32, copy=True))
        return (
            np.stack(body_pos, axis=0),
            np.stack(body_quat, axis=0),
            np.stack(body_lin_vel, axis=0),
            np.stack(body_ang_vel, axis=0),
        )


class SingleActorCollector:
    def __init__(
        self,
        sim_config: Dict[str, Any],
        node,
        simulator: LockedBaseSimulator,
        train_cfg_path: str,
        policy_path: str,
        source_motion_path: str,
        template_motion_path: str,
        output_path: str,
        num_steps: int,
    ):
        self.sim_config = sim_config
        self.node = node
        self.rate = self.node.create_rate(50)
        self.simulator = simulator
        self.robot = Robot(sim_config)
        self.state_processor = StateProcessor(sim_config) if HAS_UNITREE_SDK else None
        self.command_sender = CommandSender(sim_config) if HAS_UNITREE_SDK else None
        self.use_unitree_sdk = HAS_UNITREE_SDK
        self.num_steps = int(num_steps)
        self.output_path = output_path

        train_cfg = _load_yaml(train_cfg_path)
        self.policy_cfg = _validate_actor_obs_config(train_cfg)
        _, self.source_motion = _load_motion_dict(source_motion_path)
        self.template_key, self.template_motion = _load_motion_dict(template_motion_path)
        self.cycle_time = _infer_cycle_time(self.source_motion)

        self.onnx_session, self.onnx_input_name, self.onnx_output_name = _load_model_signature(policy_path)

        self.full_dof_names = self._build_full_dof_names()
        self.selected_dof_indices = np.array(
            [self.full_dof_names.index(name) for name in self.policy_cfg["dof_names"]],
            dtype=np.int64,
        )
        self.full_default_dof = np.asarray(self.sim_config["DEFAULT_DOF_ANGLES"], dtype=np.float32)
        self.selected_default_dof = self.full_default_dof[self.selected_dof_indices].copy()

        if not np.allclose(self.selected_default_dof, self.policy_cfg["default_dof_pos"], atol=1e-6):
            raise ValueError(
                "default_dof_pos mismatch between sim2real config and training config for selected 23 dofs"
            )

        self.full_dof_axis = self._parse_full_dof_axis()
        self.selected_dof_axis = self.full_dof_axis[self.selected_dof_indices].copy()
        self.last_action = np.zeros(self.policy_cfg["num_actions"], dtype=np.float32)
        self.history = {
            "base_ang_vel": np.zeros((self.policy_cfg["history_len"], 3), dtype=np.float32),
            "projected_gravity": np.zeros((self.policy_cfg["history_len"], 3), dtype=np.float32),
            "dof_pos": np.zeros((self.policy_cfg["history_len"], self.policy_cfg["num_actions"]), dtype=np.float32),
            "dof_vel": np.zeros((self.policy_cfg["history_len"], self.policy_cfg["num_actions"]), dtype=np.float32),
            "actions": np.zeros((self.policy_cfg["history_len"], self.policy_cfg["num_actions"]), dtype=np.float32),
            "ref_motion_phase": np.zeros((self.policy_cfg["history_len"], 1), dtype=np.float32),
        }
        self.action_abs_thresholds = (10.0, 20.0, 50.0, 90.0)
        self.first_trigger_steps: Dict[str, int] = {}
        self.node.get_logger().info(
            "pkl['action'] source: clipped ONNX action directly from collector.step_policy, "
            "not inferred back from q_target."
        )
        self._log_torque_limit_diagnostics()

    def _log_torque_limit_diagnostics(self):
        config_limit = np.asarray(self.sim_config["motor_effort_limit_list"], dtype=np.float32)
        ctrl_limit = self.simulator.actuator_ctrl_limit
        effective_limit = self.simulator.effective_torque_limit
        finite_ctrl = np.isfinite(ctrl_limit)
        if np.any(finite_ctrl):
            mismatch = np.where(np.abs(config_limit[finite_ctrl] - ctrl_limit[finite_ctrl]) > 1e-4)[0]
            if mismatch.size > 0:
                self.node.get_logger().warning(
                    f"torque clip mismatch config vs actuator ctrlrange at indices={mismatch.tolist()}"
                )
        self.node.get_logger().info(
            "torque clip validation "
            f"config_limit={_stats_text('cfg', config_limit)} "
            f"ctrlrange_limit={_stats_text('ctrl', ctrl_limit[np.isfinite(ctrl_limit)]) if np.any(finite_ctrl) else 'ctrl[unlimited]'} "
            f"effective_limit={_stats_text('effective', effective_limit)}"
        )

    def _build_full_dof_names(self) -> List[str]:
        names = []
        for joint_id in range(self.simulator.mj_model.njnt):
            if self.simulator.mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(self.simulator.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            names.append(name)
        return names

    def _parse_full_dof_axis(self) -> np.ndarray:
        dof_axis = []
        for joint_id in range(self.simulator.mj_model.njnt):
            if self.simulator.mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_axis.append(np.asarray(self.simulator.mj_model.jnt_axis[joint_id], dtype=np.float32))
        dof_axis = np.stack(dof_axis, axis=0)
        if dof_axis.shape[0] != len(self.full_dof_names):
            raise ValueError(f"full dof axis size mismatch: {dof_axis.shape[0]} vs {len(self.full_dof_names)}")
        return dof_axis

    def wait_for_first_low_state(self, timeout_s: float):
        if not self.use_unitree_sdk:
            time.sleep(0.1)
            return
        start = time.time()
        while time.time() - start < timeout_s:
            if self.state_processor._prepare_low_state() is not None:
                return
            time.sleep(0.05)
        raise TimeoutError(f"No low state received within {timeout_s:.2f}s")

    def compute_ref_motion_phase(self, step_idx: int) -> np.float32:
        sim_time = (step_idx + 1) * float(self.sim_config["SIMULATE_DT"]) * 4.0
        phase_time = min(sim_time, self.cycle_time - 1e-6)
        return np.float32(np.clip(phase_time / self.cycle_time, 0.0, 1.0))

    def initialize_sim_from_motion(self, frame_idx: int = 0):
        motion = self.source_motion
        frame_idx = int(np.clip(frame_idx, 0, motion["dof"].shape[0] - 1))
        root_pos = np.asarray(motion["root_trans_offset"][frame_idx], dtype=np.float32)
        root_rot_xyzw = np.asarray(motion["root_rot"][frame_idx], dtype=np.float32)
        root_rot_wxyz = root_rot_xyzw[[3, 0, 1, 2]].copy()
        dof = np.asarray(motion["dof"][frame_idx], dtype=np.float32)
        root_lin_vel = np.asarray(motion.get("root_lin_vel", np.zeros((1, 3), dtype=np.float32))[frame_idx], dtype=np.float32)
        root_ang_vel = np.asarray(motion.get("root_ang_vel", np.zeros((1, 3), dtype=np.float32))[frame_idx], dtype=np.float32)
        dof_vel_src = motion.get("dof_vel", None)
        if dof_vel_src is None:
            dof_vel = np.zeros_like(dof)
        else:
            dof_vel = np.asarray(dof_vel_src[frame_idx], dtype=np.float32)

        with self.simulator.mj_lock:
            qpos = self.simulator.mj_data.qpos.copy()
            qvel = self.simulator.mj_data.qvel.copy()
            qpos[:3] = root_pos
            qpos[3:7] = root_rot_wxyz
            qpos[7:] = self.full_default_dof
            qpos[7:][self.selected_dof_indices] = dof
            qvel[:] = 0.0
            qvel[:3] = root_lin_vel
            qvel[3:6] = root_ang_vel
            qvel[6:][self.selected_dof_indices] = dof_vel
            self.simulator.mj_data.qpos[:] = qpos
            self.simulator.mj_data.qvel[:] = qvel
            self.simulator.mj_data.qacc[:] = 0.0
            self.simulator.mj_data.ctrl[:] = 0.0
            self.simulator.torques[:] = 0.0
            mujoco.mj_forward(self.simulator.mj_model, self.simulator.mj_data)

        self.last_action[:] = 0.0
        for key in self.history:
            self.history[key][:] = 0.0
        self.node.get_logger().info(
            f"initialized MuJoCo state from source motion frame={frame_idx} "
            f"root_height={float(root_pos[2]):.6f}"
        )

    def build_actor_obs(self, step_idx: int) -> np.ndarray:
        obs_state = self.simulator.get_obs_state(self.selected_dof_indices)
        ref_motion_phase = self.compute_ref_motion_phase(step_idx)

        current_single = np.zeros((76,), dtype=np.float32)
        current_single[0:23] = self.last_action * self.policy_cfg["obs_scale_actions"]
        current_single[23:26] = obs_state["base_ang_vel_local"] * self.policy_cfg["obs_scale_base_ang_vel"]
        current_single[26:49] = (
            (obs_state["dof_pos"] - self.selected_default_dof) * self.policy_cfg["obs_scale_dof_pos"]
        )
        current_single[49:72] = obs_state["dof_vel"] * self.policy_cfg["obs_scale_dof_vel"]
        current_single[72:75] = obs_state["projected_gravity"] * self.policy_cfg["obs_scale_projected_gravity"]
        current_single[75] = ref_motion_phase * self.policy_cfg["obs_scale_ref_motion_phase"]

        history_cat = np.concatenate(
            [
                self.history["actions"].reshape(-1),
                self.history["base_ang_vel"].reshape(-1),
                self.history["dof_pos"].reshape(-1),
                self.history["dof_vel"].reshape(-1),
                self.history["projected_gravity"].reshape(-1),
                self.history["ref_motion_phase"].reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        actor_obs = np.zeros((1, 380), dtype=np.float32)
        actor_obs[0, 0:23] = current_single[0:23]
        actor_obs[0, 23:26] = current_single[23:26]
        actor_obs[0, 26:49] = current_single[26:49]
        actor_obs[0, 49:72] = current_single[49:72]
        actor_obs[0, 72:376] = history_cat * self.policy_cfg["obs_scale_history_actor"]
        actor_obs[0, 376:379] = current_single[72:75]
        actor_obs[0, 379] = current_single[75]
        actor_obs = np.clip(actor_obs, -self.policy_cfg["clip_observations"], self.policy_cfg["clip_observations"])

        self._update_history(current_single)
        return actor_obs

    def _update_history(self, current_single: np.ndarray):
        mapping = {
            "actions": current_single[0:23],
            "base_ang_vel": current_single[23:26],
            "dof_pos": current_single[26:49],
            "dof_vel": current_single[49:72],
            "projected_gravity": current_single[72:75],
            "ref_motion_phase": current_single[75:76],
        }
        for key, value in mapping.items():
            self.history[key][1:] = self.history[key][:-1]
            self.history[key][0] = value

    def step_policy(self, step_idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        if self.use_unitree_sdk:
            _ = self.state_processor._prepare_low_state()
        actor_obs = self.build_actor_obs(step_idx)
        raw_action = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: actor_obs})[0]
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        clipped_action = np.clip(
            raw_action,
            -self.policy_cfg["action_clip_value"],
            self.policy_cfg["action_clip_value"],
        )

        q_target_full = self.full_default_dof.copy()
        q_target_selected = self.selected_default_dof + clipped_action * self.policy_cfg["action_scale"]
        q_target_full[self.selected_dof_indices] = q_target_selected
        q_target_full = np.clip(
            q_target_full,
            np.asarray(self.sim_config["motor_pos_lower_limit_list"], dtype=np.float32),
            np.asarray(self.sim_config["motor_pos_upper_limit_list"], dtype=np.float32),
        )
        q_target_selected = q_target_full[self.selected_dof_indices].copy()

        if self.use_unitree_sdk:
            cmd_dq = np.zeros(self.robot.NUM_JOINTS, dtype=np.float32)
            cmd_tau = np.zeros(self.robot.NUM_JOINTS, dtype=np.float32)
            self.command_sender.send_command(q_target_full, cmd_dq, cmd_tau)
        else:
            self._apply_local_pd_target(q_target_full)
        self.last_action = clipped_action.copy()

        frame = self.simulator.get_frame_data(self.selected_dof_indices)
        frame["action"] = clipped_action.copy()
        frame["pose_aa"] = self.build_pose_aa(frame["root_rot"], frame["dof"])
        frame["fps"] = np.int32(50)
        sim_diag = self.simulator.get_diagnostics(self.selected_dof_indices)
        diag = {
            "actor_obs": actor_obs.reshape(-1).copy(),
            "onnx_action_raw": raw_action.copy(),
            "clipped_action": clipped_action.copy(),
            "q_target": q_target_selected.astype(np.float32, copy=True),
            "dof_q": sim_diag["dof_q"],
            "dof_vel": sim_diag["dof_vel"],
            "tau": sim_diag["tau"],
            "root_height": float(sim_diag["root_height"]),
            "qacc_max_abs": float(sim_diag["qacc_max_abs"]),
            "qacc_nonfinite": bool(sim_diag["qacc_nonfinite"]),
        }
        return actor_obs, frame, diag

    def build_pose_aa(self, root_rot_xyzw: np.ndarray, dof: np.ndarray) -> np.ndarray:
        root_rot_vec = sRot.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
        extend_pose = np.zeros((len(EXTEND_OFFSETS), 3), dtype=np.float32)
        return np.concatenate(
            [root_rot_vec[None, :], self.selected_dof_axis * dof[:, None], extend_pose],
            axis=0,
        )

    def _apply_local_pd_target(self, q_target_full: np.ndarray):
        kp = np.asarray(self.sim_config["MOTOR_KP"], dtype=np.float32)
        kd = np.asarray(self.sim_config["MOTOR_KD"], dtype=np.float32)
        tau_limit = self.simulator.effective_torque_limit
        with self.simulator.mj_lock:
            qpos_all = self.simulator.mj_data.qpos[7 : 7 + self.robot.NUM_JOINTS].copy()
            qvel_all = self.simulator.mj_data.qvel[6 : 6 + self.robot.NUM_JOINTS].copy()
            torques = kp * (q_target_full - qpos_all) - kd * qvel_all
            torques = np.clip(torques, -tau_limit, tau_limit)
            self.simulator.torques[:] = torques

    def maybe_log_step_diagnostics(self, step_idx: int, diag: Dict[str, Any], log_every: int):
        if log_every <= 0:
            return
        if step_idx % log_every != 0:
            return
        self.node.get_logger().info(
            f"step={step_idx} "
            f"{_stats_text('actor_obs', diag['actor_obs'])} "
            f"{_stats_text('onnx_raw', diag['onnx_action_raw'])} "
            f"{_stats_text('clipped_action', diag['clipped_action'])} "
            f"{_stats_text('q_target', diag['q_target'])} "
            f"{_stats_text('dof_q', diag['dof_q'])} "
            f"{_stats_text('dof_vel', diag['dof_vel'])} "
            f"{_stats_text('tau', diag['tau'])} "
            f"root_height={diag['root_height']:.6f} "
            f"qacc_max_abs={diag['qacc_max_abs']:.6f}"
        )

    def update_first_trigger_steps(
        self,
        step_idx: int,
        diag: Dict[str, Any],
        qacc_warn_threshold: float,
        torque_limit_margin: float,
    ):
        action_absmax = float(np.max(np.abs(diag["clipped_action"])))
        tau_absmax = float(np.max(np.abs(diag["tau"])))
        tau_limit = self.simulator.effective_torque_limit
        tau_limit_max = float(np.max(tau_limit)) if tau_limit.size > 0 else 0.0
        tau_ratio = float(np.max(np.abs(diag["tau"]) / np.maximum(tau_limit, 1e-6))) if tau_limit.size > 0 else 0.0
        qacc_issue = bool(diag["qacc_nonfinite"]) or float(diag["qacc_max_abs"]) > qacc_warn_threshold

        for threshold in self.action_abs_thresholds:
            key = f"action_absmax_gt_{int(threshold)}"
            if key not in self.first_trigger_steps and action_absmax > threshold:
                self.first_trigger_steps[key] = step_idx
                self.node.get_logger().warning(
                    f"first_trigger {key} step={step_idx} value={action_absmax:.6f}"
                )

        key = "tau_absmax_gt_90pct_limit"
        if key not in self.first_trigger_steps and tau_ratio > torque_limit_margin:
            self.first_trigger_steps[key] = step_idx
            self.node.get_logger().warning(
                f"first_trigger {key} step={step_idx} "
                f"tau_absmax={tau_absmax:.6f} tau_limit_max={tau_limit_max:.6f} tau_ratio={tau_ratio:.6f}"
            )

        key = "qacc_warn_or_threshold"
        if key not in self.first_trigger_steps and qacc_issue:
            self.first_trigger_steps[key] = step_idx
            self.node.get_logger().warning(
                f"first_trigger {key} step={step_idx} "
                f"qacc_max_abs={diag['qacc_max_abs']:.6f} qacc_nonfinite={diag['qacc_nonfinite']}"
            )

    def should_truncate_rollout(
        self,
        diag: Dict[str, Any],
        qacc_stop_threshold: float,
        action_stop_threshold: float,
    ) -> Tuple[bool, Optional[str]]:
        action_absmax = float(np.max(np.abs(diag["clipped_action"])))
        if action_absmax > action_stop_threshold:
            return True, f"action_absmax={action_absmax:.6f}>{action_stop_threshold:.6f}"
        if bool(diag["qacc_nonfinite"]):
            return True, "qacc_nonfinite=True"
        if float(diag["qacc_max_abs"]) > qacc_stop_threshold:
            return True, f"qacc_max_abs={diag['qacc_max_abs']:.6f}>{qacc_stop_threshold:.6f}"
        return False, None

    def log_first_trigger_summary(self):
        for key in [
            "action_absmax_gt_10",
            "action_absmax_gt_20",
            "action_absmax_gt_50",
            "action_absmax_gt_90",
            "tau_absmax_gt_90pct_limit",
            "qacc_warn_or_threshold",
        ]:
            step = self.first_trigger_steps.get(key, None)
            self.node.get_logger().info(f"first_step[{key}]={step}")

    def build_output_motion(self, rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
        output_motion = dict(self.template_motion)
        num_steps = int(rollout["action"].shape[0])
        output_motion["root_trans_offset"] = rollout["root_trans_offset"]
        output_motion["root_rot"] = rollout["root_rot"]
        output_motion["dof"] = rollout["dof"]
        output_motion["dof_vel"] = rollout["dof_vel"]
        output_motion["root_lin_vel"] = rollout["root_lin_vel"]
        output_motion["root_ang_vel"] = rollout["root_ang_vel"]
        output_motion["action"] = rollout["action"]
        output_motion["pose_aa"] = rollout["pose_aa"]
        output_motion["body_pos"] = rollout["body_pos"]
        output_motion["body_pos_w"] = rollout["body_pos"]
        output_motion["body_quat"] = rollout["body_quat"]
        output_motion["body_quat_w"] = rollout["body_quat"]
        output_motion["body_lin_vel"] = rollout["body_lin_vel"]
        output_motion["body_ang_vel"] = rollout["body_ang_vel"]
        output_motion["fps"] = int(50)
        output_motion["dt"] = float(1.0 / 50.0)
        output_motion["motion_length"] = num_steps
        return output_motion


def _stack_rollout_buffer(buffer: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    stacked = {}
    for key, value in buffer.items():
        if key == "fps":
            stacked[key] = np.asarray(value, dtype=np.int32)
        else:
            stacked[key] = np.stack(value, axis=0)
    return stacked


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
    parser = argparse.ArgumentParser(description="CR7 level2 single-actor MuJoCo collector")
    parser.add_argument("--config", type=str, default="sim2real/config/g1_29dof_hist.yaml", help="sim2real config")
    parser.add_argument("--num_steps", type=int, default=134, help="number of action steps to collect")
    parser.add_argument("--wait_for_state_timeout_s", type=float, default=10.0, help="timeout for first lowstate")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="output motion path")
    parser.add_argument("--init_from_motion_frame", type=int, default=0, help="initialize MuJoCo state from source motion frame")
    parser.add_argument("--log_every", type=int, default=5, help="print low-frequency diagnostics every N steps")
    parser.add_argument("--qacc_warn_threshold", type=float, default=1e3, help="warn threshold for |qacc| max")
    parser.add_argument("--qacc_stop_threshold", type=float, default=1e4, help="stop threshold for |qacc| max")
    parser.add_argument("--action_stop_threshold", type=float, default=50.0, help="stop when |action| max exceeds this")
    args = parser.parse_args()

    if not Path(POLICY_PATH).is_file():
        raise FileNotFoundError(f"policy_path not found: {POLICY_PATH}")
    if not Path(TRAIN_CONFIG_PATH).is_file():
        raise FileNotFoundError(f"train_config not found: {TRAIN_CONFIG_PATH}")
    if not Path(SOURCE_MOTION_PATH).is_file():
        raise FileNotFoundError(f"source_motion_path not found: {SOURCE_MOTION_PATH}")
    if not Path(TEMPLATE_MOTION_PATH).is_file():
        raise FileNotFoundError(f"template_motion_path not found: {TEMPLATE_MOTION_PATH}")

    sim_config = _load_sim_config(args.config)

    if HAS_UNITREE_SDK:
        if sim_config.get("INTERFACE", None):
            ChannelFactoryInitialize(sim_config["DOMAIN_ID"], sim_config["INTERFACE"])
        else:
            ChannelFactoryInitialize(sim_config["DOMAIN_ID"])

    if rclpy is not None:
        rclpy.init(args=None)
        node = rclpy.create_node("cr7_level2_single_actor_collect")
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
    else:
        node = _FallbackNode()

    simulator = LockedBaseSimulator(sim_config, node)
    simulator.sim_thread.start()

    collector = SingleActorCollector(
        sim_config=sim_config,
        node=node,
        simulator=simulator,
        train_cfg_path=TRAIN_CONFIG_PATH,
        policy_path=POLICY_PATH,
        source_motion_path=SOURCE_MOTION_PATH,
        template_motion_path=TEMPLATE_MOTION_PATH,
        output_path=args.output_path,
        num_steps=args.num_steps,
    )
    collector.initialize_sim_from_motion(args.init_from_motion_frame)
    collector.wait_for_first_low_state(args.wait_for_state_timeout_s)

    rollout_buffer = {
        "root_trans_offset": [],
        "root_rot": [],
        "dof": [],
        "dof_vel": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "action": [],
        "pose_aa": [],
        "fps": [],
        "body_pos": [],
        "body_quat": [],
        "body_lin_vel": [],
        "body_ang_vel": [],
    }

    truncated = False
    truncation_reason = None
    try:
        for step in range(args.num_steps):
            _, frame, diag = collector.step_policy(step)
            for key, value in frame.items():
                rollout_buffer[key].append(value)
            collector.maybe_log_step_diagnostics(step, diag, args.log_every)
            collector.update_first_trigger_steps(
                step_idx=step,
                diag=diag,
                qacc_warn_threshold=args.qacc_warn_threshold,
                torque_limit_margin=0.9,
            )
            truncated, truncation_reason = collector.should_truncate_rollout(
                diag=diag,
                qacc_stop_threshold=args.qacc_stop_threshold,
                action_stop_threshold=args.action_stop_threshold,
            )
            if truncated:
                logger.warning(
                    f"truncate rollout at step={step} collected_frames={len(rollout_buffer['action'])} "
                    f"reason={truncation_reason}"
                )
                break
            collector.rate.sleep()
    finally:
        simulator.close()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    rollout = _stack_rollout_buffer(rollout_buffer)
    output_motion = collector.build_output_motion(rollout)
    output_data = {collector.template_key: output_motion}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_pickle(output_data, str(output_path))
    if truncated:
        logger.warning(
            f"saved truncated rollout pkl path={output_path} frames={int(output_motion['motion_length'])} "
            f"reason={truncation_reason}"
        )
    collector.log_first_trigger_summary()
    _print_summary(str(output_path), collector.template_key, output_motion)


if __name__ == "__main__":
    main()

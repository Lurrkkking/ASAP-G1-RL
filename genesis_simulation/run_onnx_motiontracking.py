import os
import sys
from pathlib import Path

import genesis as gs
import numpy as np
import onnxruntime as ort
import torch

sys.path.append("/root/autodl-tmp/ASAP")
from humanoidverse.utils.torch_utils import quat_rotate_inverse


URDF_PATH = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf"
ONNX_PATH = os.environ.get(
    "ONNX_PATH",
    "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260409_165618-MotionTracking_CR7_FullSystem_V2_Fresh_Resume-motion_tracking-g1_29dof_anneal_23dof/exported/model_8600.onnx",
)
OUTPUT_VIDEO = os.environ.get("OUT_VIDEO", "g1_siuuu_genesis.mp4")

TRAIN_CONFIG_PATH_ENV = os.environ.get("TRAIN_CONFIG_PATH", "")
USE_TRAIN_CONFIG = os.environ.get("USE_TRAIN_CONFIG", "1").lower() not in ("0", "false", "no")

SCALES = {
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "actions": 1.0,
    "ref_motion_phase": 1.0,
}

OBS_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "dof_pos": 23,
    "dof_vel": 23,
    "actions": 23,
    "ref_motion_phase": 1,
}

# Match config.obs.obs_auxiliary.history_actor for motion_tracking.
HISTORY_CFG = {
    "base_ang_vel": 4,
    "projected_gravity": 4,
    "dof_pos": 4,
    "dof_vel": 4,
    "actions": 4,
    "ref_motion_phase": 4,
}

# Training concatenates obs keys using sorted(obs_config).
ACTOR_OBS_ORDER = sorted([
    "base_ang_vel",
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "actions",
    "ref_motion_phase",
    "history_actor",
])
HISTORY_KEY_ORDER = sorted(HISTORY_CFG.keys())

ACTION_SCALE = float(os.environ.get("ACTION_SCALE", "0.25"))
ACTION_CLIP_VALUE = float(os.environ.get("ACTION_CLIP_VALUE", "100.0"))
ACTION_FILTER_ALPHA = float(os.environ.get("ACTION_FILTER_ALPHA", "1.0"))
CONTROL_DECIMATION = int(os.environ.get("CONTROL_DECIMATION", "4"))
SIM_FPS = float(os.environ.get("SIM_FPS", "200"))
SIM_DT = 1.0 / SIM_FPS
SIM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Default uses motion-lib reported length for CR7 clip (4.4333s).
MOTION_DURATION = float(os.environ.get("MOTION_DURATION", "4.4333333333"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "600"))
KD_SCALE = float(os.environ.get("KD_SCALE", "1.0"))
TIME_OFFSET_ENV = os.environ.get("TIME_OFFSET")
TIME_OFFSET = float(TIME_OFFSET_ENV) if TIME_OFFSET_ENV is not None else 0.0
FLOOR_FRICTION = float(os.environ.get("FLOOR_FRICTION", "1.0"))
USE_MOTION_INIT = os.environ.get("USE_MOTION_INIT", "0").lower() not in ("0", "false", "no")
INIT_STATE_NPZ = Path(os.environ.get("INIT_STATE_NPZ", "/tmp/g1_init_t1p5.npz"))
PHASE_WRAP = os.environ.get("PHASE_WRAP", "0").lower() in ("1", "true", "yes")
STOP_AT_MOTION_END = os.environ.get("STOP_AT_MOTION_END", "1").lower() not in ("0", "false", "no")
USE_IMPLICIT_PD = os.environ.get("USE_IMPLICIT_PD", "0").lower() not in ("0", "false", "no")
USE_SELF_COLLISION = os.environ.get("USE_SELF_COLLISION", "1").lower() not in ("0", "false", "no")
DIAG_OBS = os.environ.get("DIAG_OBS", "0").lower() not in ("0", "false", "no")
DIAG_WARMUP_STEPS = int(os.environ.get("DIAG_WARMUP_STEPS", "20"))
DIAG_LOG_EVERY = int(os.environ.get("DIAG_LOG_EVERY", "50"))
DIAG_OUT_NPZ = os.environ.get("DIAG_OUT_NPZ", "/tmp/actor_obs_diag_genesis.npz")
NO_RECORD = os.environ.get("NO_RECORD", "0").lower() in ("1", "true", "yes")
RECORD_FPS_ENV = os.environ.get("RECORD_FPS", "").strip()
TRACE_OUT_NPZ = os.environ.get("TRACE_OUT_NPZ", "")
METRICS_OUT_NPZ = os.environ.get("METRICS_OUT_NPZ", "")
FALL_ROOT_Z = float(os.environ.get("FALL_ROOT_Z", "0.45"))
STOP_ON_FALL = os.environ.get("STOP_ON_FALL", "1").lower() not in ("0", "false", "no")
TORQUE_DIAG_SECONDS = float(os.environ.get("TORQUE_DIAG_SECONDS", "2.0"))
TORQUE_DIAG_OUT_NPZ = os.environ.get("TORQUE_DIAG_OUT_NPZ", "")
SWEEP_TAG = os.environ.get("SWEEP_TAG", "")

RL_JOINT_NAMES = [
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

# Keep only training-side body set, same as locomotion Genesis runner.
BODY_NAMES = [
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

DEFAULT_DOF_POS = np.array(
    [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

KP = np.array(
    [
        100,
        100,
        100,
        200,
        20,
        20,
        100,
        100,
        100,
        200,
        20,
        20,
        400,
        400,
        400,
        90,
        60,
        20,
        60,
        90,
        60,
        20,
        60,
    ],
    dtype=np.float32,
)

KD = np.array(
    [
        2.5,
        2.5,
        2.5,
        5.0,
        0.2,
        0.1,
        2.5,
        2.5,
        2.5,
        5.0,
        0.2,
        0.1,
        5.0,
        5.0,
        5.0,
        2.0,
        1.0,
        0.4,
        1.0,
        2.0,
        1.0,
        0.4,
        1.0,
    ],
    dtype=np.float32,
)
KD = KD * KD_SCALE

TORQUE_LIMITS = np.array(
    [
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
    ],
    dtype=np.float32,
)


def _env_has(name):
    return name in os.environ


def _load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_train_config_path():
    if TRAIN_CONFIG_PATH_ENV:
        p = Path(TRAIN_CONFIG_PATH_ENV)
        return p if p.is_file() else None

    onnx = Path(ONNX_PATH)
    candidates = [
        onnx.parent.parent / "config.yaml",
        onnx.parent.parent / ".hydra" / ".hydra" / "config.yaml",
        onnx.parent.parent / ".hydra" / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_motion_duration_from_file(motion_file):
    mf = Path(motion_file)
    if not mf.is_absolute():
        mf = Path('/root/autodl-tmp/ASAP') / mf
    if not mf.is_file():
        return None

    try:
        import joblib

        data = joblib.load(mf)
    except Exception:
        return None

    entry = data
    if isinstance(data, dict) and len(data) > 0:
        first_key = next(iter(data.keys()))
        entry = data[first_key]

    if not isinstance(entry, dict):
        return None

    fps = entry.get("fps", None)
    if fps is None:
        return None

    frames = None
    for key in ("dof", "root_trans_offset", "pose_aa", "root_rot"):
        arr = entry.get(key, None)
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            frames = int(arr.shape[0])
            break

    if frames is None or frames < 2:
        return None

    return float((frames - 1) / float(fps))


def _match_control_key(joint_name, table):
    matches = [k for k in table.keys() if k in joint_name]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def apply_training_config_overrides():
    global URDF_PATH
    global SCALES, OBS_DIMS, HISTORY_CFG, HISTORY_KEY_ORDER, ACTOR_OBS_ORDER
    global ACTION_SCALE, ACTION_CLIP_VALUE, CONTROL_DECIMATION, SIM_FPS, SIM_DT
    global MOTION_DURATION, FLOOR_FRICTION, USE_SELF_COLLISION
    global RL_JOINT_NAMES, BODY_NAMES, DEFAULT_DOF_POS, KP, KD, TORQUE_LIMITS

    if not USE_TRAIN_CONFIG:
        print('[INFO] USE_TRAIN_CONFIG disabled; keep hardcoded runner defaults')
        return

    cfg_path = _resolve_train_config_path()
    if cfg_path is None:
        print('[WARN] Could not resolve training config.yaml near ONNX; keep hardcoded defaults')
        return

    try:
        cfg = _load_yaml(cfg_path)
    except Exception as exc:
        print(f'[WARN] Failed to parse training config {cfg_path}: {exc}')
        return

    robot = ((cfg or {}).get('robot') or {})
    env_cfg = (((cfg or {}).get('env') or {}).get('config') or {})
    obs_cfg = (env_cfg.get('obs') or {})
    if not isinstance(obs_cfg, dict):
        obs_cfg = (cfg.get('obs') or {})

    dof_names = robot.get('dof_names') or []
    body_names = robot.get('body_names') or []
    if dof_names:
        RL_JOINT_NAMES = list(dof_names)
    if body_names:
        BODY_NAMES = list(body_names)

    asset = robot.get('asset') or {}
    asset_root = asset.get('asset_root')
    urdf_file = asset.get('urdf_file')
    if isinstance(urdf_file, str) and ('${robot.asset.robot_type}' in urdf_file):
        robot_type = str(asset.get('robot_type', ''))
        urdf_file = urdf_file.replace('${robot.asset.robot_type}', robot_type)
    if (not _env_has('URDF_PATH')) and asset_root and urdf_file:
        URDF_PATH = str((Path('/root/autodl-tmp/ASAP') / asset_root / urdf_file).resolve())

    default_joint_angles = ((robot.get('init_state') or {}).get('default_joint_angles') or {})
    if default_joint_angles and RL_JOINT_NAMES:
        DEFAULT_DOF_POS = np.array([float(default_joint_angles.get(n, 0.0)) for n in RL_JOINT_NAMES], dtype=np.float32)

    control = robot.get('control') or {}
    if (not _env_has('ACTION_SCALE')) and ('action_scale' in control):
        ACTION_SCALE = float(control['action_scale'])
    if (not _env_has('ACTION_CLIP_VALUE')) and ('action_clip_value' in control):
        ACTION_CLIP_VALUE = float(control['action_clip_value'])

    stiffness = control.get('stiffness') or {}
    damping = control.get('damping') or {}
    if stiffness and damping and RL_JOINT_NAMES:
        kp = []
        kd = []
        for j in RL_JOINT_NAMES:
            k1 = _match_control_key(j, stiffness)
            k2 = _match_control_key(j, damping)
            kp.append(float(stiffness[k1]) if k1 else 0.0)
            kd.append(float(damping[k2]) if k2 else 0.0)
        KP = np.asarray(kp, dtype=np.float32)
        KD = np.asarray(kd, dtype=np.float32)

    dof_effort = robot.get('dof_effort_limit_list') or []
    if dof_effort and len(dof_effort) == len(RL_JOINT_NAMES):
        TORQUE_LIMITS = np.asarray(dof_effort, dtype=np.float32)

    sim = (((cfg.get('simulator') or {}).get('config') or {}).get('sim') or {})
    if (not _env_has('SIM_FPS')) and ('fps' in sim):
        SIM_FPS = float(sim['fps'])
        SIM_DT = 1.0 / SIM_FPS
    if (not _env_has('CONTROL_DECIMATION')) and ('control_decimation' in sim):
        CONTROL_DECIMATION = int(sim['control_decimation'])

    plane = (((cfg.get('simulator') or {}).get('config') or {}).get('plane') or {})
    if (not _env_has('FLOOR_FRICTION')) and ('static_friction' in plane):
        FLOOR_FRICTION = float(plane['static_friction'])

    obs_scales = obs_cfg.get('obs_scales') or {}
    for k in SCALES.keys():
        if k in obs_scales:
            SCALES[k] = float(obs_scales[k])

    obs_dims_list = obs_cfg.get('obs_dims') or []
    if isinstance(obs_dims_list, list):
        dim_map = {}
        for it in obs_dims_list:
            if isinstance(it, dict):
                dim_map.update(it)
        for k in OBS_DIMS.keys():
            v = dim_map.get(k, None)
            if isinstance(v, int):
                OBS_DIMS[k] = int(v)

    obs_aux = obs_cfg.get('obs_auxiliary') or {}
    hist_actor = obs_aux.get('history_actor') or {}
    if hist_actor:
        HISTORY_CFG = {k: int(v) for k, v in hist_actor.items()}
        HISTORY_KEY_ORDER = sorted(HISTORY_CFG.keys())

    obs_dict = obs_cfg.get('obs_dict') or {}
    actor_obs = obs_dict.get('actor_obs') or []
    if actor_obs:
        ACTOR_OBS_ORDER = sorted(list(actor_obs))

    if not _env_has('MOTION_DURATION'):
        motion_file = ((robot.get('motion') or {}).get('motion_file') or '')
        duration = _load_motion_duration_from_file(motion_file)
        if duration is not None:
            MOTION_DURATION = float(duration)

    if not _env_has('USE_SELF_COLLISION'):
        has_self_collision = (robot.get('motion') or {}).get('has_self_collision', None)
        if has_self_collision is not None:
            USE_SELF_COLLISION = bool(has_self_collision)

    KD = KD * KD_SCALE

    print(
        f"[INFO] Loaded runner defaults from train config: {cfg_path}\n"
        f"       joints={len(RL_JOINT_NAMES)} bodies={len(BODY_NAMES)} fps={SIM_FPS} decim={CONTROL_DECIMATION}\n"
        f"       action_scale={ACTION_SCALE} action_clip={ACTION_CLIP_VALUE} floor_friction={FLOOR_FRICTION}\n"
        f"       motion_duration={MOTION_DURATION:.6f} self_collision={int(USE_SELF_COLLISION)}"
    )


apply_training_config_overrides()


def maybe_autofill_time_offset():
    """If TIME_OFFSET is not set explicitly, align it to init snapshot time_offset."""
    global TIME_OFFSET
    if TIME_OFFSET_ENV is not None or (not USE_MOTION_INIT) or (not INIT_STATE_NPZ.is_file()):
        return
    try:
        data = np.load(INIT_STATE_NPZ)
        if "time_offset" in data:
            TIME_OFFSET = float(np.array(data["time_offset"]).reshape(-1)[0])
            print(f"[INFO] Auto TIME_OFFSET={TIME_OFFSET:.4f} from {INIT_STATE_NPZ}")
    except Exception as exc:
        print(f"[WARN] Failed to parse time_offset from {INIT_STATE_NPZ}: {exc}")


def resolve_record_fps():
    if RECORD_FPS_ENV:
        try:
            fps = int(RECORD_FPS_ENV)
            if fps > 0:
                return fps
        except Exception:
            pass
        print(f"[WARN] Invalid RECORD_FPS='{RECORD_FPS_ENV}', fallback to control-rate fps")

    control_hz = 1.0 / max(SIM_DT * CONTROL_DECIMATION, 1e-8)
    return max(1, int(round(control_hz)))


def as_tensor_2d(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=SIM_DEVICE)
    else:
        x = x.to(device=SIM_DEVICE, dtype=torch.float32)
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def to_numpy_1d(x):
    x = as_tensor_2d(x)
    return x[0].detach().cpu().numpy().astype(np.float32)


def try_apply_reference_init(robot, motor_dofs):
    if not USE_MOTION_INIT:
        print("[INFO] USE_MOTION_INIT disabled; use standing init")
        return False
    if not INIT_STATE_NPZ.is_file():
        print(f"[WARN] INIT_STATE_NPZ not found: {INIT_STATE_NPZ}; use standing init")
        return False

    data = np.load(INIT_STATE_NPZ)
    has_root_quat = ("root_quat" in data) or ("root_rot_xyzw" in data)
    required = ["root_pos", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]
    missing = [k for k in required if k not in data]
    if not has_root_quat:
        missing.append("root_quat/root_rot_xyzw")
    if missing:
        print(f"[WARN] INIT_STATE_NPZ missing keys {missing}; use standing init")
        return False

    root_pos = torch.tensor(data["root_pos"], dtype=torch.float32, device=SIM_DEVICE)
    root_rot_xyzw = data["root_quat"].astype(np.float32) if "root_quat" in data else data["root_rot_xyzw"].astype(np.float32)
    root_rot_wxyz = torch.tensor(root_rot_xyzw[[3, 0, 1, 2]], dtype=torch.float32, device=SIM_DEVICE)
    root_vel = torch.tensor(data["root_vel"], dtype=torch.float32, device=SIM_DEVICE)
    root_ang_vel = torch.tensor(data["root_ang_vel"], dtype=torch.float32, device=SIM_DEVICE)
    dof_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=SIM_DEVICE)
    dof_vel = torch.tensor(data["dof_vel"], dtype=torch.float32, device=SIM_DEVICE)

    try:
        robot.set_pos(root_pos, zero_velocity=False)
        robot.set_quat(root_rot_wxyz, zero_velocity=False)
    except TypeError:
        robot.set_pos(root_pos)
        robot.set_quat(root_rot_wxyz)

    robot.set_dofs_velocity(velocity=root_vel, dofs_idx_local=[0, 1, 2])
    robot.set_dofs_velocity(velocity=root_ang_vel, dofs_idx_local=[3, 4, 5])
    robot.set_dofs_position(position=dof_pos, dofs_idx_local=motor_dofs)
    robot.set_dofs_velocity(velocity=dof_vel, dofs_idx_local=motor_dofs)

    if "time_offset" in data:
        init_t = float(np.array(data["time_offset"]).reshape(-1)[0])
        if abs(init_t - TIME_OFFSET) > 1e-4:
            print(f"[WARN] TIME_OFFSET ({TIME_OFFSET:.4f}) != INIT_STATE_NPZ.time_offset ({init_t:.4f})")

    print(f"[INFO] Applied reference init from {INIT_STATE_NPZ}")
    return True


def make_history_buffers():
    return {
        key: np.zeros((HISTORY_CFG[key], OBS_DIMS[key]), dtype=np.float32)
        for key in HISTORY_CFG
    }


def query_history_actor(history_buffers):
    chunks = []
    for key in HISTORY_KEY_ORDER:
        chunks.append(history_buffers[key].reshape(-1))
    history_actor = np.concatenate(chunks, axis=0).astype(np.float32)
    return history_actor


def update_history_buffers(history_buffers, curr_features):
    # Match HistoryHandler.add: shift old -> index+1, write newest at index 0.
    for key in HISTORY_CFG:
        history_buffers[key][1:] = history_buffers[key][:-1]
        history_buffers[key][0] = curr_features[key]


def build_curr_features(dof_pos, dof_vel, base_ang_vel_body, projected_gravity, last_actions, phase):
    return {
        "actions": (last_actions * SCALES["actions"]).astype(np.float32),
        "base_ang_vel": (base_ang_vel_body * SCALES["base_ang_vel"]).astype(np.float32),
        "dof_pos": ((dof_pos - DEFAULT_DOF_POS) * SCALES["dof_pos"]).astype(np.float32),
        "dof_vel": (dof_vel * SCALES["dof_vel"]).astype(np.float32),
        "projected_gravity": (projected_gravity * SCALES["projected_gravity"]).astype(np.float32),
        "ref_motion_phase": np.array([phase * SCALES["ref_motion_phase"]], dtype=np.float32),
    }


def build_actor_obs(curr_features, history_actor, return_parts=False):
    parts = {
        "actions": curr_features["actions"],
        "base_ang_vel": curr_features["base_ang_vel"],
        "dof_pos": curr_features["dof_pos"],
        "dof_vel": curr_features["dof_vel"],
        "history_actor": history_actor,
        "projected_gravity": curr_features["projected_gravity"],
        "ref_motion_phase": curr_features["ref_motion_phase"],
    }
    obs = np.concatenate([parts[k] for k in ACTOR_OBS_ORDER], axis=0).astype(np.float32)
    if obs.shape[0] != 380:
        raise RuntimeError(f"actor_obs dim mismatch: got {obs.shape[0]}, expected 380")
    if return_parts:
        return obs, parts
    return obs


def init_diag_stats():
    dims = {
        "actor_obs": 380,
        "actions": OBS_DIMS["actions"],
        "base_ang_vel": OBS_DIMS["base_ang_vel"],
        "dof_pos": OBS_DIMS["dof_pos"],
        "dof_vel": OBS_DIMS["dof_vel"],
        "history_actor": sum(HISTORY_CFG[k] * OBS_DIMS[k] for k in HISTORY_CFG),
        "projected_gravity": OBS_DIMS["projected_gravity"],
        "ref_motion_phase": OBS_DIMS["ref_motion_phase"],
    }
    stats = {}
    for key, dim in dims.items():
        stats[key] = {
            "sum": np.zeros(dim, dtype=np.float64),
            "sum2": np.zeros(dim, dtype=np.float64),
            "min": np.full(dim, np.inf, dtype=np.float64),
            "max": np.full(dim, -np.inf, dtype=np.float64),
            "count": 0,
        }
    return stats


def update_diag_stats(diag_stats, actor_obs, obs_parts):
    samples = {"actor_obs": actor_obs}
    samples.update(obs_parts)
    for key, vec in samples.items():
        x = np.asarray(vec, dtype=np.float64).reshape(-1)
        st = diag_stats[key]
        st["sum"] += x
        st["sum2"] += x * x
        st["min"] = np.minimum(st["min"], x)
        st["max"] = np.maximum(st["max"], x)
        st["count"] += 1


def finalize_diag_stats(diag_stats, out_npz):
    out = {}
    for key, st in diag_stats.items():
        c = max(int(st["count"]), 1)
        mean = st["sum"] / c
        var = np.maximum(st["sum2"] / c - mean * mean, 0.0)
        std = np.sqrt(var)
        out[f"{key}_mean"] = mean.astype(np.float32)
        out[f"{key}_std"] = std.astype(np.float32)
        out[f"{key}_min"] = st["min"].astype(np.float32)
        out[f"{key}_max"] = st["max"].astype(np.float32)
        out[f"{key}_count"] = np.array([st["count"]], dtype=np.int32)
    np.savez(out_npz, **out)
    print(f"[DIAG] saved actor_obs stats to {out_npz}")


def main():
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not os.path.isfile(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    maybe_autofill_time_offset()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(enable_self_collision=USE_SELF_COLLISION),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=FLOOR_FRICTION))

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=URDF_PATH,
            merge_fixed_links=True,
            links_to_keep=BODY_NAMES,
            pos=(0.0, 0.0, 0.8),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )

    cam = None
    if not NO_RECORD:
        cam = scene.add_camera(
            res=(640, 480),
            pos=(2.0, 2.0, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=60,
            GUI=False,
        )
    scene.build()

    motor_dofs = [robot.get_joint(name).dof_idx_local for name in RL_JOINT_NAMES]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX_PATH={ONNX_PATH}")
    print(f"[INFO] ONNX input={session.get_inputs()[0].shape}, output={session.get_outputs()[0].shape}")
    control_hz = 1.0 / (SIM_DT * CONTROL_DECIMATION)
    torque_diag_steps = int(np.ceil(max(TORQUE_DIAG_SECONDS, 0.0) / SIM_DT))
    record_fps = resolve_record_fps()
    print(
        f"[INFO] MOTION_DURATION={MOTION_DURATION:.4f}, TIME_OFFSET={TIME_OFFSET:.4f}, KD_SCALE={KD_SCALE:.3f}, "
        f"FLOOR_FRICTION={FLOOR_FRICTION:.3f}, ACTION_FILTER_ALPHA={ACTION_FILTER_ALPHA:.2f}, "
        f"PHASE_WRAP={int(PHASE_WRAP)}, STOP_AT_MOTION_END={int(STOP_AT_MOTION_END)}, "
        f"USE_IMPLICIT_PD={int(USE_IMPLICIT_PD)}, USE_SELF_COLLISION={int(USE_SELF_COLLISION)}, "
        f"USE_MOTION_INIT={int(USE_MOTION_INIT)}, INIT_STATE_NPZ={INIT_STATE_NPZ}, NO_RECORD={int(NO_RECORD)}"
    )
    print(f"[INFO] Timing: sim_dt={SIM_DT:.6f}s decim={CONTROL_DECIMATION} control_hz={control_hz:.2f} record_fps={record_fps}")
    if torque_diag_steps > 0:
        print(f"[INFO] TorqueDiag: window={TORQUE_DIAG_SECONDS:.3f}s ({torque_diag_steps} physics steps), out_npz={TORQUE_DIAG_OUT_NPZ if TORQUE_DIAG_OUT_NPZ else '<disabled>'}")
    if USE_IMPLICIT_PD and torque_diag_steps > 0:
        print("[WARN] TorqueDiag requires explicit torque-PD path (USE_IMPLICIT_PD=0); detailed torque recording disabled")

    applied_ref_init = try_apply_reference_init(robot, motor_dofs)
    if not applied_ref_init:
        robot.set_dofs_position(position=torch.tensor(DEFAULT_DOF_POS, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)
        robot.set_dofs_velocity(velocity=torch.zeros(23, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)

    if USE_IMPLICIT_PD:
        robot.set_dofs_kp(kp=torch.tensor(KP, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)
        robot.set_dofs_kv(kv=torch.tensor(KD, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)

    gravity_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=SIM_DEVICE)
    last_actions = np.zeros(23, dtype=np.float32)
    history_buffers = make_history_buffers()
    diag_stats = init_diag_stats() if DIAG_OBS else None
    root_z_hist = []
    action_norm_hist = []
    phase_hist = []
    trace_actor_obs = []
    trace_action = []
    fell = False
    stop_reason = "num_steps_reached"

    hip_joint_ids = np.array([i for i, n in enumerate(RL_JOINT_NAMES) if "hip" in n], dtype=np.int32)
    knee_joint_ids = np.array([i for i, n in enumerate(RL_JOINT_NAMES) if "knee" in n], dtype=np.int32)
    torque_diag_req = []
    torque_diag_actual = []
    torque_diag_clipped = []
    torque_diag_step_idx = []

    if cam is not None:
        cam.start_recording()
    print("[INFO] Start motion-tracking ONNX rollout")

    for step in range(NUM_STEPS):
        dof_pos = to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
        dof_vel = to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))

        quat_wxyz = as_tensor_2d(robot.get_quat())
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        base_ang_vel_world = as_tensor_2d(robot.get_ang())

        base_ang_vel_body = to_numpy_1d(quat_rotate_inverse(quat_xyzw, base_ang_vel_world))
        projected_gravity = to_numpy_1d(quat_rotate_inverse(quat_xyzw, gravity_world))

        # Phase progression: default is non-wrapping single-motion playback.
        t = TIME_OFFSET + (step + 1) * (SIM_DT * CONTROL_DECIMATION)
        if PHASE_WRAP:
            phase = np.float32((t % MOTION_DURATION) / MOTION_DURATION)
        else:
            if STOP_AT_MOTION_END and t >= MOTION_DURATION:
                print(f"[INFO] Reached motion end at step={step}, t={t:.3f}s; stop rollout")
                stop_reason = "motion_end"
                break
            phase_time = min(t, MOTION_DURATION - 1e-6)
            phase = np.float32(phase_time / MOTION_DURATION)

        curr_features = build_curr_features(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            base_ang_vel_body=base_ang_vel_body,
            projected_gravity=projected_gravity,
            last_actions=last_actions,
            phase=phase,
        )

        # Match training timing: actor_obs uses history from previous steps (not including current step).
        history_actor = query_history_actor(history_buffers)
        if DIAG_OBS:
            actor_obs, obs_parts = build_actor_obs(curr_features, history_actor, return_parts=True)
            if step >= DIAG_WARMUP_STEPS:
                update_diag_stats(diag_stats, actor_obs, obs_parts)
            if step % DIAG_LOG_EVERY == 0:
                print(f"[DIAG step {step}] actor_obs_mean={float(np.mean(actor_obs)):.4f} actor_obs_std={float(np.std(actor_obs)):.4f}")
        else:
            actor_obs = build_actor_obs(curr_features, history_actor)

        action = session.run([output_name], {input_name: actor_obs[None, :]})[0][0].astype(np.float32)
        action = np.clip(action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        if ACTION_FILTER_ALPHA < 0.999:
            # First-order low-pass filter on policy output to suppress jitter.
            action = ACTION_FILTER_ALPHA * action + (1.0 - ACTION_FILTER_ALPHA) * last_actions

        # Update history after observation/inference.
        update_history_buffers(history_buffers, curr_features)
        last_actions = action.copy()
        action_norm_hist.append(float(np.linalg.norm(action)))
        phase_hist.append(float(phase))
        if TRACE_OUT_NPZ:
            trace_actor_obs.append(actor_obs.copy())
            trace_action.append(action.copy())

        target_pos = action * ACTION_SCALE + DEFAULT_DOF_POS

        for substep in range(CONTROL_DECIMATION):
            if USE_IMPLICIT_PD:
                robot.control_dofs_position(position=torch.tensor(target_pos, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)
            else:
                dof_pos_sub = to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
                dof_vel_sub = to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
                torques_req = KP * (target_pos - dof_pos_sub) - KD * dof_vel_sub
                torques = np.clip(torques_req, -TORQUE_LIMITS, TORQUE_LIMITS)

                if torque_diag_steps > 0:
                    physics_step_idx = step * CONTROL_DECIMATION + substep
                    if physics_step_idx < torque_diag_steps:
                        clipped_mask = np.abs(torques_req) > TORQUE_LIMITS
                        torque_diag_req.append(torques_req.astype(np.float32))
                        torque_diag_actual.append(torques.astype(np.float32))
                        torque_diag_clipped.append(clipped_mask)
                        torque_diag_step_idx.append(physics_step_idx)

                robot.control_dofs_force(torch.tensor(torques, dtype=torch.float32, device=SIM_DEVICE), dofs_idx_local=motor_dofs)
            scene.step()

        root_pos = to_numpy_1d(robot.get_pos())
        root_z_hist.append(float(root_pos[2]))
        if cam is not None:
            cam.set_pose(
                pos=(root_pos[0] + 2.0, root_pos[1] + 2.0, 1.0),
                lookat=(root_pos[0], root_pos[1], 0.5),
                up=(0.0, 0.0, 1.0),
            )
            cam.render()

        if STOP_ON_FALL and root_pos[2] < FALL_ROOT_Z:
            fell = True
            stop_reason = "fall_detected"
            print(f"[INFO] Fall detected at step={step}, root_z={root_pos[2]:.4f} < FALL_ROOT_Z={FALL_ROOT_Z:.4f}")
            break

        if step % 50 == 0:
            print(
                f"[step {step}] phase={phase:.3f}, action_norm={np.linalg.norm(action):.3f}, "
                f"root_z={root_pos[2]:.3f}"
            )

    if cam is not None:
        cam.stop_recording(save_to_filename=OUTPUT_VIDEO, fps=record_fps)
    if DIAG_OBS:
        finalize_diag_stats(diag_stats, DIAG_OUT_NPZ)
    if cam is not None:
        print(f"[DONE] saved video: {os.path.abspath(OUTPUT_VIDEO)}")

    steps_executed = len(root_z_hist)
    root_z_arr = np.asarray(root_z_hist, dtype=np.float32)
    action_norm_arr = np.asarray(action_norm_hist, dtype=np.float32)
    phase_arr = np.asarray(phase_hist, dtype=np.float32)
    torque_diag_metrics = {}
    if torque_diag_clipped:
        torque_diag_req_arr = np.asarray(torque_diag_req, dtype=np.float32)
        torque_diag_actual_arr = np.asarray(torque_diag_actual, dtype=np.float32)
        torque_diag_clipped_arr = np.asarray(torque_diag_clipped, dtype=np.bool_)
        torque_diag_step_idx_arr = np.asarray(torque_diag_step_idx, dtype=np.int32)

        hip_any = np.any(torque_diag_clipped_arr[:, hip_joint_ids], axis=1) if hip_joint_ids.size > 0 else np.zeros(torque_diag_clipped_arr.shape[0], dtype=np.bool_)
        knee_any = np.any(torque_diag_clipped_arr[:, knee_joint_ids], axis=1) if knee_joint_ids.size > 0 else np.zeros(torque_diag_clipped_arr.shape[0], dtype=np.bool_)

        hip_any_ratio = float(np.mean(hip_any)) if hip_any.size > 0 else np.nan
        knee_any_ratio = float(np.mean(knee_any)) if knee_any.size > 0 else np.nan
        hip_joint_ratio = np.mean(torque_diag_clipped_arr[:, hip_joint_ids], axis=0).astype(np.float32) if hip_joint_ids.size > 0 else np.zeros((0,), dtype=np.float32)
        knee_joint_ratio = np.mean(torque_diag_clipped_arr[:, knee_joint_ids], axis=0).astype(np.float32) if knee_joint_ids.size > 0 else np.zeros((0,), dtype=np.float32)

        torque_diag_metrics = {
            "torque_diag_samples": np.array([torque_diag_clipped_arr.shape[0]], dtype=np.int32),
            "torque_diag_window_s_effective": np.array([torque_diag_clipped_arr.shape[0] * SIM_DT], dtype=np.float32),
            "hip_clip_ratio_any": np.array([hip_any_ratio], dtype=np.float32),
            "knee_clip_ratio_any": np.array([knee_any_ratio], dtype=np.float32),
            "hip_clip_ratio_mean_joint": np.array([float(np.mean(hip_joint_ratio)) if hip_joint_ratio.size > 0 else np.nan], dtype=np.float32),
            "knee_clip_ratio_mean_joint": np.array([float(np.mean(knee_joint_ratio)) if knee_joint_ratio.size > 0 else np.nan], dtype=np.float32),
        }

        print(
            f"[TORQUE_DIAG] samples={torque_diag_clipped_arr.shape[0]} "
            f"window={torque_diag_clipped_arr.shape[0] * SIM_DT:.3f}s "
            f"hip_any_clip_ratio={hip_any_ratio:.3f} knee_any_clip_ratio={knee_any_ratio:.3f}"
        )

        if TORQUE_DIAG_OUT_NPZ:
            np.savez(
                TORQUE_DIAG_OUT_NPZ,
                joint_names=np.array(RL_JOINT_NAMES, dtype=object),
                hip_joint_ids=hip_joint_ids,
                hip_joint_names=np.array([RL_JOINT_NAMES[i] for i in hip_joint_ids], dtype=object),
                knee_joint_ids=knee_joint_ids,
                knee_joint_names=np.array([RL_JOINT_NAMES[i] for i in knee_joint_ids], dtype=object),
                step_idx=torque_diag_step_idx_arr,
                tau_req=torque_diag_req_arr,
                tau_actual=torque_diag_actual_arr,
                clipped=torque_diag_clipped_arr.astype(np.uint8),
                clip_limit=np.asarray(TORQUE_LIMITS, dtype=np.float32),
                hip_clip_ratio_per_joint=hip_joint_ratio,
                knee_clip_ratio_per_joint=knee_joint_ratio,
                hip_clip_ratio_any=np.array([hip_any_ratio], dtype=np.float32),
                knee_clip_ratio_any=np.array([knee_any_ratio], dtype=np.float32),
                sim_dt=np.array([SIM_DT], dtype=np.float32),
                diag_window_seconds=np.array([TORQUE_DIAG_SECONDS], dtype=np.float32),
            )
            print(f"[TORQUE_DIAG] saved detailed torque trace to {TORQUE_DIAG_OUT_NPZ}")

    metrics = {
        "steps_executed": np.array([steps_executed], dtype=np.int32),
        "fell": np.array([int(fell)], dtype=np.int32),
        "stop_reason": np.array([stop_reason], dtype=object),
        "min_root_z": np.array([float(root_z_arr.min()) if steps_executed > 0 else np.nan], dtype=np.float32),
        "mean_root_z": np.array([float(root_z_arr.mean()) if steps_executed > 0 else np.nan], dtype=np.float32),
        "std_root_z": np.array([float(root_z_arr.std()) if steps_executed > 0 else np.nan], dtype=np.float32),
        "mean_action_norm": np.array([float(action_norm_arr.mean()) if action_norm_arr.size > 0 else np.nan], dtype=np.float32),
        "std_action_norm": np.array([float(action_norm_arr.std()) if action_norm_arr.size > 0 else np.nan], dtype=np.float32),
        "fall_root_z_threshold": np.array([FALL_ROOT_Z], dtype=np.float32),
        "time_offset": np.array([TIME_OFFSET], dtype=np.float32),
        "kd_scale": np.array([KD_SCALE], dtype=np.float32),
        "floor_friction": np.array([FLOOR_FRICTION], dtype=np.float32),
        "use_motion_init": np.array([int(USE_MOTION_INIT)], dtype=np.int32),
        "use_self_collision": np.array([int(USE_SELF_COLLISION)], dtype=np.int32),
        "use_implicit_pd": np.array([int(USE_IMPLICIT_PD)], dtype=np.int32),
        "sweep_tag": np.array([SWEEP_TAG], dtype=object),
    }
    metrics.update(torque_diag_metrics)

    print(
        f"[METRIC] steps={steps_executed} fell={int(fell)} stop_reason={stop_reason} "
        f"min_root_z={metrics['min_root_z'][0]:.4f} mean_root_z={metrics['mean_root_z'][0]:.4f} "
        f"mean_action_norm={metrics['mean_action_norm'][0]:.4f}"
    )

    if METRICS_OUT_NPZ:
        np.savez(METRICS_OUT_NPZ, **metrics)
        print(f"[METRIC] saved metrics to {METRICS_OUT_NPZ}")

    if TRACE_OUT_NPZ:
        trace_data = {
            "actor_obs": np.asarray(trace_actor_obs, dtype=np.float32),
            "action": np.asarray(trace_action, dtype=np.float32),
            "phase": phase_arr,
            "root_z": root_z_arr,
            "stop_reason": np.array([stop_reason], dtype=object),
        }
        if torque_diag_clipped:
            trace_data.update({
                "torque_step_idx": np.asarray(torque_diag_step_idx, dtype=np.int32),
                "tau_req": np.asarray(torque_diag_req, dtype=np.float32),
                "tau_actual": np.asarray(torque_diag_actual, dtype=np.float32),
                "tau_clipped": np.asarray(torque_diag_clipped, dtype=np.uint8),
                "torque_limit": np.asarray(TORQUE_LIMITS, dtype=np.float32),
            })
        np.savez(TRACE_OUT_NPZ, **trace_data)
        print(f"[TRACE] saved rollout trace to {TRACE_OUT_NPZ}")


if __name__ == "__main__":
    main()

import argparse
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import onnxruntime
import yaml
from scipy.spatial.transform import Rotation as R

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


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


def read_conf(config_file: str):
    cfg = SimpleNamespace()
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.cycle_time = config["cycle_time"]
    cfg.frame_stack = config["frame_stack"]
    cfg.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)
    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]
    cfg.clip_observations = config["clip_observations"]
    cfg.kps = np.array(config["kps"], dtype=np.float32)
    cfg.kds = np.array(config["kds"], dtype=np.float32)
    cfg.xml_path = config["xml_path"]
    cfg.num_actions = config["num_actions"]
    cfg.policy_path = config["policy_path"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.clip_actions = config["clip_actions"]
    cfg.action_scale = config["action_scale"]
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
    return cfg


def _shape_of(value: Any):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


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


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return quat_wxyz[[1, 2, 3, 0]].copy()


def _load_motion_dict(path: str) -> Tuple[str, Dict[str, Any]]:
    data = _load_pickle(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Expected non-empty dict motion pkl, got {type(data).__name__}")
    key = next(iter(data.keys()))
    motion = data[key]
    if not isinstance(motion, dict):
        raise ValueError(f"Expected dict motion entry, got {type(motion).__name__}")
    return key, motion


def get_mujoco_obs_data(data: mujoco.MjData) -> Dict[str, np.ndarray]:
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    quat_xyzw = np.array([q[4], q[5], q[6], q[3]], dtype=np.float64)
    rot = R.from_quat(quat_xyzw)
    return {
        "dof_pos": q[7:].astype(np.float32),
        "dof_vel": dq[6:].astype(np.float32),
        "base_ang_vel": rot.apply(dq[3:6], inverse=True).astype(np.float32),
        "gvec": rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float32),
    }


def update_hist_obs(hist_dict: Dict[str, np.ndarray], obs_single: np.ndarray) -> np.ndarray:
    slices = {
        "actions": slice(0, 23),
        "base_ang_vel": slice(23, 26),
        "dof_pos": slice(26, 49),
        "dof_vel": slice(49, 72),
        "projected_gravity": slice(72, 75),
        "ref_motion_phase": slice(75, 76),
    }
    for key, slc in slices.items():
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_single[0, slc], arr))
        hist_dict[key] = arr
    history_keys = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "ref_motion_phase"]
    hist_obs = [hist_dict[key].reshape(1, -1) for key in history_keys]
    return np.concatenate(hist_obs, axis=1).astype(np.float32)


def get_obs(
    hist_obs_c: np.ndarray,
    hist_dict: Dict[str, np.ndarray],
    mujoco_data: Dict[str, np.ndarray],
    action: np.ndarray,
    counter: int,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    ref_motion_phase = np.clip((counter + 1) * cfg.simulation_dt / cfg.cycle_time, 0.0, 1.0)
    num_obs_input = (cfg.frame_stack + 1) * cfg.num_single_obs

    obs_all = np.zeros((1, num_obs_input), dtype=np.float32)
    obs_single = np.zeros((1, cfg.num_single_obs), dtype=np.float32)
    obs_single[0, 0:23] = action
    obs_single[0, 23:26] = mujoco_data["base_ang_vel"] * cfg.obs_scale_base_ang_vel
    obs_single[0, 26:49] = (mujoco_data["dof_pos"] - cfg.default_dof_pos) * cfg.obs_scale_dof_pos
    obs_single[0, 49:72] = mujoco_data["dof_vel"] * cfg.obs_scale_dof_vel
    obs_single[0, 72:75] = mujoco_data["gvec"] * cfg.obs_scale_gvec
    obs_single[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion

    obs_all[0, 0:23] = obs_single[0, 0:23]
    obs_all[0, 23:26] = obs_single[0, 23:26]
    obs_all[0, 26:49] = obs_single[0, 26:49]
    obs_all[0, 49:72] = obs_single[0, 49:72]
    obs_all[0, 72:376] = hist_obs_c[0] * cfg.obs_scale_hist
    obs_all[0, 376:379] = obs_single[0, 72:75]
    obs_all[0, 379] = obs_single[0, 75]

    hist_obs_cat = update_hist_obs(hist_dict, obs_single)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    return obs_all, hist_obs_cat


def pd_control(target_pos: np.ndarray, dof_pos: np.ndarray, target_vel: np.ndarray, dof_vel: np.ndarray, cfg):
    return (target_pos - dof_pos) * cfg.kps + (target_vel - dof_vel) * cfg.kds


def _build_dof_axis(model: mujoco.MjModel) -> np.ndarray:
    dof_axis = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_axis.append(np.asarray(model.jnt_axis[joint_id], dtype=np.float32))
    return np.stack(dof_axis, axis=0)


def _capture_body_data(model: mujoco.MjModel, data: mujoco.MjData):
    body_pos = []
    body_quat = []
    body_lin_vel = []
    body_ang_vel = []
    for body_name in BODY_NAMES_24:
        body_id = model.body(body_name).id
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
        body_pos.append(data.xpos[body_id].astype(np.float32, copy=True))
        body_quat.append(_wxyz_to_xyzw(data.xquat[body_id]))
        body_lin_vel.append(vel6[3:6].astype(np.float32, copy=True))
        body_ang_vel.append(vel6[0:3].astype(np.float32, copy=True))
    return (
        np.stack(body_pos, axis=0),
        np.stack(body_quat, axis=0),
        np.stack(body_lin_vel, axis=0),
        np.stack(body_ang_vel, axis=0),
    )


def _capture_frame(model: mujoco.MjModel, data: mujoco.MjData, dof_axis: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    body_pos, body_quat, body_lin_vel, body_ang_vel = _capture_body_data(model, data)
    root_rot_xyzw = _wxyz_to_xyzw(qpos[3:7])
    root_rot_vec = R.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    dof = qpos[7:].astype(np.float32, copy=True)
    pose_aa = np.concatenate(
        [root_rot_vec[None, :], dof_axis * dof[:, None], np.zeros((len(EXTEND_OFFSETS), 3), dtype=np.float32)],
        axis=0,
    )
    return {
        "root_trans_offset": qpos[:3].astype(np.float32, copy=True),
        "root_rot": root_rot_xyzw,
        "dof": dof,
        "dof_vel": qvel[6:].astype(np.float32, copy=True),
        "root_lin_vel": qvel[:3].astype(np.float32, copy=True),
        "root_ang_vel": qvel[3:6].astype(np.float32, copy=True),
        "action": action.astype(np.float32, copy=True),
        "pose_aa": pose_aa,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_lin_vel": body_lin_vel,
        "body_ang_vel": body_ang_vel,
    }


def _stack_rollout_buffer(buffer: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack(value, axis=0) for key, value in buffer.items()}


def _build_output_motion(template_motion: Dict[str, Any], rollout: Dict[str, np.ndarray]) -> Dict[str, Any]:
    output_motion = dict(template_motion)
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
    output_motion["fps"] = 50
    output_motion["dt"] = 1.0 / 50.0
    output_motion["motion_length"] = int(rollout["action"].shape[0])
    return output_motion


def _check_array(name: str, value: Any):
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        return
    print(f"check[{name}] has_nan={bool(np.isnan(arr).any())} has_inf={bool(np.isinf(arr).any())}")


def _print_summary(output_path: str, motion_key: str, output_motion: Dict[str, Any]):
    print(f"output_path={output_path}")
    print(f"motion_key={motion_key}")
    for key, value in output_motion.items():
        print(f"output[{key}] shape={_shape_of(value)} type={type(value).__name__}")
    action = np.asarray(output_motion["action"], dtype=np.float64)
    print(
        "action_stats "
        f"mean={float(action.mean()):.6f} "
        f"std={float(action.std()):.6f} "
        f"min={float(action.min()):.6f} "
        f"max={float(action.max()):.6f}"
    )
    for key, value in output_motion.items():
        _check_array(key, value)


def run_and_collect(cfg, template_motion_path: str, output_path: str, num_steps: int):
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt
    data.qpos[-cfg.num_actions:] = cfg.default_dof_pos
    mujoco.mj_step(model, data)

    policy = onnxruntime.InferenceSession(cfg.policy_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    target_dof_pos = cfg.default_dof_pos.copy()
    action = np.zeros(cfg.num_actions, dtype=np.float32)
    hist_dict = {
        "actions": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        "base_ang_vel": np.zeros((cfg.frame_stack, 3), dtype=np.float32),
        "dof_pos": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        "dof_vel": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        "projected_gravity": np.zeros((cfg.frame_stack, 3), dtype=np.float32),
        "ref_motion_phase": np.zeros((cfg.frame_stack, 1), dtype=np.float32),
    }
    history_keys = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "ref_motion_phase"]
    hist_obs = [hist_dict[key].reshape(1, -1) for key in history_keys]
    hist_obs_c = np.concatenate(hist_obs, axis=1)
    dof_axis = _build_dof_axis(model)

    rollout_buffer = {
        "root_trans_offset": [],
        "root_rot": [],
        "dof": [],
        "dof_vel": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "action": [],
        "pose_aa": [],
        "body_pos": [],
        "body_quat": [],
        "body_lin_vel": [],
        "body_ang_vel": [],
    }

    counter = 0
    sim_steps = int(cfg.simulation_duration / cfg.simulation_dt)
    for _ in range(sim_steps):
        if len(rollout_buffer["action"]) >= num_steps:
            break
        mujoco_data = get_mujoco_obs_data(data)
        tau = pd_control(target_dof_pos, mujoco_data["dof_pos"], np.zeros_like(cfg.kds), mujoco_data["dof_vel"], cfg)
        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        if counter % cfg.control_decimation != 0:
            continue

        obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg)
        raw_action = policy.run([output_name], {input_name: obs_buff})[0]
        action = np.asarray(raw_action).reshape(-1).astype(np.float32)
        action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
        target_dof_pos = action * cfg.action_scale + cfg.default_dof_pos
        frame = _capture_frame(model, data, dof_axis, action)
        for key, value in frame.items():
            rollout_buffer[key].append(value)
        if len(rollout_buffer["action"]) == 1 or len(rollout_buffer["action"]) == num_steps:
            print(
                f"step={len(rollout_buffer['action']) - 1} "
                f"action_mean={float(action.mean()):.6f} "
                f"action_std={float(action.std()):.6f} "
                f"action_min={float(action.min()):.6f} "
                f"action_max={float(action.max()):.6f}"
            )

    if len(rollout_buffer["action"]) != num_steps:
        raise RuntimeError(
            f"Collected {len(rollout_buffer['action'])} steps, expected {num_steps}. "
            "Increase simulation_duration or inspect rollout termination behavior."
        )

    rollout = _stack_rollout_buffer(rollout_buffer)
    template_key, template_motion = _load_motion_dict(template_motion_path)
    output_motion = _build_output_motion(template_motion, rollout)
    output_data = {template_key: output_motion}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_pickle(output_data, str(output_path))
    print(f"[DONE] saved motion pkl: {output_path.resolve()}")
    _print_summary(str(output_path.resolve()), template_key, output_motion)


def _default_config_path():
    cwd_candidate = Path(os.getcwd()) / "mujoco_simulation" / "g1_config" / "mujoco_config.yaml"
    if cwd_candidate.is_file():
        return str(cwd_candidate)
    script_candidate = Path(__file__).resolve().parent / "g1_config" / "mujoco_config.yaml"
    return str(script_candidate)


def main():
    parser = argparse.ArgumentParser(description="Stable MuJoCo sim2sim pkl extractor")
    parser.add_argument("--config", type=str, default=_default_config_path())
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--xml-path", type=str, default="")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=134)
    parser.add_argument("--template-motion-path", type=str, default=TEMPLATE_MOTION_PATH)
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    cfg = read_conf(args.config)
    cfg_dir = Path(args.config).resolve().parent
    if not os.path.isabs(cfg.xml_path):
        xml_candidates = [
            (cfg_dir / cfg.xml_path).resolve(),
            (cfg_dir.parent / cfg.xml_path).resolve(),
        ]
        cfg.xml_path = str(next((p for p in xml_candidates if p.is_file()), xml_candidates[0]))
    if not os.path.isabs(cfg.policy_path):
        policy_candidates = [
            (cfg_dir / cfg.policy_path).resolve(),
            (cfg_dir.parent / cfg.policy_path).resolve(),
        ]
        cfg.policy_path = str(next((p for p in policy_candidates if p.is_file()), policy_candidates[0]))
    if args.xml_path:
        cfg.xml_path = args.xml_path
    if args.policy_path:
        cfg.policy_path = args.policy_path
    if args.duration > 0:
        cfg.simulation_duration = float(args.duration)

    if not os.path.isfile(cfg.xml_path):
        raise FileNotFoundError(f"xml_path not found: {cfg.xml_path}")
    if not os.path.isfile(cfg.policy_path):
        raise FileNotFoundError(f"policy_path not found: {cfg.policy_path}")
    if not os.path.isfile(args.template_motion_path):
        raise FileNotFoundError(f"template_motion_path not found: {args.template_motion_path}")

    run_and_collect(cfg, args.template_motion_path, args.output_path, int(args.num_steps))


if __name__ == "__main__":
    main()

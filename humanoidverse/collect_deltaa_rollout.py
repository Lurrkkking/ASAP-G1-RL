import os
import sys
from pathlib import Path
from typing import Any, Dict

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import isaacgym  # noqa: F401
import hydra
import joblib
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation as sRot
from humanoidverse.utils.helpers import pre_process_config

import logging


DEFAULT_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
)
DEFAULT_OUTPUT_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_gym_rollout_with_action.pkl"
)


def _shape_of(value: Any):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


def _dtype_of(value: Any):
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    return None


def _to_numpy(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _summarize_motion_dict(motion_key: str, motion_dict: Dict[str, Any]):
    logger.info(f"Loaded source motion entry: {motion_key}")
    for key, value in motion_dict.items():
        logger.info(
            f"source[{key}] type={type(value).__name__} shape={_shape_of(value)} dtype={_dtype_of(value)}"
        )


def _load_source_motion(path: str):
    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Expected non-empty dict motion pkl, got {type(data).__name__}")
    motion_key = next(iter(data.keys()))
    motion_dict = data[motion_key]
    if not isinstance(motion_dict, dict):
        raise ValueError(f"Expected dict motion entry, got {type(motion_dict).__name__}")
    _summarize_motion_dict(motion_key, motion_dict)
    return data, motion_key, motion_dict


def _load_eval_config(override_config: OmegaConf) -> OmegaConf:
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

            return OmegaConf.merge(train_config, override_config)

    if override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = OmegaConf.create(eval_overrides)
        return OmegaConf.merge(config, eval_overrides)

    return override_config


def _get_collect_cfg(config: OmegaConf):
    collect_cfg = config.get("collect", OmegaConf.create())
    return {
        "source_motion_path": str(collect_cfg.get("source_motion_path", DEFAULT_MOTION_PATH)),
        "output_path": str(collect_cfg.get("output_path", DEFAULT_OUTPUT_PATH)),
        "num_steps": collect_cfg.get("num_steps", None),
        "env_id": int(collect_cfg.get("env_id", 0)),
        "print_debug": bool(collect_cfg.get("print_debug", True)),
    }


def _capture_frame(env, env_id: int):
    root_state = env.simulator.robot_root_states[env_id]
    frame = {
        "dof": _to_numpy(env.simulator.dof_pos[env_id]).astype(np.float32, copy=True),
        "dof_vel": _to_numpy(env.simulator.dof_vel[env_id]).astype(np.float32, copy=True),
        "root_trans_offset": _to_numpy(root_state[0:3]).astype(np.float32, copy=True),
        "root_rot": _to_numpy(root_state[3:7]).astype(np.float32, copy=True),
        "root_lin_vel": _to_numpy(root_state[7:10]).astype(np.float32, copy=True),
        "root_ang_vel": _to_numpy(root_state[10:13]).astype(np.float32, copy=True),
        "body_pos": _to_numpy(env.simulator._rigid_body_pos[env_id]).astype(np.float32, copy=True),
        "body_pos_w": _to_numpy(env.simulator._rigid_body_pos[env_id]).astype(np.float32, copy=True),
        "body_quat": _to_numpy(env.simulator._rigid_body_rot[env_id]).astype(np.float32, copy=True),
        "body_quat_w": _to_numpy(env.simulator._rigid_body_rot[env_id]).astype(np.float32, copy=True),
        "body_lin_vel": _to_numpy(env.simulator._rigid_body_vel[env_id]).astype(np.float32, copy=True),
        "body_ang_vel": _to_numpy(env.simulator._rigid_body_ang_vel[env_id]).astype(np.float32, copy=True),
    }
    return frame


def _build_pose_aa(root_rot_xyzw: np.ndarray, dof: np.ndarray, env) -> np.ndarray:
    root_rot_vec = sRot.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    dof_axis = _to_numpy(env._motion_lib.mesh_parsers.dof_axis).astype(np.float32)
    num_augment_joint = len(getattr(env.config.robot.motion, "extend_config", []))
    pose_aa = np.concatenate(
        [
            root_rot_vec[None, :],
            dof_axis * dof[:, None],
            np.zeros((num_augment_joint, 3), dtype=np.float32),
        ],
        axis=0,
    )
    return pose_aa


def _infer_num_steps(collect_cfg: Dict[str, Any], source_motion_dict: Dict[str, Any], env) -> int:
    requested = collect_cfg["num_steps"]
    if requested is not None:
        return int(requested)

    source_len = None
    for key in ("dof", "root_trans_offset", "pose_aa", "root_rot"):
        if key in source_motion_dict and hasattr(source_motion_dict[key], "shape"):
            source_len = int(source_motion_dict[key].shape[0])
            break
    if source_len is not None:
        return source_len

    return int(env.max_episode_length)


def _build_output_motion(
    source_motion_dict: Dict[str, Any],
    rollout: Dict[str, np.ndarray],
    env,
    num_steps: int,
    print_debug: bool,
):
    output_motion: Dict[str, Any] = {}
    output_motion.update(source_motion_dict)

    output_motion["root_trans_offset"] = rollout["root_trans_offset"]
    output_motion["root_rot"] = rollout["root_rot"]
    output_motion["dof"] = rollout["dof"]
    output_motion["dof_vel"] = rollout["dof_vel"]
    output_motion["root_lin_vel"] = rollout["root_lin_vel"]
    output_motion["root_ang_vel"] = rollout["root_ang_vel"]
    output_motion["action"] = rollout["action"]
    output_motion["body_pos"] = rollout["body_pos"]
    output_motion["body_pos_w"] = rollout["body_pos_w"]
    output_motion["body_quat"] = rollout["body_quat"]
    output_motion["body_quat_w"] = rollout["body_quat_w"]
    output_motion["body_lin_vel"] = rollout["body_lin_vel"]
    output_motion["body_ang_vel"] = rollout["body_ang_vel"]
    output_motion["fps"] = int(round(1.0 / env.dt))
    output_motion["dt"] = float(env.dt)
    output_motion["motion_length"] = int(num_steps)

    pose_aa = np.stack(
        [
            _build_pose_aa(output_motion["root_rot"][i], output_motion["dof"][i], env)
            for i in range(num_steps)
        ],
        axis=0,
    ).astype(np.float32)
    output_motion["pose_aa"] = pose_aa

    if "smpl_joints" in source_motion_dict:
        src_smpl = _to_numpy(source_motion_dict["smpl_joints"])
        if src_smpl is not None and src_smpl.ndim == 3 and src_smpl.shape[1:] == rollout["body_pos"].shape[1:]:
            output_motion["smpl_joints"] = rollout["body_pos"].astype(np.float32)
            if print_debug:
                logger.info("Replaced source smpl_joints with rollout body_pos because shapes match.")
        else:
            if print_debug:
                logger.warning(
                    "Keeping source smpl_joints unchanged because rollout body_pos shape does not match."
                )

    return output_motion


def _stack_rollout_buffer(buffer: Dict[str, list]):
    return {key: np.stack(value, axis=0) for key, value in buffer.items()}


def _check_array(name: str, value: Any):
    arr = _to_numpy(value)
    if arr is None or not np.issubdtype(arr.dtype, np.number):
        return
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    logger.info(f"check[{name}] has_nan={has_nan} has_inf={has_inf}")


def _print_output_summary(output_path: str, motion_key: str, output_motion: Dict[str, Any], env, policy_action_dim: int):
    logger.info(f"output_path={output_path}")
    logger.info(f"output_motion_key={motion_key}")
    for key, value in output_motion.items():
        logger.info(f"output[{key}] shape={_shape_of(value)} type={type(value).__name__}")

    action = output_motion["action"]
    logger.info(
        "action_stats "
        f"mean={float(action.mean()):.6f} "
        f"std={float(action.std()):.6f} "
        f"min={float(action.min()):.6f} "
        f"max={float(action.max()):.6f}"
    )
    logger.info(f"dof_pos shape={output_motion['dof'].shape}")
    logger.info(
        "root_state shape="
        f"{output_motion['root_trans_offset'].shape},"
        f"{output_motion['root_rot'].shape},"
        f"{output_motion['root_lin_vel'].shape},"
        f"{output_motion['root_ang_vel'].shape}"
    )
    logger.info(
        "action_dim_check "
        f"action.shape[-1]={action.shape[-1]} "
        f"env.num_actions={env.dim_actions} "
        f"policy_action_dim={policy_action_dim}"
    )

    for key, value in output_motion.items():
        _check_array(key, value)


def _log_termination_debug(env, env_id: int, step: int):
    root_state = _to_numpy(env.simulator.robot_root_states[env_id])
    projected_gravity = _to_numpy(env.projected_gravity[env_id])
    log_parts = [
        f"termination_debug step={step}",
        f"time_out={bool(env.time_out_buf[env_id].item())}",
        f"reset={bool(env.reset_buf[env_id].item())}",
        f"episode_length={int(env.episode_length_buf[env_id].item())}",
        f"motion_len={float(env.motion_len[env_id].item()) if hasattr(env, 'motion_len') else 'NA'}",
        f"base_height={float(root_state[2]):.6f}",
        f"projected_gravity_x={float(projected_gravity[0]):.6f}",
        f"projected_gravity_y={float(projected_gravity[1]):.6f}",
        f"gravity_thresh_x={float(env.config.termination_scales.termination_gravity_x):.6f}",
        f"gravity_thresh_y={float(env.config.termination_scales.termination_gravity_y):.6f}",
    ]
    if hasattr(env, "dif_global_body_pos"):
        motion_far = _to_numpy(env.dif_global_body_pos[env_id])
        motion_far_norm = np.linalg.norm(motion_far, axis=-1)
        log_parts.extend(
            [
                f"motion_far_max={float(motion_far_norm.max()):.6f}",
                f"motion_far_mean={float(motion_far_norm.mean()):.6f}",
                f"motion_far_thresh={float(getattr(env, 'terminate_when_motion_far_threshold', np.nan)):.6f}",
            ]
        )
    logger.warning(" ".join(log_parts))


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    from humanoidverse.utils import config_utils as _config_utils  # noqa: F401
    from humanoidverse.utils.logging import HydraLoggerBridge

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "collect_deltaa_rollout.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stderr, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    config = _load_eval_config(override_config)
    collect_cfg = _get_collect_cfg(config)

    source_data, source_motion_key, source_motion_dict = _load_source_motion(collect_cfg["source_motion_path"])

    auto_record = bool(config.get("auto_record", False))
    auto_record_num_frames = int(config.get("auto_record_num_frames", 600))
    offscreen_record = bool(config.get("offscreen_record", False))
    offscreen_record_width = int(config.get("offscreen_record_width", 1600))
    offscreen_record_height = int(config.get("offscreen_record_height", 900))
    offscreen_record_fps = int(config.get("offscreen_record_fps", 50))

    if offscreen_record:
        config.headless = True

    OmegaConf.update(config, "headless", True, force_add=True)
    OmegaConf.update(config, "num_envs", 1, force_add=True)
    OmegaConf.update(config, "env.config.headless", True, force_add=True)
    OmegaConf.update(config, "env.config.num_envs", 1, force_add=True)
    OmegaConf.update(config, "env.config.auto_record", auto_record, force_add=True)
    OmegaConf.update(config, "env.config.auto_record_num_frames", auto_record_num_frames, force_add=True)
    OmegaConf.update(config, "env.config.offscreen_record", offscreen_record, force_add=True)
    OmegaConf.update(config, "env.config.offscreen_record_width", offscreen_record_width, force_add=True)
    OmegaConf.update(config, "env.config.offscreen_record_height", offscreen_record_height, force_add=True)
    OmegaConf.update(config, "env.config.offscreen_record_fps", offscreen_record_fps, force_add=True)
    OmegaConf.update(
        config,
        "env.config.robot.motion.motion_file",
        collect_cfg["source_motion_path"],
        force_add=True,
    )
    OmegaConf.update(config, "env.config.save_motion", False, force_add=True)

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        raise NotImplementedError("This collector currently supports IsaacGym sanity check only.")
    import torch

    pre_process_config(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs

    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    checkpoint = Path(config.checkpoint)
    ckpt_num = checkpoint.stem.split("_")[-1]
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent)

    with open_dict(config.env.config):
        config.env.config.auto_record = auto_record
        config.env.config.auto_record_num_frames = auto_record_num_frames
        config.env.config.offscreen_record = offscreen_record
        config.env.config.offscreen_record_width = offscreen_record_width
        config.env.config.offscreen_record_height = offscreen_record_height
        config.env.config.offscreen_record_fps = offscreen_record_fps

    env = instantiate(config.env, device=device)
    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    eval_policy = algo._get_inference_policy()
    obs_dict = env.reset_all()

    env_id = collect_cfg["env_id"]
    if env_id < 0 or env_id >= env.num_envs:
        raise ValueError(f"collect.env_id={env_id} out of range for num_envs={env.num_envs}")

    num_steps = _infer_num_steps(collect_cfg, source_motion_dict, env)
    logger.info(
        "Collecting rollout with aligned current-state convention: "
        "state[t] is sampled before env.step(action[t]), so action[t] maps s_t -> s_(t+1). "
        "To stay compatible with the motion loader, action length is set to T and state length is also T."
    )
    logger.info(f"num_steps={num_steps} dt={env.dt} fps={1.0 / env.dt}")

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

    policy_action_dim = int(env.dim_actions)
    clip_action_limit = float(config.robot.control.action_clip_value)
    logger.info(
        "saved action is the action actually passed to env.step and expected by "
        "delta_a_open_loop before multiplying action_scale. "
        f"clip_action_limit={clip_action_limit:.6f}"
    )

    for step in range(num_steps):
        frame = _capture_frame(env, env_id)
        for key, value in frame.items():
            rollout_buffer[key].append(value)

        with torch.no_grad():
            raw_actions = eval_policy(obs_dict["actor_obs"])
        actions_to_step = torch.clip(raw_actions, -clip_action_limit, clip_action_limit)
        raw_action_np = _to_numpy(raw_actions[env_id]).astype(np.float32, copy=True)
        saved_action_np = _to_numpy(actions_to_step[env_id]).astype(np.float32, copy=True)
        rollout_buffer["action"].append(saved_action_np)

        actor_state = {"obs": obs_dict, "actions": actions_to_step}
        obs_dict, _, dones, _ = env.step(actor_state)

        if collect_cfg["print_debug"] and (step == 0 or step == num_steps - 1):
            logger.info(
                f"step={step} raw_action_stats "
                f"mean={float(raw_action_np.mean()):.6f} std={float(raw_action_np.std()):.6f} "
                f"min={float(raw_action_np.min()):.6f} max={float(raw_action_np.max()):.6f}"
            )
            logger.info(
                f"step={step} saved_clipped_action_stats "
                f"mean={float(saved_action_np.mean()):.6f} std={float(saved_action_np.std()):.6f} "
                f"min={float(saved_action_np.min()):.6f} max={float(saved_action_np.max()):.6f}"
            )

        if bool(dones[env_id].item()):
            _log_termination_debug(env, env_id, step)
            logger.warning(
                f"Environment {env_id} terminated immediately after step={step}. "
                "Stopping collection to avoid mixing auto-reset states into the same trajectory."
            )
            break

    rollout = _stack_rollout_buffer(rollout_buffer)
    num_steps = int(rollout["action"].shape[0])
    output_motion = _build_output_motion(
        source_motion_dict=source_motion_dict,
        rollout=rollout,
        env=env,
        num_steps=num_steps,
        print_debug=collect_cfg["print_debug"],
    )

    output_path = Path(collect_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {source_motion_key: output_motion}
    joblib.dump(output_data, output_path)

    _print_output_summary(str(output_path), source_motion_key, output_motion, env, policy_action_dim)


if __name__ == "__main__":
    main()

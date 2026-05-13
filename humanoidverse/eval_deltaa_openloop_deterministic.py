from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict

torch = None
quat_conjugate = None
quat_mul = None
quat_rotate_inverse = None
PPOActor = None


DEFAULT_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_mujoco_rollout_with_action.pkl"
)
DEFAULT_NUM_STEPS = 134
DEFAULT_NUM_ENVS = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = [
    "root_pos_error",
    "root_rot_error",
    "dof_pos_error",
    "dof_vel_error",
    "body_pos_mpjpe",
    "body_rot_error",
    "body_vel_error",
    "body_ang_vel_error",
    "total_diff_norm",
]
ACTION_STAT_NAMES = [
    "deterministic_delta_action_mean_abs",
    "deterministic_delta_action_max_abs",
    "motion_action_mean_abs",
    "delta_over_motion_mean",
]


def tensor_to_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    return np.asarray(x).tolist()


def disable_domain_rand(cfg):
    if not hasattr(cfg, "domain_rand"):
        return
    for key in list(cfg.domain_rand.keys()):
        value = cfg.domain_rand[key]
        if isinstance(value, bool) and (
            key.startswith("randomize") or key.startswith("push") or key.endswith("noise")
        ):
            cfg.domain_rand[key] = False


def disable_terminations(cfg):
    termination = cfg.env.config.termination
    for key in list(termination.keys()):
        if isinstance(termination[key], bool):
            termination[key] = False


def get_checkpoint_config_path(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find config.yaml next to checkpoint {checkpoint_path}. "
        f"Tried: {candidates}"
    )


def load_runtime_config(override_config: OmegaConf) -> OmegaConf:
    eval_cfg = override_config.get("eval_deltaa", OmegaConf.create())
    checkpoint = eval_cfg.get("checkpoint", None)
    if checkpoint is None:
        config = override_config
    else:
        config_path = get_checkpoint_config_path(Path(str(checkpoint)))
        logger.info(f"Loading training config from {config_path}")
        train_config = OmegaConf.load(config_path)
        if train_config.get("eval_overrides", None) is not None:
            train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
        config = OmegaConf.merge(train_config, override_config)

    motion_file = str(eval_cfg.get("motion_file", DEFAULT_MOTION_PATH))
    num_steps = int(eval_cfg.get("num_steps", DEFAULT_NUM_STEPS))
    num_envs = int(eval_cfg.get("num_envs", DEFAULT_NUM_ENVS))

    if not Path(motion_file).exists():
        raise FileNotFoundError(f"motion_file does not exist: {motion_file}")
    if checkpoint is None:
        raise ValueError("Missing required +eval_deltaa.checkpoint=... override")
    if not Path(str(checkpoint)).exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    with open_dict(config):
        config.headless = bool(eval_cfg.get("headless", True))
        config.num_envs = num_envs
        config.use_wandb = False
        config.checkpoint = None
        config.eval_name = config.get("eval_name", "deltaa_openloop_deterministic")
        config.env.config.headless = config.headless
        config.env.config.num_envs = num_envs
        config.env.config.max_episode_length_s = max(float(config.env.config.max_episode_length_s), float(num_steps + 10))
        config.env.config.noise_to_initial_level = 0
        config.env.config.enforce_randomize_motion_start_eval = False
        config.env.config.randomize_motion_start_train = False
        config.env.config.save_motion = False
        config.env.config.robot.motion.motion_file = motion_file
        config.algo.config.load_optimizer = False
        config.eval_deltaa = OmegaConf.create(
            {
                "checkpoint": str(checkpoint),
                "motion_file": motion_file,
                "num_steps": num_steps,
                "num_envs": num_envs,
                "headless": bool(config.headless),
            }
        )

    disable_domain_rand(config)
    disable_terminations(config)
    return config


def get_device(config):
    if config.get("device", None):
        return config.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def quat_angle_error(q_ref, q_sim):
    q_diff = quat_mul(q_ref, quat_conjugate(q_sim, w_last=True), w_last=True)
    xyz = q_diff[..., :3]
    w = torch.clamp(q_diff[..., 3], -1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1)
    return 2.0 * torch.atan2(sin_half, torch.abs(w))


def compute_motion_state(env, motion_time: float):
    motion_times = torch.full((env.num_envs,), float(motion_time), dtype=torch.float32, device=env.device)
    return env._motion_lib.get_motion_state(env.motion_ids, motion_times, offset=env.env_origins)


def refresh_obs(env):
    env._refresh_sim_tensors()
    env._pre_compute_observations_callback()
    env._compute_observations()
    return {key: value.clone() for key, value in env.obs_buf_dict.items()}


def set_manual_state(env, motion_time: float, action_time: float):
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    motion_res = compute_motion_state(env, motion_time)

    env.simulator.robot_root_states[env_ids, :3] = motion_res["root_pos"]
    env.simulator.robot_root_states[env_ids, 3:7] = motion_res["root_rot"]
    env.simulator.robot_root_states[env_ids, 7:10] = motion_res["root_vel"]
    env.simulator.robot_root_states[env_ids, 10:13] = motion_res["root_ang_vel"]
    env.simulator.dof_pos[env_ids] = motion_res["dof_pos"]
    env.simulator.dof_vel[env_ids] = motion_res["dof_vel"]

    env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
    env.simulator.set_dof_state_tensor(env_ids, env.simulator.dof_state)
    env._refresh_sim_tensors()

    env.motion_start_times[:] = float(action_time - env.dt)
    env.episode_length_buf.zero_()
    env.last_episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.actions.zero_()
    env.actions_after_delay.zero_()
    env.last_actions.zero_()
    env.last_dof_pos[:] = env.simulator.dof_pos
    env.last_dof_vel[:] = env.simulator.dof_vel
    env.last_root_vel[:] = env.simulator.robot_root_states[:, 7:13]
    env.feet_air_time.zero_()
    env.last_contacts.zero_()
    env.last_contacts_filt.zero_()
    env.feet_air_max_height.zero_()
    env.need_to_refresh_envs.zero_()
    env.history_handler.reset(env_ids)
    env.base_quat[:] = env.simulator.base_quat[:]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.simulator.robot_root_states[:, 7:10], w_last=True)
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.simulator.robot_root_states[:, 10:13], w_last=True)
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec, w_last=True)
    return motion_res


def compute_error_row(env, target_motion_time: float, step_idx: int, mode: str):
    motion_res = compute_motion_state(env, target_motion_time)

    root_pos_error = torch.linalg.norm(motion_res["root_pos"][0] - env.simulator.robot_root_states[0, :3])
    root_rot_error = quat_angle_error(
        motion_res["root_rot"][0:1],
        env.simulator.robot_root_states[0:1, 3:7],
    )[0]
    dof_pos_error = torch.linalg.norm(motion_res["dof_pos"][0] - env.simulator.dof_pos[0])
    dof_vel_error = torch.linalg.norm(motion_res["dof_vel"][0] - env.simulator.dof_vel[0])

    body_pos_err = motion_res["rg_pos"][0] - env.simulator._rigid_body_pos[0]
    body_pos_mpjpe = torch.linalg.norm(body_pos_err, dim=-1).mean()
    body_rot_error = quat_angle_error(motion_res["rb_rot"][0], env.simulator._rigid_body_rot[0]).mean()
    body_vel_error = torch.linalg.norm(
        motion_res["body_vel"][0] - env.simulator._rigid_body_vel[0], dim=-1
    ).mean()
    body_ang_vel_error = torch.linalg.norm(
        motion_res["body_ang_vel"][0] - env.simulator._rigid_body_ang_vel[0], dim=-1
    ).mean()

    total_diff_norm = torch.sqrt(
        root_pos_error.square()
        + root_rot_error.square()
        + dof_pos_error.square()
        + dof_vel_error.square()
        + body_pos_mpjpe.square()
        + body_rot_error.square()
        + body_vel_error.square()
        + body_ang_vel_error.square()
    )

    row = {
        "mode": mode,
        "step": int(step_idx),
        "target_motion_time": float(target_motion_time),
        "root_pos_error": float(root_pos_error.detach().cpu().item()),
        "root_rot_error": float(root_rot_error.detach().cpu().item()),
        "dof_pos_error": float(dof_pos_error.detach().cpu().item()),
        "dof_vel_error": float(dof_vel_error.detach().cpu().item()),
        "body_pos_mpjpe": float(body_pos_mpjpe.detach().cpu().item()),
        "body_rot_error": float(body_rot_error.detach().cpu().item()),
        "body_vel_error": float(body_vel_error.detach().cpu().item()),
        "body_ang_vel_error": float(body_ang_vel_error.detach().cpu().item()),
        "total_diff_norm": float(total_diff_norm.detach().cpu().item()),
    }
    return row


def summarize_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return summary


def summarize_action_stats(
    delta_actions: List[torch.Tensor],
    motion_actions: List[torch.Tensor],
) -> Dict[str, float]:
    delta_tensor = torch.cat([x.reshape(-1) for x in delta_actions], dim=0)
    motion_tensor = torch.cat([x.reshape(-1) for x in motion_actions], dim=0)
    return {
        "deterministic_delta_action_mean_abs": float(delta_tensor.abs().mean().cpu().item()),
        "deterministic_delta_action_max_abs": float(delta_tensor.abs().max().cpu().item()),
        "motion_action_mean_abs": float(motion_tensor.abs().mean().cpu().item()),
        "delta_over_motion_mean": float(
            (delta_tensor.abs() / (motion_tensor.abs() + 1e-6)).mean().cpu().item()
        ),
    }


def load_deterministic_actor(config: OmegaConf, env, device: str, checkpoint_path: str):
    actor = PPOActor(
        obs_dim_dict=env.config.robot.algo_obs_dim_dict,
        module_config_dict=config.algo.config.module_dict.actor,
        num_actions=env.config.robot.actions_dim,
        init_noise_std=float(config.algo.config.init_noise_std),
        learn_sigma=bool(config.algo.config.get("learn_sigma", True)),
    ).to(device)
    loaded_dict = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(loaded_dict["actor_model_state_dict"])
    actor.eval()
    return actor, loaded_dict


def run_replay(env, mode: str, num_steps: int, actor=None) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    assert mode in {"zero_delta", "deterministic_deltaa"}
    set_manual_state(env, motion_time=0.0, action_time=0.0)
    obs_dict = refresh_obs(env)
    rows = []
    delta_actions = []
    motion_actions = []

    for step_idx in range(num_steps):
        motion_action = env.get_open_loop_action_at_current_timestep().detach().clone()
        if mode == "zero_delta":
            delta_action = torch.zeros(
                (env.num_envs, env.dim_actions),
                dtype=torch.float32,
                device=env.device,
            )
        else:
            with torch.inference_mode():
                delta_action = actor.act_inference(obs_dict["actor_obs"]).detach()

        motion_actions.append(motion_action)
        delta_actions.append(delta_action)

        obs_dict, _, dones, _ = env.step({"actions": delta_action})
        target_motion_time = float((step_idx + 1) * env.dt)
        rows.append(compute_error_row(env, target_motion_time, step_idx, mode))

        if bool(dones[0].item()):
            logger.warning(
                f"{mode} reached done=True at step={step_idx}; continuing because terminations were disabled."
            )

    return rows, summarize_action_stats(delta_actions, motion_actions)


def compute_improvements(
    zero_summary: Dict[str, Dict[str, float]],
    det_summary: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    improvements = {}
    for metric in METRIC_NAMES:
        zero_mean = zero_summary[metric]["mean"]
        det_mean = det_summary[metric]["mean"]
        if abs(zero_mean) < 1e-12:
            improvements[metric] = 0.0
        else:
            improvements[metric] = float((zero_mean - det_mean) / zero_mean)
    return improvements


def compute_improvement_percent(improvement_ratio: Dict[str, float]) -> Dict[str, float]:
    return {key: float(100.0 * value) for key, value in improvement_ratio.items()}


def format_metric_table(
    zero_summary: Dict[str, Dict[str, float]],
    det_summary: Dict[str, Dict[str, float]],
    improvements: Dict[str, float],
) -> str:
    lines = []
    header = (
        f"{'metric':<24} "
        f"{'zero_mean':>12} {'zero_max':>12} {'zero_min':>12} "
        f"{'delta_mean':>12} {'delta_max':>12} {'delta_min':>12} "
        f"{'improve_%':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for metric in METRIC_NAMES:
        lines.append(
            f"{metric:<24} "
            f"{zero_summary[metric]['mean']:12.6f} {zero_summary[metric]['max']:12.6f} {zero_summary[metric]['min']:12.6f} "
            f"{det_summary[metric]['mean']:12.6f} {det_summary[metric]['max']:12.6f} {det_summary[metric]['min']:12.6f} "
            f"{100.0 * improvements[metric]:12.2f}"
        )
    return "\n".join(lines)


def format_action_table(
    zero_action_stats: Dict[str, float],
    det_action_stats: Dict[str, float],
) -> str:
    lines = []
    header = f"{'action_stat':<36} {'zero_delta':>14} {'deterministic_deltaa':>24}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in ACTION_STAT_NAMES:
        lines.append(
            f"{name:<36} {zero_action_stats[name]:14.6f} {det_action_stats[name]:24.6f}"
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Dict):
    json_path = output_dir / "deltaa_openloop_deterministic_summary.json"
    txt_path = output_dir / "deltaa_openloop_deterministic_summary.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metric_table = format_metric_table(
        payload["zero_delta"]["metric_summary"],
        payload["deterministic_deltaa"]["metric_summary"],
        payload["improvement_ratio"],
    )
    action_table = format_action_table(
        payload["zero_delta"]["action_stats"],
        payload["deterministic_deltaa"]["action_stats"],
    )
    lines = [
        "[CONFIG]",
        json.dumps(payload["config"], indent=2),
        "",
        "[METRIC_TABLE]",
        metric_table,
        "",
        "[ACTION_TABLE]",
        action_table,
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, txt_path


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    global torch, quat_conjugate, quat_mul, quat_rotate_inverse, PPOActor

    config = load_runtime_config(override_config)
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        raise NotImplementedError("This evaluator currently supports IsaacGym-style execution only.")
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401
    import torch as _torch
    from isaac_utils.rotations import (
        quat_conjugate as _quat_conjugate,
        quat_mul as _quat_mul,
        quat_rotate_inverse as _quat_rotate_inverse,
    )
    from humanoidverse.agents.modules.ppo_modules import PPOActor as _PPOActor

    torch = _torch
    quat_conjugate = _quat_conjugate
    quat_mul = _quat_mul
    quat_rotate_inverse = _quat_rotate_inverse
    PPOActor = _PPOActor

    import logging
    from humanoidverse.utils import config_utils as _config_utils  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge

    hydra_log_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        "eval_deltaa_openloop_deterministic.log",
    )
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(REPO_ROOT)

    pre_process_config(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs

    device = get_device(config)
    env = instantiate(config.env, device=device)
    _ = env.reset_all()
    env.motion_start_times.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.episode_length_buf.zero_()
    env.last_episode_length_buf.zero_()

    checkpoint_path = str(config.eval_deltaa.checkpoint)
    actor, loaded_dict = load_deterministic_actor(config, env, device, checkpoint_path)
    max_pairs = int(env._motion_lib.get_motion_num_steps(env.motion_ids)[0].item()) - 1
    num_steps = min(int(config.eval_deltaa.num_steps), max_pairs)

    logger.info(
        f"Running deterministic deltaA open-loop eval with motion_file={config.eval_deltaa.motion_file} "
        f"checkpoint={checkpoint_path} num_steps={num_steps} num_envs={env.num_envs} dt={float(env.dt):.6f}"
    )

    zero_rows, zero_action_stats = run_replay(env, mode="zero_delta", num_steps=num_steps, actor=None)
    det_rows, det_action_stats = run_replay(env, mode="deterministic_deltaa", num_steps=num_steps, actor=actor)

    zero_summary = summarize_metric_rows(zero_rows)
    det_summary = summarize_metric_rows(det_rows)
    improvements = compute_improvements(zero_summary, det_summary)
    improvement_percent = compute_improvement_percent(improvements)

    metric_table = format_metric_table(zero_summary, det_summary, improvements)
    action_table = format_action_table(zero_action_stats, det_action_stats)
    print("\n[METRIC_TABLE]")
    print(metric_table)
    print("\n[ACTION_TABLE]")
    print(action_table)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    payload = {
        "config": {
            "checkpoint": checkpoint_path,
            "motion_file": str(config.eval_deltaa.motion_file),
            "num_steps": num_steps,
            "num_envs": int(env.num_envs),
            "dt": float(env.dt),
            "device": str(device),
            "hydra_output_dir": str(output_dir),
            "checkpoint_iter": int(loaded_dict.get("iter", 0)),
            "action_scale": float(env.config.robot.control.action_scale),
        },
        "zero_delta": {
            "metric_summary": zero_summary,
            "action_stats": zero_action_stats,
            "rows": zero_rows,
        },
        "deterministic_deltaa": {
            "metric_summary": det_summary,
            "action_stats": det_action_stats,
            "rows": det_rows,
        },
        "improvement_ratio": improvements,
        "improvement_percent": improvement_percent,
    }
    json_path, txt_path = write_outputs(output_dir, payload)
    logger.info(f"Saved JSON summary to {json_path}")
    logger.info(f"Saved text summary to {txt_path}")


if __name__ == "__main__":
    main()

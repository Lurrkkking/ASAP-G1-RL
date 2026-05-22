from __future__ import annotations

import json
import os
import subprocess
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
    "raw_deterministic_delta_action_mean_abs",
    "masked_deterministic_delta_action_mean_abs",
    "ankle_delta_mean_abs",
    "non_ankle_delta_after_mask_mean_abs",
    "motion_action_mean_abs",
    "delta_over_motion_mean",
    "final_action_mean_abs",
    "final_action_max_abs",
]


def final_action_matches_zero_delta(config: OmegaConf) -> bool:
    eval_cfg = config.get("eval_deltaa", OmegaConf.create())
    if bool(eval_cfg.get("force_zero_delta", False)):
        return True
    return abs(float(eval_cfg.get("delta_scale", 1.0))) < 1e-12


def get_delta_action_mask_mode(config: OmegaConf) -> str:
    algo_cfg = config.get("algo", None)
    if algo_cfg is None:
        return "full"
    algo_inner_cfg = algo_cfg.get("config", None)
    if algo_inner_cfg is None:
        return "full"
    return str(algo_inner_cfg.get("delta_action_mask_mode", "full"))


def get_ankle_delta_indices(robot_cfg: OmegaConf) -> List[int]:
    joint_names = list(robot_cfg.dof_names)
    ankle_joint_names = [
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    indices = []
    for joint_name in ankle_joint_names:
        if joint_name not in joint_names:
            raise ValueError(
                f"Could not find ankle joint '{joint_name}' in robot.dof_names={joint_names}"
            )
        indices.append(joint_names.index(joint_name))
    return indices


def build_delta_action_mask(robot_cfg: OmegaConf, device: str, mode: str, dtype=None):
    if mode not in {"full", "ankle_only"}:
        raise ValueError(
            f"Unsupported delta_action_mask_mode={mode}. Expected one of ['full', 'ankle_only']."
        )
    num_actions = int(robot_cfg.actions_dim)
    if dtype is None:
        dtype = torch.float32
    mask = torch.ones(num_actions, dtype=dtype, device=device)
    ankle_indices = get_ankle_delta_indices(robot_cfg)
    if mode == "ankle_only":
        mask.zero_()
        mask[ankle_indices] = 1.0
    return mask, ankle_indices


def compute_body_mean_l2(sim_body_pos: torch.Tensor, target_body_pos: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(sim_body_pos - target_body_pos, dim=-1).mean()


def _stack_frames(frames: List[torch.Tensor]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("No frames were collected for trajectory stacking")
    return torch.stack(frames, dim=0)


def compute_paper_metrics(
    sim_body_pos: torch.Tensor,
    target_body_pos: torch.Tensor,
    sim_root_pos: torch.Tensor,
    target_root_pos: torch.Tensor,
    sim_root_vel: torch.Tensor,
    target_root_vel: torch.Tensor,
    dt: float,
) -> Dict[str, float]:
    if sim_body_pos.shape != target_body_pos.shape:
        raise ValueError(
            f"sim_body_pos shape {list(sim_body_pos.shape)} does not match target_body_pos shape {list(target_body_pos.shape)}"
        )
    if sim_root_pos.shape != target_root_pos.shape:
        raise ValueError(
            f"sim_root_pos shape {list(sim_root_pos.shape)} does not match target_root_pos shape {list(target_root_pos.shape)}"
        )
    if sim_root_vel.shape != target_root_vel.shape:
        raise ValueError(
            f"sim_root_vel shape {list(sim_root_vel.shape)} does not match target_root_vel shape {list(target_root_vel.shape)}"
        )
    if sim_body_pos.shape[0] != sim_root_pos.shape[0]:
        raise ValueError(
            f"Body frame count {sim_body_pos.shape[0]} does not match root frame count {sim_root_pos.shape[0]}"
        )
    if sim_body_pos.shape[0] < 3:
        raise ValueError("Need at least 3 frames to compute Eacc from second differences")

    # Units:
    # - body/root positions are in meters
    # - Eg/Emp/Eacc are reported in mm
    # - Evel is reported in mm/frame
    eg_mpjpe_mm = compute_body_mean_l2(sim_body_pos, target_body_pos) * 1000.0
    sim_rel_body_pos = sim_body_pos - sim_root_pos[:, None, :]
    target_rel_body_pos = target_body_pos - target_root_pos[:, None, :]
    empjpe_mm = compute_body_mean_l2(sim_rel_body_pos, target_rel_body_pos) * 1000.0

    sim_acc = sim_body_pos[2:] - 2.0 * sim_body_pos[1:-1] + sim_body_pos[:-2]
    target_acc = target_body_pos[2:] - 2.0 * target_body_pos[1:-1] + target_body_pos[:-2]
    eacc_mm_per_frame2 = torch.linalg.norm(sim_acc - target_acc, dim=-1).mean() * 1000.0

    if sim_root_vel is not None and target_root_vel is not None:
        # root linear velocity is in m/s; multiply by dt to convert to mm/frame.
        evel_mm_per_frame = torch.linalg.norm(sim_root_vel - target_root_vel, dim=-1).mean() * dt * 1000.0
        evel_source = "root_lin_vel"
    else:
        sim_root_delta = sim_root_pos[1:] - sim_root_pos[:-1]
        target_root_delta = target_root_pos[1:] - target_root_pos[:-1]
        evel_mm_per_frame = torch.linalg.norm(sim_root_delta - target_root_delta, dim=-1).mean() * 1000.0
        evel_source = "root_pos_diff"

    return {
        "Eg_mpjpe_mm": float(eg_mpjpe_mm.detach().cpu().item()),
        "Empjpe_mm": float(empjpe_mm.detach().cpu().item()),
        "Eacc_mm_per_frame2": float(eacc_mm_per_frame2.detach().cpu().item()),
        "Evel_mm_per_frame": float(evel_mm_per_frame.detach().cpu().item()),
        "_Evel_source": evel_source,
    }


def compute_improve_percent(zero_value: float, det_value: float) -> float:
    if abs(zero_value) < 1e-12:
        return 0.0
    return float((zero_value - det_value) / zero_value * 100.0)


def compute_paper_summary(zero_metrics: Dict[str, float], det_metrics: Dict[str, float]) -> Dict[str, float]:
    metric_names = ["Eg_mpjpe_mm", "Empjpe_mm", "Eacc_mm_per_frame2", "Evel_mm_per_frame"]
    improve = {name: compute_improve_percent(zero_metrics[name], det_metrics[name]) for name in metric_names}
    normalized_ratios = []
    for name in metric_names:
        zero_value = zero_metrics[name]
        det_value = det_metrics[name]
        normalized_ratios.append(0.0 if abs(zero_value) < 1e-12 else float(det_value / zero_value))
    paper_normalized_error_ratio = float(sum(normalized_ratios) / len(normalized_ratios))
    paper_normalized_improve = float((1.0 - paper_normalized_error_ratio) * 100.0)
    paper_mean_improve = float(sum(improve.values()) / len(improve))
    return {
        **{f"{name}_improve_%": improve[name] for name in metric_names},
        "paper_mean_improve_%": paper_mean_improve,
        "paper_normalized_error_ratio": paper_normalized_error_ratio,
        "paper_normalized_improve_%": paper_normalized_improve,
    }


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
                "force_zero_delta": bool(eval_cfg.get("force_zero_delta", False)),
                "delta_scale": float(eval_cfg.get("delta_scale", 1.0)),
                "single_mode": eval_cfg.get("single_mode", None),
                "output_json": eval_cfg.get("output_json", None),
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
    return row, motion_res


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
    raw_delta_actions: List[torch.Tensor],
    masked_delta_actions: List[torch.Tensor],
    final_actions: List[torch.Tensor],
    motion_actions: List[torch.Tensor],
    ankle_delta_indices: List[int],
) -> Dict[str, float]:
    raw_delta_tensor = torch.cat([x.reshape(-1) for x in raw_delta_actions], dim=0)
    masked_delta_tensor = torch.cat([x.reshape(-1) for x in masked_delta_actions], dim=0)
    final_action_tensor = torch.cat([x.reshape(-1) for x in final_actions], dim=0)
    motion_tensor = torch.cat([x.reshape(-1) for x in motion_actions], dim=0)
    masked_delta_matrix = torch.cat(masked_delta_actions, dim=0)
    final_action_matrix = torch.cat(final_actions, dim=0)
    motion_action_matrix = torch.cat(motion_actions, dim=0)
    ankle_delta_tensor = masked_delta_matrix[:, ankle_delta_indices].reshape(-1)
    non_ankle_indices = [idx for idx in range(masked_delta_matrix.shape[1]) if idx not in ankle_delta_indices]
    if len(non_ankle_indices) > 0:
        non_ankle_delta_tensor = masked_delta_matrix[:, non_ankle_indices].reshape(-1)
        non_ankle_delta_after_mask_mean_abs = float(non_ankle_delta_tensor.abs().mean().cpu().item())
    else:
        non_ankle_delta_after_mask_mean_abs = 0.0
    return {
        "raw_deterministic_delta_action_mean_abs": float(raw_delta_tensor.abs().mean().cpu().item()),
        "masked_deterministic_delta_action_mean_abs": float(masked_delta_tensor.abs().mean().cpu().item()),
        "ankle_delta_mean_abs": float(ankle_delta_tensor.abs().mean().cpu().item()),
        "non_ankle_delta_after_mask_mean_abs": non_ankle_delta_after_mask_mean_abs,
        "motion_action_mean_abs": float(motion_tensor.abs().mean().cpu().item()),
        "delta_over_motion_mean": float(
            (masked_delta_tensor.abs() / (motion_tensor.abs() + 1e-6)).mean().cpu().item()
        ),
        "final_action_mean_abs": float(final_action_tensor.abs().mean().cpu().item()),
        "final_action_max_abs": float(final_action_tensor.abs().max().cpu().item()),
    }


def summarize_rows(rows: List[Dict[str, float]], metric_names: List[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return summary


def format_diagnostic_metric_table(
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


def format_paper_metric_table(zero_metrics: Dict[str, float], det_metrics: Dict[str, float]) -> str:
    metric_names = ["Eg_mpjpe_mm", "Empjpe_mm", "Eacc_mm_per_frame2", "Evel_mm_per_frame"]
    header = f"{'metric':<24} {'zero_delta':>14} {'deterministic_deltaa':>24} {'improve_%':>12}"
    lines = [header, "-" * len(header)]
    for name in metric_names:
        lines.append(
            f"{name:<24} {zero_metrics[name]:14.6f} {det_metrics[name]:24.6f} {compute_improve_percent(zero_metrics[name], det_metrics[name]):12.2f}"
        )
    lines.append(
        f"{'paper_mean_improve_%':<24} {'':>14} {'':>24} {compute_paper_summary(zero_metrics, det_metrics)['paper_mean_improve_%']:12.2f}"
    )
    lines.append(
        f"{'paper_normalized_improve_%':<24} {'':>14} {'':>24} {compute_paper_summary(zero_metrics, det_metrics)['paper_normalized_improve_%']:12.2f}"
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Dict):
    json_path = output_dir / "deltaa_openloop_deterministic_summary.json"
    txt_path = output_dir / "deltaa_openloop_deterministic_summary.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    comparison = payload.get("comparison", {})
    zero_metric_summary = comparison.get(
        "zero_metric_summary_for_comparison",
        payload["zero_delta"]["metric_summary"],
    )
    det_metric_summary = comparison.get(
        "det_metric_summary_for_comparison",
        payload["deterministic_deltaa"]["metric_summary"],
    )
    metric_table = format_diagnostic_metric_table(
        zero_metric_summary,
        det_metric_summary,
        payload["improvement_ratio"],
    )
    paper_table = format_paper_metric_table(
        comparison["paper_zero_metrics"],
        comparison["paper_det_metrics"],
    )
    action_table = format_action_table(
        payload["zero_delta"]["action_stats"],
        payload["deterministic_deltaa"]["action_stats"],
    )
    lines = [
        "[CONFIG]",
        json.dumps(payload["config"], indent=2),
        "",
        "[PAPER_METRIC_TABLE]",
        paper_table,
        "",
        "[DIAGNOSTIC_METRIC_TABLE]",
        metric_table,
        "",
        "[ACTION_TABLE]",
        action_table,
        "",
        "[NOTES]",
        "total_diff_norm is a raw mixed diagnostic, not a paper Table III metric.",
        "dof_vel_error is diagnostic only and is excluded from paper_mean_improve_%.",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, txt_path




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


def run_replay(env, mode: str, num_steps: int, actor=None, mask_mode: str = "full") -> Tuple[List[Dict[str, float]], Dict[str, float], Dict[str, torch.Tensor]]:
    assert mode in {"zero_delta", "deterministic_deltaa"}
    _ = env.reset_all()
    env.motion_start_times.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.episode_length_buf.zero_()
    env.last_episode_length_buf.zero_()
    set_manual_state(env, motion_time=0.0, action_time=0.0)
    obs_dict = refresh_obs(env)
    rows = []
    raw_delta_actions = []
    masked_delta_actions = []
    final_actions = []
    motion_actions = []
    sim_body_pos_frames = []
    target_body_pos_frames = []
    sim_root_pos_frames = []
    target_root_pos_frames = []
    sim_root_vel_frames = []
    target_root_vel_frames = []
    force_zero_delta = bool(getattr(env.config.eval_deltaa, "force_zero_delta", False))
    delta_scale = float(getattr(env.config.eval_deltaa, "delta_scale", 1.0))
    previous_add_extra_action = bool(env.config["add_extra_action"])
    env.config["add_extra_action"] = False

    try:
        for step_idx in range(num_steps):
            motion_action = env.get_open_loop_action_at_current_timestep().detach().clone()
            if mode == "zero_delta":
                raw_delta_action = torch.zeros((env.num_envs, env.dim_actions), dtype=torch.float32, device=env.device)
            else:
                with torch.inference_mode():
                    raw_delta_action = actor.act_inference(obs_dict["actor_obs"]).detach()

            if force_zero_delta:
                raw_delta_action = torch.zeros_like(raw_delta_action)

            delta_action_mask, ankle_delta_indices = build_delta_action_mask(
                env.config.robot,
                raw_delta_action.device,
                mask_mode,
                dtype=raw_delta_action.dtype,
            )
            env_device = torch.device(env.device)
            assert delta_action_mask.shape in {(env.dim_actions,), (1, env.dim_actions)}, delta_action_mask.shape
            assert delta_action_mask.device == env_device, (delta_action_mask.device, env_device)
            assert delta_action_mask.dtype == raw_delta_action.dtype, (delta_action_mask.dtype, raw_delta_action.dtype)
            non_ankle_indices = [idx for idx in range(raw_delta_action.shape[1]) if idx not in ankle_delta_indices]
            masked_delta_action = raw_delta_action * delta_action_mask
            masked_delta_action = delta_scale * masked_delta_action
            final_action = motion_action + masked_delta_action
            raw_delta_non_ankle_mean_abs = raw_delta_action[:, non_ankle_indices].abs().mean().item() if len(non_ankle_indices) > 0 else 0.0
            masked_delta_non_ankle_mean_abs = masked_delta_action[:, non_ankle_indices].abs().mean().item() if len(non_ankle_indices) > 0 else 0.0
            masked_delta_non_ankle_max_abs = masked_delta_action[:, non_ankle_indices].abs().max().item() if len(non_ankle_indices) > 0 else 0.0
            final_action_non_ankle_minus_motion_max_abs = (final_action[:, non_ankle_indices] - motion_action[:, non_ankle_indices]).abs().max().item() if len(non_ankle_indices) > 0 else 0.0
            assert masked_delta_non_ankle_max_abs < 1e-6, masked_delta_non_ankle_max_abs
            assert torch.allclose(final_action[:, non_ankle_indices], motion_action[:, non_ankle_indices], atol=1e-6), final_action_non_ankle_minus_motion_max_abs
            if mask_mode == "ankle_only":
                assert masked_delta_non_ankle_mean_abs < 1e-6, masked_delta_non_ankle_mean_abs

            motion_actions.append(motion_action)
            raw_delta_actions.append(raw_delta_action)
            masked_delta_actions.append(masked_delta_action)
            final_actions.append(final_action)

            if step_idx == 0:
                logger.info(
                    f"[eval_deltaa_mask] mode={mask_mode} ankle_delta_indices={ankle_delta_indices} "
                    f"mask={tensor_to_list(delta_action_mask)} "
                    f"mask_shape={list(delta_action_mask.shape)} mask_dtype={delta_action_mask.dtype} "
                    f"mask_device={delta_action_mask.device} force_zero_delta={force_zero_delta} delta_scale={delta_scale:.6f}"
                )

            if step_idx < 3:
                logger.info(
                    "[eval_deltaa_mask_stats] "
                    f"mode={mask_mode} "
                    f"step={step_idx} "
                    f"raw_delta_non_ankle_mean_abs={raw_delta_non_ankle_mean_abs:.6f} "
                    f"masked_delta_non_ankle_mean_abs={masked_delta_non_ankle_mean_abs:.6f} "
                    f"masked_delta_non_ankle_max_abs={masked_delta_non_ankle_max_abs:.6f} "
                    f"raw_deterministic_delta_action_mean_abs={raw_delta_action.abs().mean().item():.6f} "
                    f"masked_deterministic_delta_action_mean_abs={masked_delta_action.abs().mean().item():.6f} "
                    f"ankle_delta_mean_abs={masked_delta_action[:, ankle_delta_indices].abs().mean().item():.6f} "
                    f"non_ankle_delta_after_mask_mean_abs={masked_delta_non_ankle_mean_abs:.6f} "
                    f"motion_action_mean_abs={motion_action.abs().mean().item():.6f} "
                    f"final_action_mean_abs={final_action.abs().mean().item():.6f} "
                    f"final_action_max_abs={final_action.abs().max().item():.6f} "
                    f"non_ankle_final_minus_motion_max_abs={final_action_non_ankle_minus_motion_max_abs:.6f} "
                    f"delta_over_motion_mean={((masked_delta_action.abs() / (motion_action.abs() + 1e-6)).mean()).item():.6f}"
                )

            obs_dict, _, dones, _ = env.step({"actions": final_action})
            target_motion_time = float((step_idx + 1) * env.dt)
            row, motion_res = compute_error_row(env, target_motion_time, step_idx, mode)
            rows.append(row)
            sim_body_pos_frames.append(env.simulator._rigid_body_pos[0].detach().clone())
            target_body_pos_frames.append(motion_res["rg_pos"][0].detach().clone())
            sim_root_pos_frames.append(env.simulator.robot_root_states[0, :3].detach().clone())
            target_root_pos_frames.append(motion_res["root_pos"][0].detach().clone())
            sim_root_vel_frames.append(env.simulator.robot_root_states[0, 7:10].detach().clone())
            target_root_vel_frames.append(motion_res["root_vel"][0].detach().clone())

            if bool(dones[0].item()):
                logger.warning(
                    f"{mode} reached done=True at step={step_idx}; continuing because terminations were disabled."
                )
    finally:
        env.config["add_extra_action"] = previous_add_extra_action

    trajectories = {
        "sim_body_pos": _stack_frames(sim_body_pos_frames),
        "target_body_pos": _stack_frames(target_body_pos_frames),
        "sim_root_pos": _stack_frames(sim_root_pos_frames),
        "target_root_pos": _stack_frames(target_root_pos_frames),
        "sim_root_vel": _stack_frames(sim_root_vel_frames),
        "target_root_vel": _stack_frames(target_root_vel_frames),
    }

    return rows, summarize_action_stats(
        raw_delta_actions,
        masked_delta_actions,
        final_actions,
        motion_actions,
        ankle_delta_indices,
    ), trajectories

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

    comparison = payload["comparison"]
    metric_table = format_diagnostic_metric_table(
        comparison["zero_metric_summary_for_comparison"],
        comparison["det_metric_summary_for_comparison"],
        payload["diagnostic_improvement_ratio"],
    )
    paper_table = format_paper_metric_table(
        comparison["paper_zero_metrics"],
        comparison["paper_det_metrics"],
    )
    action_table = format_action_table(
        payload["zero_delta"]["action_stats"],
        payload["deterministic_deltaa"]["action_stats"],
    )
    lines = [
        "[CONFIG]",
        json.dumps(payload["config"], indent=2),
        "",
        "[PAPER_METRIC_TABLE]",
        paper_table,
        "",
        "[DIAGNOSTIC_METRIC_TABLE]",
        metric_table,
        "",
        "[ACTION_TABLE]",
        action_table,
        "",
        "[NOTES]",
        "total_diff_norm is a raw mixed diagnostic, not a paper Table III metric.",
        "dof_vel_error is diagnostic only and is excluded from paper_mean_improve_%.",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, txt_path



def write_single_mode_output(output_json_path: str, payload: Dict):
    output_path = Path(str(output_json_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def spawn_single_mode_eval(mode: str, output_json_path: Path):
    task_overrides = list(HydraConfig.get().overrides.task)
    forwarded_overrides = []
    for override in task_overrides:
        if override.startswith("+eval_deltaa.single_mode="):
            continue
        if override.startswith("+eval_deltaa.output_json="):
            continue
        forwarded_overrides.append(override)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *forwarded_overrides,
        f"+eval_deltaa.single_mode={mode}",
        f"+eval_deltaa.output_json={output_json_path}",
    ]
    logger.info(f"Spawning child eval for mode={mode}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


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
    from humanoidverse.utils.common import seeding

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

    seed = int(config.get("seed", 0))
    torch_deterministic = bool(config.get("torch_deterministic", False))
    seeding(seed, torch_deterministic=torch_deterministic)

    pre_process_config(config)
    single_mode = config.eval_deltaa.get("single_mode", None)
    if single_mode is None:
        parent_output_dir = Path(HydraConfig.get().runtime.output_dir)
        zero_json = parent_output_dir / "zero_delta_child.json"
        det_json = parent_output_dir / "deterministic_deltaa_child.json"
        spawn_single_mode_eval("zero_delta", zero_json)
        spawn_single_mode_eval("deterministic_deltaa", det_json)
        with open(zero_json, "r", encoding="utf-8") as f:
            zero_payload = json.load(f)
        with open(det_json, "r", encoding="utf-8") as f:
            det_payload = json.load(f)

        zero_summary = zero_payload["metric_summary"]
        det_summary = det_payload["metric_summary"]
        zero_action_stats = zero_payload["action_stats"]
        det_action_stats = det_payload["action_stats"]
        zero_traj = {k: torch.tensor(v) for k, v in zero_payload["trajectories"].items()}
        det_traj = {k: torch.tensor(v) for k, v in det_payload["trajectories"].items()}
        effective_same_final_action = final_action_matches_zero_delta(config)

        zero_paper_metrics = compute_paper_metrics(
            zero_traj["sim_body_pos"],
            zero_traj["target_body_pos"],
            zero_traj["sim_root_pos"],
            zero_traj["target_root_pos"],
            zero_traj["sim_root_vel"],
            zero_traj["target_root_vel"],
            float(zero_payload["config"]["dt"]),
        )
        det_paper_metrics = compute_paper_metrics(
            det_traj["sim_body_pos"],
            det_traj["target_body_pos"],
            det_traj["sim_root_pos"],
            det_traj["target_root_pos"],
            det_traj["sim_root_vel"],
            det_traj["target_root_vel"],
            float(det_payload["config"]["dt"]),
        )
        paper_summary = compute_paper_summary(zero_paper_metrics, det_paper_metrics)
        paper_metric_improvements = {
            name: paper_summary[f"{name}_improve_%"]
            for name in ["Eg_mpjpe_mm", "Empjpe_mm", "Eacc_mm_per_frame2", "Evel_mm_per_frame"]
        }

        if effective_same_final_action:
            logger.info(
                "delta_scale is effectively zero or force_zero_delta is enabled; paper-style metrics should be ~0-improve because both modes step the env with identical final_action."
            )
            zero_summary_for_comparison = zero_summary
            det_summary_for_comparison = zero_summary
        else:
            zero_summary_for_comparison = zero_summary
            det_summary_for_comparison = det_summary
        diagnostic_improvements = compute_improvements(zero_summary_for_comparison, det_summary_for_comparison)
        diagnostic_improvement_percent = compute_improvement_percent(diagnostic_improvements)

        if effective_same_final_action:
            for name, improve in paper_metric_improvements.items():
                if abs(improve) > 1e-3:
                    logger.warning(f"Sanity check: {name} improve is {improve:.6f} even though final_action should match zero_delta.")

        diagnostic_metric_table = format_diagnostic_metric_table(
            zero_summary_for_comparison,
            det_summary_for_comparison,
            diagnostic_improvements,
        )
        paper_metric_table = format_paper_metric_table(zero_paper_metrics, det_paper_metrics)
        action_table = format_action_table(zero_action_stats, det_action_stats)
        print("\n[PAPER_METRIC_TABLE]")
        print(paper_metric_table)
        print("\n[DIAGNOSTIC_METRIC_TABLE]")
        print(diagnostic_metric_table)
        print("\n[ACTION_TABLE]")
        print(action_table)

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        payload = {
            "config": {
                "checkpoint": str(config.eval_deltaa.checkpoint),
                "motion_file": str(config.eval_deltaa.motion_file),
                "num_steps": int(config.eval_deltaa.num_steps),
                "num_envs": int(config.eval_deltaa.num_envs),
                "device": str(get_device(config)),
                "hydra_output_dir": str(output_dir),
                "delta_action_mask_mode": get_delta_action_mask_mode(config),
                "force_zero_delta": bool(config.eval_deltaa.force_zero_delta),
                "delta_scale": float(config.eval_deltaa.delta_scale),
                "seed": seed,
                "torch_deterministic": torch_deterministic,
                "comparison_mode": (
                    "separate_process_children_with_zero_scale_canonicalization"
                    if effective_same_final_action
                    else "separate_process_children"
                ),
            },
            "zero_delta": zero_payload,
            "deterministic_deltaa": det_payload,
            "comparison": {
                "effective_same_final_action": effective_same_final_action,
                "paper_zero_metrics": zero_paper_metrics,
                "paper_det_metrics": det_paper_metrics,
                "paper_summary": paper_summary,
                "zero_metric_summary_for_comparison": zero_summary_for_comparison,
                "det_metric_summary_for_comparison": det_summary_for_comparison,
            },
            "paper_metric_improvements": paper_metric_improvements,
            "paper_mean_improve_%": paper_summary["paper_mean_improve_%"],
            "paper_normalized_error_ratio": paper_summary["paper_normalized_error_ratio"],
            "paper_normalized_improve_%": paper_summary["paper_normalized_improve_%"],
            "diagnostic_improvement_ratio": diagnostic_improvements,
            "diagnostic_improvement_percent": diagnostic_improvement_percent,
            "diagnostic_total_diff_norm_improve": diagnostic_improvements["total_diff_norm"],
            "diagnostic_dof_vel_improve": diagnostic_improvements["dof_vel_error"],
        }
        json_path, txt_path = write_outputs(output_dir, payload)
        logger.info(f"Saved JSON summary to {json_path}")
        logger.info(f"Saved text summary to {txt_path}")
        return

    assert single_mode in {"zero_delta", "deterministic_deltaa"}, single_mode
    mask_mode = get_delta_action_mask_mode(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs
        config.env.config.eval_deltaa = config.eval_deltaa
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
        f"Running single-mode deterministic deltaA open-loop eval with mode={single_mode} "
        f"motion_file={config.eval_deltaa.motion_file} checkpoint={checkpoint_path} "
        f"num_steps={num_steps} num_envs={env.num_envs} dt={float(env.dt):.6f} "
        f"delta_action_mask_mode={mask_mode}"
    )

    rows, action_stats, trajectories = run_replay(
        env,
        mode=single_mode,
        num_steps=num_steps,
        actor=(None if single_mode == "zero_delta" else actor),
        mask_mode=mask_mode,
    )
    metric_summary = summarize_metric_rows(rows)
    payload = {
        "mode": single_mode,
        "metric_summary": metric_summary,
        "action_stats": action_stats,
        "rows": rows,
        "trajectories": {k: v.detach().cpu().tolist() for k, v in trajectories.items()},
        "config": {
            "checkpoint": checkpoint_path,
            "motion_file": str(config.eval_deltaa.motion_file),
            "num_steps": num_steps,
            "num_envs": int(env.num_envs),
            "dt": float(env.dt),
            "device": str(device),
            "checkpoint_iter": int(loaded_dict.get("iter", 0)),
            "action_scale": float(env.config.robot.control.action_scale),
            "delta_action_mask_mode": get_delta_action_mask_mode(config),
            "ankle_delta_indices": get_ankle_delta_indices(env.config.robot),
            "force_zero_delta": bool(config.eval_deltaa.force_zero_delta),
            "delta_scale": float(config.eval_deltaa.delta_scale),
            "seed": seed,
            "torch_deterministic": torch_deterministic,
            "env_step_action_variable": "final_action",
            "final_action_formula": "motion_action + delta_scale * masked_delta_action",
            "masked_delta_formula": "raw_delta_action * mask",
        },
    }
    output_json = config.eval_deltaa.get("output_json", None)
    if output_json is not None:
        output_path = write_single_mode_output(output_json, payload)
        logger.info(f"Saved single-mode JSON to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()

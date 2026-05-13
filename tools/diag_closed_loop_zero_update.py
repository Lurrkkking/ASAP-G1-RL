#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import isaacgym  # noqa: F401
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict

from humanoidverse.utils.config_utils import *  # noqa: F401,F403
from humanoidverse.utils.helpers import pre_process_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "closed_loop_zero_update_diag"
DEFAULT_BASELINE_CFG = REPO_ROOT / "logs" / "TEST_CR7_Siuuu" / "baseline13000group" / "config.yaml"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "logs" / "TEST_CR7_Siuuu" / "baseline13000group" / "model_13000.pt"
DEFAULT_CLOSED_LOOP_CFG = (
    REPO_ROOT
    / "logs"
    / "DeltaA_MuJoCo"
    / "20260513_180908-closedloop_cr7_level2_mujoco_deltaA600-delta_a-g1_29dof_anneal_23dof"
    / "config.yaml"
)
DEFAULT_DELTA_CKPT = (
    REPO_ROOT
    / "logs"
    / "DeltaA_MuJoCo"
    / "20260513_134050-openloop_mujoco_cr7_readtest-delta_a-g1_29dof_anneal_23dof"
    / "model_600.pt"
)


def tensor_to_cpu(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return x


def scalar_float(x):
    x = tensor_to_cpu(x)
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def mean_abs_and_max_abs(x):
    x = tensor_to_cpu(x).float()
    return float(x.abs().mean().item()), float(x.abs().max().item())


def load_config(config_path: Path, checkpoint: Path, output_dir: Path, num_steps: int):
    cfg = OmegaConf.load(config_path)
    if cfg.get("eval_overrides") is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)

    with open_dict(cfg):
        cfg.headless = True
        cfg.num_envs = 1
        cfg.use_wandb = False
        cfg.auto_load_latest = False
        cfg.checkpoint = str(checkpoint)
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.env.config.headless = True
        cfg.env.config.num_envs = 1
        cfg.env.config.max_episode_length_s = max(float(cfg.env.config.max_episode_length_s), 100000.0)
        cfg.env.config.save_rendering_dir = str(output_dir)
        cfg.env.config.ckpt_dir = str(checkpoint.parent)
        cfg.env.config.auto_record = True
        cfg.env.config.auto_record_num_frames = int(num_steps)
        cfg.env.config.offscreen_record = True
        cfg.env.config.offscreen_record_width = 1600
        cfg.env.config.offscreen_record_height = 900
        cfg.env.config.offscreen_record_fps = 50
        cfg.algo.config.eval_steps = int(num_steps)

    pre_process_config(cfg)
    return cfg


def compute_ref_dof(env):
    motion_times = env.episode_length_buf.float() * env.dt + env.motion_start_times
    motion_res = env._motion_lib.get_motion_state(env.motion_ids, motion_times, offset=env.env_origins)
    return motion_times.clone(), motion_res["dof_pos"].clone()


def collect_state(env):
    motion_times, ref_dof = compute_ref_dof(env)
    return {
        "motion_time": scalar_float(motion_times[0]),
        "ref_dof": tensor_to_cpu(ref_dof[0]).float(),
        "current_dof": tensor_to_cpu(env.simulator.dof_pos[0]).float(),
        "root_height": scalar_float(env.simulator.robot_root_states[0, 2]),
        "episode_length": int(tensor_to_cpu(env.episode_length_buf[0]).item()),
        "upper_body_diff_norm": scalar_float(env.log_dict.get("upper_body_diff_norm", 0.0)),
        "lower_body_diff_norm": scalar_float(env.log_dict.get("lower_body_diff_norm", 0.0)),
        "joint_pos_diff_norm": scalar_float(env.log_dict.get("joint_pos_diff_norm", 0.0)),
        "action_clip_frac": scalar_float(env.log_dict.get("action_clip_frac", 0.0)),
    }


def run_case(case_name, cfg, checkpoint, alpha, output_dir):
    device = cfg.device
    env = instantiate(cfg.env, device=device)
    algo = instantiate(cfg.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(checkpoint))
    algo._eval_mode()
    if hasattr(env, "set_is_evaluating"):
        env.set_is_evaluating()

    eval_policy = algo._get_inference_policy()
    frozen_policy = getattr(getattr(algo, "loaded_policy", None), "eval_policy", None)

    clip_action_limit = float(env.config.robot.control.action_clip_value)
    lower_body_dim = int(getattr(env.config.robot, "lower_body_actions_dim", 15))
    ankle_indices = [4, 5, 10, 11]

    obs_dict = env.reset_all()
    prev_state = collect_state(env)
    rows = []

    for step in range(int(cfg.algo.config.eval_steps)):
        with torch.no_grad():
            trainable_action = eval_policy(obs_dict["actor_obs"])
            trainable_clipped = torch.clamp(trainable_action, -clip_action_limit, clip_action_limit)

            if frozen_policy is None:
                frozen_raw = torch.zeros_like(trainable_action)
                frozen_scaled = frozen_raw
            else:
                frozen_raw = frozen_policy(obs_dict["closed_loop_actor_obs"]).detach()
                frozen_scaled = frozen_raw * float(alpha)

            frozen_clipped = torch.clamp(frozen_scaled, -clip_action_limit, clip_action_limit)
            final_action = trainable_clipped + frozen_clipped

            if frozen_policy is None:
                actor_state = {"actions": trainable_action}
            else:
                actor_state = {
                    "actions": trainable_action,
                    "actions_closed_loop": frozen_scaled,
                }

            obs_dict, rewards, dones, infos = env.step(actor_state)

        current_state = collect_state(env)
        ref_dof_delta = torch.linalg.norm(current_state["ref_dof"] - prev_state["ref_dof"]).item()
        current_dof_delta = torch.linalg.norm(current_state["current_dof"] - prev_state["current_dof"]).item()

        trainable_mean_abs, trainable_max_abs = mean_abs_and_max_abs(trainable_clipped[0])
        lower_mean_abs, lower_max_abs = mean_abs_and_max_abs(final_action[0, :lower_body_dim])
        ankle_mean_abs, ankle_max_abs = mean_abs_and_max_abs(final_action[0, ankle_indices])
        frozen_mean_abs, frozen_max_abs = mean_abs_and_max_abs(frozen_clipped[0])
        final_mean_abs, final_max_abs = mean_abs_and_max_abs(final_action[0])

        row = {
            "step": step,
            "done": int(tensor_to_cpu(dones[0]).item()),
            "reward": scalar_float(rewards[0]),
            "motion_time": current_state["motion_time"],
            "ref_dof_delta_l2": float(ref_dof_delta),
            "current_dof_delta_l2": float(current_dof_delta),
            "trainable_policy_action_mean_abs": trainable_mean_abs,
            "trainable_policy_action_max_abs": trainable_max_abs,
            "lower_body_action_mean_abs": lower_mean_abs,
            "lower_body_action_max_abs": lower_max_abs,
            "ankle_action_mean_abs": ankle_mean_abs,
            "ankle_action_max_abs": ankle_max_abs,
            "frozen_delta_action_mean_abs": frozen_mean_abs,
            "frozen_delta_action_max_abs": frozen_max_abs,
            "final_action_mean_abs": final_mean_abs,
            "final_action_max_abs": final_max_abs,
            "action_clip_frac": current_state["action_clip_frac"],
            "root_height": current_state["root_height"],
            "episode_length": current_state["episode_length"],
            "upper_body_diff_norm": current_state["upper_body_diff_norm"],
            "lower_body_diff_norm": current_state["lower_body_diff_norm"],
            "joint_pos_diff_norm": current_state["joint_pos_diff_norm"],
        }
        rows.append(row)
        prev_state = current_state

    if hasattr(env, "simulator") and hasattr(env.simulator, "finalize_recording"):
        env.simulator.finalize_recording()

    video_path = None
    if hasattr(env.simulator, "offscreen_video_path"):
        video_path = env.simulator.offscreen_video_path

    return {
        "case_name": case_name,
        "env_target": cfg.env._target_,
        "checkpoint": str(checkpoint),
        "policy_checkpoint": str(getattr(cfg.algo.config, "policy_checkpoint", "")),
        "alpha": float(alpha),
        "randomize_ctrl_delay": bool(cfg.domain_rand.randomize_ctrl_delay),
        "clip_action_value": clip_action_limit,
        "video_path": video_path,
        "rows": rows,
    }


def print_case_report(report):
    rows = report["rows"]
    print(f"\n=== {report['case_name']} ===")
    print(f"env={report['env_target']}")
    print(f"checkpoint={report['checkpoint']}")
    if report["policy_checkpoint"]:
        print(f"policy_checkpoint={report['policy_checkpoint']}")
    print(f"alpha={report['alpha']:.3f}")
    print(f"randomize_ctrl_delay={int(report['randomize_ctrl_delay'])}")
    if report["video_path"]:
        print(f"video_path={report['video_path']}")

    def print_series(key):
        vals = [row[key] for row in rows[:20]]
        formatted = ", ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)
        print(f"{key}[0:20]={formatted}")

    print_series("motion_time")
    print_series("ref_dof_delta_l2")
    print_series("current_dof_delta_l2")
    print_series("root_height")
    print_series("episode_length")
    print_series("upper_body_diff_norm")
    print_series("lower_body_diff_norm")

    for key in [
        "trainable_policy_action_mean_abs",
        "trainable_policy_action_max_abs",
        "lower_body_action_mean_abs",
        "lower_body_action_max_abs",
        "ankle_action_mean_abs",
        "ankle_action_max_abs",
        "frozen_delta_action_mean_abs",
        "frozen_delta_action_max_abs",
        "final_action_mean_abs",
        "final_action_max_abs",
        "action_clip_frac",
    ]:
        series = [row[key] for row in rows]
        print(
            f"{key}: first={series[0]:.6f} mean={sum(series) / len(series):.6f} max={max(series):.6f}"
        )


def save_report(report, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    csv_path = output_dir / f"{report['case_name']}.csv"
    json_path = output_dir / f"{report['case_name']}.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        payload = {k: v for k, v in report.items() if k != "rows"}
        payload["rows"] = rows
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="0-update baseline vs closed-loop rollout diagnostic.")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Which cases to run: A=baseline, B=closed-loop alpha0, C=closed-loop alpha1",
    )
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CFG)
    parser.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--closed-loop-config", type=Path, default=DEFAULT_CLOSED_LOOP_CFG)
    parser.add_argument("--closed-loop-main-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--delta-ckpt", type=Path, default=DEFAULT_DELTA_CKPT)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "WARNING").upper())

    cases = []

    if "A" in args.cases:
        baseline_out = args.output_dir / "A_baseline"
        cfg_a = load_config(args.baseline_config, args.baseline_ckpt, baseline_out, args.num_steps)
        cases.append(("A_baseline", cfg_a, args.baseline_ckpt, 0.0, baseline_out))

    if "B" in args.cases:
        case_out = args.output_dir / "B_closed_loop_alpha0"
        cfg_b = load_config(args.closed_loop_config, args.closed_loop_main_ckpt, case_out, args.num_steps)
        with open_dict(cfg_b):
            cfg_b.algo.config.policy_checkpoint = str(args.delta_ckpt)
        cases.append(("B_closed_loop_alpha0", cfg_b, args.closed_loop_main_ckpt, 0.0, case_out))

    if "C" in args.cases:
        case_out = args.output_dir / "C_closed_loop_alpha1"
        cfg_c = load_config(args.closed_loop_config, args.closed_loop_main_ckpt, case_out, args.num_steps)
        with open_dict(cfg_c):
            cfg_c.algo.config.policy_checkpoint = str(args.delta_ckpt)
        cases.append(("C_closed_loop_alpha1", cfg_c, args.closed_loop_main_ckpt, 1.0, case_out))

    for case_name, cfg, checkpoint, alpha, output_dir in cases:
        report = run_case(case_name, cfg, checkpoint, alpha, output_dir)
        csv_path, json_path = save_report(report, output_dir)
        print_case_report(report)
        print(f"saved_csv={csv_path}")
        print(f"saved_json={json_path}")


if __name__ == "__main__":
    main()

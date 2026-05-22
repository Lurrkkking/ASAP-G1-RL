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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "closed_loop_scale0_sanity"
DEFAULT_MOTION_FILE = (
    REPO_ROOT
    / "humanoidverse"
    / "data"
    / "motions"
    / "g1_29dof_anneal_23dof"
    / "TairanTestbed"
    / "singles"
    / "0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
)
DEFAULT_BASELINE_CFG = REPO_ROOT / "logs" / "TEST_CR7_Siuuu" / "baseline13000group" / "config.yaml"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "logs" / "TEST_CR7_Siuuu" / "baseline13000group" / "model_13000.pt"
DEFAULT_CLOSED_LOOP_CFG = (
    REPO_ROOT
    / "logs"
    / "DeltaA_MuJoCo"
    / "20260522_100923-closedloop_cr7_ankle_delta6000_scale0p5_smoke-delta_a-g1_29dof_anneal_23dof"
    / "config.yaml"
)
DEFAULT_DELTA_CKPT = (
    REPO_ROOT
    / "logs"
    / "DeltaA_MuJoCo"
    / "20260521_205236-openloop_mujoco_cr7_ankle_only_smoke-delta_a-g1_29dof_anneal_23dof"
    / "model_6000.pt"
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


def mean_value(rows, key):
    return float(sum(row[key] for row in rows) / max(len(rows), 1))


def max_value(rows, key):
    return float(max(row[key] for row in rows))


def min_value(rows, key):
    return float(min(row[key] for row in rows))


def load_config(config_path: Path, checkpoint: Path, output_dir: Path, num_steps: int, motion_file: Path):
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
        if "robot" in cfg and "asset" in cfg.robot:
            asset_root = Path(str(cfg.robot.asset.asset_root))
            if not asset_root.is_absolute():
                abs_asset_root = (REPO_ROOT / asset_root).resolve()
                cfg.robot.asset.asset_root = str(abs_asset_root)
                if "env" in cfg and "config" in cfg.env and "robot" in cfg.env.config and "asset" in cfg.env.config.robot:
                    cfg.env.config.robot.asset.asset_root = str(abs_asset_root)
        if "robot" in cfg and "motion" in cfg.robot:
            cfg.robot.motion.motion_file = str(motion_file)
            if "asset" in cfg.robot and "motion" in cfg.robot.motion and "asset" in cfg.robot.motion:
                motion_asset_root = Path(str(cfg.robot.motion.asset.assetRoot))
                if not motion_asset_root.is_absolute():
                    abs_motion_asset_root = (REPO_ROOT / motion_asset_root).resolve()
                    cfg.robot.motion.asset.assetRoot = str(abs_motion_asset_root)
        if "env" in cfg and "config" in cfg.env and "robot" in cfg.env.config and "motion" in cfg.env.config.robot:
            cfg.env.config.robot.motion.motion_file = str(motion_file)
            if "asset" in cfg.env.config.robot.motion:
                motion_asset_root = Path(str(cfg.env.config.robot.motion.asset.assetRoot))
                if not motion_asset_root.is_absolute():
                    abs_motion_asset_root = (REPO_ROOT / motion_asset_root).resolve()
                    cfg.env.config.robot.motion.asset.assetRoot = str(abs_motion_asset_root)

    pre_process_config(cfg)
    return cfg


def collect_state(env):
    return {
        "root_height": scalar_float(env.simulator.robot_root_states[0, 2]),
        "episode_length": int(tensor_to_cpu(env.episode_length_buf[0]).item()),
        "upper_body_diff_norm": scalar_float(env.log_dict.get("upper_body_diff_norm", 0.0)),
        "lower_body_diff_norm": scalar_float(env.log_dict.get("lower_body_diff_norm", 0.0)),
        "joint_pos_diff_norm": scalar_float(env.log_dict.get("joint_pos_diff_norm", 0.0)),
        "action_clip_frac": scalar_float(
            env.log_dict.get("closed_loop/final_action_clip_frac", env.log_dict.get("action_clip_frac", 0.0))
        ),
    }


def build_summary(rows):
    done_steps = next((row["step"] + 1 for row in rows if row["done"] > 0), len(rows))
    return {
        "done_steps": int(done_steps),
        "reward_mean": mean_value(rows, "reward"),
        "reward_min": min_value(rows, "reward"),
        "root_height_min": min_value(rows, "root_height"),
        "root_height_max": max_value(rows, "root_height"),
        "upper_body_diff_norm": mean_value(rows, "upper_body_diff_norm"),
        "lower_body_diff_norm": mean_value(rows, "lower_body_diff_norm"),
        "joint_pos_diff_norm": mean_value(rows, "joint_pos_diff_norm"),
        "policy_action_mean_abs": mean_value(rows, "policy_action_mean_abs"),
        "policy_action_max_abs": max_value(rows, "policy_action_max_abs"),
        "final_action_mean_abs": mean_value(rows, "final_action_mean_abs"),
        "final_action_max_abs": max_value(rows, "final_action_max_abs"),
        "frozen_delta_masked_mean_abs": mean_value(rows, "frozen_delta_masked_mean_abs"),
        "delta_over_policy_mean": mean_value(rows, "delta_over_policy_mean"),
        "action_clip_frac": mean_value(rows, "action_clip_frac"),
        "scale_zero_final_equals_main": bool(all(row["scale_zero_final_equals_main"] for row in rows)),
    }


def run_case(case_name, cfg, checkpoint: Path, delta_action_scale: float):
    env = instantiate(cfg.env, device=cfg.device)
    algo = instantiate(cfg.algo, env=env, device=cfg.device, log_dir=None)
    algo.setup()
    algo.load(str(checkpoint))
    algo._eval_mode()
    if hasattr(env, "set_is_evaluating"):
        env.set_is_evaluating()

    eval_policy = algo._get_inference_policy()
    frozen_policy = getattr(getattr(algo, "loaded_policy", None), "eval_policy", None)
    delta_action_mask = getattr(algo, "delta_action_mask", None)
    if delta_action_mask is None:
        delta_action_mask = torch.ones(env.dim_actions, device=cfg.device)

    obs_dict = env.reset_all()
    rows = []

    for step in range(int(cfg.algo.config.eval_steps)):
        with torch.no_grad():
            main_policy_action = eval_policy(obs_dict["actor_obs"]).detach()
            if frozen_policy is None:
                frozen_delta_raw = torch.zeros_like(main_policy_action)
            else:
                frozen_delta_raw = frozen_policy(obs_dict["closed_loop_actor_obs"]).detach()

            frozen_delta_masked = frozen_delta_raw * delta_action_mask
            frozen_delta_scaled = frozen_delta_masked * float(delta_action_scale)
            final_action = main_policy_action + frozen_delta_scaled

            scale_zero_equal = True
            if abs(delta_action_scale) <= 1e-12:
                scale_zero_equal = torch.allclose(final_action, main_policy_action, atol=1e-6)
                assert scale_zero_equal, (
                    f"{case_name}: scale=0 but final_action != main_policy_action. "
                    f"max_abs_diff={(final_action - main_policy_action).abs().max().item():.9f}"
                )

            actor_state = {"actions": main_policy_action}
            if frozen_policy is not None:
                actor_state["actions_closed_loop"] = frozen_delta_scaled
            obs_dict, rewards, dones, infos = env.step(actor_state)

        state = collect_state(env)
        policy_mean_abs, policy_max_abs = mean_abs_and_max_abs(main_policy_action[0])
        final_mean_abs, final_max_abs = mean_abs_and_max_abs(final_action[0])
        frozen_masked_mean_abs, _ = mean_abs_and_max_abs(frozen_delta_masked[0])
        delta_over_policy_mean = float(
            (frozen_delta_scaled.abs() / (main_policy_action.abs() + 1e-6)).mean().item()
        )

        rows.append(
            {
                "step": int(step),
                "done": int(tensor_to_cpu(dones[0]).item()),
                "reward": scalar_float(rewards[0]),
                "root_height": state["root_height"],
                "upper_body_diff_norm": state["upper_body_diff_norm"],
                "lower_body_diff_norm": state["lower_body_diff_norm"],
                "joint_pos_diff_norm": state["joint_pos_diff_norm"],
                "policy_action_mean_abs": policy_mean_abs,
                "policy_action_max_abs": policy_max_abs,
                "final_action_mean_abs": final_mean_abs,
                "final_action_max_abs": final_max_abs,
                "frozen_delta_masked_mean_abs": frozen_masked_mean_abs,
                "delta_over_policy_mean": delta_over_policy_mean,
                "action_clip_frac": state["action_clip_frac"],
                "scale_zero_final_equals_main": bool(scale_zero_equal),
            }
        )
        if int(tensor_to_cpu(dones[0]).item()) > 0:
            break

    if hasattr(env, "simulator") and hasattr(env.simulator, "finalize_recording"):
        env.simulator.finalize_recording()

    video_path = getattr(env.simulator, "offscreen_video_path", None)
    return {
        "case_name": case_name,
        "env_target": cfg.env._target_,
        "checkpoint": str(checkpoint),
        "policy_checkpoint": str(getattr(cfg.algo.config, "policy_checkpoint", "")),
        "delta_action_scale": float(delta_action_scale),
        "delta_action_mask_mode": str(getattr(cfg.algo.config, "delta_action_mask_mode", "none")),
        "video_path": video_path,
        "summary": build_summary(rows),
        "rows": rows,
    }


def save_report(report, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{report['case_name']}.csv"
    json_path = output_dir / f"{report['case_name']}.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["rows"])

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return csv_path, json_path


def print_summary(report):
    summary = report["summary"]
    print(f"\n=== {report['case_name']} ===")
    print(f"env={report['env_target']}")
    print(f"checkpoint={report['checkpoint']}")
    if report["policy_checkpoint"]:
        print(f"policy_checkpoint={report['policy_checkpoint']}")
    print(f"delta_action_scale={report['delta_action_scale']:.6f}")
    print(f"delta_action_mask_mode={report['delta_action_mask_mode']}")
    print(f"done_steps={summary['done_steps']}")
    print(f"reward_mean={summary['reward_mean']:.6f}")
    print(f"reward_min={summary['reward_min']:.6f}")
    print(f"root_height_min={summary['root_height_min']:.6f}")
    print(f"root_height_max={summary['root_height_max']:.6f}")
    print(f"upper_body_diff_norm={summary['upper_body_diff_norm']:.6f}")
    print(f"lower_body_diff_norm={summary['lower_body_diff_norm']:.6f}")
    print(f"joint_pos_diff_norm={summary['joint_pos_diff_norm']:.6f}")
    print(f"policy_action_mean_abs={summary['policy_action_mean_abs']:.6f}")
    print(f"policy_action_max_abs={summary['policy_action_max_abs']:.6f}")
    print(f"final_action_mean_abs={summary['final_action_mean_abs']:.6f}")
    print(f"final_action_max_abs={summary['final_action_max_abs']:.6f}")
    print(f"frozen_delta_masked_mean_abs={summary['frozen_delta_masked_mean_abs']:.6f}")
    print(f"delta_over_policy_mean={summary['delta_over_policy_mean']:.6f}")
    print(f"action_clip_frac={summary['action_clip_frac']:.6f}")
    print(f"scale_zero_final_equals_main={int(summary['scale_zero_final_equals_main'])}")
    if report["video_path"]:
        print(f"video_path={report['video_path']}")


def build_comparison(baseline_report, closed_loop_report):
    baseline = baseline_report["summary"]
    closed_loop = closed_loop_report["summary"]
    comparison = {}
    for key in baseline.keys():
        if isinstance(baseline[key], bool):
            comparison[key] = {
                "baseline": bool(baseline[key]),
                "closed_loop_scale0": bool(closed_loop[key]),
            }
        else:
            comparison[key] = {
                "baseline": float(baseline[key]),
                "closed_loop_scale0": float(closed_loop[key]),
                "closed_loop_minus_baseline": float(closed_loop[key] - baseline[key]),
            }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Closed-loop delta_action_scale=0 sanity check.")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CFG)
    parser.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--closed-loop-config", type=Path, default=DEFAULT_CLOSED_LOOP_CFG)
    parser.add_argument("--closed-loop-main-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--delta-ckpt", type=Path, default=DEFAULT_DELTA_CKPT)
    parser.add_argument("--delta-action-scale", type=float, default=0.0)
    parser.add_argument("--delta-action-mask-mode", type=str, default="ankle_only")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "WARNING").upper())

    baseline_out = args.output_dir / "baseline"
    baseline_cfg = load_config(args.baseline_config, args.baseline_ckpt, baseline_out, args.num_steps, args.motion_file)
    baseline_report = run_case("baseline_motion_tracking", baseline_cfg, args.baseline_ckpt, delta_action_scale=0.0)
    baseline_csv, baseline_json = save_report(baseline_report, baseline_out)
    print_summary(baseline_report)
    print(f"saved_csv={baseline_csv}")
    print(f"saved_json={baseline_json}")

    closed_loop_out = args.output_dir / "closed_loop_scale0"
    closed_loop_cfg = load_config(
        args.closed_loop_config,
        args.closed_loop_main_ckpt,
        closed_loop_out,
        args.num_steps,
        args.motion_file,
    )
    with open_dict(closed_loop_cfg):
        closed_loop_cfg.algo.config.policy_checkpoint = str(args.delta_ckpt)
        closed_loop_cfg.algo.config.delta_action_scale = float(args.delta_action_scale)
        closed_loop_cfg.algo.config.delta_action_mask_mode = str(args.delta_action_mask_mode)
        if (
            "action_anchor" in closed_loop_cfg.algo.config
            and closed_loop_cfg.algo.config.action_anchor is not None
            and "enabled" in closed_loop_cfg.algo.config.action_anchor
        ):
            closed_loop_cfg.algo.config.action_anchor.enabled = False

    closed_loop_report = run_case(
        "closed_loop_scale0",
        closed_loop_cfg,
        args.closed_loop_main_ckpt,
        delta_action_scale=args.delta_action_scale,
    )
    closed_loop_csv, closed_loop_json = save_report(closed_loop_report, closed_loop_out)
    print_summary(closed_loop_report)
    print(f"saved_csv={closed_loop_csv}")
    print(f"saved_json={closed_loop_json}")

    comparison = build_comparison(baseline_report, closed_loop_report)
    comparison_path = args.output_dir / "comparison.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"comparison_json={comparison_path}")


if __name__ == "__main__":
    main()

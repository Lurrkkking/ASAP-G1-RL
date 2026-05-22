#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
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
DEFAULT_MOTION_FILE = REPO_ROOT / "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
DEFAULT_BASELINE_CFG = REPO_ROOT / "logs/TEST_CR7_Siuuu/baseline13000group/config.yaml"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "logs/TEST_CR7_Siuuu/baseline13000group/model_13000.pt"
DEFAULT_CLOSED_LOOP_CFG = REPO_ROOT / "logs/DeltaA_MuJoCo/20260522_100923-closedloop_cr7_ankle_delta6000_scale0p5_smoke-delta_a-g1_29dof_anneal_23dof/config.yaml"
DEFAULT_DELTA_CKPT = REPO_ROOT / "logs/DeltaA_MuJoCo/20260521_205236-openloop_mujoco_cr7_ankle_only_smoke-delta_a-g1_29dof_anneal_23dof/model_6000.pt"


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
        "upper_body_diff_norm": scalar_float(env.log_dict.get("upper_body_diff_norm", 0.0)),
        "lower_body_diff_norm": scalar_float(env.log_dict.get("lower_body_diff_norm", 0.0)),
        "joint_pos_diff_norm": scalar_float(env.log_dict.get("joint_pos_diff_norm", 0.0)),
        "action_clip_frac": scalar_float(
            env.log_dict.get("closed_loop/final_action_clip_frac", env.log_dict.get("action_clip_frac", 0.0))
        ),
    }


def summarize(rows):
    done_steps = next((row["step"] + 1 for row in rows if row["done"] > 0), len(rows))
    return {
        "done_steps": int(done_steps),
        "reward_mean": float(sum(r["reward"] for r in rows) / max(len(rows), 1)),
        "reward_min": float(min(r["reward"] for r in rows)),
        "root_height_min": float(min(r["root_height"] for r in rows)),
        "root_height_max": float(max(r["root_height"] for r in rows)),
        "upper_body_diff_norm": float(sum(r["upper_body_diff_norm"] for r in rows) / max(len(rows), 1)),
        "lower_body_diff_norm": float(sum(r["lower_body_diff_norm"] for r in rows) / max(len(rows), 1)),
        "joint_pos_diff_norm": float(sum(r["joint_pos_diff_norm"] for r in rows) / max(len(rows), 1)),
        "policy_action_mean_abs": float(sum(r["policy_action_mean_abs"] for r in rows) / max(len(rows), 1)),
        "policy_action_max_abs": float(max(r["policy_action_max_abs"] for r in rows)),
        "final_action_mean_abs": float(sum(r["final_action_mean_abs"] for r in rows) / max(len(rows), 1)),
        "final_action_max_abs": float(max(r["final_action_max_abs"] for r in rows)),
        "frozen_delta_masked_mean_abs": float(sum(r["frozen_delta_masked_mean_abs"] for r in rows) / max(len(rows), 1)),
        "delta_over_policy_mean": float(sum(r["delta_over_policy_mean"] for r in rows) / max(len(rows), 1)),
        "action_clip_frac": float(sum(r["action_clip_frac"] for r in rows) / max(len(rows), 1)),
        "scale_zero_final_equals_main": bool(all(r["scale_zero_final_equals_main"] for r in rows)),
    }


def run_case(case_name, cfg, checkpoint: Path, delta_action_scale: float, output_dir: Path):
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
            frozen_delta_raw = frozen_policy(obs_dict["closed_loop_actor_obs"]).detach() if frozen_policy is not None else torch.zeros_like(main_policy_action)
            frozen_delta_masked = frozen_delta_raw * delta_action_mask
            frozen_delta_scaled = frozen_delta_masked * float(delta_action_scale)
            final_action = main_policy_action + frozen_delta_scaled

            scale_zero_equal = True
            if abs(delta_action_scale) <= 1e-12:
                scale_zero_equal = torch.allclose(final_action, main_policy_action, atol=1e-6)
                assert scale_zero_equal, (
                    f"{case_name}: scale=0 but final_action != main_policy_action; "
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
        delta_over_policy_mean = float((frozen_delta_scaled.abs() / (main_policy_action.abs() + 1e-6)).mean().item())

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
    report = {
        "case_name": case_name,
        "env_target": cfg.env._target_,
        "checkpoint": str(checkpoint),
        "policy_checkpoint": str(getattr(cfg.algo.config, "policy_checkpoint", "")),
        "delta_action_scale": float(delta_action_scale),
        "delta_action_mask_mode": str(getattr(cfg.algo.config, "delta_action_mask_mode", "none")),
        "video_path": video_path,
        "summary": summarize(rows),
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{case_name}.csv"
    json_path = output_dir / f"{case_name}.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    return report, csv_path, json_path


def print_report(report, csv_path, json_path):
    s = report["summary"]
    print(f"\n=== {report['case_name']} ===")
    print(f"env={report['env_target']}")
    print(f"checkpoint={report['checkpoint']}")
    if report["policy_checkpoint"]:
        print(f"policy_checkpoint={report['policy_checkpoint']}")
    print(f"delta_action_scale={report['delta_action_scale']:.6f}")
    print(f"delta_action_mask_mode={report['delta_action_mask_mode']}")
    print(f"done_steps={s['done_steps']}")
    print(f"reward_mean={s['reward_mean']:.6f}")
    print(f"reward_min={s['reward_min']:.6f}")
    print(f"root_height_min={s['root_height_min']:.6f}")
    print(f"root_height_max={s['root_height_max']:.6f}")
    print(f"upper_body_diff_norm={s['upper_body_diff_norm']:.6f}")
    print(f"lower_body_diff_norm={s['lower_body_diff_norm']:.6f}")
    print(f"joint_pos_diff_norm={s['joint_pos_diff_norm']:.6f}")
    print(f"policy_action_mean_abs={s['policy_action_mean_abs']:.6f}")
    print(f"policy_action_max_abs={s['policy_action_max_abs']:.6f}")
    print(f"final_action_mean_abs={s['final_action_mean_abs']:.6f}")
    print(f"final_action_max_abs={s['final_action_max_abs']:.6f}")
    print(f"frozen_delta_masked_mean_abs={s['frozen_delta_masked_mean_abs']:.6f}")
    print(f"delta_over_policy_mean={s['delta_over_policy_mean']:.6f}")
    print(f"action_clip_frac={s['action_clip_frac']:.6f}")
    print(f"scale_zero_final_equals_main={int(s['scale_zero_final_equals_main'])}")
    if report["video_path"]:
        print(f"video_path={report['video_path']}")
    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")


def build_comparison(a, b):
    out = {}
    for key in a["summary"].keys():
        if isinstance(a["summary"][key], bool):
            out[key] = {"baseline": bool(a["summary"][key]), "closed_loop_scale0": bool(b["summary"][key])}
        else:
            out[key] = {
                "baseline": float(a["summary"][key]),
                "closed_loop_scale0": float(b["summary"][key]),
                "closed_loop_minus_baseline": float(b["summary"][key] - a["summary"][key]),
            }
    return out


def run_single_case(args):
    if args.single_case == "baseline":
        cfg = load_config(args.baseline_config, args.baseline_ckpt, args.output_dir, args.num_steps, args.motion_file)
        report, csv_path, json_path = run_case("baseline_motion_tracking", cfg, args.baseline_ckpt, 0.0, args.output_dir)
    else:
        cfg = load_config(args.closed_loop_config, args.closed_loop_main_ckpt, args.output_dir, args.num_steps, args.motion_file)
        with open_dict(cfg):
            cfg.algo.config.policy_checkpoint = str(args.delta_ckpt)
            cfg.algo.config.delta_action_scale = float(args.delta_action_scale)
            cfg.algo.config.delta_action_mask_mode = str(args.delta_action_mask_mode)
            if "action_anchor" in cfg.algo.config and cfg.algo.config.action_anchor is not None:
                cfg.algo.config.action_anchor.enabled = False
        report, csv_path, json_path = run_case("closed_loop_scale0", cfg, args.closed_loop_main_ckpt, args.delta_action_scale, args.output_dir)
    print_report(report, csv_path, json_path)


def spawn_case(args, mode, case_name, output_dir):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case",
        mode,
        "--output-dir",
        str(output_dir),
        "--case-name",
        case_name,
        "--num-steps",
        str(args.num_steps),
        "--motion-file",
        str(args.motion_file),
        "--baseline-config",
        str(args.baseline_config),
        "--baseline-ckpt",
        str(args.baseline_ckpt),
        "--closed-loop-config",
        str(args.closed_loop_config),
        "--closed-loop-main-ckpt",
        str(args.closed_loop_main_ckpt),
        "--delta-ckpt",
        str(args.delta_ckpt),
        "--delta-action-scale",
        str(args.delta_action_scale),
        "--delta-action-mask-mode",
        str(args.delta_action_mask_mode),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} case failed with exit code {proc.returncode}")
    json_path = output_dir / f"{case_name}.json"
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Closed-loop delta_action_scale=0 sanity check.")
    parser.add_argument("--single-case", choices=["baseline", "closed_loop"], default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-name", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=50)
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

    if args.single_case is not None:
        run_single_case(args)
        return

    baseline_out = args.output_dir / "baseline"
    closed_loop_out = args.output_dir / "closed_loop_scale0"
    baseline_report = spawn_case(args, "baseline", "baseline_motion_tracking", baseline_out)
    closed_loop_report = spawn_case(args, "closed_loop", "closed_loop_scale0", closed_loop_out)

    comparison = build_comparison(baseline_report, closed_loop_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"comparison_json={comparison_path}")


if __name__ == "__main__":
    main()

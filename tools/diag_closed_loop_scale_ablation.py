#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

if os.environ.get('OMP_NUM_THREADS') in {'', '0'}:
    os.environ['OMP_NUM_THREADS'] = '1'

try:
    import ninja
    os.environ['PATH'] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get('PATH', '')
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'tmp' / 'closed_loop_scale_ablation'
DEFAULT_BASELINE_CFG = REPO_ROOT / 'logs' / 'TEST_CR7_Siuuu' / 'baseline13000group' / 'config.yaml'
DEFAULT_BASELINE_CKPT = REPO_ROOT / 'logs' / 'TEST_CR7_Siuuu' / 'baseline13000group' / 'model_13000.pt'
DEFAULT_CLOSED_LOOP_CFG = REPO_ROOT / 'logs' / 'DeltaA_MuJoCo' / '20260522_100923-closedloop_cr7_ankle_delta6000_scale0p5_smoke-delta_a-g1_29dof_anneal_23dof' / 'config.yaml'
DEFAULT_DELTA_CKPT = REPO_ROOT / 'logs' / 'DeltaA_MuJoCo' / '20260521_205236-openloop_mujoco_cr7_ankle_only_smoke-delta_a-g1_29dof_anneal_23dof' / 'model_6000.pt'


def tensor_to_cpu(x):
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
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
    if cfg.get('eval_overrides') is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
    with open_dict(cfg):
        cfg.headless = True
        cfg.num_envs = 1
        cfg.use_wandb = False
        cfg.auto_load_latest = False
        cfg.checkpoint = str(checkpoint)
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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


def collect_state(env):
    return {
        'root_height': scalar_float(env.simulator.robot_root_states[0, 2]),
        'episode_length': int(tensor_to_cpu(env.episode_length_buf[0]).item()),
        'upper_body_diff_norm': scalar_float(env.log_dict.get('upper_body_diff_norm', 0.0)),
        'lower_body_diff_norm': scalar_float(env.log_dict.get('lower_body_diff_norm', 0.0)),
        'joint_pos_diff_norm': scalar_float(env.log_dict.get('joint_pos_diff_norm', 0.0)),
        'action_clip_frac': scalar_float(env.log_dict.get('action_clip_frac', 0.0)),
    }


def run_case(case_name, cfg, checkpoint, alpha, output_dir):
    env = instantiate(cfg.env, device=cfg.device)
    algo = instantiate(cfg.algo, env=env, device=cfg.device, log_dir=None)
    algo.setup()
    algo.load(str(checkpoint))
    algo._eval_mode()
    if hasattr(env, 'set_is_evaluating'):
        env.set_is_evaluating()

    eval_policy = algo._get_inference_policy()
    frozen_policy = getattr(getattr(algo, 'loaded_policy', None), 'eval_policy', None)
    clip_action_limit = float(env.config.robot.control.action_clip_value)
    lower_body_dim = int(getattr(env.config.robot, 'lower_body_actions_dim', 15))
    ankle_indices = [4, 5, 10, 11]
    hip_indices = list(range(12))

    obs_dict = env.reset_all()
    rows = []
    for step in range(int(cfg.algo.config.eval_steps)):
        with torch.no_grad():
            trainable_action = eval_policy(obs_dict['actor_obs'])
            trainable_clipped = torch.clamp(trainable_action, -clip_action_limit, clip_action_limit)
            if frozen_policy is None:
                frozen_raw = torch.zeros_like(trainable_action)
                frozen_scaled = frozen_raw
            else:
                frozen_raw = frozen_policy(obs_dict['closed_loop_actor_obs']).detach()
                frozen_scaled = frozen_raw * float(alpha)
            final_action = trainable_clipped + frozen_scaled
            actor_state = {'actions': trainable_action, 'actions_closed_loop': frozen_scaled} if frozen_policy is not None else {'actions': trainable_action}
            obs_dict, rewards, dones, infos = env.step(actor_state)

        state = collect_state(env)
        row = {
            'step': step,
            'done': int(tensor_to_cpu(dones[0]).item()),
            'reward': scalar_float(rewards[0]),
            'root_height': state['root_height'],
            'episode_length': state['episode_length'],
            'upper_body_diff_norm': state['upper_body_diff_norm'],
            'lower_body_diff_norm': state['lower_body_diff_norm'],
            'joint_pos_diff_norm': state['joint_pos_diff_norm'],
            'action_clip_frac': state['action_clip_frac'],
            'trainable_policy_action_mean_abs': mean_abs_and_max_abs(trainable_clipped[0])[0],
            'frozen_delta_action_mean_abs': mean_abs_and_max_abs(frozen_scaled[0])[0],
            'final_action_mean_abs': mean_abs_and_max_abs(final_action[0])[0],
            'final_action_max_abs': mean_abs_and_max_abs(final_action[0])[1],
            'delta_over_policy_mean': float((frozen_scaled.abs() / (trainable_clipped.abs() + 1e-6)).mean().item()),
            'lower_body_action_mean_abs': mean_abs_and_max_abs(final_action[0, :lower_body_dim])[0],
            'ankle_action_mean_abs': mean_abs_and_max_abs(final_action[0, ankle_indices])[0],
            'hip_action_mean_abs': mean_abs_and_max_abs(final_action[0, hip_indices])[0],
        }
        rows.append(row)
        if int(tensor_to_cpu(dones[0]).item()) > 0:
            break

    return {
        'case_name': case_name,
        'checkpoint': str(checkpoint),
        'policy_checkpoint': str(getattr(cfg.algo.config, 'policy_checkpoint', '')),
        'alpha': float(alpha),
        'rows': rows,
    }


def save_report(report, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{report['case_name']}.csv"
    json_path = output_dir / f"{report['case_name']}.json"
    rows = report['rows']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w') as f:
        payload = {k: v for k, v in report.items() if k != 'rows'}
        payload['rows'] = rows
        json.dump(payload, f, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--baseline-config', type=Path, default=DEFAULT_BASELINE_CFG)
    parser.add_argument('--baseline-ckpt', type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument('--closed-loop-config', type=Path, default=DEFAULT_CLOSED_LOOP_CFG)
    parser.add_argument('--closed-loop-main-ckpt', type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument('--delta-ckpt', type=Path, default=DEFAULT_DELTA_CKPT)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.1, 0.25, 0.5])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get('LOGURU_LEVEL', 'WARNING').upper())

    reports = []
    for alpha in args.alphas:
        case_name = f'scale_{str(alpha).replace(".", "p")}'
        case_out = args.output_dir / case_name
        cfg = load_config(args.closed_loop_config, args.closed_loop_main_ckpt, case_out, args.num_steps)
        with open_dict(cfg):
            cfg.algo.config.policy_checkpoint = str(args.delta_ckpt)
        report = run_case(case_name, cfg, args.closed_loop_main_ckpt, alpha, case_out)
        csv_path, json_path = save_report(report, case_out)
        print(f'case={case_name} csv={csv_path} json={json_path}')
        reports.append(report)

    summary_path = args.output_dir / 'summary.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(reports, f, indent=2)
    print(f'summary={summary_path}')


if __name__ == '__main__':
    main()

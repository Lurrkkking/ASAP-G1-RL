#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humanoidverse.eval_deltaa_openloop_deterministic import compute_paper_metrics, compute_paper_summary
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.config_utils import *  # noqa: F401,F403
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mujoco_simulation.fdd_asap_sim2sim import (
    ACTION_CLIP_VALUE,
    ACTION_SCALE,
    ANKLE_INDICES,
    HIP_KNEE_INDICES,
    LOWER_BODY_INDICES,
    create_history,
    get_obs,
    pd_control,
)

DEFAULT_MOTION_FILE = REPO_ROOT / 'humanoidverse' / 'data' / 'motions' / 'g1_29dof_anneal_23dof' / 'TairanTestbed' / 'singles' / '0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl'
DEFAULT_MUJOCO_CFG = REPO_ROOT / 'mujoco_simulation' / 'g1_config' / 'mujoco_config.yaml'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'tmp' / 'closed_loop_eval_table'


def parse_case(spec: str):
    parts = spec.split('|')
    if len(parts) != 3:
        raise ValueError('case format must be name|checkpoint|scale')
    return {'name': parts[0], 'checkpoint': Path(parts[1]), 'scale': float(parts[2])}


def export_onnx(checkpoint: Path) -> Path:
    export_dir = checkpoint.parent / 'exported'
    onnx_path = export_dir / f'{checkpoint.stem}.onnx'
    if onnx_path.is_file():
        return onnx_path
    cmd = [sys.executable, str(REPO_ROOT / 'humanoidverse' / 'export_pt_to_onnx.py'), f'checkpoint={checkpoint}', '+simulator=isaacgym', '+device=cuda:0', '+headless=true']
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    return onnx_path


def load_eval_cfg(checkpoint: Path, motion_file: Path):
    cfg_path = checkpoint.parent / 'config.yaml'
    if not cfg_path.is_file():
        cfg_path = checkpoint.parent.parent / 'config.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Could not find config.yaml near {checkpoint}')
    cfg = OmegaConf.load(cfg_path)
    if cfg.get('eval_overrides') is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
    cfg.headless = True
    cfg.num_envs = 1
    cfg.use_wandb = False
    cfg.auto_load_latest = False
    cfg.checkpoint = str(checkpoint)
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.env.config.headless = True
    cfg.env.config.num_envs = 1
    cfg.env.config.max_episode_length_s = max(float(cfg.env.config.max_episode_length_s), 100000.0)
    cfg.env.config.save_rendering_dir = str(checkpoint.parent / 'renderings_eval')
    cfg.env.config.ckpt_dir = str(checkpoint.parent)
    cfg.env.config.auto_record = True
    cfg.env.config.auto_record_num_frames = 50
    cfg.env.config.offscreen_record = True
    cfg.env.config.offscreen_record_width = 1600
    cfg.env.config.offscreen_record_height = 900
    cfg.env.config.offscreen_record_fps = 50
    cfg.env.config.robot.motion.motion_file = str(motion_file)
    pre_process_config(cfg)
    return cfg


def tensor_to_np(x):
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def build_body_ids(model):
    ids = []
    for i in range(1, model.nbody):
        ids.append(i)
    return np.asarray(ids, dtype=np.int32)


def mean_abs_and_max_abs(x):
    x = tensor_to_np(x).astype(np.float32)
    return float(np.abs(x).mean()), float(np.abs(x).max())


def rollout_case(case, baseline_session, baseline_input_name, baseline_output_name, ref_env, mj_cfg, output_dir):
    onnx_path = export_onnx(case['checkpoint'])
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    model = mujoco.MjModel.from_xml_path(str(mj_cfg.xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = mj_cfg.simulation_dt
    if mj_cfg.solver_iterations > 0:
        model.opt.iterations = mj_cfg.solver_iterations
    if hasattr(model.opt, 'ls_iterations') and mj_cfg.solver_ls_iterations > 0:
        model.opt.ls_iterations = mj_cfg.solver_ls_iterations
    data.qpos[-mj_cfg.num_actions:] = mj_cfg.default_dof_pos
    mujoco.mj_step(model, data)

    body_ids = build_body_ids(model)
    import imageio.v2 as imageio
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{case['name']}.mp4"
    renderer = mujoco.Renderer(model, height=900, width=1600)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.distance = 2.8; cam.azimuth = 135.0; cam.elevation = -12.0

    hist_dict, hist_obs_c = create_history(mj_cfg)
    action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    prev_final_action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    rows = []
    frames = []
    sim_body_pos_frames = []
    target_body_pos_frames = []
    sim_root_pos_frames = []
    target_root_pos_frames = []
    sim_root_vel_frames = []
    target_root_vel_frames = []
    per_joint_diff_accum = []

    sim_steps = int(mj_cfg.simulation_duration / mj_cfg.simulation_dt)
    stop_reason = 'motion_end'
    fall_step = None
    for step in range(sim_steps):
        mj = {'mujoco_dof_pos': data.qpos[7:].copy().astype(np.float32), 'mujoco_dof_vel': data.qvel[6:].copy().astype(np.float32), 'mujoco_base_angvel': data.qvel[3:6].copy().astype(np.float32), 'mujoco_gvec': np.array([0.0, 0.0, -1.0], dtype=np.float32)}
        obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mj, action, step, mj_cfg)
        actor_obs = obs_buff.astype(np.float32)

        case_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)
        baseline_action = baseline_session.run([baseline_output_name], {baseline_input_name: actor_obs})[0][0].astype(np.float32)
        case_action = np.clip(case_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        baseline_action = np.clip(baseline_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        final_action = case_action

        target_dof_pos = final_action * ACTION_SCALE + mj_cfg.default_dof_pos
        baseline_target_dof_pos = baseline_action * ACTION_SCALE + mj_cfg.default_dof_pos
        action_rate = float(np.mean(np.abs(final_action - prev_final_action)))
        action_clip_frac = float(np.mean(np.abs(final_action) >= ACTION_CLIP_VALUE))
        action_diff = final_action - baseline_action
        q_target_diff = target_dof_pos - baseline_target_dof_pos

        data.ctrl[:] = pd_control(target_dof_pos, mj['mujoco_dof_pos'], np.zeros_like(mj_cfg.kds), mj['mujoco_dof_vel'], mj_cfg)
        data.ctrl[:] = np.clip(data.ctrl, -mj_cfg.tau_limit, mj_cfg.tau_limit)
        mujoco.mj_step(model, data)

        ref_motion_time = float((step + 1) * mj_cfg.simulation_dt)
        motion_t = torch.tensor([ref_motion_time], dtype=torch.float32, device=ref_env.device)
        motion_res = ref_env._motion_lib.get_motion_state(ref_env.motion_ids, motion_t, offset=ref_env.env_origins)

        sim_body_pos = data.xpos[body_ids].copy().astype(np.float32)
        target_body_pos = tensor_to_np(motion_res['rg_pos'][0])[body_ids].astype(np.float32)
        sim_root_pos = data.qpos[:3].copy().astype(np.float32)
        target_root_pos = tensor_to_np(motion_res['root_pos'][0]).astype(np.float32)
        sim_root_vel = data.qvel[:3].copy().astype(np.float32)
        target_root_vel = tensor_to_np(motion_res['root_vel'][0]).astype(np.float32)

        sim_body_pos_frames.append(torch.tensor(sim_body_pos))
        target_body_pos_frames.append(torch.tensor(target_body_pos))
        sim_root_pos_frames.append(torch.tensor(sim_root_pos))
        target_root_pos_frames.append(torch.tensor(target_root_pos))
        sim_root_vel_frames.append(torch.tensor(sim_root_vel))
        target_root_vel_frames.append(torch.tensor(target_root_vel))

        root_height = float(data.qpos[2])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or root_height < 0.45:
            stop_reason = 'fall'
            fall_step = step
            rows.append({'step': step, 'done': 1, 'root_height': root_height, 'action_mean_abs': float(np.abs(final_action).mean()), 'action_max_abs': float(np.abs(final_action).max()), 'action_rate': action_rate, 'action_clip_frac': action_clip_frac, 'ankle_action_mean_abs': float(np.abs(final_action[list(ANKLE_INDICES)]).mean()), 'hip_action_mean_abs': float(np.abs(final_action[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_mean_abs': float(np.abs(final_action[list(LOWER_BODY_INDICES)]).mean()), 'action_diff_mean_abs': float(np.abs(action_diff).mean()), 'ankle_action_diff_mean_abs': float(np.abs(action_diff[list(ANKLE_INDICES)]).mean()), 'hip_action_diff_mean_abs': float(np.abs(action_diff[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_diff_mean_abs': float(np.abs(action_diff[list(LOWER_BODY_INDICES)]).mean()), 'q_target_diff_mean_abs': float(np.abs(q_target_diff).mean())})
            break

        rows.append({'step': step, 'done': 0, 'root_height': root_height, 'action_mean_abs': float(np.abs(final_action).mean()), 'action_max_abs': float(np.abs(final_action).max()), 'action_rate': action_rate, 'action_clip_frac': action_clip_frac, 'ankle_action_mean_abs': float(np.abs(final_action[list(ANKLE_INDICES)]).mean()), 'hip_action_mean_abs': float(np.abs(final_action[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_mean_abs': float(np.abs(final_action[list(LOWER_BODY_INDICES)]).mean()), 'action_diff_mean_abs': float(np.abs(action_diff).mean()), 'ankle_action_diff_mean_abs': float(np.abs(action_diff[list(ANKLE_INDICES)]).mean()), 'hip_action_diff_mean_abs': float(np.abs(action_diff[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_diff_mean_abs': float(np.abs(action_diff[list(LOWER_BODY_INDICES)]).mean()), 'q_target_diff_mean_abs': float(np.abs(q_target_diff).mean())})
        per_joint_diff_accum.append(np.abs(action_diff))
        prev_final_action = final_action
        action = final_action
        if step % 2 == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    imageio.mimsave(str(video_path), frames, fps=50)

    sim_body_pos = torch.stack(sim_body_pos_frames, dim=0)
    target_body_pos = torch.stack(target_body_pos_frames, dim=0)
    sim_root_pos = torch.stack(sim_root_pos_frames, dim=0)
    target_root_pos = torch.stack(target_root_pos_frames, dim=0)
    sim_root_vel = torch.stack(sim_root_vel_frames, dim=0)
    target_root_vel = torch.stack(target_root_vel_frames, dim=0)
    paper_metrics = compute_paper_metrics(sim_body_pos, target_body_pos, sim_root_pos, target_root_pos, sim_root_vel, target_root_vel, mj_cfg.simulation_dt)

    summary = {'checkpoint': str(case['checkpoint']), 'onnx': str(onnx_path), 'scale': float(case['scale']), 'fall_step': int(fall_step if fall_step is not None else len(rows)), 'root_height_min': float(min(r['root_height'] for r in rows)), 'root_height_curve': [float(r['root_height']) for r in rows], 'action_mean_abs': float(np.mean([r['action_mean_abs'] for r in rows])), 'action_max_abs': float(np.max([r['action_max_abs'] for r in rows])), 'action_rate_mean': float(np.mean([r['action_rate'] for r in rows])), 'action_clip_frac': float(np.mean([r['action_clip_frac'] for r in rows])), 'ankle_action_mean_abs': float(np.mean([r['ankle_action_mean_abs'] for r in rows])), 'hip_action_mean_abs': float(np.mean([r['hip_action_mean_abs'] for r in rows])), 'ankle_action_diff_mean_abs': float(np.mean([r['ankle_action_diff_mean_abs'] for r in rows])), 'hip_action_diff_mean_abs': float(np.mean([r['hip_action_diff_mean_abs'] for r in rows])), 'lower_body_action_diff_mean_abs': float(np.mean([r['lower_body_action_diff_mean_abs'] for r in rows])), 'q_target_diff_mean_abs': float(np.mean([r['q_target_diff_mean_abs'] for r in rows])), **paper_metrics, 'video_path': str(video_path), 'per_joint_mean_abs_action_diff': np.mean(np.stack(per_joint_diff_accum, axis=0), axis=0).tolist() if per_joint_diff_accum else []}

    json_path = out_dir / f"{case['name']}.json"
    csv_path = out_dir / f"{case['name']}.csv"
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'rows': rows}, f, indent=2)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return summary, csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description='Closed-loop MuJoCo evaluation table')
    parser.add_argument('--baseline-ckpt', type=Path, required=True)
    parser.add_argument('--motion-file', type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument('--mujoco-config', type=Path, default=DEFAULT_MUJOCO_CFG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--case', action='append', required=True, help='name|checkpoint|scale')
    args = parser.parse_args()

    cases = [parse_case(spec) for spec in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = load_eval_cfg(args.baseline_ckpt, args.motion_file)
    baseline_env = instantiate(baseline_cfg.env, device=baseline_cfg.device)
    baseline_env.set_is_evaluating()

    mj_cfg = read_conf(str(args.mujoco_config))
    mj_cfg.xml_path = str((REPO_ROOT / 'humanoidverse' / 'data' / 'robots' / 'g1' / 'g1_29dof_anneal_23dof.xml'))
    mj_cfg.simulation_duration = float(mj_cfg.cycle_time)

    baseline_onnx = export_onnx(args.baseline_ckpt)
    baseline_session = ort.InferenceSession(str(baseline_onnx), providers=['CPUExecutionProvider'])
    baseline_input_name = baseline_session.get_inputs()[0].name
    baseline_output_name = baseline_session.get_outputs()[0].name

    reports = []
    for case in cases:
        report, csv_path, json_path = rollout_case(case, baseline_session, baseline_input_name, baseline_output_name, baseline_env, mj_cfg, args.output_dir / case['name'])
        reports.append(report)
        print(f'saved_csv={csv_path}')
        print(f'saved_json={json_path}')

    baseline = next(r for r in reports if r['scale'] == 0.0)
    baseline_paper = {'Eg_mpjpe_mm': baseline['Eg_mpjpe_mm'], 'Empjpe_mm': baseline['Empjpe_mm'], 'Eacc_mm_per_frame2': baseline['Eacc_mm_per_frame2'], 'Evel_mm_per_frame': baseline['Evel_mm_per_frame']}
    table = []
    for r in reports:
        det = {'Eg_mpjpe_mm': r['Eg_mpjpe_mm'], 'Empjpe_mm': r['Empjpe_mm'], 'Eacc_mm_per_frame2': r['Eacc_mm_per_frame2'], 'Evel_mm_per_frame': r['Evel_mm_per_frame']}
        improve = compute_paper_summary(baseline_paper, det)['paper_mean_improve_%']
        table.append({'checkpoint': r['checkpoint'], 'scale': r['scale'], 'fall_step': r['fall_step'], 'Eg_mpjpe': r['Eg_mpjpe_mm'], 'Empjpe': r['Empjpe_mm'], 'Eacc': r['Eacc_mm_per_frame2'], 'Evel': r['Evel_mm_per_frame'], 'paper_mean_improve_vs_baseline': improve, 'ankle_action_diff': r['ankle_action_diff_mean_abs'], 'hip_action_diff': r['hip_action_diff_mean_abs'], 'action_rate': r['action_rate_mean'], 'video_path': r['video_path']})

    table_path = args.output_dir / 'table.json'
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()

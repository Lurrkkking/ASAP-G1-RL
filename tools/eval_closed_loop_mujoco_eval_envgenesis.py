#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path('/root/autodl-tmp/ASAP')
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humanoidverse.eval_deltaa_openloop_deterministic import compute_paper_metrics, compute_paper_summary
import humanoidverse.eval_deltaa_openloop_deterministic as eval_deltaa_mod
eval_deltaa_mod.torch = torch
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.config_utils import *  # noqa: F401,F403
from hydra.utils import instantiate
from omegaconf import OmegaConf

DEFAULT_MOTION_FILE = REPO_ROOT / 'humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl'
DEFAULT_MUJOCO_CFG = REPO_ROOT / 'mujoco_simulation/g1_config/mujoco_config.yaml'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'tmp/closed_loop_eval_table'

LOWER_BODY_INDICES = tuple(range(12))
ANKLE_INDICES = (4, 5, 10, 11)
HIP_KNEE_INDICES = (0, 1, 2, 3, 6, 7, 8, 9)
BODY_NAMES = [
    'pelvis',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'waist_yaw_link',
    'waist_roll_link',
    'torso_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
]
ACTION_SCALE = float(__import__('os').environ.get('ACTION_SCALE', '0.25'))
ACTION_CLIP_VALUE = float(__import__('os').environ.get('ACTION_CLIP_VALUE', '100.0'))


def read_conf(config_file):
    import yaml

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    class Cfg:
        pass

    out = Cfg()
    out.xml_path = str(REPO_ROOT / 'humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml')
    out.num_actions = 23
    out.simulation_dt = float(cfg['simulation_dt'])
    out.cycle_time = float(cfg['cycle_time'])
    out.default_dof_pos = np.array(cfg['default_dof_pos'], dtype=np.float32)
    out.kps = np.array(cfg['kps'], dtype=np.float32)
    out.kds = np.array(cfg['kds'], dtype=np.float32)
    out.tau_limit = np.array(cfg['tau_limit'], dtype=np.float32)
    out.solver_iterations = int(cfg.get('solver_iterations', 100))
    out.solver_ls_iterations = int(cfg.get('solver_ls_iterations', 50))
    out.clip_observations = float(cfg.get('clip_observations', 100.0))
    out.obs_scale_base_ang_vel = float(cfg.get('obs_scale_base_ang_vel', 0.25))
    out.obs_scale_dof_pos = float(cfg.get('obs_scale_dof_pos', 1.0))
    out.obs_scale_dof_vel = float(cfg.get('obs_scale_dof_vel', 0.05))
    out.obs_scale_gvec = float(cfg.get('obs_scale_gvec', 1.0))
    out.obs_scale_refmotion = float(cfg.get('obs_scale_refmotion', 1.0))
    out.obs_scale_hist = float(cfg.get('obs_scale_hist', 1.0))
    out.frame_stack = int(cfg.get('frame_stack', 4))
    out.num_single_obs = int(cfg.get('num_single_obs', 76))
    out.time_offset = float(cfg.get('time_offset', 0.0))
    out.phase_wrap = bool(cfg.get('phase_wrap', False))
    out.stop_at_motion_end = bool(cfg.get('stop_at_motion_end', True))
    out.action_filter_alpha = float(cfg.get('action_filter_alpha', 1.0))
    out.target_pos_rate_limit = float(cfg.get('target_pos_rate_limit', 0.0))
    out.safety_check_nonfinite = bool(cfg.get('safety_check_nonfinite', True))
    out.max_abs_qacc = float(cfg.get('max_abs_qacc', 5.0e4))
    return out


def create_history(cfg):
    hist_dict = {
        'actions': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        'base_ang_vel': np.zeros((cfg.frame_stack, 3), dtype=np.float32),
        'dof_pos': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        'dof_vel': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.float32),
        'projected_gravity': np.zeros((cfg.frame_stack, 3), dtype=np.float32),
        'ref_motion_phase': np.zeros((cfg.frame_stack, 1), dtype=np.float32),
    }
    hist_obs_c = np.concatenate([hist_dict[k].reshape(1, -1) for k in ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']], axis=1)
    return hist_dict, hist_obs_c


def _update_history(hist_dict, obs_single):
    slices = {
        'actions': slice(0, 23),
        'base_ang_vel': slice(23, 26),
        'dof_pos': slice(26, 49),
        'dof_vel': slice(49, 72),
        'projected_gravity': slice(72, 75),
        'ref_motion_phase': slice(75, 76),
    }
    for key, slc in slices.items():
        hist_dict[key] = np.vstack([obs_single[0, slc], hist_dict[key][:-1]])
    return np.concatenate([hist_dict[k].reshape(1, -1) for k in ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']], axis=1)


def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    obs_single = np.zeros((1, cfg.num_single_obs), dtype=np.float32)
    obs_single[0, 0:23] = action
    obs_single[0, 23:26] = mujoco_data['mujoco_base_angvel'] * cfg.obs_scale_base_ang_vel
    obs_single[0, 26:49] = (mujoco_data['mujoco_dof_pos'] - cfg.default_dof_pos) * cfg.obs_scale_dof_pos
    obs_single[0, 49:72] = mujoco_data['mujoco_dof_vel'] * cfg.obs_scale_dof_vel
    obs_single[0, 72:75] = mujoco_data['mujoco_gvec'] * cfg.obs_scale_gvec
    phase = min((counter + 1) * cfg.simulation_dt / cfg.cycle_time, 1.0)
    obs_single[0, 75] = phase * cfg.obs_scale_refmotion
    obs_all = np.zeros((1, (cfg.frame_stack + 1) * cfg.num_single_obs), dtype=np.float32)
    obs_all[0, 0:76] = obs_single[0]
    obs_all[0, 76:76 + hist_obs_c.shape[1]] = hist_obs_c[0] * cfg.obs_scale_hist
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    return obs_all, _update_history(hist_dict, obs_single)


def pd_control(target_pos, dof_pos, target_vel, dof_vel, cfg):
    return (target_pos - dof_pos) * cfg.kps + (target_vel - dof_vel) * cfg.kds


def parse_case(spec):
    name, checkpoint, scale = spec.split('|')
    return {'name': name, 'checkpoint': Path(checkpoint), 'scale': float(scale)}


def export_onnx(checkpoint):
    onnx_path = checkpoint.parent / 'exported' / f'{checkpoint.stem}.onnx'
    if onnx_path.is_file():
        return onnx_path
    raise FileNotFoundError(onnx_path)


def load_motion_lib(checkpoint, motion_file):
    cfg_path = checkpoint.parent / 'config.yaml'
    if not cfg_path.is_file():
        cfg_path = checkpoint.parent.parent / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    if cfg.get('eval_overrides') is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
    cfg.headless = True
    cfg.num_envs = 1
    cfg.use_wandb = False
    cfg.auto_load_latest = False
    cfg.checkpoint = str(checkpoint)
    cfg.device = 'cpu'
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
    motion_cfg = cfg.robot.motion if cfg.get('robot') is not None else cfg.env.config.robot.motion
    if getattr(motion_cfg, 'asset', None) is not None:
        asset_root = getattr(motion_cfg.asset, 'asset_root', None) or getattr(motion_cfg.asset, 'assetRoot', None)
        asset_file = getattr(motion_cfg.asset, 'assetFileName', None) or getattr(motion_cfg.asset, 'asset_file', None)
        if asset_root:
            resolved_root = str((REPO_ROOT / asset_root).resolve())
            if 'asset_root' in motion_cfg.asset:
                motion_cfg.asset.asset_root = resolved_root
            if 'assetRoot' in motion_cfg.asset:
                motion_cfg.asset.assetRoot = resolved_root
        else:
            resolved_root = str((REPO_ROOT / 'humanoidverse/data/robots').resolve())
        if asset_file:
            resolved_file = str((Path(resolved_root) / asset_file).resolve())
            if 'assetFileName' in motion_cfg.asset:
                motion_cfg.asset.assetFileName = resolved_file
            if 'asset_file' in motion_cfg.asset:
                motion_cfg.asset.asset_file = resolved_file
    ml = MotionLibRobot(motion_cfg, num_envs=1, device='cpu')
    ml.load_motions(random_sample=False)
    motion_ids = torch.zeros(1, dtype=torch.long)
    env_origins = torch.zeros((1, 3), dtype=torch.float32)
    return ml, motion_ids, env_origins


def tensor_to_np(x):
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def mean_abs_max(x):
    x = tensor_to_np(x).astype(np.float32)
    return float(np.abs(x).mean()), float(np.abs(x).max())


def build_body_ids(model):
    ids = []
    for name in BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f'missing MuJoCo body: {name}')
        ids.append(body_id)
    return np.asarray(ids, dtype=np.int32)


def rollout(case, baseline_session, baseline_input, baseline_output, motion_lib, motion_ids, env_origins, mj_cfg, out_dir):
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

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = None

    hist_dict, hist_obs_c = create_history(mj_cfg)
    action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    prev_action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    rows = []
    sim_body_pos_frames = []
    target_body_pos_frames = []
    sim_root_pos_frames = []
    target_root_pos_frames = []
    sim_root_vel_frames = []
    target_root_vel_frames = []
    diff_accum = []
    fall_step = None
    stop_reason = 'motion_end'
    sim_steps = int(mj_cfg.simulation_duration / mj_cfg.simulation_dt)
    for step in range(sim_steps):
        mj = {'mujoco_dof_pos': data.qpos[7:].copy().astype(np.float32), 'mujoco_dof_vel': data.qvel[6:].copy().astype(np.float32), 'mujoco_base_angvel': data.qvel[3:6].copy().astype(np.float32), 'mujoco_gvec': np.array([0.0, 0.0, -1.0], dtype=np.float32)}
        obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mj, action, step, mj_cfg)
        actor_obs = obs_buff.astype(np.float32)
        case_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)
        baseline_action = baseline_session.run([baseline_output], {baseline_input: actor_obs})[0][0].astype(np.float32)
        case_action = np.clip(case_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        baseline_action = np.clip(baseline_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        final_action = case_action
        target_dof_pos = final_action * ACTION_SCALE + mj_cfg.default_dof_pos
        baseline_target_dof_pos = baseline_action * ACTION_SCALE + mj_cfg.default_dof_pos
        action_rate = float(np.abs(final_action - prev_action).mean())
        action_clip_frac = float((np.abs(final_action) >= ACTION_CLIP_VALUE).mean())
        action_diff = final_action - baseline_action
        q_target_diff = target_dof_pos - baseline_target_dof_pos
        data.ctrl[:] = pd_control(target_dof_pos, mj['mujoco_dof_pos'], np.zeros_like(mj_cfg.kds), mj['mujoco_dof_vel'], mj_cfg)
        data.ctrl[:] = np.clip(data.ctrl, -mj_cfg.tau_limit, mj_cfg.tau_limit)
        mujoco.mj_step(model, data)

        ref_motion_time = float((step + 1) * mj_cfg.simulation_dt)
        motion_t = torch.tensor([ref_motion_time], dtype=torch.float32)
        motion_res = motion_lib.get_motion_state(motion_ids, motion_t, offset=env_origins)
        sim_body_pos = data.xpos[body_ids].copy().astype(np.float32)
        target_body_pos = tensor_to_np(motion_res['rg_pos'][0]).astype(np.float32)
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
        qpos_finite = bool(np.isfinite(data.qpos).all())
        qvel_finite = bool(np.isfinite(data.qvel).all())
        qacc_finite = bool(np.isfinite(data.qacc).all())
        qacc_max_abs = float(np.max(np.abs(data.qacc))) if qacc_finite else float('inf')

        step_stop_reason = 'running'
        if not qpos_finite:
            step_stop_reason = 'numerical_instability_nonfinite_qpos'
        elif not qvel_finite:
            step_stop_reason = 'numerical_instability_nonfinite_qvel'
        elif not qacc_finite:
            step_stop_reason = 'numerical_instability_nonfinite_qacc'
        elif qacc_max_abs > mj_cfg.max_abs_qacc:
            step_stop_reason = 'numerical_instability_qacc'
        elif root_height < 0.45:
            step_stop_reason = 'fall'

        rows.append({'step': step, 'root_height': root_height, 'qacc_max_abs': qacc_max_abs, 'stop_reason': step_stop_reason, 'action_mean_abs': float(np.abs(final_action).mean()), 'action_max_abs': float(np.abs(final_action).max()), 'action_rate': action_rate, 'action_clip_frac': action_clip_frac, 'ankle_action_mean_abs': float(np.abs(final_action[list(ANKLE_INDICES)]).mean()), 'hip_action_mean_abs': float(np.abs(final_action[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_mean_abs': float(np.abs(final_action[list(LOWER_BODY_INDICES)]).mean()), 'action_diff_mean_abs': float(np.abs(action_diff).mean()), 'ankle_action_diff_mean_abs': float(np.abs(action_diff[list(ANKLE_INDICES)]).mean()), 'hip_action_diff_mean_abs': float(np.abs(action_diff[list(HIP_KNEE_INDICES)]).mean()), 'lower_body_action_diff_mean_abs': float(np.abs(action_diff[list(LOWER_BODY_INDICES)]).mean()), 'q_target_diff_mean_abs': float(np.abs(q_target_diff).mean())})
        if step_stop_reason != 'running':
            stop_reason = step_stop_reason
            fall_step = step
            break
        diff_accum.append(np.abs(action_diff))
        prev_action = final_action
        action = final_action

    sim_body_pos = torch.stack(sim_body_pos_frames, dim=0)
    target_body_pos = torch.stack(target_body_pos_frames, dim=0)
    sim_root_pos = torch.stack(sim_root_pos_frames, dim=0)
    target_root_pos = torch.stack(target_root_pos_frames, dim=0)
    sim_root_vel = torch.stack(sim_root_vel_frames, dim=0)
    target_root_vel = torch.stack(target_root_vel_frames, dim=0)
    paper_metrics = compute_paper_metrics(sim_body_pos, target_body_pos, sim_root_pos, target_root_pos, sim_root_vel, target_root_vel, mj_cfg.simulation_dt)
    summary = {'checkpoint': str(case['checkpoint']), 'onnx': str(onnx_path), 'scale': float(case['scale']), 'fall_step': int(fall_step if fall_step is not None else len(rows)), 'stop_reason': stop_reason, 'root_height_min': float(min(r['root_height'] for r in rows)), 'root_height_curve': [float(r['root_height']) for r in rows], 'action_mean_abs': float(np.mean([r['action_mean_abs'] for r in rows])), 'action_max_abs': float(np.max([r['action_max_abs'] for r in rows])), 'action_rate_mean': float(np.mean([r['action_rate'] for r in rows])), 'action_clip_frac': float(np.mean([r['action_clip_frac'] for r in rows])), 'ankle_action_mean_abs': float(np.mean([r['ankle_action_mean_abs'] for r in rows])), 'hip_action_mean_abs': float(np.mean([r['hip_action_mean_abs'] for r in rows])), 'ankle_action_diff_mean_abs': float(np.mean([r['ankle_action_diff_mean_abs'] for r in rows])), 'hip_action_diff_mean_abs': float(np.mean([r['hip_action_diff_mean_abs'] for r in rows])), 'lower_body_action_diff_mean_abs': float(np.mean([r['lower_body_action_diff_mean_abs'] for r in rows])), 'q_target_diff_mean_abs': float(np.mean([r['q_target_diff_mean_abs'] for r in rows])), **paper_metrics, 'video_path': str(video_path) if video_path is not None else '', 'per_joint_mean_abs_action_diff': np.mean(np.stack(diff_accum, axis=0), axis=0).tolist() if diff_accum else []}
    json_path = out_dir / f"{case['name']}.json"
    csv_path = out_dir / f"{case['name']}.csv"
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'rows': rows}, f, indent=2)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ['step'])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    return summary, csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ckpt', type=Path, required=True)
    parser.add_argument('--motion-file', type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument('--mujoco-config', type=Path, default=DEFAULT_MUJOCO_CFG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--case', action='append', required=True)
    args = parser.parse_args()
    cases = [parse_case(s) for s in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    motion_lib, motion_ids, env_origins = load_motion_lib(args.baseline_ckpt, args.motion_file)

    mj_cfg = read_conf(str(args.mujoco_config))
    mj_cfg.xml_path = str((REPO_ROOT / 'humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml'))
    mj_cfg.simulation_duration = float(mj_cfg.cycle_time)

    baseline_onnx = export_onnx(args.baseline_ckpt)
    baseline_session = ort.InferenceSession(str(baseline_onnx), providers=['CPUExecutionProvider'])
    baseline_input = baseline_session.get_inputs()[0].name
    baseline_output = baseline_session.get_outputs()[0].name

    reports = []
    for case in cases:
        rep, csv_path, json_path = rollout(case, baseline_session, baseline_input, baseline_output, motion_lib, motion_ids, env_origins, mj_cfg, args.output_dir / case['name'])
        reports.append(rep)
        print(f'saved_csv={csv_path}')
        print(f'saved_json={json_path}')

    baseline = next(r for r in reports if r['scale'] == 0.0)
    baseline_paper = {'Eg_mpjpe_mm': baseline['Eg_mpjpe_mm'], 'Empjpe_mm': baseline['Empjpe_mm'], 'Eacc_mm_per_frame2': baseline['Eacc_mm_per_frame2'], 'Evel_mm_per_frame': baseline['Evel_mm_per_frame']}
    table = []
    for r in reports:
        det = {'Eg_mpjpe_mm': r['Eg_mpjpe_mm'], 'Empjpe_mm': r['Empjpe_mm'], 'Eacc_mm_per_frame2': r['Eacc_mm_per_frame2'], 'Evel_mm_per_frame': r['Evel_mm_per_frame']}
        improve = compute_paper_summary(baseline_paper, det)['paper_mean_improve_%']
        table.append({'checkpoint': r['checkpoint'], 'scale': r['scale'], 'fall_step': r['fall_step'], 'Eg_mpjpe': r['Eg_mpjpe_mm'], 'Empjpe': r['Empjpe_mm'], 'Eacc': r['Eacc_mm_per_frame2'], 'Evel': r['Evel_mm_per_frame'], 'paper_mean_improve_vs_baseline': improve, 'ankle_action_diff': r['ankle_action_diff_mean_abs'], 'hip_action_diff': r['hip_action_diff_mean_abs'], 'action_rate': r['action_rate_mean'], 'video_path': r['video_path']})

    with open(args.output_dir / 'table.json', 'w') as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))

if __name__ == '__main__':
    main()

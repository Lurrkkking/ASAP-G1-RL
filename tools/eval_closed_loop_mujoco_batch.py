
#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humanoidverse.eval_deltaa_openloop_deterministic import compute_paper_metrics, compute_paper_summary
from mujoco_simulation.fdd_asap_sim2sim import (
    BODY_NAMES,
    ANKLE_INDICES,
    HIP_KNEE_INDICES,
    LOWER_BODY_INDICES,
    ACTION_SCALE,
    ACTION_CLIP_VALUE,
    DEFAULT_DOF_POS,
    DEFAULT_BASELINE_POLICY_PATH,
    read_conf,
    load_onnx_policy,
    create_history,
    get_obs,
    pd_control,
    compute_ref_motion_phase,
)

DEFAULT_MUJOCO_CFG = REPO_ROOT / "mujoco_simulation" / "g1_config" / "mujoco_config.yaml"
DEFAULT_MOTION_FILE = REPO_ROOT / "humanoidverse" / "data" / "motions" / "g1_29dof_anneal_23dof" / "TairanTestbed" / "singles" / "0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "closed_loop_eval_mujoco"
DEFAULT_EXPORT_SCRIPT = REPO_ROOT / "humanoidverse" / "export_pt_to_onnx.py"


def parse_case(spec: str):
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError("case format must be name|checkpoint|scale")
    name, checkpoint, scale = parts
    return {"name": name, "checkpoint": Path(checkpoint), "scale": float(scale)}


def load_checkpoint_config(checkpoint: Path):
    cfg_candidates = [checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"]
    for cfg_path in cfg_candidates:
        if cfg_path.is_file():
            import isaacgym  # noqa: F401
            from omegaconf import OmegaConf
            from humanoidverse.utils.helpers import pre_process_config
            cfg = OmegaConf.load(cfg_path)
            if cfg.get("eval_overrides") is not None:
                cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
            cfg.headless = True
            cfg.num_envs = 1
            cfg.use_wandb = False
            cfg.checkpoint = str(checkpoint)
            cfg.env.config.headless = True
            cfg.env.config.num_envs = 1
            cfg.env.config.noise_to_initial_level = 0
            cfg.env.config.enforce_randomize_motion_start_eval = False
            cfg.env.config.randomize_motion_start_train = False
            cfg.env.config.save_motion = False
            cfg.env.config.robot.motion.motion_file = str(DEFAULT_MOTION_FILE)
            cfg.env.config.max_episode_length_s = max(float(cfg.env.config.max_episode_length_s), 100000.0)
            pre_process_config(cfg)
            return cfg
    raise FileNotFoundError(f"Could not find config.yaml near {checkpoint}")


def export_onnx(checkpoint: Path):
    export_dir = checkpoint.parent / "exported"
    onnx_path = export_dir / f"{checkpoint.stem}.onnx"
    if onnx_path.is_file():
        return onnx_path
    cmd = [
        sys.executable,
        str(DEFAULT_EXPORT_SCRIPT),
        f"checkpoint={checkpoint}",
        "+simulator=isaacgym",
        "+device=cuda:0",
        "+headless=true",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX export failed: {onnx_path}")
    return onnx_path


def make_session(onnx_path: Path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def tensor_to_np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def build_body_ids(model):
    ids = []
    for name in BODY_NAMES:
        try:
            ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        except Exception:
            pass
    return np.asarray(ids, dtype=np.int32)


def run_case(case, baseline_session, baseline_input_name, baseline_output_name, ref_env, mj_cfg, output_dir):
    onnx_path = export_onnx(case["checkpoint"])
    session = make_session(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    model = mujoco.MjModel.from_xml_path(str(mj_cfg.xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = mj_cfg.simulation_dt
    if mj_cfg.solver_iterations > 0:
        model.opt.iterations = mj_cfg.solver_iterations
    if hasattr(model.opt, "ls_iterations") and mj_cfg.solver_ls_iterations > 0:
        model.opt.ls_iterations = mj_cfg.solver_ls_iterations
    data.qpos[-mj_cfg.num_actions:] = mj_cfg.default_dof_pos
    mujoco.mj_step(model, data)

    body_ids = build_body_ids(model)
    if len(body_ids) == 0:
        raise RuntimeError("No BODY_NAMES matched the MuJoCo model")

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required for mp4 output") from exc

    video_path = output_dir / f"{case['name']}.mp4"
    renderer = mujoco.Renderer(model, height=900, width=1600)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.8
    cam.azimuth = 135.0
    cam.elevation = -12.0

    hist_dict, hist_obs_c = create_history(mj_cfg)
    prev_final_action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    frames = []
    rows = []
    sim_body_pos_frames = []
    target_body_pos_frames = []
    sim_root_pos_frames = []
    target_root_pos_frames = []
    sim_root_vel_frames = []
    target_root_vel_frames = []
    per_joint_diff_accum = []

    sim_steps = int(mj_cfg.simulation_duration / mj_cfg.simulation_dt)
    fall_step = None
    stop_reason = "motion_end"

    motion_time = 0.0
    action = np.zeros(mj_cfg.num_actions, dtype=np.float32)
    default_dof_pos = np.asarray(mj_cfg.default_dof_pos, dtype=np.float32)

    for step in range(sim_steps):
        mj = {
            "mujoco_dof_pos": data.qpos[7:].copy().astype(np.float32),
            "mujoco_dof_vel": data.qvel[6:].copy().astype(np.float32),
            "mujoco_base_angvel": data.qvel[3:6].copy().astype(np.float32),
            "mujoco_gvec": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mj, action, step, mj_cfg)
        actor_obs = obs_buff.astype(np.float32)

        with np.errstate(over="ignore", invalid="ignore"):
            case_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)
            baseline_action = baseline_session.run([baseline_output_name], {baseline_input_name: actor_obs})[0][0].astype(np.float32)

        case_action = np.clip(case_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        baseline_action = np.clip(baseline_action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        final_action = case_action

        target_dof_pos = final_action * ACTION_SCALE + default_dof_pos
        baseline_target_dof_pos = baseline_action * ACTION_SCALE + default_dof_pos
        action_rate = float(np.mean(np.abs(final_action - prev_final_action)))
        action_clip_frac = float(np.mean(np.abs(final_action) >= ACTION_CLIP_VALUE))
        action_diff = final_action - baseline_action
        q_target_diff = target_dof_pos - baseline_target_dof_pos

        data.ctrl[:] = pd_control(target_dof_pos, mj["mujoco_dof_pos"], np.zeros_like(mj_cfg.kds), mj["mujoco_dof_vel"], mj_cfg)
        data.ctrl[:] = np.clip(data.ctrl, -mj_cfg.tau_limit, mj_cfg.tau_limit)
        mujoco.mj_step(model, data)

        ref_motion_time = float((step + 1) * mj_cfg.simulation_dt)
        motion_t = torch.tensor([ref_motion_time], dtype=torch.float32, device=ref_env.device)
        motion_res = ref_env._motion_lib.get_motion_state(ref_env.motion_ids, motion_t, offset=ref_env.env_origins)

        sim_body_pos = data.xpos[body_ids].copy().astype(np.float32)
        target_body_pos = tensor_to_np(motion_res["rg_pos"][0])[body_ids].astype(np.float32)
        sim_root_pos = data.qpos[:3].copy().astype(np.float32)
        target_root_pos = tensor_to_np(motion_res["root_pos"][0]).astype(np.float32)
        sim_root_vel = data.qvel[:3].copy().astype(np.float32)
        target_root_vel = tensor_to_np(motion_res["root_vel"][0]).astype(np.float32)

        sim_body_pos_frames.append(torch.tensor(sim_body_pos))
        target_body_pos_frames.append(torch.tensor(target_body_pos))
        sim_root_pos_frames.append(torch.tensor(sim_root_pos))
        target_root_pos_frames.append(torch.tensor(target_root_pos))
        sim_root_vel_frames.append(torch.tensor(sim_root_vel))
        target_root_vel_frames.append(torch.tensor(target_root_vel))

        root_height = float(data.qpos[2])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or root_height < 0.45:
            stop_reason = "fall"
            fall_step = step
            rows.append({
                "step": step,
                "root_height": root_height,
                "action_rate": action_rate,
                "action_clip_frac": action_clip_frac,
                "ankle_action_mean_abs": float(np.mean(np.abs(final_action[list(ANKLE_INDICES)]))),
                "hip_action_mean_abs": float(np.mean(np.abs(final_action[list(HIP_KNEE_INDICES)]))),
                "lower_body_action_mean_abs": float(np.mean(np.abs(final_action[list(LOWER_BODY_INDICES)]))),
                "action_diff_mean_abs": float(np.mean(np.abs(action_diff))),
                "ankle_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(ANKLE_INDICES)]))),
                "hip_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(HIP_KNEE_INDICES)]))),
                "lower_body_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(LOWER_BODY_INDICES)]))),
                "q_target_diff_mean_abs": float(np.mean(np.abs(q_target_diff))),
            })
            break

        rows.append({
            "step": step,
            "root_height": root_height,
            "action_rate": action_rate,
            "action_clip_frac": action_clip_frac,
            "action_mean_abs": float(np.mean(np.abs(final_action))),
            "action_max_abs": float(np.max(np.abs(final_action))),
            "ankle_action_mean_abs": float(np.mean(np.abs(final_action[list(ANKLE_INDICES)]))),
            "hip_action_mean_abs": float(np.mean(np.abs(final_action[list(HIP_KNEE_INDICES)]))),
            "lower_body_action_mean_abs": float(np.mean(np.abs(final_action[list(LOWER_BODY_INDICES)]))),
            "action_diff_mean_abs": float(np.mean(np.abs(action_diff))),
            "ankle_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(ANKLE_INDICES)]))),
            "hip_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(HIP_KNEE_INDICES)]))),
            "lower_body_action_diff_mean_abs": float(np.mean(np.abs(action_diff[list(LOWER_BODY_INDICES)]))),
            "q_target_diff_mean_abs": float(np.mean(np.abs(q_target_diff))),
        })

        per_joint_diff_accum.append(np.abs(action_diff))
        prev_final_action = final_action
        prev_baseline_action = baseline_action
        action = final_action

        if step % 2 == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    imageio.mimsave(str(video_path), frames, fps=50)

    if len(rows) == 0:
        raise RuntimeError(f"No rollout steps collected for {case['name']}")

    sim_body_pos = torch.stack(sim_body_pos_frames, dim=0)
    target_body_pos = torch.stack(target_body_pos_frames, dim=0)
    sim_root_pos = torch.stack(sim_root_pos_frames, dim=0)
    target_root_pos = torch.stack(target_root_pos_frames, dim=0)
    sim_root_vel = torch.stack(sim_root_vel_frames, dim=0)
    target_root_vel = torch.stack(target_root_vel_frames, dim=0)
    paper_metrics = compute_paper_metrics(sim_body_pos, target_body_pos, sim_root_pos, target_root_pos, sim_root_vel, target_root_vel, mj_cfg.simulation_dt)

    summary = {
        "checkpoint": str(case["checkpoint"]),
        "onnx": str(onnx_path),
        "scale": float(case["scale"]),
        "fall_step": int(fall_step if fall_step is not None else len(rows)),
        "root_height_min": float(min(r["root_height"] for r in rows)),
        "root_height_curve": [float(r["root_height"]) for r in rows],
        "action_mean_abs": float(np.mean([r["action_mean_abs"] for r in rows])),
        "action_max_abs": float(np.max([r["action_max_abs"] for r in rows])),
        "action_rate_mean": float(np.mean([r["action_rate"] for r in rows])),
        "action_clip_frac": float(np.mean([r["action_clip_frac"] for r in rows])),
        "ankle_action_mean_abs": float(np.mean([r["ankle_action_mean_abs"] for r in rows])),
        "hip_action_mean_abs": float(np.mean([r["hip_action_mean_abs"] for r in rows])),
        "ankle_action_diff_mean_abs": float(np.mean([r["ankle_action_diff_mean_abs"] for r in rows])),
        "hip_action_diff_mean_abs": float(np.mean([r["hip_action_diff_mean_abs"] for r in rows])),
        "lower_body_action_diff_mean_abs": float(np.mean([r["lower_body_action_diff_mean_abs"] for r in rows])),
        "q_target_diff_mean_abs": float(np.mean([r["q_target_diff_mean_abs"] for r in rows])),
        **paper_metrics,
        "video_path": str(video_path),
        "per_joint_mean_abs_action_diff": np.mean(np.stack(per_joint_diff_accum, axis=0), axis=0).tolist(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Closed-loop MuJoCo batch evaluation for CR7")
    parser.add_argument("--case", action="append", required=True, help="name|checkpoint|scale; repeat for each case")
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mujoco-config", type=Path, default=DEFAULT_MUJOCO_CFG)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--baseline-onnx", type=Path, default=Path(DEFAULT_BASELINE_POLICY_PATH))
    args = parser.parse_args()

    cases = [parse_case(spec) for spec in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mj_cfg = read_conf(str(args.mujoco_config))
    mj_cfg.policy_path = str(args.baseline_onnx)
    mj_cfg.xml_path = str((REPO_ROOT / "humanoidverse" / "data" / "robots" / "g1" / "g1_29dof_anneal_23dof.xml"))
    mj_cfg.simulation_duration = float(mj_cfg.cycle_time)

    import isaacgym  # noqa: F401
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from humanoidverse.utils.helpers import pre_process_config

    baseline_case = next((c for c in cases if c["name"] == args.baseline_name), None)
    if baseline_case is None:
        raise ValueError(f"baseline case '{args.baseline_name}' not found in --case list")

    baseline_cfg = load_checkpoint_config(baseline_case["checkpoint"])
    baseline_cfg.env.config.robot.motion.motion_file = str(args.motion_file)
    pre_process_config(baseline_cfg)
    ref_env = instantiate(baseline_cfg.env, device=baseline_cfg.device)
    ref_env.set_is_evaluating()

    baseline_onnx = export_onnx(baseline_case["checkpoint"])
    baseline_session = make_session(baseline_onnx)
    baseline_input_name = baseline_session.get_inputs()[0].name
    baseline_output_name = baseline_session.get_outputs()[0].name

    reports = []
    for case in cases:
        report = run_case(case, baseline_session, baseline_input_name, baseline_output_name, ref_env, mj_cfg, args.output_dir / case["name"])
        reports.append(report)

    baseline_report = next(r for r in reports if Path(r["checkpoint"]) == baseline_case["checkpoint"])
    baseline_paper = {
        "Eg_mpjpe_mm": baseline_report["Eg_mpjpe_mm"],
        "Empjpe_mm": baseline_report["Empjpe_mm"],
        "Eacc_mm_per_frame2": baseline_report["Eacc_mm_per_frame2"],
        "Evel_mm_per_frame": baseline_report["Evel_mm_per_frame"],
    }
    for report in reports:
        det_paper = {
            "Eg_mpjpe_mm": report["Eg_mpjpe_mm"],
            "Empjpe_mm": report["Empjpe_mm"],
            "Eacc_mm_per_frame2": report["Eacc_mm_per_frame2"],
            "Evel_mm_per_frame": report["Evel_mm_per_frame"],
        }
        report["paper_mean_improve_vs_baseline"] = float(compute_paper_summary(baseline_paper, det_paper)["paper_mean_improve_%"])

    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "scale", "fall_step", "root_height_min", "Eg_mpjpe_mm", "Empjpe_mm", "Eacc_mm_per_frame2", "Evel_mm_per_frame",
            "paper_mean_improve_vs_baseline", "ankle_action_diff_mean_abs", "hip_action_diff_mean_abs", "action_rate_mean", "video_path"
        ])
        writer.writeheader()
        for r in reports:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"csv={csv_path}")
    print(f"json={json_path}")
    for r in reports:
        print(
            f"{Path(r['checkpoint']).name} scale={r['scale']:.3f} fall={r['fall_step']} Eg={r['Eg_mpjpe_mm']:.2f} Emp={r['Empjpe_mm']:.2f} "
            f"Eacc={r['Eacc_mm_per_frame2']:.2f} Evel={r['Evel_mm_per_frame']:.2f} paper_improve={r['paper_mean_improve_vs_baseline']:.2f} "
            f"ankle_diff={r['ankle_action_diff_mean_abs']:.4f} hip_diff={r['hip_action_diff_mean_abs']:.4f} rate={r['action_rate_mean']:.4f} video={r['video_path']}"
        )


if __name__ == '__main__':
    main()

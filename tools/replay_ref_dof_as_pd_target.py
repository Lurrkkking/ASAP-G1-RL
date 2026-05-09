import argparse
import csv
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


def disable_domain_rand(cfg):
    if not hasattr(cfg, "domain_rand"):
        return
    for key in list(cfg.domain_rand.keys()):
        value = cfg.domain_rand[key]
        if isinstance(value, bool) and (
            key.startswith("randomize") or key.startswith("push") or key.endswith("noise")
        ):
            cfg.domain_rand[key] = False


def parse_force_root_components(args):
    if args.force_root:
        return {"root_pos", "root_rot", "root_lin_vel", "root_ang_vel"}

    components = set()
    if args.force_root_components:
        for item in args.force_root_components.split(","):
            item = item.strip()
            if item:
                components.add(item)
    valid = {"root_pos", "root_rot", "root_lin_vel", "root_ang_vel"}
    invalid = sorted(components - valid)
    if invalid:
        raise ValueError(f"Unsupported force-root components: {invalid}")
    return components


def force_root_state_from_motion(env, motion_res, force_components):
    import torch

    if not force_components:
        return

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if "root_pos" in force_components:
        env.simulator.robot_root_states[env_ids, :3] = motion_res["root_pos"].to(env.device)
    if "root_rot" in force_components:
        env.simulator.robot_root_states[env_ids, 3:7] = motion_res["root_rot"].to(env.device)
    if "root_lin_vel" in force_components:
        env.simulator.robot_root_states[env_ids, 7:10] = motion_res["root_vel"].to(env.device)
    if "root_ang_vel" in force_components:
        env.simulator.robot_root_states[env_ids, 10:13] = motion_res["root_ang_vel"].to(env.device)
    env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
    env.simulator.refresh_sim_tensors()


def compute_motion_res(env, motion_time_s):
    import torch

    motion_t = torch.full((env.num_envs,), float(motion_time_s), dtype=torch.float32, device=env.device)
    return env._motion_lib.get_motion_state(env.motion_ids, motion_t, offset=env.env_origins)


def compute_required_action_stats(env, motion_length, speed_scale):
    import torch

    total_steps = max(1, int(math.ceil(motion_length / (env.dt * max(speed_scale, 1e-8)))))
    all_actions = []
    for step in range(total_steps):
        motion_time_s = min((step + 1) * env.dt * speed_scale, motion_length)
        motion_res = compute_motion_res(env, motion_time_s)
        ref_dof_pos = motion_res["dof_pos"]
        req_actions = (ref_dof_pos - env.default_dof_pos) / float(env.config.robot.control.action_scale)
        all_actions.append(req_actions[0].detach().cpu())
    all_actions = torch.stack(all_actions, dim=0)

    print("[ACTION_STATS] required_action = (ref_dof_pos - default_dof_pos) / action_scale")
    print(
        "[ACTION_STATS] "
        f"frames={all_actions.shape[0]} action_scale={float(env.config.robot.control.action_scale):.6f} "
        f"global_max_abs={float(all_actions.abs().max()):.4f} "
        f"frac_abs_gt_1={float((all_actions.abs() > 1.0).float().mean()):.4f} "
        f"frac_abs_gt_2={float((all_actions.abs() > 2.0).float().mean()):.4f}"
    )
    for joint_idx, name in enumerate(env.dof_names):
        v = all_actions[:, joint_idx]
        print(
            "[ACTION_STATS] "
            f"{name}: min={float(v.min()):.4f} max={float(v.max()):.4f} "
            f"max_abs={float(v.abs().max()):.4f} "
            f"frac_gt_1={float((v.abs() > 1.0).float().mean()):.4f} "
            f"frac_gt_2={float((v.abs() > 2.0).float().mean()):.4f}"
        )


def collect_step_diag(env, motion_res, step_idx, motion_time_s, initial_root_xy, initial_left_foot_xy):
    import torch
    from isaac_utils.rotations import (
        get_euler_xyz_in_tensor,
        quat_angle_axis,
        quat_conjugate,
        quat_mul,
    )

    name_to_idx = {name: idx for idx, name in enumerate(env.dof_names)}
    tracked_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    joint_diag = {}
    ref_dof_pos = motion_res["dof_pos"][0]
    sim_dof_pos = env.simulator.dof_pos[0]
    torques = env.torques[0]
    torque_ratio = env.torques.abs() / env.torque_limits.clamp(min=1e-6)
    for joint_name in tracked_joint_names:
        idx = name_to_idx[joint_name]
        joint_diag[f"{joint_name}_err"] = float((ref_dof_pos[idx] - sim_dof_pos[idx]).detach().cpu().item())
        joint_diag[f"{joint_name}_torque"] = float(torques[idx].detach().cpu().item())
        joint_diag[f"{joint_name}_torque_ratio"] = float(torque_ratio[0, idx].detach().cpu().item())

    left_foot_body_idx = env.body_names.index(getattr(env.config.robot, "left_foot_name", "left_ankle_roll_link"))
    right_foot_body_idx = env.body_names.index(getattr(env.config.robot, "right_foot_name", "right_ankle_roll_link"))

    sim_root_state = env.simulator.robot_root_states[0]
    sim_root_pos = sim_root_state[:3]
    sim_root_rot = sim_root_state[3:7]
    sim_root_lin_vel = sim_root_state[7:10]
    sim_root_ang_vel = sim_root_state[10:13]
    ref_root_pos = motion_res["root_pos"][0]
    ref_root_rot = motion_res["root_rot"][0]
    ref_root_lin_vel = motion_res["root_vel"][0]
    ref_root_ang_vel = motion_res["root_ang_vel"][0]

    root_rot_diff = quat_mul(
        ref_root_rot.unsqueeze(0),
        quat_conjugate(sim_root_rot.unsqueeze(0), w_last=True),
        w_last=True,
    )
    root_rot_err_rad = float(quat_angle_axis(root_rot_diff, w_last=True)[0][0].detach().cpu().item())
    root_rot_rpy_err = get_euler_xyz_in_tensor(root_rot_diff)[0].detach().cpu()
    ref_root_rpy = get_euler_xyz_in_tensor(ref_root_rot.unsqueeze(0))[0].detach().cpu()
    sim_root_rpy = get_euler_xyz_in_tensor(sim_root_rot.unsqueeze(0))[0].detach().cpu()
    wrapped_rpy_err = ((ref_root_rpy - sim_root_rpy + math.pi) % (2 * math.pi)) - math.pi
    root_lin_vel_err_vec = (ref_root_lin_vel - sim_root_lin_vel).detach().cpu()
    root_ang_vel_err_vec = (ref_root_ang_vel - sim_root_ang_vel).detach().cpu()

    foot_force_z = env.simulator.contact_forces[0, env.feet_indices, 2].detach().cpu()
    foot_force_norm = torch.norm(env.simulator.contact_forces[0, env.feet_indices, :], dim=-1).detach().cpu()
    left_slip = float(torch.norm(env.simulator._rigid_body_vel[0, left_foot_body_idx, :2]).detach().cpu().item())
    right_slip = float(torch.norm(env.simulator._rigid_body_vel[0, right_foot_body_idx, :2]).detach().cpu().item())
    left_foot_pos = env.simulator._rigid_body_pos[0, left_foot_body_idx].detach().cpu()
    right_foot_pos = env.simulator._rigid_body_pos[0, right_foot_body_idx].detach().cpu()
    left_contact_flag = int(foot_force_norm[0].item() > 1.0)
    right_contact_flag = int(foot_force_norm[1].item() > 1.0)

    body_names = list(getattr(env, "body_names", getattr(env.simulator, "_body_list", [])))
    body_pos_err_vecs = env.dif_global_body_pos[0]
    body_pos_err_norms = torch.norm(body_pos_err_vecs, dim=-1)
    topk = min(5, body_pos_err_norms.shape[0])
    top_vals, top_ids = torch.topk(body_pos_err_norms, k=topk, largest=True, sorted=True)

    right_foot_pos_err_vec = env.dif_global_body_pos[0, right_foot_body_idx]
    right_foot_vel_err_vec = env.dif_global_body_vel[0, right_foot_body_idx]

    lower_body_err = float(env.dif_global_body_pos[:, env.lower_body_id, :].norm(dim=-1).mean().detach().cpu().item())
    gravity = env.projected_gravity[0].detach().cpu()
    gravity_margin_x = float(
        env.config.termination_scales.termination_gravity_x - abs(gravity[0].item())
    )
    gravity_margin_y = float(
        env.config.termination_scales.termination_gravity_y - abs(gravity[1].item())
    )
    base_height_margin = float(
        sim_root_pos[2].detach().cpu().item() - env.config.termination_scales.termination_min_base_height
    )
    body_pos_err_norm = torch.norm(env.dif_global_body_pos[0], dim=-1)
    motion_far_peak = float(body_pos_err_norm.max().detach().cpu().item())
    motion_far_margin = float(env.terminate_when_motion_far_threshold - motion_far_peak)
    contact_peak = float(
        torch.norm(env.simulator.contact_forces[0, env.termination_contact_indices, :], dim=-1).max().detach().cpu().item()
    ) if len(env.termination_contact_indices) > 0 else 0.0
    contact_margin = float(1.0 - contact_peak)
    torque_limit_margin = float((1.0 - torque_ratio[0]).min().detach().cpu().item())
    termination_margin = min(
        gravity_margin_x,
        gravity_margin_y,
        base_height_margin,
        motion_far_margin,
        contact_margin,
        torque_limit_margin,
    )
    diag = {
        "step": int(step_idx),
        "motion_time_s": float(motion_time_s),
        "joint_err_norm": float((ref_dof_pos - sim_dof_pos).norm().detach().cpu().item()),
        "lower_body_err": lower_body_err,
        "torque_clip_ratio": float((torque_ratio > 0.98).float().mean().detach().cpu().item()),
        "left_foot_contact_z": float(foot_force_z[0].item()),
        "right_foot_contact_z": float(foot_force_z[1].item()),
        "left_foot_contact_norm": float(foot_force_norm[0].item()),
        "right_foot_contact_norm": float(foot_force_norm[1].item()),
        "left_foot_contact_flag": left_contact_flag,
        "right_foot_contact_flag": right_contact_flag,
        "left_foot_slip_xy": left_slip,
        "right_foot_slip_xy": right_slip,
        "left_foot_x": float(left_foot_pos[0].item()),
        "left_foot_y": float(left_foot_pos[1].item()),
        "left_foot_z": float(left_foot_pos[2].item()),
        "right_foot_x": float(right_foot_pos[0].item()),
        "right_foot_y": float(right_foot_pos[1].item()),
        "right_foot_z": float(right_foot_pos[2].item()),
        "projected_gravity_x": float(gravity[0].item()),
        "projected_gravity_y": float(gravity[1].item()),
        "projected_gravity_z": float(gravity[2].item()),
        "root_pos_err": float(torch.norm(ref_root_pos - sim_root_pos).detach().cpu().item()),
        "root_rot_err_rad": root_rot_err_rad,
        "root_lin_vel_err": float(torch.norm(ref_root_lin_vel - sim_root_lin_vel).detach().cpu().item()),
        "root_ang_vel_err": float(torch.norm(ref_root_ang_vel - sim_root_ang_vel).detach().cpu().item()),
        "root_rot_err_roll": float(root_rot_rpy_err[0].item()),
        "root_rot_err_pitch": float(root_rot_rpy_err[1].item()),
        "root_rot_err_yaw": float(root_rot_rpy_err[2].item()),
        "ref_root_roll": float(ref_root_rpy[0].item()),
        "ref_root_pitch": float(ref_root_rpy[1].item()),
        "ref_root_yaw": float(ref_root_rpy[2].item()),
        "root_roll_err_wrapped": float(wrapped_rpy_err[0].item()),
        "root_pitch_err_wrapped": float(wrapped_rpy_err[1].item()),
        "root_yaw_err_wrapped": float(wrapped_rpy_err[2].item()),
        "root_lin_vel_err_x": float(root_lin_vel_err_vec[0].item()),
        "root_lin_vel_err_y": float(root_lin_vel_err_vec[1].item()),
        "root_lin_vel_err_z": float(root_lin_vel_err_vec[2].item()),
        "root_ang_vel_err_x": float(root_ang_vel_err_vec[0].item()),
        "root_ang_vel_err_y": float(root_ang_vel_err_vec[1].item()),
        "root_ang_vel_err_z": float(root_ang_vel_err_vec[2].item()),
        "root_roll": float(sim_root_rpy[0].item()),
        "root_pitch": float(sim_root_rpy[1].item()),
        "root_yaw": float(sim_root_rpy[2].item()),
        "root_height": float(sim_root_pos[2].detach().cpu().item()),
        "root_xy_drift": float(torch.norm(sim_root_pos[:2] - initial_root_xy).detach().cpu().item()),
        "root_x": float(sim_root_pos[0].detach().cpu().item()),
        "root_y": float(sim_root_pos[1].detach().cpu().item()),
        "left_foot_xy_drift": float(torch.norm(left_foot_pos[:2] - initial_left_foot_xy).item()),
        "right_foot_pos_err": float(torch.norm(right_foot_pos_err_vec).detach().cpu().item()),
        "right_foot_vel_err": float(torch.norm(right_foot_vel_err_vec).detach().cpu().item()),
        "right_foot_pos_err_x": float(right_foot_pos_err_vec[0].detach().cpu().item()),
        "right_foot_pos_err_y": float(right_foot_pos_err_vec[1].detach().cpu().item()),
        "right_foot_pos_err_z": float(right_foot_pos_err_vec[2].detach().cpu().item()),
        "right_foot_vel_err_x": float(right_foot_vel_err_vec[0].detach().cpu().item()),
        "right_foot_vel_err_y": float(right_foot_vel_err_vec[1].detach().cpu().item()),
        "right_foot_vel_err_z": float(right_foot_vel_err_vec[2].detach().cpu().item()),
        "max_body_pos_err": motion_far_peak,
        "termination_margin": termination_margin,
        "termination_gravity_margin_x": gravity_margin_x,
        "termination_gravity_margin_y": gravity_margin_y,
        "termination_base_height_margin": base_height_margin,
        "termination_motion_far_margin": motion_far_margin,
        "termination_motion_far_peak": motion_far_peak,
        "termination_contact_margin": contact_margin,
        "termination_contact_peak": contact_peak,
        "termination_torque_margin": torque_limit_margin,
        "reset": int(env.reset_buf[0].item()),
        "timeout": int(env.time_out_buf[0].item()),
    }
    for rank, (body_idx, body_err) in enumerate(zip(top_ids.detach().cpu().tolist(), top_vals.detach().cpu().tolist()), start=1):
        err_vec = body_pos_err_vecs[body_idx].detach().cpu()
        ref_pos = env.ref_body_pos_extend[0, body_idx].detach().cpu()
        sim_pos = env._rigid_body_pos_extend[0, body_idx].detach().cpu()
        body_name = body_names[body_idx] if body_idx < len(body_names) else f"body_{body_idx}"
        diag[f"top{rank}_body_idx"] = int(body_idx)
        diag[f"top{rank}_body_name"] = body_name
        diag[f"top{rank}_body_err"] = float(body_err)
        diag[f"top{rank}_body_err_x"] = float(err_vec[0].item())
        diag[f"top{rank}_body_err_y"] = float(err_vec[1].item())
        diag[f"top{rank}_body_err_z"] = float(err_vec[2].item())
        diag[f"top{rank}_body_ref_x"] = float(ref_pos[0].item())
        diag[f"top{rank}_body_ref_y"] = float(ref_pos[1].item())
        diag[f"top{rank}_body_ref_z"] = float(ref_pos[2].item())
        diag[f"top{rank}_body_sim_x"] = float(sim_pos[0].item())
        diag[f"top{rank}_body_sim_y"] = float(sim_pos[1].item())
        diag[f"top{rank}_body_sim_z"] = float(sim_pos[2].item())
    diag["projected_gravity"] = [
        diag["projected_gravity_x"],
        diag["projected_gravity_y"],
        diag["projected_gravity_z"],
    ]
    diag.update(joint_diag)
    return diag


def print_diag(diag):
    print(
        f"[STEP {diag['step']:04d}] "
        f"t={diag['motion_time_s']:.3f}s "
        f"joint_err={diag['joint_err_norm']:.4f} "
        f"lower_body_err={diag['lower_body_err']:.4f} "
        f"root_pos_err={diag['root_pos_err']:.4f} "
        f"root_rot_err={diag['root_rot_err_rad']:.4f} "
        f"torque_clip_ratio={diag['torque_clip_ratio']:.4f} "
        f"lFz={diag['left_foot_contact_z']:.2f} "
        f"rFz={diag['right_foot_contact_z']:.2f} "
        f"lSlip={diag['left_foot_slip_xy']:.3f} "
        f"rSlip={diag['right_foot_slip_xy']:.3f} "
        f"gravity={diag['projected_gravity']} "
        f"reset={diag['reset']} timeout={diag['timeout']}"
    )
    print(
        "[RIGHT_LEG] "
        f"hip_pitch err={diag['right_hip_pitch_joint_err']:.4f} "
        f"knee err={diag['right_knee_joint_err']:.4f} "
        f"ankle_pitch err={diag['right_ankle_pitch_joint_err']:.4f}"
    )
    print(
        "[ROOT] "
        f"rpy=({diag['root_roll']:.4f}, {diag['root_pitch']:.4f}, {diag['root_yaw']:.4f}) "
        f"h={diag['root_height']:.4f} xy_drift={diag['root_xy_drift']:.4f}"
    )
    print(
        "[RIGHT_FOOT] "
        f"contact={diag['right_foot_contact_flag']} "
        f"pos_err={diag['right_foot_pos_err']:.4f} "
        f"vel_err={diag['right_foot_vel_err']:.4f}"
    )


def print_final_diag(diag):
    gravity = diag["projected_gravity"]
    print(
        "[FINAL_DIAG] "
        f"step={diag['step']} "
        f"t={diag['motion_time_s']:.6f} "
        f"joint_err={diag['joint_err_norm']:.6f} "
        f"lower_body_err={diag['lower_body_err']:.6f} "
        f"projected_gravity=({gravity[0]:.6f},{gravity[1]:.6f},{gravity[2]:.6f}) "
        f"root_pos_err={diag['root_pos_err']:.6f} "
        f"root_rot_err_rad={diag['root_rot_err_rad']:.6f} "
        f"torque_clip_ratio={diag['torque_clip_ratio']:.6f} "
        f"left_support_contact={diag['left_foot_contact_flag']} "
        f"left_support_force={diag['left_foot_contact_norm']:.6f} "
        f"left_support_slip={diag['left_foot_slip_xy']:.6f} "
        f"right_contact={diag['right_foot_contact_flag']} "
        f"right_contact_force={diag['right_foot_contact_norm']:.6f} "
        f"right_foot_pos_err={diag['right_foot_pos_err']:.6f} "
        f"right_foot_vel_err={diag['right_foot_vel_err']:.6f}"
    )


def summarize_contact_timing(step_diags, key):
    contact_steps = [diag["step"] for diag in step_diags if diag[key] > 0]
    if not contact_steps:
        return "none"

    segments = []
    start = contact_steps[0]
    prev = contact_steps[0]
    for step in contact_steps[1:]:
        if step == prev + 1:
            prev = step
            continue
        segments.append((start, prev))
        start = step
        prev = step
    segments.append((start, prev))
    return ",".join(f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in segments)


def export_step_diags_csv(step_diags, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for diag in step_diags:
        for key, value in diag.items():
            if isinstance(value, list):
                continue
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for diag in step_diags:
            row = {key: value for key, value in diag.items() if not isinstance(value, list)}
            writer.writerow(row)


def default_csv_path(args, force_components):
    motion_stem = Path(args.motion_file).stem
    force_tag = "none" if not force_components else "_".join(sorted(force_components))
    speed_tag = f"{args.speed_scale:.2f}".replace(".", "p")
    return REPO_ROOT / "tmp" / "replay_diag" / f"{motion_stem}__force_{force_tag}__speed_{speed_tag}.csv"


def build_config(args):
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/base.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/robot/robot_base.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/simulator/isaacgym.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/terrain/terrain_locomotion_plane.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/domain_rand/domain_rand_base.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/obs/motion_tracking/motion_tracking.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/env/legged_base.yaml"),
        OmegaConf.load(REPO_ROOT / "humanoidverse/config/env/motion_tracking_kick.yaml"),
        OmegaConf.create({"algo": {"config": {"module_dict": {}}}}),
    )

    with open_dict(cfg):
        cfg.headless = bool(args.headless)
        cfg.num_envs = 1
        cfg.seed = int(args.seed)
        cfg.device = args.device
        cfg.use_wandb = False
        cfg.save_rendering_dir = None
        cfg.auto_record = False
        cfg.auto_record_num_frames = -1
        cfg.offscreen_record = False
        cfg.offscreen_record_width = 1600
        cfg.offscreen_record_height = 900
        cfg.offscreen_record_fps = 50
        cfg.experiment_name = "ref_dof_pd_replay"
        cfg.robot.asset.asset_root = str(REPO_ROOT / "humanoidverse/data/robots")
        cfg.robot.motion.asset.assetRoot = str(REPO_ROOT / "humanoidverse/data/robots/g1")
        cfg.robot.motion.motion_file = str(Path(args.motion_file).resolve())
        cfg.env.config.headless = bool(args.headless)
        cfg.env.config.num_envs = 1
        cfg.env.config.env_spacing = cfg.env.config.get("env_spacing", cfg.env_spacing)
        cfg.env.config.simulator = cfg.simulator
        cfg.env.config.save_rendering_dir = None
        cfg.env.config.max_episode_length_s = float(args.max_episode_length_s)
        cfg.env.config.enforce_randomize_motion_start_eval = False
        cfg.env.config.randomize_motion_start_train = False
        cfg.env.config.resample_motion_when_training = False
        cfg.env.config.noise_to_initial_level = 0.0
        cfg.env.config.normalization.clip_actions = 100.0
        cfg.env.config.normalization.clip_observations = 100.0
        cfg.env.config.auto_record = False
        cfg.env.config.auto_record_num_frames = -1
        cfg.env.config.offscreen_record = False
        cfg.env.config.offscreen_record_width = 1600
        cfg.env.config.offscreen_record_height = 900
        cfg.env.config.offscreen_record_fps = 50
        cfg.env.config.obs.add_noise_currculum = False
        cfg.env.config.obs.noise_initial_value = 0.0
        cfg.env.config.obs.noise_value_max = 1.0
        cfg.env.config.obs.noise_value_min = 0.0
        cfg.env.config.obs.soft_dof_pos_curriculum_degree = 0.00001
        if args.disable_early_termination:
            cfg.env.config.termination.terminate_by_gravity = False
            cfg.env.config.termination.terminate_when_motion_far = False
            cfg.env.config.termination.terminate_by_contact = False
        elif args.disable_motion_far_termination:
            cfg.env.config.termination.terminate_when_motion_far = False

    disable_domain_rand(cfg.env.config)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Replay reference dof_pos as direct PD targets in motion_tracking_kick."
    )
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 means replay until motion end or reset.")
    parser.add_argument("--max-episode-length-s", type=float, default=100000.0)
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Reference playback speed for PD target generation.")
    parser.add_argument("--force-root", action="store_true", help="Force all root state components to reference each control step.")
    parser.add_argument(
        "--force-root-components",
        type=str,
        default="",
        help="Comma-separated subset of root_pos,root_rot,root_lin_vel,root_ang_vel.",
    )
    parser.add_argument("--report-action-stats", action="store_true", help="Print required normalized action range over the whole motion.")
    parser.add_argument("--disable-motion-far-termination", action="store_true", help="Disable only motion-far termination.")
    parser.add_argument("--export-csv", type=str, default="", help="Export per-step diagnostics to this CSV path.")
    parser.add_argument("--window-start", type=float, default=-1.0, help="Optional diagnostic window start time in seconds.")
    parser.add_argument("--window-end", type=float, default=-1.0, help="Optional diagnostic window end time in seconds.")
    parser.add_argument(
        "--disable-early-termination",
        action="store_true",
        help="Disable gravity/motion-far/contact termination to inspect pure open-loop PD drift.",
    )
    args = parser.parse_args()

    import humanoidverse.utils.config_utils  # noqa: F401

    cfg = build_config(args)
    if str(cfg.simulator._target_).endswith("IsaacGym"):
        import isaacgym  # noqa: F401
    import torch
    from hydra.utils import instantiate

    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(cfg)

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = instantiate(cfg.env, device=device)
    env.set_is_evaluating()
    env.reset_all()
    force_components = parse_force_root_components(args)
    initial_root_xy = env.simulator.robot_root_states[0, :2].detach().clone()
    left_foot_body_idx = env.body_names.index(getattr(env.config.robot, "left_foot_name", "left_ankle_roll_link"))
    initial_left_foot_xy = env.simulator._rigid_body_pos[0, left_foot_body_idx, :2].detach().cpu().clone()

    motion_id = torch.zeros(1, dtype=torch.long, device=env.device)
    motion_length = float(env._motion_lib.get_motion_length(motion_id)[0].item())
    action_scale = float(env.config.robot.control.action_scale)
    total_steps = int(torch.ceil(torch.tensor(motion_length / env.dt)).item())
    if args.max_steps > 0:
        total_steps = min(total_steps, int(args.max_steps))
    elif args.speed_scale != 1.0:
        total_steps = int(math.ceil(motion_length / (env.dt * max(args.speed_scale, 1e-8))))

    print(f"[INFO] motion_file={Path(args.motion_file).resolve()}")
    print(f"[INFO] device={device}")
    print(f"[INFO] dt={env.dt:.6f}s motion_length={motion_length:.6f}s total_steps={total_steps}")
    print(f"[INFO] action_scale={action_scale:.6f}")
    print(
        f"[INFO] speed_scale={float(args.speed_scale):.6f} "
        f"force_root={bool(force_components)} force_components={sorted(force_components)}"
    )
    print(
        "[INFO] term_flags="
        f"gravity={bool(env.config.termination.terminate_by_gravity)} "
        f"motion_far={bool(env.config.termination.terminate_when_motion_far)} "
        f"contact={bool(env.config.termination.terminate_by_contact)} "
        f"motion_end={bool(env.config.termination.terminate_when_motion_end)}"
    )

    if args.report_action_stats:
        compute_required_action_stats(env, motion_length, args.speed_scale)

    failure_step = None
    success_timeout = False
    step_diags = []

    for step in range(total_steps):
        motion_time_s = min((step + 1) * env.dt * args.speed_scale, motion_length)
        motion_res = compute_motion_res(env, motion_time_s)
        ref_dof_pos = motion_res["dof_pos"]
        actions = (ref_dof_pos - env.default_dof_pos) / action_scale

        force_root_state_from_motion(env, motion_res, force_components)
        obs, rew, reset, extras = env.step({"actions": actions})
        del obs, rew, extras

        diag = collect_step_diag(env, motion_res, step + 1, motion_time_s, initial_root_xy, initial_left_foot_xy)
        step_diags.append(diag)
        if step == 0 or (step + 1) % 20 == 0 or bool(reset[0].item()):
            print_diag(diag)

        if bool(reset[0].item()):
            failure_step = step + 1
            success_timeout = bool(env.time_out_buf[0].item())
            break

    if failure_step is not None and not success_timeout:
        lo = max(0, failure_step - 3)
        hi = min(len(step_diags), failure_step + 1)
        print(f"[DIAG_WINDOW] failure_step={failure_step}")
        for diag in step_diags[lo:hi]:
            print_diag(diag)

    if step_diags:
        print_final_diag(step_diags[-1])
        print(
            "[CONTACT_TIMING] "
            f"left={summarize_contact_timing(step_diags, 'left_foot_contact_flag')} "
            f"right={summarize_contact_timing(step_diags, 'right_foot_contact_flag')}"
        )

    if args.window_start >= 0.0 and args.window_end >= args.window_start:
        window_diags = [diag for diag in step_diags if args.window_start <= diag["motion_time_s"] <= args.window_end]
        print(
            f"[WINDOW] start={args.window_start:.6f} end={args.window_end:.6f} "
            f"num_steps={len(window_diags)}"
        )
        for diag in window_diags:
            print_diag(diag)

    csv_path = args.export_csv
    if not csv_path and (args.window_start >= 0.0 or force_components):
        csv_path = str(default_csv_path(args, force_components))
    if csv_path:
        export_step_diags_csv(step_diags, csv_path)
        print(f"[CSV] wrote={csv_path}")

    if failure_step is None:
        print(f"[RESULT] replay_finished_without_reset steps={total_steps}")
    elif success_timeout:
        print(
            f"[RESULT] success_reached_motion_end step={failure_step} "
            f"sim_time={failure_step * env.dt:.6f}s motion_length={motion_length:.6f}s"
        )
    else:
        print(
            f"[RESULT] failed_before_motion_end step={failure_step} "
            f"sim_time={failure_step * env.dt:.6f}s motion_length={motion_length:.6f}s"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Q1 Goalkeeper Side-Slip Physics Diagnostic
===========================================

Diagnoses root causes of lateral slipping: foot collision geometry,
foot/ground/ball friction, ball impulse, PD-induced slip, support polygon.

Does NOT: train PPO, modify reward/URDF/config, or affect G1.

Usage (from /root/autodl-tmp/ASAP):
  python tools/diagnose_q1_side_slip_physics.py \
      +exp=q1_goalkeeper_smoke \
      "robot.asset.asset_root=/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/robots/q1" \
      "robot.asset.urdf_file=q1_22dof_goalkeeper_collision.urdf" \
      num_envs=1 headless=True
"""

import os, sys, json, csv, math, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import hydra
from omegaconf import OmegaConf, open_dict

# OmegaConf resolvers
try:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda p, a, b: a if p else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception:
    pass

import isaacgym  # noqa
from isaacgym import gymapi
import torch

# ────────────────────────────────────────────────────────────
# Config helpers
# ────────────────────────────────────────────────────────────
DT = 1.0 / 50.0  # sim dt (50 Hz)
CONTROL_DECIMATION = 3  # control at ~16.67 Hz
SIM_STEPS_PER_FRAME = CONTROL_DECIMATION
FRAME_DT = DT * CONTROL_DECIMATION  # ~0.06s per frame

# ────────────────────────────────────────────────────────────
@hydra.main(config_path="../humanoidverse/config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    orig_cwd = hydra.utils.get_original_cwd()
    out_dir = Path(orig_cwd) / "outputs/q1_side_slip_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    from humanoidverse.utils.helpers import pre_process_config
    pre_process_config(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs
        if hasattr(config, "algo") and hasattr(config.algo, "config"):
            config.env.config.algo = config.algo

    config.env.config.save_rendering_dir = str(out_dir / "renderings")

    from humanoidverse.envs.base_task.base_task import BaseTask
    from hydra.utils import instantiate

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 65)
    print("  Q1 SIDE-SLIP PHYSICS DIAGNOSTIC")
    print("=" * 65)
    print(f"  URDF: {config.robot.asset.asset_root}/{config.robot.asset.urdf_file}")
    print(f"  root_z: {config.robot.init_state.pos[2]}")
    print(f"  action_scale: {config.robot.control.action_scale}")
    print(f"  control_type: {config.robot.control.control_type}")

    # ── Create env ──
    print("\n[1/8] Creating environment...")
    env: BaseTask = instantiate(config=config.env, device=device)
    sim = env.simulator
    gym = sim.gym
    env_ptr = sim.envs[0]
    robot_handle = sim.robot_handles[0]
    num_envs = 1
    num_dof = sim.num_dof
    num_bodies = sim.num_bodies
    body_names = list(sim.body_names)
    dof_names = list(sim.dof_names)

    # Foot / ball body indices
    left_foot_name = config.robot.left_foot_name
    right_foot_name = config.robot.right_foot_name
    left_foot_idx = body_names.index(left_foot_name)
    right_foot_idx = body_names.index(right_foot_name)
    ball_body_env_idx = getattr(env, 'ball_body_env_idx',
                                 sim._cubeA_id)  # index for contact forces

    # PD gains and default pose
    p_gains = env.p_gains.clone()
    d_gains = env.d_gains.clone()
    default_dof_pos = env.default_dof_pos.clone()
    action_scale = config.robot.control.action_scale
    dof_pos_limits = env.dof_pos_limits
    torque_limits = env.torque_limits

    print(f"  Bodies: {num_bodies}  DOFs: {num_dof}")
    print(f"  Left foot: {left_foot_name}[{left_foot_idx}]")
    print(f"  Right foot: {right_foot_name}[{right_foot_idx}]")

    # ═══════════════════════════════════════════════════
    # PART 1: Physics parameters
    # ═══════════════════════════════════════════════════
    print("\n[2/8] Physics parameters...")

    # Get robot shape properties
    actor_shape_props = gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
    asset_shape_ranges = gym.get_asset_rigid_body_shape_indices(sim.robot_asset)

    # Ground properties
    ground_friction = config.simulator.config.terrain.static_friction
    ground_restitution = config.simulator.config.terrain.restitution

    # Ball properties (stored on simulator)
    ball_radius = sim.ball_radius
    ball_mass = sim.ball_mass
    ball_friction = sim.ball_friction
    ball_restitution = sim.ball_restitution

    # Foot shape details
    foot_shapes = {}
    for foot_name, foot_idx in [(left_foot_name, left_foot_idx),
                                 (right_foot_name, right_foot_idx)]:
        ir = asset_shape_ranges[foot_idx]
        shapes = []
        for k in range(ir.count):
            si = ir.start + k
            if si < len(actor_shape_props):
                p = actor_shape_props[si]
                shapes.append({
                    "shape_idx": int(si),
                    "friction": float(p.friction),
                    "restitution": float(p.restitution),
                    "contact_offset": float(p.contact_offset),
                    "rest_offset": float(p.rest_offset),
                })
        foot_shapes[foot_name] = shapes

    # Print key params
    print(f"\n  ── Ground ──")
    print(f"    friction={ground_friction}  restitution={ground_restitution}")
    print(f"\n  ── Ball ──")
    print(f"    radius={ball_radius}  mass={ball_mass}")
    print(f"    friction={ball_friction}  restitution={ball_restitution}")
    print(f"\n  ── Foot collision shapes ──")
    for fn, shapes in foot_shapes.items():
        for s in shapes:
            print(f"    {fn}[s{s['shape_idx']}]: "
                  f"friction={s['friction']}  restitution={s['restitution']}  "
                  f"contact_offset={s['contact_offset']}  rest_offset={s['rest_offset']}")

    # Save shape properties
    shape_props = {
        "ground": {"friction": ground_friction, "restitution": ground_restitution},
        "ball": {"radius": ball_radius, "mass": ball_mass,
                 "friction": ball_friction, "restitution": ball_restitution},
        "foot_shapes": {fn: shapes for fn, shapes in foot_shapes.items()},
    }
    with open(out_dir / "shape_properties.json", "w") as f:
        json.dump(shape_props, f, indent=2, default=float)

    # CSV version
    with open(out_dir / "shape_properties.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "body", "shape_idx", "friction", "restitution",
                     "contact_offset", "rest_offset", "radius", "mass"])
        w.writerow(["ground", "-", "-", ground_friction, ground_restitution, "-", "-", "-", "-"])
        w.writerow(["ball", "-", "-", ball_friction, ball_restitution, "-", "-",
                     ball_radius, ball_mass])
        for fn, shapes in foot_shapes.items():
            for s in shapes:
                w.writerow(["foot", fn, s["shape_idx"], s["friction"],
                            s["restitution"], s["contact_offset"], s["rest_offset"],
                            "-", "-"])

    # ═══════════════════════════════════════════════════
    # Helper: manual physics step
    # ═══════════════════════════════════════════════════
    def reset_env_with_pose():
        """Reset robot to config default standing pose and push to GPU sim."""
        env_ids = torch.tensor([0], device=device, dtype=torch.long)
        # Set dofs to config default_joint_angles
        cfg_angles = config.robot.init_state.default_joint_angles
        for i, name in enumerate(dof_names):
            if name in cfg_angles:
                val = float(cfg_angles[name])
                sim.dof_pos[0, i] = val
                default_dof_pos[0, i] = val  # sync PD target
        sim.dof_vel[0, :] = 0.0
        # Set root to init_state
        init_xyz = [float(v) for v in config.robot.init_state.pos]
        init_quat = [float(v) for v in config.robot.init_state.rot]
        sim.robot_root_states[0, :3] = torch.tensor(init_xyz, device=device)
        sim.robot_root_states[0, 3:7] = torch.tensor(init_quat, device=device)
        sim.robot_root_states[0, 7:13] = 0.0
        # Push to GPU sim
        dof_states = torch.cat([sim.dof_pos[env_ids].unsqueeze(-1),
                                sim.dof_vel[env_ids].unsqueeze(-1)], dim=-1)
        sim.set_dof_state_tensor(env_ids, dof_states)
        sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)
        # Settle physics briefly
        for _ in range(10):
            sim.simulate_at_each_physics_step()
        sim.refresh_sim_tensors()

    def set_ball_far_away():
        """Move ball far away (effectively disabled)."""
        sim.cubeA_state[0, 0] = 100.0   # x far
        sim.cubeA_state[0, 1] = 0.0
        sim.cubeA_state[0, 2] = 100.0  # z high
        sim.cubeA_state[0, 7:10] = 0.0  # zero vel
        sim.cubeA_state[0, 10:13] = 0.0
        eids = torch.tensor([0], device=device, dtype=torch.int32)
        sim.set_actor_root_state_tensor(eids, sim.all_root_states)

    def set_ball_static_in_front():
        """Place ball stationary 2m in front of robot."""
        root_x = sim.robot_root_states[0, 0].item()
        sim.cubeA_state[0, 0] = root_x + 2.0
        sim.cubeA_state[0, 1] = 0.0
        sim.cubeA_state[0, 2] = 0.15  # just above ground
        sim.cubeA_state[0, 7:10] = 0.0
        sim.cubeA_state[0, 10:13] = 0.0
        eids = torch.tensor([0], device=device, dtype=torch.int32)
        sim.set_actor_root_state_tensor(eids, sim.all_root_states)

    def set_ball_moving():
        """Set ball moving toward robot at moderate speed."""
        root_x = sim.robot_root_states[0, 0].item()
        sim.cubeA_state[0, 0] = root_x + 2.5
        sim.cubeA_state[0, 1] = 0.0
        sim.cubeA_state[0, 2] = 0.25  # mid-height shot
        sim.cubeA_state[0, 7] = -3.0   # toward robot at 3 m/s
        sim.cubeA_state[0, 8] = 0.0
        sim.cubeA_state[0, 9] = 0.0
        sim.cubeA_state[0, 10:13] = 0.0
        eids = torch.tensor([0], device=device, dtype=torch.int32)
        sim.set_actor_root_state_tensor(eids, sim.all_root_states)

    def compute_pd_torque(actions):
        """Compute PD torque from actions (same as env._compute_torques but raw)."""
        actions_scaled = actions * action_scale
        torques = p_gains * (actions_scaled + default_dof_pos - sim.dof_pos) \
                  - d_gains * sim.dof_vel
        torques = torch.clamp(torques, -torque_limits, torque_limits)
        return torques

    def run_physics_frame(torques, hold_ball_static=False, hold_ball_far=False):
        """Run one control frame (control_decimation physics steps).
        Returns dict of per-step data from the LAST sim step.
        """
        for _ in range(CONTROL_DECIMATION):
            sim.apply_torques_at_dof(torques)
            sim.simulate_at_each_physics_step()

            if hold_ball_far:
                set_ball_far_away()
            elif hold_ball_static:
                # Keep ball static at current position
                sim.cubeA_state[0, 7:13] = 0.0

        sim.refresh_sim_tensors()

        # Extract data
        root_state = sim.robot_root_states[0].cpu().numpy()
        dof_pos = sim.dof_pos[0].cpu().numpy()
        dof_vel = sim.dof_vel[0].cpu().numpy()
        body_pos = sim._rigid_body_pos[0].cpu().numpy()
        body_vel = sim._rigid_body_vel[0].cpu().numpy()
        body_rot = sim._rigid_body_rot[0].cpu().numpy()
        contact_f = sim.contact_forces[0].cpu().numpy()

        # Foot data
        lf_pos = body_pos[left_foot_idx]
        rf_pos = body_pos[right_foot_idx]
        lf_vel = body_vel[left_foot_idx]
        rf_vel = body_vel[right_foot_idx]
        lf_contact = contact_f[left_foot_idx]
        rf_contact = contact_f[right_foot_idx]
        lf_contact_norm = float(np.linalg.norm(lf_contact))
        rf_contact_norm = float(np.linalg.norm(rf_contact))

        # Ball data
        ball_pos = sim.cubeA_state[0, :3].cpu().numpy()
        ball_vel = sim.cubeA_state[0, 7:10].cpu().numpy()
        ball_contact = contact_f[ball_body_env_idx]
        ball_contact_norm = float(np.linalg.norm(ball_contact))

        # Max contact force and body
        body_contact_norms = np.linalg.norm(contact_f[:num_bodies], axis=1)
        max_contact_body_idx = int(np.argmax(body_contact_norms))
        max_contact_force = float(body_contact_norms[max_contact_body_idx])

        # Foot clearance
        lf_clearance = float(lf_pos[2])  # z relative to ground (z=0)
        rf_clearance = float(rf_pos[2])

        # Projected gravity (approximate from root quat)
        root_quat = root_state[3:7]  # xyzw
        # Gravity in world: [0,0,-1]. In base frame: rotate by inverse quat
        # Simplified: project world Z onto base Z via quat rotation
        qx, qy, qz, qw = root_quat
        # R * [0,0,1] for quaternion xyzw
        gz = 2*(qx*qz - qw*qy)
        gy = 2*(qw*qx + qy*qz)
        gx = 1 - 2*(qx*qx + qy*qy)
        proj_g = np.array([gx, gy, gz])

        return {
            "root_x": float(root_state[0]),
            "root_y": float(root_state[1]),
            "root_z": float(root_state[2]),
            "root_qx": float(root_state[3]),
            "root_qy": float(root_state[4]),
            "root_qz": float(root_state[5]),
            "root_qw": float(root_state[6]),
            "root_vx": float(root_state[7]),
            "root_vy": float(root_state[8]),
            "root_vz": float(root_state[9]),
            "root_wx": float(root_state[10]),
            "root_wy": float(root_state[11]),
            "root_wz": float(root_state[12]),
            "proj_gx": float(proj_g[0]),
            "proj_gy": float(proj_g[1]),
            "proj_gz": float(proj_g[2]),
            "lf_pos_x": float(lf_pos[0]), "lf_pos_y": float(lf_pos[1]),
            "lf_pos_z": float(lf_pos[2]),
            "lf_vel_x": float(lf_vel[0]), "lf_vel_y": float(lf_vel[1]),
            "lf_vel_z": float(lf_vel[2]),
            "lf_contact_fx": float(lf_contact[0]),
            "lf_contact_fy": float(lf_contact[1]),
            "lf_contact_fz": float(lf_contact[2]),
            "lf_contact_norm": lf_contact_norm,
            "lf_clearance": lf_clearance,
            "rf_pos_x": float(rf_pos[0]), "rf_pos_y": float(rf_pos[1]),
            "rf_pos_z": float(rf_pos[2]),
            "rf_vel_x": float(rf_vel[0]), "rf_vel_y": float(rf_vel[1]),
            "rf_vel_z": float(rf_vel[2]),
            "rf_contact_fx": float(rf_contact[0]),
            "rf_contact_fy": float(rf_contact[1]),
            "rf_contact_fz": float(rf_contact[2]),
            "rf_contact_norm": rf_contact_norm,
            "ball_pos_x": float(ball_pos[0]),
            "ball_pos_y": float(ball_pos[1]),
            "ball_pos_z": float(ball_pos[2]),
            "ball_vel_x": float(ball_vel[0]),
            "ball_vel_y": float(ball_vel[1]),
            "ball_vel_z": float(ball_vel[2]),
            "ball_contact_norm": ball_contact_norm,
            "max_contact_force": max_contact_force,
            "max_contact_body": body_names[max_contact_body_idx]
            if max_contact_body_idx < len(body_names) else f"actor_{max_contact_body_idx}",
        }

    def run_episode(num_frames, torque_mode="zero", ball_mode="none",
                    set_friction=None):
        """Run one episode and return per-frame data + summary metrics.

        torque_mode: "zero" (true zero torque), "pd_zero_action" (PD with zero action)
        ball_mode: "none" (far away), "static" (in front), "moving" (shot toward robot)
        set_friction: if not None, temporarily set all shapes to this friction value
        """
        reset_env_with_pose()

        if ball_mode == "none":
            set_ball_far_away()
        elif ball_mode == "static":
            set_ball_static_in_front()
        elif ball_mode == "moving":
            set_ball_moving()

        # Apply runtime friction if requested
        original_friction = None
        if set_friction is not None:
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
            original_friction = [float(p.friction) for p in shape_props]
            for p in shape_props:
                p.friction = float(set_friction)
            gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, shape_props)

        root_y0 = sim.robot_root_states[0, 1].item()
        frames = []
        ball_contacted = False
        ball_contact_frame = -1

        for fi in range(num_frames):
            if torque_mode == "zero":
                torques = torch.zeros(1, num_dof, device=device)
            elif torque_mode == "pd_zero_action":
                torques = compute_pd_torque(torch.zeros(1, num_dof, device=device))
            else:
                torques = torch.zeros(1, num_dof, device=device)

            hold_static = (ball_mode == "static")
            hold_far = (ball_mode == "none")
            data = run_physics_frame(torques, hold_ball_static=hold_static,
                                     hold_ball_far=hold_far)
            data["frame"] = fi
            data["root_y0"] = root_y0
            data["root_disp_y"] = data["root_y"] - root_y0
            frames.append(data)

            # Track ball contact
            if not ball_contacted and data["ball_contact_norm"] > 1.0:
                ball_contacted = True
                ball_contact_frame = fi

        # Restore friction
        if original_friction is not None:
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
            for i, p in enumerate(shape_props):
                if i < len(original_friction):
                    p.friction = original_friction[i]
            gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, shape_props)

        # Compute summary metrics
        root_vy_arr = np.array([f["root_vy"] for f in frames])
        root_disp_y_arr = np.array([f["root_disp_y"] for f in frames])
        lf_contact_arr = np.array([f["lf_contact_norm"] for f in frames])
        rf_contact_arr = np.array([f["rf_contact_norm"] for f in frames])
        lf_vel_y_arr = np.array([f["lf_vel_y"] for f in frames])
        rf_vel_y_arr = np.array([f["rf_vel_y"] for f in frames])

        contact_thresh = 1.0  # N
        foot_slip_thresh = 0.05  # m/s

        side_slip_frames = np.sum(np.abs(root_vy_arr) > 0.15)
        large_disp_frames = np.sum(np.abs(root_disp_y_arr) > 0.05)
        side_slip_rate = side_slip_frames / num_frames
        large_disp_rate = large_disp_frames / num_frames

        max_side_vel = float(np.max(np.abs(root_vy_arr)))
        max_side_disp = float(np.max(np.abs(root_disp_y_arr)))
        final_side_disp = float(root_disp_y_arr[-1])

        # Foot lateral slip when in contact
        lf_in_contact = lf_contact_arr > contact_thresh
        rf_in_contact = rf_contact_arr > contact_thresh
        lf_slip = np.abs(lf_vel_y_arr) > foot_slip_thresh
        rf_slip = np.abs(rf_vel_y_arr) > foot_slip_thresh
        lf_slip_frames = np.sum(lf_in_contact & lf_slip)
        rf_slip_frames = np.sum(rf_in_contact & rf_slip)

        # Contact force imbalance: abs(left_z - right_z) / (left_z + right_z)
        lf_z = np.array([f["lf_contact_fz"] for f in frames])
        rf_z = np.array([f["rf_contact_fz"] for f in frames])
        total_z = lf_z + rf_z + 1e-6
        imbalance = np.mean(np.abs(lf_z - rf_z) / total_z)

        # Fall detection: projected gravity z < 0.5 (robot tilted > 60 deg)
        proj_gz_arr = np.array([f["proj_gz"] for f in frames])
        fell = np.any(proj_gz_arr < 0.5)

        # Ball contact stats
        ball_contact_frames = np.sum(np.array([f["ball_contact_norm"] for f in frames]) > 1.0)

        # Velocity jump at ball contact
        vel_jump = 0.0
        if ball_contact_frame >= 0 and ball_contact_frame < num_frames - 1:
            vy_before = root_vy_arr[ball_contact_frame]
            vy_after = root_vy_arr[min(ball_contact_frame + 2, num_frames - 1)]
            vel_jump = float(vy_after - vy_before)

        summary = {
            "side_slip_rate": side_slip_rate,
            "large_disp_rate": large_disp_rate,
            "max_side_vel": max_side_vel,
            "max_side_disp": max_side_disp,
            "final_side_disp": final_side_disp,
            "lf_slip_frames": int(lf_slip_frames),
            "rf_slip_frames": int(rf_slip_frames),
            "contact_imbalance": float(imbalance),
            "fell": bool(fell),
            "ball_contacted": ball_contacted,
            "ball_contact_frame": ball_contact_frame,
            "ball_contact_frames": int(ball_contact_frames),
            "vel_jump_at_contact": vel_jump,
            "mean_root_vy": float(np.mean(root_vy_arr)),
            "std_root_vy": float(np.std(root_vy_arr)),
        }
        return frames, summary

    # ═══════════════════════════════════════════════════
    # PART 3: Four diagnostic cases
    # ═══════════════════════════════════════════════════
    print("\n[3/8] Running diagnostic rollouts...")

    NUM_RESETS = 5
    NUM_RESETS = 200
FRAMES_PER_RESET = 200  # ~12 seconds at 16.67Hz

    cases = [
        ("A: no_ball_zero_torque", "zero", "none"),
        ("B: no_ball_zero_action", "pd_zero_action", "none"),
        ("C: static_ball_zero_action", "pd_zero_action", "static"),
        ("D: moving_ball_zero_action", "pd_zero_action", "moving"),
    ]

    all_case_summaries = {}
    all_frames = []  # aggregated

    for case_name, torque_mode, ball_mode in cases:
        print(f"\n  Running {case_name} ({NUM_RESETS} resets × {FRAMES_PER_RESET} frames)...")
        summaries = []
        t0 = time.time()
        for ri in range(NUM_RESETS):
            frames, summary = run_episode(FRAMES_PER_RESET,
                                          torque_mode=torque_mode,
                                          ball_mode=ball_mode)
            summary["reset_idx"] = ri
            summary["case"] = case_name
            summaries.append(summary)
            # Append frames with case tag (sample every 10 frames to keep CSV small)
            for fi, f in enumerate(frames):
                if fi % 10 == 0 or fi < 5 or fi > FRAMES_PER_RESET - 5:
                    f["case"] = case_name
                    f["reset_idx"] = ri
                    all_frames.append(f)
            if (ri + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {ri+1}/{NUM_RESETS} done ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        avg_ssr = np.mean([s["side_slip_rate"] for s in summaries])
        avg_mv = np.mean([s["max_side_vel"] for s in summaries])
        avg_fell = np.mean([s["fell"] for s in summaries])
        print(f"    Done in {elapsed:.1f}s. avg side_slip_rate={avg_ssr:.4f} "
              f"max_vel={avg_mv:.3f} fell_rate={avg_fell:.3f}")

        all_case_summaries[case_name] = summaries

    # ═══════════════════════════════════════════════════
    # PART 4: Compute metrics & save frames CSV
    # ═══════════════════════════════════════════════════
    print("\n[4/8] Saving frame data...")
    csv_fields = [
        "case", "reset_idx", "frame",
        "root_x", "root_y", "root_z", "root_disp_y",
        "root_vx", "root_vy", "root_vz",
        "root_wx", "root_wy", "root_wz",
        "proj_gx", "proj_gy", "proj_gz",
        "lf_pos_x", "lf_pos_y", "lf_pos_z", "lf_clearance",
        "lf_vel_x", "lf_vel_y", "lf_vel_z",
        "lf_contact_fx", "lf_contact_fy", "lf_contact_fz", "lf_contact_norm",
        "rf_pos_x", "rf_pos_y", "rf_pos_z", "rf_clearance",
        "rf_vel_x", "rf_vel_y", "rf_vel_z",
        "rf_contact_fx", "rf_contact_fy", "rf_contact_fz", "rf_contact_norm",
        "ball_pos_x", "ball_pos_y", "ball_pos_z",
        "ball_vel_x", "ball_vel_y", "ball_vel_z",
        "ball_contact_norm",
        "max_contact_force", "max_contact_body",
    ]
    with open(out_dir / "side_slip_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        # Write in batches
        for i in range(0, len(all_frames), 10000):
            w.writerows(all_frames[i:i+10000])
    print(f"  ✓ side_slip_frames.csv ({len(all_frames)} rows)")

    # ═══════════════════════════════════════════════════
    # PART 5: Friction sweep
    # ═══════════════════════════════════════════════════
    print("\n[5/8] Running friction sweep...")

    friction_values = [0.4, 0.6, 0.8, 1.0, 1.2]
    friction_results = {}

    for fric in friction_values:
        print(f"  Friction={fric} ...")
        summaries = []
        for ri in range(10):  # 10 resets per friction value
            _, summary = run_episode(100, torque_mode="pd_zero_action",
                                     ball_mode="none",
                                     set_friction=fric)
            summary["reset_idx"] = ri
            summaries.append(summary)

        avg_ssr = np.mean([s["side_slip_rate"] for s in summaries])
        avg_mv = np.mean([s["max_side_vel"] for s in summaries])
        fell_rate = np.mean([s["fell"] for s in summaries])
        friction_results[fric] = {
            "side_slip_rate": float(avg_ssr),
            "max_side_vel": float(avg_mv),
            "fell_rate": float(fell_rate),
        }
        print(f"    side_slip_rate={avg_ssr:.4f}  max_vel={avg_mv:.3f}  fell_rate={fell_rate:.3f}")

    # ═══════════════════════════════════════════════════
    # PART 6: Foot support geometry
    # ═══════════════════════════════════════════════════
    print("\n[6/8] Computing foot support geometry...")

    # Reset once and get foot bbox
    reset_env_with_pose()
    sim.refresh_sim_tensors()
    body_pos = sim._rigid_body_pos[0].cpu().numpy()

    # URDF collision shapes for feet
    import xml.etree.ElementTree as ET
    asset_path = os.path.join(config.robot.asset.asset_root,
                               config.robot.asset.urdf_file)
    urdf_shapes = defaultdict(list)
    tree = ET.parse(asset_path)
    for link in tree.getroot().findall("link"):
        for coll in link.findall("collision"):
            geom = coll.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None:
                continue
            origin = coll.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0,0,0]
            size = [float(v) for v in box.get("size").split()]
            urdf_shapes[link.get("name")].append({"type": "box", "size": size, "xyz": xyz})

    support_info = {}
    for foot_name, foot_idx in [(left_foot_name, left_foot_idx),
                                 (right_foot_name, right_foot_idx)]:
        shapes = urdf_shapes.get(foot_name, [])
        if shapes:
            s = shapes[0]  # first shape
            support_info[foot_name] = {
                "box_size_x": s["size"][0],
                "box_size_y": s["size"][1],
                "box_size_z": s["size"][2],
                "local_offset": s["xyz"],
            }

    # Support polygon in world frame (approximate from foot centers)
    lf_pos = body_pos[left_foot_idx]
    rf_pos = body_pos[right_foot_idx]
    lf_w = support_info.get(left_foot_name, {}).get("box_size_y", 0.07) / 2
    rf_w = support_info.get(right_foot_name, {}).get("box_size_y", 0.07) / 2

    support_y_min = min(lf_pos[1] - lf_w, rf_pos[1] - rf_w)
    support_y_max = max(lf_pos[1] + lf_w, rf_pos[1] + rf_w)
    support_x_min = min(lf_pos[0] - support_info.get(left_foot_name, {}).get("box_size_x", 0.16) / 2,
                        rf_pos[0] - support_info.get(right_foot_name, {}).get("box_size_x", 0.16) / 2)
    support_x_max = max(lf_pos[0] + support_info.get(left_foot_name, {}).get("box_size_x", 0.16) / 2,
                        rf_pos[0] + support_info.get(right_foot_name, {}).get("box_size_x", 0.16) / 2)

    root_y = sim.robot_root_states[0, 1].item()
    root_x = sim.robot_root_states[0, 0].item()

    foot_support = {
        "left_foot": support_info.get(left_foot_name, {}),
        "right_foot": support_info.get(right_foot_name, {}),
        "left_foot_pos": {"x": float(lf_pos[0]), "y": float(lf_pos[1]), "z": float(lf_pos[2])},
        "right_foot_pos": {"x": float(rf_pos[0]), "y": float(rf_pos[1]), "z": float(rf_pos[2])},
        "support_polygon_y_min": float(support_y_min),
        "support_polygon_y_max": float(support_y_max),
        "support_polygon_x_min": float(support_x_min),
        "support_polygon_x_max": float(support_x_max),
        "support_width_y": float(support_y_max - support_y_min),
        "support_length_x": float(support_x_max - support_x_min),
        "root_y": float(root_y),
        "root_y_offset_from_center": float(root_y - (support_y_min + support_y_max) / 2),
        "root_x": float(root_x),
        "root_x_in_support": bool(support_x_min <= root_x <= support_x_max),
        "root_y_in_support": bool(support_y_min <= root_y <= support_y_max),
    }

    print(f"  Foot box sizes: L={support_info.get(left_foot_name, {})}  "
          f"R={support_info.get(right_foot_name, {})}")
    print(f"  Support Y: [{support_y_min:.3f}, {support_y_max:.3f}] width={support_y_max-support_y_min:.3f}")
    print(f"  Root Y: {root_y:.3f} offset_from_center={root_y - (support_y_min+support_y_max)/2:.3f}")
    print(f"  Foot Y width per foot: L={2*lf_w:.3f} R={2*rf_w:.3f}")

    # ═══════════════════════════════════════════════════
    # PART 7: Ball impulse analysis (Case D only)
    # ═══════════════════════════════════════════════════
    print("\n[7/8] Ball impulse analysis (Case D)...")
    case_d = all_case_summaries.get("D: moving_ball_zero_action", [])
    ball_contact_episodes = [s for s in case_d if s["ball_contacted"]]
    no_contact_episodes = [s for s in case_d if not s["ball_contacted"]]

    ball_analysis = {
        "total_episodes": len(case_d),
        "ball_contact_episodes": len(ball_contact_episodes),
        "no_contact_episodes": len(no_contact_episodes),
    }
    if ball_contact_episodes:
        bc = ball_contact_episodes
        ball_analysis["mean_vel_jump"] = float(np.mean([s["vel_jump_at_contact"] for s in bc]))
        ball_analysis["max_vel_jump"] = float(np.max([s["vel_jump_at_contact"] for s in bc]))
        ball_analysis["mean_side_slip_rate_with_contact"] = float(np.mean([s["side_slip_rate"] for s in bc]))
        ball_analysis["mean_side_slip_rate_without_contact"] = float(
            np.mean([s["side_slip_rate"] for s in no_contact_episodes])) if no_contact_episodes else 0.0
    print(f"  Ball contacted in {len(ball_contact_episodes)}/{len(case_d)} episodes")
    if ball_contact_episodes:
        print(f"  Mean vel jump at contact: {ball_analysis['mean_vel_jump']:.4f} m/s")
        print(f"  Slip rate with contact: {ball_analysis['mean_side_slip_rate_with_contact']:.4f}")
        print(f"  Slip rate without contact: {ball_analysis['mean_side_slip_rate_without_contact']:.4f}")

    # ═══════════════════════════════════════════════════
    # PART 8: FINAL REPORT
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("[8/8] FINAL REPORT")
    print("=" * 65)

    # Table 1: Physics parameters
    print("\n  ── Table 1: Physics Parameters ──")
    print(f"  {'Object':<20s} | {'friction':>10s} | {'restitution':>12s} | {'contact_offset':>14s} | {'rest_offset':>11s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*11}")
    print(f"  {'ground':<20s} | {ground_friction:>10.3f} | {ground_restitution:>12.3f} | {'N/A':>14s} | {'N/A':>11s}")
    print(f"  {'ball':<20s} | {ball_friction:>10.3f} | {ball_restitution:>12.3f} | {'N/A':>14s} | {'N/A':>11s}")
    for fn, shapes in foot_shapes.items():
        for s in shapes:
            print(f"  {fn[:20]:<20s} | {s['friction']:>10.3f} | {s['restitution']:>12.3f} | "
                  f"{s['contact_offset']:>14.4f} | {s['rest_offset']:>11.4f}")

    # Table 2: Side-slip bifurcation
    print("\n  ── Table 2: Side-Slip Diagnostic Cases ──")
    print(f"  {'case':<35s} | {'ball':>6s} | {'control':>12s} | {'slip_rate':>9s} | {'max_vy':>7s} | {'max_disp':>8s} | {'fell':>5s} | conclusion")
    print(f"  {'-'*35}-+-{'-'*6}-+-{'-'*12}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*5}-+-{'-'*30}")

    case_labels = {
        "A: no_ball_zero_torque": ("no", "zero torque"),
        "B: no_ball_zero_action": ("no", "zero action"),
        "C: static_ball_zero_action": ("static", "zero action"),
        "D: moving_ball_zero_action": ("moving", "zero action"),
    }

    conclusions = []
    for case_name, summaries in all_case_summaries.items():
        ball_lbl, ctrl_lbl = case_labels[case_name]
        ssr = np.mean([s["side_slip_rate"] for s in summaries])
        mv = np.mean([s["max_side_vel"] for s in summaries])
        md = np.mean([s["max_side_disp"] for s in summaries])
        fell = np.mean([s["fell"] for s in summaries])

        # Determine conclusion
        if ssr < 0.05:
            conc = "stable"
        elif ssr < 0.2:
            conc = "minor slip"
        elif "zero_torque" in case_name and ssr > 0.3:
            conc = "collapses → not friction; check COM/pose"
        elif "no_ball" in case_name and ssr > 0.2:
            conc = "slips WITHOUT ball → foot/ground/friction/PD"
        elif "moving_ball" in case_name:
            bc_eps = [s for s in summaries if s["ball_contacted"]]
            if len(bc_eps) > 0:
                bc_ssr = np.mean([s["side_slip_rate"] for s in bc_eps])
                if bc_ssr > 0.3:
                    conc = "ball impulse → slip"
                else:
                    conc = "ball contact but stable"
            else:
                conc = "no ball contact"
        else:
            conc = "inconclusive"

        conclusions.append((case_name, conc))
        print(f"  {case_name:<35s} | {ball_lbl:>6s} | {ctrl_lbl:>12s} | {ssr:>9.4f} | {mv:>7.3f} | {md:>8.3f} | {fell:>5.2f} | {conc}")

    # Table 3: Ball impulse
    print("\n  ── Table 3: Ball Impulse Analysis (Case D only) ──")
    if case_d:
        bc_eps = [s for s in case_d if s["ball_contacted"]]
        nc_eps = [s for s in case_d if not s["ball_contacted"]]
        bc_ssr = np.mean([s["side_slip_rate"] for s in bc_eps]) if bc_eps else 0
        nc_ssr = np.mean([s["side_slip_rate"] for s in nc_eps]) if nc_eps else 0
        bc_mv = np.mean([s["vel_jump_at_contact"] for s in bc_eps]) if bc_eps else 0
        print(f"  Ball contact rate:  {len(bc_eps)}/{len(case_d)}")
        print(f"  Slip rate WITH ball contact:    {bc_ssr:.4f}")
        print(f"  Slip rate WITHOUT ball contact: {nc_ssr:.4f}")
        print(f"  Mean side-vel jump at contact:  {bc_mv:.4f} m/s")

    else:
        print("  No data")

    # Table 4: Foot support
    print("\n  ── Table 4: Foot Support Geometry ──")
    fs = foot_support
    lf_box = fs["left_foot"] or {}
    rf_box = fs["right_foot"] or {}
    lf_size_y = lf_box.get("box_size_y", 0)
    rf_size_y = rf_box.get("box_size_y", 0)
    supp_w = fs["support_width_y"]
    supp_l = fs["support_length_x"]
    root_off = fs["root_y_offset_from_center"]

    lf_box_str = f'{lf_box.get("box_size_x",0):.3f}x{lf_box.get("box_size_y",0):.3f}x{lf_box.get("box_size_z",0):.3f}'
    rf_box_str = f'{rf_box.get("box_size_x",0):.3f}x{rf_box.get("box_size_y",0):.3f}x{rf_box.get("box_size_z",0):.3f}'
    lf_narrow_warn = '⚠️ narrow' if lf_size_y < 0.05 else ''
    rf_narrow_warn = '⚠️ narrow' if rf_size_y < 0.05 else ''
    supp_narrow_warn = '⚠️ < 0.15m' if supp_w < 0.15 else ''
    root_off_warn = '⚠️' if abs(root_off) > 0.03 else ''

    print(f"  {'metric':<35s} | {'value':>15s} | warning")
    print(f"  {'-'*35}-+-{'-'*15}-+-{'-'*30}")
    print(f"  {'left foot box size (xyz)':<35s} | {lf_box_str:>15s} | {lf_narrow_warn}")
    print(f"  {'right foot box size (xyz)':<35s} | {rf_box_str:>15s} | {rf_narrow_warn}")
    print(f"  {'support polygon Y width':<35s} | {supp_w:>15.4f} | {supp_narrow_warn}")
    print(f"  {'support polygon X length':<35s} | {supp_l:>15.4f} | ")
    print(f"  {'root Y offset from center':<35s} | {root_off:>15.4f} | {root_off_warn}")
    print(f"  {'root Y in support polygon?':<35s} | {str(fs['root_y_in_support']):>15s} | ")
    print(f"  {'root X in support polygon?':<35s} | {str(fs['root_x_in_support']):>15s} | ")

    # Table 5: Friction sweep
    print("\n  ── Table 5: Friction Sweep (no ball, zero action, 50 eps × 100 frames) ──")
    print(f"  {'friction':>10s} | {'slip_rate':>10s} | {'max_vy':>8s} | {'fell_rate':>9s} | trend")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}-+-{'-'*30}")
    prev_ssr = None
    for fric in friction_values:
        r = friction_results[fric]
        trend = ""
        if prev_ssr is not None:
            if r["side_slip_rate"] < prev_ssr * 0.7:
                trend = "↓ big improvement → friction-limited"
            elif r["side_slip_rate"] < prev_ssr * 0.9:
                trend = "↘ slight improvement"
            elif r["side_slip_rate"] > prev_ssr * 1.3:
                trend = "↗ worse (pinned?)"
            else:
                trend = "→ no change → not friction-limited"
        prev_ssr = r["side_slip_rate"]
        print(f"  {fric:>10.1f} | {r['side_slip_rate']:>10.4f} | {r['max_side_vel']:>8.3f} | "
              f"{r['fell_rate']:>9.3f} | {trend}")

    # ═══════════════════════════════════════════════
    # Answer the 8 questions
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  ROOT CAUSE ANSWERS")
    print("=" * 65)

    case_a = all_case_summaries.get("A: no_ball_zero_torque", [])
    case_b = all_case_summaries.get("B: no_ball_zero_action", [])
    case_d = all_case_summaries.get("D: moving_ball_zero_action", [])

    a_ssr = np.mean([s["side_slip_rate"] for s in case_a]) if case_a else 0
    b_ssr = np.mean([s["side_slip_rate"] for s in case_b]) if case_b else 0
    a_fell = np.mean([s["fell"] for s in case_a]) if case_a else 0
    b_fell = np.mean([s["fell"] for s in case_b]) if case_b else 0

    bc_d = [s for s in case_d if s["ball_contacted"]] if case_d else []
    bc_d_ssr = np.mean([s["side_slip_rate"] for s in bc_d]) if bc_d else 0

    # Q1: no ball → slip?
    q1 = "YES" if b_ssr > 0.1 else ("minor" if b_ssr > 0.03 else "NO")
    print(f"\n  Q1: Slip WITHOUT ball?           {q1} (slip_rate={b_ssr:.4f})")

    # Q2: zero torque → slip?
    q2 = "YES" if a_ssr > 0.1 else ("minor" if a_ssr > 0.03 else "NO")
    print(f"  Q2: Slip with ZERO TORQUE?       {q2} (slip_rate={a_ssr:.4f}, fell_rate={a_fell:.3f})")

    # Q3: slip after ball contact?
    q3 = "N/A"
    if bc_d:
        nc_d = [s for s in case_d if not s["ball_contacted"]]
        nc_d_ssr = np.mean([s["side_slip_rate"] for s in nc_d]) if nc_d else 0
        if bc_d_ssr > nc_d_ssr * 1.5:
            q3 = f"YES (w/ ball={bc_d_ssr:.4f} vs w/o={nc_d_ssr:.4f})"
        else:
            q3 = f"NO significant difference (w/={bc_d_ssr:.4f}, w/o={nc_d_ssr:.4f})"
    print(f"  Q3: Slip AFTER ball contact?     {q3}")

    # Q4: friction values
    print(f"\n  Q4: Foot/ground/ball friction?")
    for fn, shapes in foot_shapes.items():
        for s in shapes:
            print(f"      {fn}: friction={s['friction']}")
    print(f"      ground: friction={ground_friction}")
    print(f"      ball: friction={ball_friction}")

    # Q5: restitution
    print(f"\n  Q5: Restitution = 0?")
    print(f"      foot: {foot_shapes.get(left_foot_name, [{}])[0].get('restitution','?') if foot_shapes.get(left_foot_name) else 'N/A'}")
    print(f"      ground: {ground_restitution}")
    print(f"      ball: {ball_restitution}")

    # Q6: contact_offset/rest_offset
    print(f"\n  Q6: contact_offset / rest_offset reasonable?")
    for fn, shapes in foot_shapes.items():
        for s in shapes:
            print(f"      {fn}: contact_offset={s['contact_offset']:.4f} rest_offset={s['rest_offset']:.4f}")

    # Q7: foot box too narrow?
    print(f"\n  Q7: Foot collision box — narrow or penetrating?")
    print(f"      left: {lf_box.get('box_size_x',0):.3f}×{lf_box.get('box_size_y',0):.3f}×{lf_box.get('box_size_z',0):.3f} m")
    print(f"      right: {rf_box.get('box_size_x',0):.3f}×{rf_box.get('box_size_y',0):.3f}×{rf_box.get('box_size_z',0):.3f} m")
    print(f"      support width Y: {supp_w:.3f} m")
    if lf_size_y < 0.05:
        print(f"      ⚠️  Foot Y width < 5cm → narrow lateral support!")
    if supp_w < 0.15:
        print(f"      ⚠️  Total support Y < 15cm → small support polygon!")

    # Q8: Root cause
    print(f"\n  Q8: ROOT CAUSE → ", end="")
    if a_ssr > 0.3 and a_fell > 0.5:
        print("PHYSICS COLLAPSE: even with zero torque, robot falls. "
              "Check init pose / COM / foot placement.")
    elif b_ssr > 0.2:
        print("PD + FOOT/GROUND: slips without ball even with PD holding pose. "
              "Likely insufficient lateral friction or narrow support polygon.")
    elif bc_d_ssr > b_ssr * 1.5:
        print("BALL IMPULSE: ball contact significantly increases side slip. "
              "Check ball mass/speed and contact geometry.")
    elif b_ssr < 0.03:
        print("STABLE at reset pose: side slip requires external disturbance "
              "(ball, action noise, asymmetric pose). Check training action distributions.")
    else:
        print("MIXED: minor inherent slip, amplified by ball. "
              "Consider increasing foot friction or lateral support width.")

    # ═══════════════════════════════════════════════
    # Save all JSON summaries
    # ═══════════════════════════════════════════════
    report = {
        "physics_params": shape_props,
        "case_summaries": {
            cn: [{k: v for k, v in s.items()} for s in summaries]
            for cn, summaries in all_case_summaries.items()
        },
        "friction_sweep": {str(k): v for k, v in friction_results.items()},
        "foot_support": foot_support,
        "ball_impulse": ball_analysis,
    }
    with open(out_dir / "diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n  All outputs → {out_dir}/")
    print(f"    side_slip_frames.csv")
    print(f"    shape_properties.json / .csv")
    print(f"    diagnostic_report.json")
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC COMPLETE — no files modified.")
    print("=" * 65)

    gym.destroy_sim(sim.sim)


if __name__ == "__main__":
    main()

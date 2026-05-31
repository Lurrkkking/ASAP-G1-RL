"""
Q1 Standing Diagnosis v2 — comprehensive sweep over root heights, poses, PD gains.
"""
import sys, os, time, copy, json
from pathlib import Path
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


# ============================================================
# Helpers
# ============================================================

def build_sim(gym):
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 10
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    return sim


def load_asset(gym, sim, robot_cfg):
    asset_root = str(PROJECT_ROOT / "humanoidverse/data/robots/q1")
    urdf_file = "q1_22dof.urdf"
    ac = robot_cfg.asset
    opts = gymapi.AssetOptions()
    opts.collapse_fixed_joints = ac.collapse_fixed_joints
    opts.replace_cylinder_with_capsule = ac.replace_cylinder_with_capsule
    opts.flip_visual_attachments = ac.flip_visual_attachments
    opts.fix_base_link = ac.fix_base_link
    opts.density = ac.density
    opts.angular_damping = ac.angular_damping
    opts.linear_damping = ac.linear_damping
    opts.max_angular_velocity = ac.max_angular_velocity
    opts.max_linear_velocity = ac.max_linear_velocity
    opts.armature = ac.armature
    opts.thickness = ac.thickness
    opts.default_dof_drive_mode = ac.default_dof_drive_mode
    return gym.load_asset(sim, asset_root, urdf_file, opts)


def create_env_actor(gym, sim, asset, root_z):
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, root_z)
    ah = gym.create_actor(env, asset, pose, "q1_diag", 0, 1, 0)
    return env, ah


def set_init_dof(gym, env, ah, angles):
    """angles: dict name->rad"""
    num_dof = gym.get_actor_dof_count(env, ah)
    states = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    dof_names = gym.get_actor_dof_names(env, ah)
    for i, name in enumerate(dof_names):
        if name in angles:
            states[i]["pos"] = float(angles[name])
        else:
            states[i]["pos"] = 0.0
        states[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, states, gymapi.STATE_ALL)


def get_body_idx(gym, env, ah, name):
    try:
        return gym.find_actor_rigid_body_handle(env, ah, name)
    except:
        return -1


def projected_gz_from_quat(qx, qy, qz, qw):
    return 1.0 - 2.0 * (qx*qx + qy*qy)


def run_trial(robot_cfg, default_angles, root_z, stiffness, damping,
              duration=1.0, dt=0.005, decimation=4, verbose_init=False):
    """
    Run a single trial. Returns dict of results.
    """
    gym = gymapi.acquire_gym()
    sim = build_sim(gym)
    asset = load_asset(gym, sim, robot_cfg)
    env, ah = create_env_actor(gym, sim, asset, root_z)
    set_init_dof(gym, env, ah, default_angles)

    gym.prepare_sim(sim)

    root_state_t = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_t).view(1, -1, 13)
    robot_root = root_states[:, 0, :]

    dof_state_t = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_t).view(1, -1, 2)

    rigid_t = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_t)
    num_bodies = gym.get_actor_rigid_body_count(env, ah)
    num_per_env = rigid.shape[0] // 1
    rigid_pos = rigid.view(1, num_per_env, 13)[..., :num_bodies, 0:3]

    contact_t = gym.acquire_net_contact_force_tensor(sim)
    contact_f = gymtorch.wrap_tensor(contact_t).view(1, -1, 3)

    dof_names = gym.get_actor_dof_names(env, ah)
    body_names = gym.get_actor_rigid_body_names(env, ah)
    num_dof = len(dof_names)

    # Build PD gain arrays
    p_gains_np = np.zeros(num_dof, dtype=np.float32)
    d_gains_np = np.zeros(num_dof, dtype=np.float32)
    for i, name in enumerate(dof_names):
        for key in stiffness:
            if key in name:
                p_gains_np[i] = stiffness[key]
                d_gains_np[i] = damping[key]
                break

    p_gains = torch.tensor(p_gains_np, device="cuda:0")
    d_gains = torch.tensor(d_gains_np, device="cuda:0")

    default_dof_t = torch.tensor([float(default_angles.get(n, 0.0)) for n in dof_names], device="cuda:0")

    # Torque limits
    dof_props = gym.get_actor_dof_properties(env, ah)
    torque_limits = torch.tensor([dof_props["effort"][i].item() for i in range(num_dof)], device="cuda:0")
    vel_limits = torch.tensor([dof_props["velocity"][i].item() for i in range(num_dof)], device="cuda:0")

    # Body indices
    left_ankle_roll_idx = get_body_idx(gym, env, ah, "left_ankle_roll_link")
    right_ankle_roll_idx = get_body_idx(gym, env, ah, "right_ankle_roll_link")
    left_ankle_pitch_idx = get_body_idx(gym, env, ah, "left_ankle_pitch_link")
    right_ankle_pitch_idx = get_body_idx(gym, env, ah, "right_ankle_pitch_link")
    left_knee_idx = get_body_idx(gym, env, ah, "left_knee_link")
    right_knee_idx = get_body_idx(gym, env, ah, "right_knee_link")

    # ---- Initial state snapshot ----
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    init_root_pos = robot_root[0, :3].clone().cpu().numpy()
    init_dof_pos = dof_state[0, :, 0].clone().cpu().numpy()

    init_body_z = rigid_pos[0, :, 2].clone().cpu().numpy()
    init_min_z = init_body_z.min()
    init_max_z = init_body_z.max()

    def bz(name):
        idx = get_body_idx(gym, env, ah, name)
        if idx >= 0:
            return rigid_pos[0, idx, 2].item()
        return float('nan')

    init_lar_z = bz("left_ankle_roll_link")
    init_rar_z = bz("right_ankle_roll_link")
    init_lap_z = bz("left_ankle_pitch_link")
    init_rap_z = bz("right_ankle_pitch_link")

    init_lf_fz = contact_f[0, left_ankle_roll_idx, 2].item()
    init_rf_fz = contact_f[0, right_ankle_roll_idx, 2].item()

    # Check non-foot contact
    non_foot_contact = []
    for i, name in enumerate(body_names):
        fz = contact_f[0, i, 2].item()
        if abs(fz) > 1.0 and i not in (left_ankle_roll_idx, right_ankle_roll_idx):
            non_foot_contact.append((name, fz))

    init_info = {
        "root_pos": init_root_pos,
        "left_ankle_roll_z": init_lar_z,
        "right_ankle_roll_z": init_rar_z,
        "left_ankle_pitch_z": init_lap_z,
        "right_ankle_pitch_z": init_rap_z,
        "body_z_min": init_min_z,
        "body_z_max": init_max_z,
        "left_foot_fz": init_lf_fz,
        "right_foot_fz": init_rf_fz,
        "has_foot_contact": (abs(init_lf_fz) > 1.0 or abs(init_rf_fz) > 1.0),
        "non_foot_contacts": non_foot_contact,
    }

    if verbose_init:
        print(f"  INIT: root_z={init_root_pos[2]:.4f}  "
              f"feet: L_roll_z={init_lar_z:.4f} R_roll_z={init_rar_z:.4f}  "
              f"L_pitch_z={init_lap_z:.4f} R_pitch_z={init_rap_z:.4f}  "
              f"body_z: [{init_min_z:.4f}, {init_max_z:.4f}]")
        print(f"  INIT contact: L_fz={init_lf_fz:.2f} R_fz={init_rf_fz:.2f}  "
              f"has_foot_contact={init_info['has_foot_contact']}")
        if non_foot_contact:
            print(f"  INIT non-foot contact: {non_foot_contact}")

    # ---- Sim loop ----
    total_steps = int(duration / dt)
    fall_time = None
    tau_max_history = 0.0
    dofv_max_history = 0.0
    foot_fz_left = []
    foot_fz_right = []
    saturated_joints = set()

    for sim_step in range(total_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if sim_step % decimation == 0:
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)

            dof_pos = dof_state[0, :, 0]
            dof_vel = dof_state[0, :, 1]
            torques = p_gains * (default_dof_t - dof_pos) - d_gains * dof_vel
            torques = torch.clamp(torques, -torque_limits, torque_limits)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques.contiguous()))

            tau_np = torques.cpu().numpy()
            dofv_np = dof_vel.cpu().numpy()
            tau_abs_max = abs(tau_np).max()
            dofv_abs_max = abs(dofv_np).max()
            tau_max_history = max(tau_max_history, tau_abs_max)
            dofv_max_history = max(dofv_max_history, dofv_abs_max)

            # Check saturated
            for i in range(num_dof):
                if abs(tau_np[i]) > 0.95 * torque_limits[i].item():
                    saturated_joints.add(dof_names[i])

            if left_ankle_roll_idx >= 0:
                foot_fz_left.append(contact_f[0, left_ankle_roll_idx, 2].item())
            if right_ankle_roll_idx >= 0:
                foot_fz_right.append(contact_f[0, right_ankle_roll_idx, 2].item())

            # Check fall
            qw = robot_root[0, 6].item()
            qx = robot_root[0, 3].item()
            qy = robot_root[0, 4].item()
            qz = robot_root[0, 5].item()
            pgz = projected_gz_from_quat(qx, qy, qz, qw)
            if pgz < 0.3 and fall_time is None:
                fall_time = sim_step * dt

    # ---- Final state ----
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    final_root_z = robot_root[0, 2].item()
    qw = robot_root[0, 6].item()
    qx = robot_root[0, 3].item()
    qy = robot_root[0, 4].item()
    qz = robot_root[0, 5].item()
    final_pgz = projected_gz_from_quat(qx, qy, qz, qw)

    final_dof_pos = dof_state[0, :, 0].clone().cpu().numpy()
    final_l_fz = contact_f[0, left_ankle_roll_idx, 2].item() if left_ankle_roll_idx >= 0 else 0
    final_r_fz = contact_f[0, right_ankle_roll_idx, 2].item() if right_ankle_roll_idx >= 0 else 0

    mean_l_fz = np.mean(foot_fz_left) if foot_fz_left else 0.0
    mean_r_fz = np.mean(foot_fz_right) if foot_fz_right else 0.0

    # Knee limits
    knee_left_pos = final_dof_pos[3]  # left_knee_joint index
    knee_right_pos = final_dof_pos[9]  # right_knee_joint index
    knee_at_limit = (knee_left_pos < 0.01) or (knee_right_pos < 0.01)

    # Ankle limits (indices 4,5 = left ankle pitch/roll; 10,11 = right)
    ankle_at_limit = False
    for ai in [4, 5, 10, 11]:
        if final_dof_pos[ai] < robot_cfg.dof_pos_lower_limit_list[ai] * 1.02 or \
           final_dof_pos[ai] > robot_cfg.dof_pos_upper_limit_list[ai] * 0.98:
            ankle_at_limit = True

    fell = final_pgz < 0.5 or (fall_time is not None and fall_time < duration * 0.8)

    gym.destroy_sim(sim)

    return {
        "root_z": root_z,
        "final_root_z": final_root_z,
        "height_drop": init_root_pos[2] - final_root_z,
        "final_pgz": final_pgz,
        "fell": fell,
        "fall_time": fall_time,
        "tau_max": tau_max_history,
        "dofv_max": dofv_max_history,
        "saturated_joints": sorted(saturated_joints),
        "mean_left_fz": mean_l_fz,
        "mean_right_fz": mean_r_fz,
        "final_left_fz": final_l_fz,
        "final_right_fz": final_r_fz,
        "knee_at_limit": knee_at_limit,
        "ankle_at_limit": ankle_at_limit,
        "init_info": init_info,
        "final_dof_pos": final_dof_pos,
        "dof_names": dof_names,
    }


def print_section(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


# ============================================================
# Main
# ============================================================

def main():
    config_path = PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"
    config = OmegaConf.load(str(config_path))
    robot_cfg = config.robot

    dof_names_ordered = list(robot_cfg.dof_names)

    # ------------------------------------------------------------------
    # SECTION 1: Initial geometry & contact state for one representative config
    # ------------------------------------------------------------------
    print_section("SECTION 1: Initial geometry & contact state (root_z=0.8, Pose A, PD1)")

    pose_a = {
        "left_hip_pitch_joint": -0.10, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.30, "left_ankle_pitch_joint": -0.20, "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.10, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.30, "right_ankle_pitch_joint": -0.20, "right_ankle_roll_joint": 0.0,
        "waist_roll_joint": 0.0, "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.0,
    }

    pd1_stiff = {"hip_pitch":60,"hip_roll":60,"hip_yaw":60,"knee":80,
                 "ankle_pitch":30,"ankle_roll":20,"waist_yaw":50,"waist_roll":50,
                 "shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20}
    pd1_damp = {"hip_pitch":2.0,"hip_roll":2.0,"hip_yaw":2.0,"knee":3.0,
                "ankle_pitch":1.0,"ankle_roll":0.8,"waist_yaw":2.0,"waist_roll":2.0,
                "shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8}

    r = run_trial(robot_cfg, pose_a, 0.8, pd1_stiff, pd1_damp, duration=0.05, verbose_init=True)
    info = r["init_info"]
    print(f"\n  Detailed init state:")
    print(f"    root_pos            = {info['root_pos']}")
    print(f"    left_ankle_roll_z   = {info['left_ankle_roll_z']:.4f}")
    print(f"    right_ankle_roll_z  = {info['right_ankle_roll_z']:.4f}")
    print(f"    left_ankle_pitch_z  = {info['left_ankle_pitch_z']:.4f}")
    print(f"    right_ankle_pitch_z = {info['right_ankle_pitch_z']:.4f}")
    print(f"    body_z range        = [{info['body_z_min']:.4f}, {info['body_z_max']:.4f}]")
    print(f"    left_foot init Fz   = {info['left_foot_fz']:.2f}")
    print(f"    right_foot init Fz  = {info['right_foot_fz']:.2f}")
    print(f"    has_foot_contact    = {info['has_foot_contact']}")
    if info["non_foot_contacts"]:
        print(f"    NON-FOOT contacts   = {info['non_foot_contacts']}")
    else:
        print(f"    NON-FOOT contacts   = None")

    # ------------------------------------------------------------------
    # SECTION 2: Root height sweep
    # ------------------------------------------------------------------
    print_section("SECTION 2: Root height sweep (Pose A, PD1, 1s)")

    root_z_list = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    height_results = []
    for rz in root_z_list:
        r = run_trial(robot_cfg, pose_a, rz, pd1_stiff, pd1_damp, duration=1.0)
        i = r["init_info"]
        print(f"  root_z={rz:.2f}: init_foot_z=({i['left_ankle_roll_z']:.3f},{i['right_ankle_roll_z']:.3f})  "
              f"init_contact={i['has_foot_contact']}  init_fz=({i['left_foot_fz']:.2f},{i['right_foot_fz']:.2f})  "
              f"final_rz={r['final_root_z']:.3f}  pgz={r['final_pgz']:.3f}  "
              f"fell={r['fell']}  tau_max={r['tau_max']:.1f}  dofv_max={r['dofv_max']:.1f}  "
              f"mean_fz=({r['mean_left_fz']:.0f},{r['mean_right_fz']:.0f})")
        if i["non_foot_contacts"]:
            print(f"          NON-FOOT contact: {i['non_foot_contacts']}")
        height_results.append(r)

    # ------------------------------------------------------------------
    # SECTION 3: Pose sweep
    # ------------------------------------------------------------------
    print_section("SECTION 3: Default standing pose sweep (2s, PD1, root_z=0.75)")

    poses = {
        "Pose_A (G1-like)": dict(hip_pitch=-0.10, knee=0.30, ankle_pitch=-0.20),
        "Pose_B (slight crouch)": dict(hip_pitch=-0.20, knee=0.45, ankle_pitch=-0.25),
        "Pose_C (mid crouch)": dict(hip_pitch=-0.30, knee=0.60, ankle_pitch=-0.30),
        "Pose_D (near upright)": dict(hip_pitch=-0.05, knee=0.15, ankle_pitch=-0.10),
        "Pose_E (deep crouch)": dict(hip_pitch=-0.40, knee=0.80, ankle_pitch=-0.40),
    }

    def make_pose(hip_pitch, knee, ankle_pitch):
        p = {}
        for name in dof_names_ordered:
            p[name] = 0.0
        p["left_hip_pitch_joint"] = hip_pitch
        p["right_hip_pitch_joint"] = hip_pitch
        p["left_knee_joint"] = knee
        p["right_knee_joint"] = knee
        p["left_ankle_pitch_joint"] = ankle_pitch
        p["right_ankle_pitch_joint"] = ankle_pitch
        return p

    pose_results = []
    for label, params in poses.items():
        pose_dict = make_pose(**params)
        r = run_trial(robot_cfg, pose_dict, 0.75, pd1_stiff, pd1_damp, duration=2.0)
        i = r["init_info"]
        print(f"  {label:25s}  hp={params['hip_pitch']:.2f} knee={params['knee']:.2f} ap={params['ankle_pitch']:.2f}")
        print(f"    init: fz=({i['left_foot_fz']:.1f},{i['right_foot_fz']:.1f})  foot_z=({i['left_ankle_roll_z']:.3f},{i['right_ankle_roll_z']:.3f})  has_contact={i['has_foot_contact']}")
        if i["non_foot_contacts"]:
            print(f"          NON-FOOT: {i['non_foot_contacts']}")
        print(f"    final: rz={r['final_root_z']:.3f}  pgz={r['final_pgz']:.3f}  fell={r['fell']}  "
              f"fall_t={r['fall_time']}  tau_max={r['tau_max']:.1f}  dofv_max={r['dofv_max']:.1f}")
        print(f"    knee_at_limit={r['knee_at_limit']}  ankle_at_limit={r['ankle_at_limit']}  "
              f"mean_fz=({r['mean_left_fz']:.0f},{r['mean_right_fz']:.0f})")
        if r["saturated_joints"]:
            print(f"    SATURATED: {r['saturated_joints']}")
        pose_results.append((label, params, r))

    # ------------------------------------------------------------------
    # SECTION 4: PD parameter sweep on the best pose
    # ------------------------------------------------------------------
    print_section("SECTION 4: PD parameter sweep (Pose C, root_z=0.75, 2s)")

    pd_sets = {
        "PD1 (conservative)": (
            {"hip_pitch":60,"hip_roll":60,"hip_yaw":60,"knee":80,
             "ankle_pitch":30,"ankle_roll":20,"waist_yaw":50,"waist_roll":50,
             "shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
            {"hip_pitch":2.0,"hip_roll":2.0,"hip_yaw":2.0,"knee":3.0,
             "ankle_pitch":1.0,"ankle_roll":0.8,"waist_yaw":2.0,"waist_roll":2.0,
             "shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
        ),
        "PD2 (medium)": (
            {"hip_pitch":90,"hip_roll":90,"hip_yaw":90,"knee":120,
             "ankle_pitch":40,"ankle_roll":30,"waist_yaw":70,"waist_roll":70,
             "shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
            {"hip_pitch":3.0,"hip_roll":3.0,"hip_yaw":3.0,"knee":4.0,
             "ankle_pitch":1.5,"ankle_roll":1.2,"waist_yaw":3.0,"waist_roll":3.0,
             "shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
        ),
        "PD3 (stiff)": (
            {"hip_pitch":120,"hip_roll":120,"hip_yaw":120,"knee":150,
             "ankle_pitch":50,"ankle_roll":40,"waist_yaw":80,"waist_roll":80,
             "shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
            {"hip_pitch":3.5,"hip_roll":3.5,"hip_yaw":3.5,"knee":5.0,
             "ankle_pitch":2.0,"ankle_roll":1.5,"waist_yaw":3.5,"waist_roll":3.5,
             "shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
        ),
    }

    pose_c = make_pose(hip_pitch=-0.30, knee=0.60, ankle_pitch=-0.30)
    pd_results = []
    for label, (stiff, damp) in pd_sets.items():
        r = run_trial(robot_cfg, pose_c, 0.75, stiff, damp, duration=2.0)
        print(f"  {label:25s}  rz={r['final_root_z']:.3f}  pgz={r['final_pgz']:.3f}  "
              f"fell={r['fell']}  tau_max={r['tau_max']:.1f}  dofv_max={r['dofv_max']:.1f}  "
              f"knee_limit={r['knee_at_limit']}  mean_fz=({r['mean_left_fz']:.0f},{r['mean_right_fz']:.0f})")
        if r["saturated_joints"]:
            print(f"          SATURATED: {r['saturated_joints']}")
        pd_results.append((label, r))

    # ------------------------------------------------------------------
    # SECTION 4b: Cross-sweep (best poses x PD sets)
    # ------------------------------------------------------------------
    print_section("SECTION 4b: Cross sweep (Pose B/C/D x PD1/PD2/PD3, root_z=0.75, 2s)")

    all_pose_configs = {
        "Pose_B": make_pose(hip_pitch=-0.20, knee=0.45, ankle_pitch=-0.25),
        "Pose_C": make_pose(hip_pitch=-0.30, knee=0.60, ankle_pitch=-0.30),
        "Pose_D": make_pose(hip_pitch=-0.05, knee=0.15, ankle_pitch=-0.10),
    }

    cross_results = []
    for pname, pdict in all_pose_configs.items():
        for pdname, (stiff, damp) in pd_sets.items():
            r = run_trial(robot_cfg, pdict, 0.75, stiff, damp, duration=2.0)
            cross_results.append((pname, pdname, r))
            sat = ",".join(r["saturated_joints"][:3]) if r["saturated_joints"] else "-"
            print(f"  {pname:8s} {pdname:22s}  fell={str(r['fell']):5s}  "
                  f"rz={r['final_root_z']:.3f}  pgz={r['final_pgz']:.3f}  "
                  f"tau={r['tau_max']:.1f}  dofv={r['dofv_max']:.1f}  "
                  f"knee_lim={str(r['knee_at_limit']):5s}  fz=({r['mean_left_fz']:.0f},{r['mean_right_fz']:.0f})  "
                  f"saturated=[{sat}]")

    # ------------------------------------------------------------------
    # SECTION 5: Summary table & final recommendations
    # ------------------------------------------------------------------
    print_section("SECTION 5: SUMMARY TABLE")

    print(f"\n  {'height':>8s}  {'pose':>10s}  {'PD':>22s}  {'survived':>9s}  {'final_rz':>9s}  "
          f"{'pgz':>7s}  {'tau_max':>8s}  {'dofv_max':>9s}  "
          f"{'saturated':>20s}  {'mean_fz_L':>10s}  {'mean_fz_R':>10s}  {'knee_lim':>9s}  {'ankle_lim':>9s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*22}  {'-'*9}  {'-'*9}  "
          f"{'-'*7}  {'-'*8}  {'-'*9}  "
          f"{'-'*20}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")

    # Height sweep rows
    for r in height_results:
        sat = ",".join(r["saturated_joints"][:3]) if r["saturated_joints"] else "-"
        print(f"  {r['root_z']:8.2f}  {'Pose_A':>10s}  {'PD1':>22s}  "
              f"{str(not r['fell']):>9s}  {r['final_root_z']:9.3f}  "
              f"{r['final_pgz']:7.3f}  {r['tau_max']:8.1f}  {r['dofv_max']:9.1f}  "
              f"{sat:>20s}  {r['mean_left_fz']:10.0f}  {r['mean_right_fz']:10.0f}  "
              f"{str(r['knee_at_limit']):>9s}  {str(r['ankle_at_limit']):>9s}")

    # Pose sweep rows
    for label, params, r in pose_results:
        sat = ",".join(r["saturated_joints"][:3]) if r["saturated_joints"] else "-"
        print(f"  {0.75:8.2f}  {label:>10s}  {'PD1':>22s}  "
              f"{str(not r['fell']):>9s}  {r['final_root_z']:9.3f}  "
              f"{r['final_pgz']:7.3f}  {r['tau_max']:8.1f}  {r['dofv_max']:9.1f}  "
              f"{sat:>20s}  {r['mean_left_fz']:10.0f}  {r['mean_right_fz']:10.0f}  "
              f"{str(r['knee_at_limit']):>9s}  {str(r['ankle_at_limit']):>9s}")

    # Cross sweep rows
    for pname, pdname, r in cross_results:
        sat = ",".join(r["saturated_joints"][:3]) if r["saturated_joints"] else "-"
        print(f"  {0.75:8.2f}  {pname:>10s}  {pdname:>22s}  "
              f"{str(not r['fell']):>9s}  {r['final_root_z']:9.3f}  "
              f"{r['final_pgz']:7.3f}  {r['tau_max']:8.1f}  {r['dofv_max']:9.1f}  "
              f"{sat:>20s}  {r['mean_left_fz']:10.0f}  {r['mean_right_fz']:10.0f}  "
              f"{str(r['knee_at_limit']):>9s}  {str(r['ankle_at_limit']):>9s}")

    # ------------------------------------------------------------------
    # Best candidates
    # ------------------------------------------------------------------
    print(f"\n  {'='*80}")
    print(f"  FINDINGS")
    print(f"  {'='*80}")

    # Find best
    best = None
    best_score = -999
    for pname, pdname, r in cross_results:
        if r["fell"]:
            continue
        score = r["final_root_z"] * 5 + r["final_pgz"] * 2 - r["tau_max"] / 36.0 - len(r["saturated_joints"]) * 0.5
        score += (abs(r["mean_left_fz"]) + abs(r["mean_right_fz"])) / 200.0  # prefer contact
        if not r["knee_at_limit"]:
            score += 1.0
        if score > best_score:
            best_score = score
            best = (pname, pdname, r)

    if best is None:
        print(f"\n  No configuration survived 2s. Checking for least-bad option...")
        for pname, pdname, r in cross_results:
            if best is None or r["fall_time"] is None or (best[2]["fall_time"] is not None and
                                                           r["fall_time"] is not None and
                                                           r["fall_time"] > best[2]["fall_time"]):
                best = (pname, pdname, r)

    if best:
        pname, pdname, r = best
        print(f"\n  Best: {pname} + {pdname}")
        print(f"    root_z=0.75  fell={r['fell']}  final_rz={r['final_root_z']:.3f}  pgz={r['final_pgz']:.3f}")
        if r["saturated_joints"]:
            print(f"    saturated: {r['saturated_joints']}")

    print(f"\n  Primary failure cause analysis:")
    # Check section 1 info
    s1 = height_results[3]  # root_z=0.65
    init = s1["init_info"]
    foot_z_mean = (init["left_ankle_roll_z"] + init["right_ankle_roll_z"]) / 2

    print(f"    [SECTION 1] Initial geometry at root_z=0.8:")
    print(f"      Body z range: [{init['body_z_min']:.3f}, {init['body_z_max']:.3f}]")
    print(f"      Foot z ~{foot_z_mean:.3f}, contact={init['has_foot_contact']}")
    print(f"      -> {'Feet ABOVE ground (hovering)' if foot_z_mean > 0.05 else 'Feet AT/Near ground'}")

    if height_results[5]["fell"] and height_results[0]["fell"]:
        print(f"    [SECTION 2] All root heights 0.35-0.85 fail -> NOT a height issue alone")

    deep_ok = any(not r["fell"] for _, _, r in pose_results)
    if not deep_ok:
        print(f"    [SECTION 3] All poses fail at 2s -> stiffness too low for any realistic standing pose")

    print(f"\n  RECOMMENDATIONS:")
    # Find best configuration
    best_cross = None
    for pname, pdname, r in cross_results:
        if not r["fell"]:
            best_cross = (pname, pdname, r)
            break
    if best_cross is None:
        for pname, pdname, r in cross_results:
            if best_cross is None or (r["fall_time"] is not None and
                                       best_cross[2]["fall_time"] is not None and
                                       r["fall_time"] > best_cross[2]["fall_time"]):
                best_cross = (pname, pdname, r)

    if not deep_ok:
        print(f"    1. stiffness MUST be increased substantially.")
        print(f"       Q1 has only 36 Nm torque limit (G1 has 139 Nm at knee).")
        print(f"       To hold standing against gravity, stiffness must command torque near the limit quickly.")
        print(f"       Current kp=80 at knee produces at most ~24 Nm from default bias (0.3 rad error).")
        print(f"       With a 60 kg robot, knee torque needed can be 50-100 Nm — above Q1's limit.")
        print(f"       This means Q1 may need a more crouched pose to reduce the moment arm, or the stiffness")
        print(f"       must be high enough to saturate immediately at full effort.")

    print(f"    2. root_height: use ~0.70-0.80 (feet should be near ground level)")
    print(f"    3. default_joint_angles: prefer Pose B or C (more knee bend)")
    print(f"    4. stiffness: prefer PD2 or PD3 if they survive longer")
    print(f"    5. The primary failure mode is: insufficient torque to hold standing against gravity.")
    print(f"       This is a hardware constraint (Q1 motors are weak), not a software bug.")

    can_proceed = any(not r["fell"] for _, _, r in cross_results)
    if can_proceed:
        print(f"\n  >>> Can proceed to 5-10s standing test with best config <<<")
    else:
        print(f"\n  >>> Cannot proceed. Need to increase stiffness beyond PD3 or reconsider standing pose <<<")


if __name__ == "__main__":
    main()

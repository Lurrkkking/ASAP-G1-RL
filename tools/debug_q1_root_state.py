"""
Root state debug — trace exactly when/why Q1 root jumps.
"""
import sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def quat_to_rpy(qx, qy, qz, qw):
    """Convert xyzw quaternion to roll, pitch, yaw in radians."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def print_root_state(prefix, root_states, actor_id=0):
    """Print root state for a specific actor."""
    pos = root_states[actor_id, 0:3].cpu().numpy()
    quat = root_states[actor_id, 3:7].cpu().numpy()
    lin_vel = root_states[actor_id, 7:10].cpu().numpy()
    ang_vel = root_states[actor_id, 10:13].cpu().numpy()
    qx, qy, qz, qw = quat
    pgz = 1.0 - 2.0 * (qx * qx + qy * qy)
    r, p, y = quat_to_rpy(qx, qy, qz, qw)
    print(f"  {prefix:50s}  pos=[{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}]  "
          f"quat(xyzw)=[{quat[0]:+.4f} {quat[1]:+.4f} {quat[2]:+.4f} {quat[3]:+.4f}]  "
          f"rpy=[{r:+6.1f} {p:+6.1f} {y:+6.1f}]  projgz={pgz:+.4f}  "
          f"lv=[{lin_vel[0]:+.4f} {lin_vel[1]:+.4f} {lin_vel[2]:+.4f}]  "
          f"av=[{ang_vel[0]:+.4f} {ang_vel[1]:+.4f} {ang_vel[2]:+.4f}]")


def main():
    conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
    rc = conf.robot

    print("=" * 80)
    print("Q1 ROOT STATE DEBUG")
    print(f"  YAML init_state: pos={rc.init_state.pos}, rot={rc.init_state.rot}")
    print(f"  YAML lin_vel={rc.init_state.lin_vel}, ang_vel={rc.init_state.ang_vel}")
    print("=" * 80)

    # ============================================================
    # Phase 1: trace root state through every init step
    # ============================================================
    print("\n--- Phase 1: init trace ---\n")

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0,0,1)
    pp.static_friction = 1.0; pp.dynamic_friction = 1.0
    gym.add_ground(sim, pp)

    # Load asset
    ac = rc.asset
    opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = ac.default_dof_drive_mode
    asset = gym.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof.urdf", opts)
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)
    ndof = len(dof_names)
    nbody = len(body_names)

    root_z = float(rc.init_state.pos[2])  # 0.39

    # ---- Step 1: create env + actor ----
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

    # Check actor count BEFORE
    actor_count_before = gym.get_actor_count(env)
    print(f"Actors in env before create: {actor_count_before}")

    start_pose = gymapi.Transform(
        p=gymapi.Vec3(float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), root_z),
        r=gymapi.Quat(float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
                      float(rc.init_state.rot[2]), float(rc.init_state.rot[3]))
    )
    print(f"start_pose: p=({start_pose.p.x},{start_pose.p.y},{start_pose.p.z})  "
          f"r=({start_pose.r.x},{start_pose.r.y},{start_pose.r.z},{start_pose.r.w})")

    ah = gym.create_actor(env, asset, start_pose, "q1", -1, 0, 0)
    actor_count_after = gym.get_actor_count(env)
    print(f"Actors in env after create: {actor_count_after}  handle={ah}")

    # Read root via rigid body states (pre-prepare)
    rb = gym.get_actor_rigid_body_states(env, ah, gymapi.STATE_POS)
    print_root_state("After create_actor (rigid body states[0])",
                     torch.tensor([[*rb[0]['pose']['p'], *rb[0]['pose']['r'], 0,0,0,0,0,0]]).view(1,-1),
                     actor_id=0)

    # ---- Step 2: read config quaternion details ----
    print(f"\n  Config rot (x,y,z,w): {list(rc.init_state.rot)}")
    print(f"  Gymapi.Quat(w,x,y,z) from config: ({start_pose.r.w},{start_pose.r.x},{start_pose.r.y},{start_pose.r.z})")
    print(f"  IsaacGym stores root_quat as (x,y,z,w) = gyapi stores as (x,y,z,w)?")

    # ---- Step 3: set DOF states ----
    defaults = dict(rc.init_state.default_joint_angles)
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i, n in enumerate(dof_names):
        dof_st[i]["pos"] = float(defaults.get(n, 0.0))
        dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    rb2 = gym.get_actor_rigid_body_states(env, ah, gymapi.STATE_POS)
    print_root_state("After set_actor_dof_states",
                     torch.tensor([[*rb2[0]['pose']['p'], *rb2[0]['pose']['r'], 0,0,0,0,0,0]]).view(1,-1),
                     actor_id=0)

    # ---- Step 4: prepare_sim ----
    gym.prepare_sim(sim)

    # Acquire tensors
    rt = gym.acquire_actor_root_state_tensor(sim)
    num_actors_total = rt.shape[0] // 13  # 13 floats per actor state
    print(f"\n  After prepare_sim: root_state_tensor total_size={rt.shape[0]}, num_actors={num_actors_total}")

    root_all = gymtorch.wrap_tensor(rt)
    print(f"  root_all shape: {root_all.shape}")

    # If there are multiple actors (0 and maybe a different one), find robot
    # actor 0 in root_all might not be the robot
    actors_in_root = root_all.shape[0] // 13
    root_reshaped = root_all.view(-1, 13)
    print(f"  Root tensor has {root_reshaped.shape[0]} entries (actors + world?)")

    # IsaacGym root state: each actor has 13 floats
    # Usually: root_all.view(num_envs, num_actors_per_env, 13)
    # World (ground) is typically at index 0 in some configurations
    num_actors_per_env = root_reshaped.shape[0]
    print(f"  num_actors_per_env (from root tensor) = {num_actors_per_env}")

    # ---- Step 5: read initial state after prepare_sim (BEFORE any simulate) ----
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    # Check each actor in root tensor
    for ai in range(num_actors_per_env):
        root_single = root_reshaped[ai:ai+1]
        print(f"\n  Actor[{ai}] in root_state_tensor:")
        print_root_state(f"  root_tensor[{ai}]", root_single, 0)

    # Also get the robot body indices
    rigid_t = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_t)
    npe = rigid.shape[0]
    rigid_v = rigid.view(-1, 1, 13)
    print(f"\n  rigid_body_state total shape: {rigid.shape}, per env: {npe}")

    cf_t = gym.acquire_net_contact_force_tensor(sim)
    cf = gymtorch.wrap_tensor(cf_t)
    print(f"  contact_force shape: {cf.shape}")
    cf_v = cf.view(-1, 1, 3)

    dt_t = gym.acquire_dof_state_tensor(sim)
    dof = gymtorch.wrap_tensor(dt_t)
    print(f"  dof_state shape: {dof.shape}")
    dof_v = dof.view(-1, 2)

    # Build PD gains
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, dtype=np.float32); dg_np = np.zeros(ndof, dtype=np.float32)
    for i, n in enumerate(dof_names):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor([float(defaults.get(n,0)) for n in dof_names], device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env,ah)["effort"][i].item() for i in range(ndof)], device="cuda:0")

    # Find robot actor index — which root tensor entry is the robot?
    robot_actor_id = -1
    for ai in range(num_actors_per_env):
        rz = root_reshaped[ai, 2].item()
        if abs(rz - root_z) < 0.5:  # robot should be near root_z
            robot_actor_id = ai
            break
    if robot_actor_id < 0:
        robot_actor_id = 0  # assume first
        print(f"\n  WARNING: Could not find robot in root tensor, assuming actor 0")
    print(f"\n  robot_actor_id in root_state_tensor = {robot_actor_id}")
    print(f"  Using this index for all root reads/writes")

    # ---- Phase 2: Sim 10 frames with ZERO TORQUE ----
    print(f"\n\n--- Phase 2: 10 frames ZERO TORQUE (no PD) ---\n")

    for frame in range(10):
        # read BEFORE simulate
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        root_pre = root_reshaped[robot_actor_id].clone()

        if frame == 0:
            print_root_state("Frame 0 BEFORE simulate", root_reshaped, robot_actor_id)
            # Also print foot positions
            Lf_idx = body_names.index("left_ankle_roll_link")
            Rf_idx = body_names.index("right_ankle_roll_link")
            rpos = rigid_v[Lf_idx:Lf_idx+1, 0, 0:3]
            print(f"  left_ankle_roll world pos: [{rpos[0,0,0].item():.4f} {rpos[0,0,1].item():.4f} {rpos[0,0,2].item():.4f}]")

        # ZERO TORQUE — no PD control
        zero_torque = torch.zeros(ndof, device="cuda:0")
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_torque.contiguous()))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        root_after = root_reshaped[robot_actor_id].clone()
        pos_diff = (root_after[0:3] - root_pre[0:3]).norm().item()
        quat_diff = (root_after[3:7] - root_pre[3:7]).norm().item()

        if frame < 5:
            print_root_state(f"Frame {frame} AFTER simulate", root_reshaped, robot_actor_id)
            print(f"         pos delta={pos_diff:.6f}  quat delta={quat_diff:.6f}")

    # ---- Phase 3: 10 frames WITH PD control ----
    print(f"\n\n--- Phase 3: 10 frames WITH PD control ---\n")

    # Reset by explicitly writing root state
    print("Resetting root state to init values...")
    root_reshaped[robot_actor_id, 0:3] = torch.tensor(
        [float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), float(rc.init_state.pos[2])],
        device="cuda:0")
    root_reshaped[robot_actor_id, 3:7] = torch.tensor(
        [float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
         float(rc.init_state.rot[2]), float(rc.init_state.rot[3])],
        device="cuda:0")
    root_reshaped[robot_actor_id, 7:13] = 0.0  # zero velocities
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_all))

    # Reset DOFs
    dof_v[0:ndof, 0] = default_t
    dof_v[0:ndof, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof))

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    print_root_state("After manual reset", root_reshaped, robot_actor_id)

    for frame in range(10):
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        root_pre = root_reshaped[robot_actor_id].clone()

        # PD control
        dp = dof_v[0:ndof, 0]; dv = dof_v[0:ndof, 1]
        tau = pg * (default_t - dp) - dg * dv
        tau = torch.clamp(tau, -tau_lim, tau_lim)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        root_after = root_reshaped[robot_actor_id].clone()
        pos_diff = (root_after[0:3] - root_pre[0:3]).norm().item()

        if frame < 5:
            # Print foot contact
            Lfz = cf_v[Lf_idx, 0, 2].item()
            Rfz = cf_v[Rf_idx, 0, 2].item()
            tau_abs = abs(tau.cpu().numpy()).max()
            print_root_state(f"Frame {frame} AFTER PD sim", root_reshaped, robot_actor_id)
            print(f"         pos_delta={pos_diff:.6f}  tau_max={tau_abs:.1f}  fz=[{Lfz:.0f},{Rfz:.0f}]")

    # ---- Phase 4: experiment — disable gravity ----
    print(f"\n\n--- Phase 4: 5 frames ZERO GRAVITY + ZERO TORQUE ---\n")
    # Reset gravity
    sp2 = gymapi.SimParams()
    sp2.dt = 0.005; sp2.up_axis = gymapi.UP_AXIS_Z
    sp2.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # ZERO gravity
    sp2.physx.solver_type = 1; sp2.physx.num_position_iterations = 4
    sp2.physx.num_velocity_iterations = 1; sp2.physx.num_threads = 10
    sp2.physx.use_gpu = True; sp2.use_gpu_pipeline = True

    gym.destroy_sim(sim)
    sim2 = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp2)
    gym.add_ground(sim2, gymapi.PlaneParams())

    asset2 = gym.load_asset(sim2, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof.urdf", opts)
    env2 = gym.create_env(sim2, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    p2 = gymapi.Transform(
        p=gymapi.Vec3(float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), float(rc.init_state.pos[2])),
        r=gymapi.Quat(float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
                      float(rc.init_state.rot[2]), float(rc.init_state.rot[3])))
    ah2 = gym.create_actor(env2, asset2, p2, "q1", -1, 0, 0)

    dof_st2 = gym.get_actor_dof_states(env2, ah2, gymapi.STATE_ALL)
    for i, n in enumerate(dof_names):
        dof_st2[i]["pos"] = float(defaults.get(n, 0.0)); dof_st2[i]["vel"] = 0.0
    gym.set_actor_dof_states(env2, ah2, dof_st2, gymapi.STATE_ALL)

    gym.prepare_sim(sim2)

    rt2 = gym.acquire_actor_root_state_tensor(sim2)
    root2_all = gymtorch.wrap_tensor(rt2)
    root2 = root2_all.view(-1, 13)

    gym.refresh_actor_root_state_tensor(sim2)
    print(f"  Zero-gravity sim:")
    print_root_state("  Frame 0 BEFORE simulate", root2, 0)

    for frame in range(5):
        zero_t = torch.zeros(ndof, device="cuda:0")
        gym.set_dof_actuation_force_tensor(sim2, gymtorch.unwrap_tensor(zero_t.contiguous()))
        gym.simulate(sim2); gym.fetch_results(sim2, True)
        gym.refresh_actor_root_state_tensor(sim2)
        print_root_state(f"  Frame {frame} AFTER (zero g, zero tau)", root2, 0)

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  1. Config init_state: pos={rc.init_state.pos}, rot={rc.init_state.rot}")
    print(f"  2. start_pose used: z={root_z}")
    print(f"  3. Actors in root tensor: {root_reshaped.shape[0]}")
    print(f"  4. Robot actor id: {robot_actor_id}")
    print(f"  5. Quat order in root_tensor: (x, y, z, w) — IsaacGym standard")
    print(f"  6. See Phase 2 output for zero-torque root trajectory")
    print(f"  7. See Phase 4 output for zero-gravity root trajectory")

    gym.destroy_sim(sim2)


if __name__ == "__main__":
    main()

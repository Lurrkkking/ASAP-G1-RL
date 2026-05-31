"""
Q1 Ground Impulse Isolation — 8 systematic experiments.
"""
import sys, numpy as np, copy, struct
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def projected_gz(qx, qy, qz, qw):
    return 1.0 - 2.0*(qx*qx + qy*qy)


def make_box_asset(gym, sim, sx, sy, sz, density=1000.0):
    opts = gymapi.AssetOptions(); opts.density = density; opts.fix_base_link = False
    opts.angular_damping = 0.0; opts.linear_damping = 0.0
    return gym.create_box(sim, sx, sy, sz, opts)


def quat_to_rpy(qx, qy, qz, qw):
    r = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    sinp = 2*(qw*qy - qz*qx)
    p = np.arcsin(np.clip(sinp, -1, 1))
    y = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return np.degrees(r), np.degrees(p), np.degrees(y)


def run_simple_trial(gym, sim, env, ah, ndof, tgt, dnames, bnames, body_names_list,
                      apply_pd, pg, dg, default_t, tau_lim, nframes, verbose):
    """Run frames and return results dict."""
    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
    if ndof > 0:
        dt = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
    else:
        d_v = torch.zeros(1, 2, device="cuda:0")
    cf_t = gym.acquire_net_contact_force_tensor(sim); cf = gymtorch.wrap_tensor(cf_t)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)

    gym.refresh_actor_root_state_tensor(sim);
    if ndof > 0: gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    nb = cf.shape[0]  # num contact bodies

    results = []
    for f in range(nframes + 1):
        if f == 0:
            # capture init state before any simulate
            pass
        else:
            gym.refresh_actor_root_state_tensor(sim)
            if ndof > 0: gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
            if apply_pd and ndof > 0:
                dp = d_v[:, 0]; dv_ = d_v[:, 1]
                tau = (pg * (default_t - dp) - dg * dv_).clamp(-tau_lim, tau_lim)
            else:
                tau = torch.zeros(max(1, ndof), device="cuda:0")
            if ndof > 0:
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))
            for _ in range(4): gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            if ndof > 0: gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        rz = r_v[0, 2].item()
        lv = r_v[0, 7:10].norm().item(); av = r_v[0, 10:13].norm().item()
        qx, qy, qz_, qw = r_v[0, 3].item(), r_v[0, 4].item(), r_v[0, 5].item(), r_v[0, 6].item()
        pgz = projected_gz(qx, qy, qz_, qw)

        cf_np = cf.view(-1, 3).norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_val = cf_np[top_i]
        top_name = bnames[top_i] if top_i < len(bnames) and bnames else f"body_{top_i}"

        rigid_v = rigid.view(rigid.shape[0], 13)
        body_z_all = rigid_v[:, 2].cpu().numpy() if rigid.shape[0] > 0 else np.array([rz])

        results.append({
            'f': f, 'rz': rz, 'lv': lv, 'av': av, 'pgz': pgz,
            'top_cf_name': top_name, 'top_cf_val': top_val,
            'bz_min': body_z_all.min(),
        })
    return results


# ==============================================================================
print("=" * 80)
print("Q1 GROUND IMPULSE ISOLATION")
print("=" * 80)

conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot
root_z_default = float(rc.init_state.pos[2])
Q1_MASS = 16.3

# ================================================================
# EXP 1: High-altitude zero-gravity test
# ================================================================
print("\n" + "=" * 80)
print("EXP 1: High-altitude zero-gravity — isolate geometric contact")
print("=" * 80)
print(f"  {'rz_init':>7s} {'rz_f1':>7s} {'lv_f1':>7s} {'av_f1':>7s} {'top_cf':>20s} {'cf_val':>8s} {'bz_min':>7s}")

for rz_test in [0.39, 0.6, 1.0, 2.0, 5.0]:
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0)  # ZERO GRAVITY
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
    g.add_ground(sim, pp)

    ac = rc.asset; opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = 3
    asset = g.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                         "q1_22dof_rl_collision.urdf", opts)
    dnames = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dnames)

    env = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    ah = g.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz_test)), "q1", -1, 0, 0)

    defaults = dict(rc.init_state.default_joint_angles)
    tgt = [float(defaults.get(n, 0)) for n in dnames]
    dof_st = g.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    g.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)
    g.prepare_sim(sim)

    # Shape props
    props = g.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(props)): props[i].contact_offset = 0.001; props[i].rest_offset = 0.0; props[i].restitution = 0.0; props[i].friction = 0.8
    g.set_actor_rigid_shape_properties(env, ah, props)

    # Tensors + fix
    rt = g.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
    dt = g.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
    r_v[0, 0:3] = torch.tensor([0.0, 0.0, rz_test], device="cuda:0")
    r_v[0, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda:0"); r_v[0, 7:13] = 0.0
    g.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    for i in range(nd): d_v[i, 0] = tgt[i]; d_v[i, 1] = 0.0
    g.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(sim); g.refresh_dof_state_tensor(sim)

    res = run_simple_trial(g, sim, env, ah, nd, tgt, dnames, bnames, bnames, False, None, None, None, None, 5, False)
    f1 = res[1]
    print(f"  {rz_test:+7.3f} {f1['rz']:+7.3f} {f1['lv']:+7.3f} {f1['av']:+7.3f} {f1['top_cf_name']:>20s} {f1['top_cf_val']:+8.0f} {f1['bz_min']:+7.3f}")
    g.destroy_sim(sim)

# ================================================================
# EXP 2: Static box ground vs infinite plane
# ================================================================
print("\n" + "=" * 80)
print("EXP 2: Static box ground vs add_ground plane")
print("=" * 80)

for ground_type, label in [("plane", "add_ground plane"), ("box", "static box ground")]:
    for grav_on, glabel in [(True, "+g"), (False, "0g")]:
        g = gymapi.acquire_gym()
        sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
        sp.gravity = gymapi.Vec3(0, 0, -9.81 if grav_on else 0.0)
        sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
        sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
        sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

        if ground_type == "plane":
            pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
            pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
            g.add_ground(sim, pp)
        else:
            # Static box as ground
            box_opts = gymapi.AssetOptions(); box_opts.fix_base_link = True
            box_opts.density = 1000.0
            box_asset = g.create_box(sim, 20.0, 20.0, 0.05, box_opts)
            # Set box shape props
            # For box ground, we need to set contact props too
            env0 = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
            box_ah = g.create_actor(env0, box_asset, gymapi.Transform(p=gymapi.Vec3(0,0,-0.025)), "ground_box", 0, -1, 0)
            # Set box shape props
            try:
                box_props = g.get_actor_rigid_shape_properties(env0, box_ah)
                for i in range(len(box_props)):
                    box_props[i].contact_offset = 0.001
                    box_props[i].rest_offset = 0.0
                    box_props[i].restitution = 0.0
                    box_props[i].friction = 0.8
                g.set_actor_rigid_shape_properties(env0, box_ah, box_props)
            except:
                pass

        ac = rc.asset; opts = gymapi.AssetOptions()
        for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
                  "fix_base_link","density","angular_damping","linear_damping",
                  "max_angular_velocity","max_linear_velocity","armature","thickness"]:
            setattr(opts, k, getattr(ac, k))
        opts.default_dof_drive_mode = 3

        asset = g.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                             "q1_22dof_rl_collision.urdf", opts)
        dnames = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dnames)

        if ground_type == "box":
            env2 = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
        else:
            env2 = env0 if ground_type == "box" else g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
        ah = g.create_actor(env2, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z_default)), "q1", -1, 0, 0)

        defaults = dict(rc.init_state.default_joint_angles)
        tgt = [float(defaults.get(n, 0)) for n in dnames]
        dof_st = g.get_actor_dof_states(env2, ah, gymapi.STATE_ALL)
        for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
        g.set_actor_dof_states(env2, ah, dof_st, gymapi.STATE_ALL)
        g.prepare_sim(sim)

        props = g.get_actor_rigid_shape_properties(env2, ah)
        for i in range(len(props)): props[i].contact_offset=0.001; props[i].rest_offset=0.0; props[i].restitution=0.0; props[i].friction=0.8
        g.set_actor_rigid_shape_properties(env2, ah, props)

        rt = g.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
        dt = g.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
        r_v[0, 0:3] = torch.tensor([0.0, 0.0, root_z_default], device="cuda:0")
        r_v[0, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda:0"); r_v[0, 7:13] = 0.0
        g.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
        for i in range(nd): d_v[i, 0] = tgt[i]; d_v[i, 1] = 0.0
        g.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
        g.refresh_actor_root_state_tensor(sim); g.refresh_dof_state_tensor(sim)

        # Count actors
        actor_count = g.get_actor_count(env2)
        print(f"  [{label}] grav={glabel} actors={actor_count}")

        res = run_simple_trial(g, sim, env2, ah, nd, tgt, dnames, bnames, bnames, False, None, None, None, None, 5, False)
        f1 = res[1]
        print(f"    f0: rz={res[0]['rz']:.3f} lv={res[0]['lv']:.3f}")
        print(f"    f1: rz={f1['rz']:.3f} lv={f1['lv']:.3f} av={f1['av']:.3f} cf={f1['top_cf_name']}:{f1['top_cf_val']:.0f} bz_min={f1['bz_min']:.3f}")
        g.destroy_sim(sim)

# ================================================================
# EXP 3: Minimal box/sphere sanity check
# ================================================================
print("\n" + "=" * 80)
print("EXP 3: Minimal box/sphere sanity check")
print("=" * 80)

for shape_type, shape_args, rz_test in [("box", (0.1, 0.1, 0.1), 0.2), ("box", (0.1, 0.1, 0.1), 1.0),
                                          ("sphere", (0.05,), 0.2), ("sphere", (0.05,), 1.0)]:
    for grav_on in [True, False]:
        g = gymapi.acquire_gym()
        sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
        sp.gravity = gymapi.Vec3(0, 0, -9.81 if grav_on else 0.0)
        sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
        sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
        sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
        pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
        pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
        g.add_ground(sim, pp)

        if shape_type == "box":
            asset = make_box_asset(g, sim, *shape_args)
        else:
            opts = gymapi.AssetOptions(); opts.density = 1000.0; opts.fix_base_link = False
            asset = g.create_sphere(sim, shape_args[0], opts)

        env = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
        ah = g.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz_test)), "test", -1, 0, 0)
        g.prepare_sim(sim)

        res = run_simple_trial(g, sim, env, ah, 0, [], [], [], [], False, None, None, None, None, 5, False)
        f1 = res[1]
        anomaly = "PUSHED UP" if (f1['rz'] - rz_test > 0.01 and not grav_on) else "OK"
        print(f"  {shape_type} rz={rz_test} grav={grav_on}: f0_rz={res[0]['rz']:.3f} f1_rz={f1['rz']:.3f} lv={f1['lv']:.3f} av={f1['av']:.3f} cf={f1['top_cf_name']}:{f1['top_cf_val']:.0f} {anomaly}")
        g.destroy_sim(sim)

# ================================================================
# EXP 4: No-collision Q1
# ================================================================
print("\n" + "=" * 80)
print("EXP 4: No-collision Q1")
print("=" * 80)

# Create temp no-collision URDF
import xml.etree.ElementTree as ET
src = str(PROJECT_ROOT / "humanoidverse/data/robots/q1/q1_22dof_rl_collision.urdf")
dst = str(PROJECT_ROOT / "humanoidverse/data/robots/q1/q1_22dof_no_collision.urdf")
tree = ET.parse(src); root_el = tree.getroot()
removed = 0
for link in root_el.iter('link'):
    for col in list(link.findall('collision')):
        link.remove(col); removed += 1
tree.write(dst, encoding='utf-8', xml_declaration=True)
print(f"  Created no-collision URDF, removed {removed} collisions")

for grav_on in [True, False]:
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81 if grav_on else 0.0)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
    g.add_ground(sim, pp)

    ac = rc.asset; opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = 3

    asset = g.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                         "q1_22dof_no_collision.urdf", opts)
    dnames = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dnames)
    shape_count = g.get_actor_rigid_shape_count(env := g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1))
    print(f"  grav={grav_on}: rigid_shapes={shape_count}")

    env2 = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    ah = g.create_actor(env2, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z_default)), "q1", -1, 0, 0)

    defaults = dict(rc.init_state.default_joint_angles)
    tgt = [float(defaults.get(n, 0)) for n in dnames]
    dof_st = g.get_actor_dof_states(env2, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    g.set_actor_dof_states(env2, ah, dof_st, gymapi.STATE_ALL)
    g.prepare_sim(sim)

    rt = g.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
    dt = g.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
    r_v[0, 0:3] = torch.tensor([0.0, 0.0, root_z_default], device="cuda:0")
    r_v[0, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda:0"); r_v[0, 7:13] = 0.0
    g.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    for i in range(nd): d_v[i, 0] = tgt[i]; d_v[i, 1] = 0.0
    g.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(sim); g.refresh_dof_state_tensor(sim)

    res = run_simple_trial(g, sim, env2, ah, nd, tgt, dnames, bnames, bnames, False, None, None, None, None, 5, False)
    f1 = res[1]
    anomaly = "PUSHED UP (no collision!)" if (f1['rz'] - root_z_default > 0.01 and not grav_on) else "OK"
    print(f"    f0: rz={res[0]['rz']:.3f}  f1: rz={f1['rz']:.3f} lv={f1['lv']:.3f} av={f1['av']:.3f} {anomaly}")
    g.destroy_sim(sim)

# ================================================================
# EXP 5: PlaneParams distance scan
# ================================================================
print("\n" + "=" * 80)
print("EXP 5: PlaneParams distance scan")
print("=" * 80)

for dist in [0.0, 1.0, -1.0]:
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1); pp.distance = dist
    pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
    g.add_ground(sim, pp)

    # Drop a box from z=0.5 to find where it stops
    box_opts = gymapi.AssetOptions(); box_opts.density = 1000.0; box_opts.fix_base_link = False
    box_asset = g.create_box(sim, 0.1, 0.1, 0.1, box_opts)
    env = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    ah = g.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0,0,0.5)), "box", -1, 0, 0)
    g.prepare_sim(sim)
    res = run_simple_trial(g, sim, env, ah, 0, [], [], ["box_body"], ["box_body"], False, None, None, None, None, 30, False)
    final_rz = res[-1]['rz']
    print(f"  plane distance={dist:+5.1f}: box settles at z={final_rz:.4f} (expected bottom at 0.05, so plane at z={final_rz-0.05:.4f})")
    g.destroy_sim(sim)

# ================================================================
# EXP 7: Actor index verification
# ================================================================
print("\n" + "=" * 80)
print("EXP 7: Actor index verification")
print("=" * 80)

g = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0)  # zero-g
sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

# Add ground (actor 0 implicitly)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
g.add_ground(sim, pp)

# Also add a static box ground
box_opts = gymapi.AssetOptions(); box_opts.fix_base_link = True
box_asset = g.create_box(sim, 20.0, 20.0, 0.05, box_opts)
env = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
box_ah = g.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0,0,-0.025)), "ground_box", 0, -1, 0)

# Now load Q1
ac = rc.asset; opts = gymapi.AssetOptions()
for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
          "fix_base_link","density","angular_damping","linear_damping",
          "max_angular_velocity","max_linear_velocity","armature","thickness"]:
    setattr(opts, k, getattr(ac, k))
opts.default_dof_drive_mode = 3
asset = g.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                     "q1_22dof_rl_collision.urdf", opts)
dnames = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dnames)
env2 = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
ah_q1 = g.create_actor(env2, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z_default)), "q1", -1, 0, 0)

actor_count = g.get_actor_count(env2)
print(f"  Actors in Q1 env: {actor_count}")
for ai in range(actor_count):
    try:
        name = g.get_actor_name(env2, ai)
        handle = g.get_actor_handle(env2, ai)
        print(f"    actor[{ai}]: name={name} handle={handle}")
    except:
        print(f"    actor[{ai}]: (error getting info)")

g.prepare_sim(sim)
rt = g.acquire_actor_root_state_tensor(sim)
print(f"  root_state_tensor shape: {rt.shape}")
r_all = gymtorch.wrap_tensor(rt)
num_roots = r_all.shape[0] // 13
r_v = r_all.view(-1, 13)
for ai in range(num_roots):
    print(f"    root[{ai}]: pos={r_v[ai,:3].cpu().numpy()}")

g.destroy_sim(sim)

# ================================================================
# EXP 8: Force estimation from momentum change (reuse Case A data)
# ================================================================
print("\n" + "=" * 80)
print("EXP 8: Momentum-based force estimation")
print("=" * 80)

# Quick re-run Case A (noPD + ground + grav) with detailed velocity tracking
g = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
sim = g.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
g.add_ground(sim, pp)

ac = rc.asset; opts = gymapi.AssetOptions()
for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
          "fix_base_link","density","angular_damping","linear_damping",
          "max_angular_velocity","max_linear_velocity","armature","thickness"]:
    setattr(opts, k, getattr(ac, k))
opts.default_dof_drive_mode = 3
asset = g.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                     "q1_22dof_rl_collision.urdf", opts)
dnames = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dnames)
env = g.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
ah = g.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z_default)), "q1", -1, 0, 0)

defaults = dict(rc.init_state.default_joint_angles)
tgt = [float(defaults.get(n, 0)) for n in dnames]
dof_st = g.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
g.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)
g.prepare_sim(sim)

props = g.get_actor_rigid_shape_properties(env, ah)
for i in range(len(props)): props[i].contact_offset=0.001; props[i].rest_offset=0.0; props[i].restitution=0.0; props[i].friction=0.8
g.set_actor_rigid_shape_properties(env, ah, props)

rt = g.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
dt = g.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
cf_t = g.acquire_net_contact_force_tensor(sim); cf = gymtorch.wrap_tensor(cf_t)

r_v[0,:3] = torch.tensor([0.0,0.0,root_z_default],device="cuda:0")
r_v[0,3:7] = torch.tensor([0,0,0,1],device="cuda:0"); r_v[0,7:13]=0.0
g.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
for i in range(nd): d_v[i,0]=tgt[i]; d_v[i,1]=0.0
g.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
g.refresh_actor_root_state_tensor(sim); g.refresh_dof_state_tensor(sim)

# Step 0 init
lv_before = np.array([r_v[0,7].item(), r_v[0,8].item(), r_v[0,9].item()])
av_before = np.array([r_v[0,10].item(), r_v[0,11].item(), r_v[0,12].item()])
print(f"  Before simulate: lv={lv_before} av={av_before}")

# Simulate 1 step (4 sub-steps)
tau = torch.zeros(nd, device="cuda:0")
g.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))
for _ in range(4): g.simulate(sim)
g.fetch_results(sim, True)
g.refresh_actor_root_state_tensor(sim); g.refresh_dof_state_tensor(sim)
g.refresh_net_contact_force_tensor(sim)

lv_after = np.array([r_v[0,7].item(), r_v[0,8].item(), r_v[0,9].item()])
av_after = np.array([r_v[0,10].item(), r_v[0,11].item(), r_v[0,12].item()])

dt_sim = 0.02
delta_v = lv_after - lv_before
delta_av = av_after - av_before
apparent_acc = delta_v / dt_sim
apparent_force = Q1_MASS * apparent_acc

cf_np = cf.view(-1,3).norm(dim=1).cpu().numpy()
cf_max = cf_np.max()
cf_total = cf_np.sum()

print(f"  After 1 step:  lv={lv_after} av={av_after}")
print(f"  delta_v = {delta_v}  |delta_v| = {np.linalg.norm(delta_v):.3f} m/s")
print(f"  delta_av = {delta_av}  |delta_av| = {np.linalg.norm(delta_av):.3f} rad/s")
print(f"  apparent_acc = {apparent_acc}  |a| = {np.linalg.norm(apparent_acc):.1f} m/s^2 ({np.linalg.norm(apparent_acc)/9.81:.1f}g)")
print(f"  apparent_force = {apparent_force}  |F| = {np.linalg.norm(apparent_force):.0f} N")
print(f"  contact_tensor max = {cf_max:.1f} N  sum = {cf_total:.1f} N")
print(f"  force_imbalance: apparent={np.linalg.norm(apparent_force):.0f}N vs contact={cf_total:.1f}N")

g.destroy_sim(sim)

# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print("""
Key findings to check in output above:

EXP 1: If rz=5.0 still gets lv~3, it's NOT geometric contact → sim/tensor issue
       If only rz=0.39 gets lv~3, it's contact-related

EXP 2: If static box ground behaves differently from add_ground → PlaneParams issue
       If both same → not specific to infinite plane

EXP 3: If minimal box/sphere also gets pushed up → sim/ground config issue
       If box/sphere is normal → Q1-specific issue

EXP 4: If no-collision Q1 still gets pushed → NOT a collision shape issue
       If no-collision Q1 is normal → hidden collision in rl_collision URDF

EXP 5: Verify plane is actually at z=0 via box drop test

EXP 7: Verify Q1 actor index is correct in root tensor

EXP 8: Compare apparent_force (from momentum) vs contact_tensor sum
       Large discrepancy → solver impulse not captured in net_contact_force_tensor
""")

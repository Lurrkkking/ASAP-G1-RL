"""
Q1 Contact/Ground Root-Cause Diagnosis — 7-section systematic debug.
"""
import sys, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def projected_gz(qx, qy, qz, qw):
    return 1.0 - 2.0 * (qx * qx + qy * qy)

# ==============================================================================
# Reusable setup
# ==============================================================================

def setup_sim(gym, gravity_z, ground_distance, fix_base, urdf_name, collision_filter):
    """Create sim with controlled params. Returns (sim, env, ah, dnames, bnames, ndof, tensors...)"""
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, gravity_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    if ground_distance is not None:
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        pp.distance = ground_distance
        pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
        gym.add_ground(sim, pp)

    conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
    rc = conf.robot
    ac = rc.asset; opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints", "replace_cylinder_with_capsule", "flip_visual_attachments",
              "fix_base_link", "density", "angular_damping", "linear_damping",
              "max_angular_velocity", "max_linear_velocity", "armature", "thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = ac.default_dof_drive_mode
    opts.fix_base_link = fix_base

    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf_name, opts)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset)
    ndof = len(dnames)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)

    root_z = float(rc.init_state.pos[2])
    start_pose = gymapi.Transform(
        p=gymapi.Vec3(float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), root_z),
        r=gymapi.Quat(float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
                      float(rc.init_state.rot[2]), float(rc.init_state.rot[3])))

    # collision filter: pass group and filter directly
    group = -1 if collision_filter is None else collision_filter
    ah = gym.create_actor(env, asset, start_pose, "q1", group, 0, 0)

    defaults = dict(rc.init_state.default_joint_angles)
    tgt = [float(defaults.get(n, 0.0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(ndof): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)
    gym.prepare_sim(sim)

    # acquire
    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); d_v = d_all.view(-1, 2)
    cft = gym.acquire_net_contact_force_tensor(sim); cf = gymtorch.wrap_tensor(cft).view(-1, 3)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    rigid_v = rigid.view(rigid.shape[0], 13)

    # FIX: clean init
    r_v[0, 0:3] = torch.tensor([0.0, 0.0, root_z], device="cuda:0")
    r_v[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
    r_v[0, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    for i in range(ndof): d_v[i, 0] = tgt[i]; d_v[i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # Verify clean
    assert abs(r_v[0, 2].item() - root_z) < 0.01
    assert r_v[0, 7:13].abs().max().item() < 1e-3
    assert d_v[:, 1].abs().max().item() < 1e-3

    # Return tensors
    pd_stiff = dict(rc.control.stiffness); pd_damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, np.float32); dg_np = np.zeros(ndof, np.float32)
    for i, n in enumerate(dnames):
        for k in pd_stiff:
            if k in n: pg_np[i] = pd_stiff[k]; dg_np[i] = pd_damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(tgt, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah)["effort"][i].item()
                            for i in range(ndof)], device="cuda:0")

    return sim, env, ah, dnames, bnames, ndof, tgt, root_z, \
           r_all, r_v, d_all, d_v, rigid_v, cf, pg, dg, default_t, tau_lim


def run_frames_with_diag(gym, sim, r_v, d_v, cf, rigid_v, bnames, dnames, ndof, tgt,
                          pg, dg, default_t, tau_lim, nframes=5, apply_pd=False,
                          label="", ground_z=0.0):
    """Run frames with PROPER contact refresh timing."""
    Lf = bnames.index("left_ankle_roll_link")
    Rf = bnames.index("right_ankle_roll_link")

    print(f"\n  [{label}] frames={nframes} PD={apply_pd} ground_z={ground_z}")
    print(f"  {'f':>3s} {'rz':>8s} {'pgz':>7s} {'lv':>7s} {'av':>8s} {'Lfz':>8s} {'Rfz':>8s} {'|tau|':>7s}  top_contacts")

    for f_no in range(nframes):
        # ---- CORRECT ORDER: read before, compute torque, simulate, fetch, refresh, read ----
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        dp = d_v[:, 0]; dv_ = d_v[:, 1]
        if apply_pd:
            tau = (pg * (default_t - dp) - dg * dv_).clamp(-tau_lim, tau_lim)
        else:
            tau = torch.zeros(ndof, device="cuda:0")
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        # CRITICAL: refresh AFTER fetch
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)  # <-- MUST refresh before reading contact

        rz = r_v[0, 2].item(); lv = r_v[0, 7:10].norm().item(); av = r_v[0, 10:13].norm().item()
        qx, qy, qz_, qw = r_v[0, 3].item(), r_v[0, 4].item(), r_v[0, 5].item(), r_v[0, 6].item()
        pgz = projected_gz(qx, qy, qz_, qw)
        Lfz = cf[Lf, 2].item(); Rfz = cf[Rf, 2].item()
        tau_abs = abs(tau.cpu().numpy()).max()

        # Contact forces — now properly refreshed
        cf_n = cf.norm(dim=1); vals, idxs = cf_n.topk(min(8, len(cf_n)))
        contacts = [f"{bnames[idxs[j].item()]}:{vals[j].item():.0f}" for j in range(len(vals)) if vals[j] > 0.5]
        contact_str = ", ".join(contacts[:5]) if contacts else "NONE"

        print(f"  {f_no:3d} {rz:+8.3f} {pgz:+7.3f} {lv:+7.3f} {av:+8.3f} {Lfz:+8.0f} {Rfz:+8.0f} {tau_abs:+7.1f}  {contact_str}")

    return r_v[0, 2].item(), r_v[0, 7:10].norm().item(), r_v[0, 10:13].norm().item()


def print_init_diag(gym, sim, env, ah, r_v, d_v, rigid_v, cf, bnames, dnames, ndof, tgt, root_z, ground_z):
    """Print detailed init state."""
    Lf = bnames.index("left_ankle_roll_link"); Rf = bnames.index("right_ankle_roll_link")
    body_z = rigid_v[:, 2].cpu().numpy()

    # Compute foot box bottom
    # Box at origin, size_z=0.04 -> bottom local = -0.02
    box_bottom_local = -0.02  # size_z=0.04
    L_box_bottom = rigid_v[Lf, 2].item() + box_bottom_local
    R_box_bottom = rigid_v[Rf, 2].item() + box_bottom_local

    print(f"  root_z={r_v[0,2].item():.4f} quat={r_v[0,3:7].cpu().numpy()} lv={r_v[0,7:10].cpu().numpy()} av={r_v[0,10:13].cpu().numpy()}")
    print(f"  body_z=[{body_z.min():.4f},{body_z.max():.4f}] foot_link_z=[{rigid_v[Lf,2].item():.4f},{rigid_v[Rf,2].item():.4f}]")
    print(f"  box_bottom_world: L={L_box_bottom:.4f} R={R_box_bottom:.4f}  ground_z={ground_z}  penetration: L={ground_z-L_box_bottom:.4f} R={ground_z-R_box_bottom:.4f}")
    print(f"  dof_vel_max={d_v[:,1].abs().max().item():.6f}  dof_pos_err_max={max(abs(d_v[i,0].item()-tgt[i]) for i in range(ndof)):.6f}")
    cf_n = cf.norm(dim=1); vals, idxs = cf_n.topk(min(10, len(cf_n)))
    contacts = [(bnames[idxs[j].item()], vals[j].item()) for j in range(len(vals)) if vals[j] > 0.1]
    print(f"  init contacts: {contacts if contacts else 'NONE'}")


def print_rigid_shape_props(gym, env, ah, bnames):
    """Print all rigid shape properties."""
    props = gym.get_actor_rigid_shape_properties(env, ah)
    # Get per-body shape indices
    num_b = gym.get_actor_rigid_body_count(env, ah)
    print(f"\n  Total rigid shapes: {len(props)}")
    shape_idx = 0
    for bi in range(num_b):
        try:
            ir = gym.get_actor_rigid_body_shape_indices(env, ah, bi)
        except:
            continue
        bname = bnames[bi] if bi < len(bnames) else f"body_{bi}"
        for si in range(ir.start, ir.start + ir.count):
            sp = props[si]
            print(f"  shape[{si}] body[{bi}]={bname:35s} "
                  f"friction={sp.friction:.2f} restitution={sp.restitution:.2f} "
                  f"contact_offset={sp.contact_offset:.4f} rest_offset={sp.rest_offset:.4f} "
                  f"thickness={sp.thickness:.4f} filter={sp.filter}")
            shape_idx += 1


def set_uniform_shape_props(gym, env, ah):
    """Set uniform reasonable shape properties."""
    props = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(props)):
        props[i].friction = 0.8
        props[i].restitution = 0.0
        props[i].contact_offset = 0.005
        props[i].rest_offset = 0.0
    gym.set_actor_rigid_shape_properties(env, ah, props)
    return props


# ==============================================================================
print("=" * 80)
print("Q1 CONTACT/GROUND DIAGNOSIS")
print("=" * 80)

conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
rc = conf.robot
conf_root_z = float(rc.init_state.pos[2])
urdf = "q1_22dof_simplified_collision.urdf"

# ====== SEC 1: Contact tensor refresh timing ======
print("\n" + "=" * 80)
print("SEC 1: Contact tensor refresh timing + re-run")
print("=" * 80)

for label, grav, gnd, apply_pd in [
    ("noPD +g +ground", -9.81, 0.0, False),
    ("noPD 0g +ground", 0.0, 0.0, False),
    ("noPD +g -ground", -9.81, None, False),
]:
    g1 = gymapi.acquire_gym()
    s1, e1, a1, dn1, bn1, nd1, tgt1, rz1, \
        ra1, rv1, da1, dv1, rig1, cf1, pg1, dg1, dt1, tl1 = \
        setup_sim(g1, grav, gnd, False, urdf, None)
    print_init_diag(g1, s1, e1, a1, rv1, dv1, rig1, cf1, bn1, dn1, nd1, tgt1, rz1, gnd if gnd else 0.0)
    run_frames_with_diag(g1, s1, rv1, dv1, cf1, rig1, bn1, dn1, nd1, tgt1,
                         pg1, dg1, dt1, tl1, nframes=5, apply_pd=apply_pd, label=label)
    g1.destroy_sim(s1)

# ====== SEC 2: Ground height experiment ======
print("\n" + "=" * 80)
print("SEC 2: Ground height experiment")
print("=" * 80)

for gnd_z, gnd_label in [(0.0, "plane z=0"), (-1.0, "plane z=-1"), (-10.0, "plane z=-10")]:
    g2 = gymapi.acquire_gym()
    s2, e2, a2, dn2, bn2, nd2, tgt2, rz2, \
        ra2, rv2, da2, dv2, rig2, cf2, pg2, dg2, dt2, tl2 = \
        setup_sim(g2, -9.81, gnd_z, False, urdf, None)
    print_init_diag(g2, s2, e2, a2, rv2, dv2, rig2, cf2, bn2, dn2, nd2, tgt2, rz2, gnd_z)
    run_frames_with_diag(g2, s2, rv2, dv2, cf2, rig2, bn2, dn2, nd2, tgt2,
                         pg2, dg2, dt2, tl2, nframes=5, apply_pd=False,
                         label=f"noPD +g {gnd_label}", ground_z=gnd_z)
    g2.destroy_sim(s2)

# ====== SEC 3: Collision filter experiment ======
print("\n" + "=" * 80)
print("SEC 3: Collision filter — disable robot-ground collision")
print("=" * 80)

# Case 1: normal collision (group=-1)
# Case 2: disable collision — use collisionGroup=0 with filter setup
# In IsaacGym, to disable collision with ground: group such that (group & groundFilter) == 0
# Ground defaults: group=0, filter=-1. To avoid: ensure (actorGroup & -1) produces no match
# Use group=0 for actor — with ground, (0 & -1) = 0 means no collision
# Actually that's wrong. Let me think...
# IsaacGym collision: (A.group & B.filter) && (B.group & A.filter)
# Ground: group=-1, filter=-1
# Actor group=X, filter=Y: collide if (X & -1)!=0 && (-1 & Y)!=0 → X!=0 && Y!=0
# So if X=0: no collision. If Y=0 AND Y is not used correctly...
# Let me try: group=-1, filter=0. (actor & ground) = (-1 & -1) = -1 != 0 ✓ (first check)
# (ground & actor) = (-1 & 0) = 0 → NO collision ✓

for case_label, coll_filter in [
    ("Case 1: normal collision  (group=-1)", None),
    ("Case 2: filter=0 (no coll?)", 0),  # actor filter=0, so (ground.group & actor.filter) = (-1 & 0) = 0
]:
    g3 = gymapi.acquire_gym()
    # Pass filter as group param — actual collision control
    # Using group=-1, but need to control filter. Let me just use the create_actor group parameter
    # For Case 2: group=0 → (0 & ground.filter) = (0 & -1) = 0 → no collision with ground

    # Actually I need to think about the correct approach. Let me just create with different groups.
    # For no collision: use group that doesn't match ground's filter bits
    # Ground filter = -1 (all bits). Any non-zero group will collide. group=0 will not collide.
    actual_group = -1 if coll_filter is None else 0  # group=0 → no ground collision

    s3, e3, a3, dn3, bn3, nd3, tgt3, rz3, \
        ra3, rv3, da3, dv3, rig3, cf3, pg3, dg3, dt3, tl3 = \
        setup_sim(g3, -9.81, 0.0, False, urdf, actual_group)
    print_init_diag(g3, s3, e3, a3, rv3, dv3, rig3, cf3, bn3, dn3, nd3, tgt3, rz3, 0.0)
    run_frames_with_diag(g3, s3, rv3, dv3, cf3, rig3, bn3, dn3, nd3, tgt3,
                         pg3, dg3, dt3, tl3, nframes=5, apply_pd=False,
                         label=f"{case_label} group={actual_group}")
    g3.destroy_sim(s3)

# Case 3: no ground (already done in SEC 1)

# ====== SEC 4: Rigid shape properties ======
print("\n" + "=" * 80)
print("SEC 4: Rigid shape properties")
print("=" * 80)

g4 = gymapi.acquire_gym()
s4, e4, a4, dn4, bn4, nd4, tgt4, rz4, \
    ra4, rv4, da4, dv4, rig4, cf4, pg4, dg4, dt4, tl4 = \
    setup_sim(g4, -9.81, 0.0, False, urdf, None)

print("\n  --- BEFORE fix ---")
print_rigid_shape_props(g4, e4, a4, bn4)

# Set uniform props
set_uniform_shape_props(g4, e4, a4)

print("\n  --- AFTER fix ---")
print_rigid_shape_props(g4, e4, a4, bn4)

# Re-acquire tensors and clean init for the modified shapes
g4.refresh_actor_root_state_tensor(s4); g4.refresh_dof_state_tensor(s4)
g4.refresh_rigid_body_state_tensor(s4); g4.refresh_net_contact_force_tensor(s4)
# Re-fix state
rv4[0, 0:3] = torch.tensor([0.0, 0.0, rz4], device="cuda:0")
rv4[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
rv4[0, 7:13] = 0.0
g4.set_actor_root_state_tensor(s4, gymtorch.unwrap_tensor(ra4))
for i in range(nd4): dv4[i, 0] = tgt4[i]; dv4[i, 1] = 0.0
g4.set_dof_state_tensor(s4, gymtorch.unwrap_tensor(da4))
g4.refresh_actor_root_state_tensor(s4); g4.refresh_dof_state_tensor(s4)
g4.refresh_rigid_body_state_tensor(s4); g4.refresh_net_contact_force_tensor(s4)

print("\n  --- With fixed shape props ---")
for label, grav, gnd in [
    ("noPD +g +ground", -9.81, 0.0),
    ("noPD 0g +ground", 0.0, 0.0),
]:
    print_init_diag(g4, s4, e4, a4, rv4, dv4, rig4, cf4, bn4, dn4, nd4, tgt4, rz4, gnd if gnd else 0.0)
    run_frames_with_diag(g4, s4, rv4, dv4, cf4, rig4, bn4, dn4, nd4, tgt4,
                         pg4, dg4, dt4, tl4, nframes=5, apply_pd=False, label=label, ground_z=gnd if gnd else 0.0)
g4.destroy_sim(s4)

# ====== SEC 5: Precise foot box bottom + root_z scan ======
print("\n" + "=" * 80)
print("SEC 5: Foot box bottom z scan")
print("=" * 80)

print(f"\n  {'rz':>7s} {'L_box_bot':>10s} {'R_box_bot':>10s} {'Lfz_f0':>9s} {'Rfz_f0':>9s} {'lv_f0':>7s} {'av_f0':>8s} {'top_contact_f0':>40s}")
BOX_HALF_Z = 0.02  # size_z=0.04

for rz_test in [0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]:
    g5 = gymapi.acquire_gym()
    s5, e5, a5, dn5, bn5, nd5, tgt5, rz5_ignore, \
        ra5, rv5, da5, dv5, rig5, cf5, pg5, dg5, dt5, tl5 = \
        setup_sim(g5, -9.81, 0.0, False, urdf, None)

    # Override root_z
    rv5[0, 2] = rz_test
    g5.set_actor_root_state_tensor(s5, gymtorch.unwrap_tensor(ra5))
    g5.refresh_actor_root_state_tensor(s5); g5.refresh_rigid_body_state_tensor(s5); g5.refresh_net_contact_force_tensor(s5)

    Lf5 = bn5.index("left_ankle_roll_link"); Rf5 = bn5.index("right_ankle_roll_link")
    L_box = rigid5_v = g5.acquire_rigid_body_state_tensor(s5)
    rigid5 = gymtorch.wrap_tensor(rigid5_v).view(rigid5_v.shape[0], 13)
    # Actually just reuse rig5 which was already acquired
    L_box_bot = rig5[Lf5, 2].item() - BOX_HALF_Z
    R_box_bot = rig5[Rf5, 2].item() - BOX_HALF_Z

    # 1 frame
    tau5 = torch.zeros(nd5, device="cuda:0")
    g5.set_dof_actuation_force_tensor(s5, gymtorch.unwrap_tensor(tau5.contiguous()))
    for _ in range(4): g5.simulate(s5)
    g5.fetch_results(s5, True)
    g5.refresh_actor_root_state_tensor(s5)
    g5.refresh_net_contact_force_tensor(s5)

    lv0 = rv5[0, 7:10].norm().item(); av0 = rv5[0, 10:13].norm().item()
    Lfz0 = cf5[Lf5, 2].item(); Rfz0 = cf5[Rf5, 2].item()
    cf_n5 = cf5.norm(dim=1); vals, idxs = cf_n5.topk(min(3, len(cf_n5)))
    top_cf = ", ".join([f"{bn5[idxs[j].item()]}:{vals[j].item():.0f}" for j in range(len(vals)) if vals[j] > 1]) or "NONE"

    print(f"  {rz_test:+.2f}  {L_box_bot:+10.4f}  {R_box_bot:+10.4f}  {Lfz0:+9.0f}  {Rfz0:+9.0f}  {lv0:+7.2f}  {av0:+8.3f}  {top_cf}")
    g5.destroy_sim(s5)

# ====== SEC 6: fix_base_link ======
print("\n" + "=" * 80)
print("SEC 6: fix_base_link=True")
print("=" * 80)

g6 = gymapi.acquire_gym()
s6, e6, a6, dn6, bn6, nd6, tgt6, rz6, \
    ra6, rv6, da6, dv6, rig6, cf6, pg6, dg6, dt6, tl6 = \
    setup_sim(g6, -9.81, 0.0, True, urdf, None)

# With fix_base, root shouldn't move. Let's just run 20 frames
Lf6 = bn6.index("left_ankle_roll_link"); Rf6 = bn6.index("right_ankle_roll_link")
print(f"  fix_base=True  root_z={rv6[0,2].item():.4f}")
print(f"  {'f':>3s} {'rz':>8s} {'|lv|':>7s} {'|av|':>7s} {'Lfz':>8s} {'Rfz':>8s} {'dof_vel_max':>11s} {'top_contacts':>50s}")

for f_no in range(20):
    g6.refresh_actor_root_state_tensor(s6); g6.refresh_dof_state_tensor(s6); g6.refresh_net_contact_force_tensor(s6)
    tau6 = torch.zeros(nd6, device="cuda:0")
    g6.set_dof_actuation_force_tensor(s6, gymtorch.unwrap_tensor(tau6.contiguous()))
    for _ in range(4): g6.simulate(s6)
    g6.fetch_results(s6, True)
    g6.refresh_actor_root_state_tensor(s6); g6.refresh_dof_state_tensor(s6); g6.refresh_net_contact_force_tensor(s6)

    rz_f = rv6[0, 2].item(); lv_f = rv6[0, 7:10].norm().item(); av_f = rv6[0, 10:13].norm().item()
    Lfz = cf6[Lf6, 2].item(); Rfz = cf6[Rf6, 2].item()
    dofvm = dv6[:, 1].abs().max().item()
    cf_n6 = cf6.norm(dim=1); vals, idxs = cf_n6.topk(min(3, len(cf_n6)))
    top = ", ".join([f"{bn6[idxs[j].item()]}:{vals[j].item():.0f}" for j in range(len(vals)) if vals[j] > 1]) or "NONE"

    if f_no < 5 or f_no % 5 == 0:
        print(f"  {f_no:3d} {rz_f:+8.4f} {lv_f:+7.3f} {av_f:+7.3f} {Lfz:+8.0f} {Rfz:+8.0f} {dofvm:+11.4f}  {top}")
g6.destroy_sim(s6)

# ====== SEC 7: Conclusions ======
print("\n" + "=" * 80)
print("SEC 7: CONCLUSIONS")
print("=" * 80)
print("""
  Check the output above and answer:
  1. contacts=NONE是否可信？看 SEC1 输出中 contact 是否真的 NONE（正确刷新后）
  2. ground z=-10 是否还飞？SEC2
  3. disable collision (group=0) 后是否还飞？SEC3 Case 2
  4. shape props contact_offset/rest_offset 是否异常？SEC4
  5. foot box bottom z 扫描结果？SEC5
  6. fix_base_link 结果？SEC6
""")

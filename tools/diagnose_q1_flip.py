"""
Q1 Flip Root-Cause Diagnosis — 5 experiments + DOF check + foot scan.
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


def make_sim(gym, gravity_z, add_ground):
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, gravity_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if add_ground: gym.add_ground(sim, gymapi.PlaneParams())
    return sim


def load_asset(gym, sim, rc):
    ac = rc.asset; opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = ac.default_dof_drive_mode
    return gym.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof.urdf", opts)


def setup_all(gym, rc, root_z, gravity_z=-9.81, add_ground=True):
    """Create sim, actor, set DOF, prepare, fix state. Returns all handles."""
    sim = make_sim(gym, gravity_z, add_ground)
    asset = load_asset(gym, sim, rc)
    dof_names = gym.get_asset_dof_names(asset); body_names = gym.get_asset_rigid_body_names(asset)
    ndof = len(dof_names)
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    start_pose = gymapi.Transform(
        p=gymapi.Vec3(float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), root_z),
        r=gymapi.Quat(float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
                      float(rc.init_state.rot[2]), float(rc.init_state.rot[3])))
    ah = gym.create_actor(env, asset, start_pose, "q1", -1, 0, 0)

    defaults = dict(rc.init_state.default_joint_angles)
    target_pos = [float(defaults.get(n, 0.0)) for n in dof_names]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(ndof): dof_st[i]["pos"] = target_pos[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)
    gym.prepare_sim(sim)

    rt = gym.acquire_actor_root_state_tensor(sim); root_all = gymtorch.wrap_tensor(rt); root_v = root_all.view(-1, 13)
    dt_t = gym.acquire_dof_state_tensor(sim); dof_all = gymtorch.wrap_tensor(dt_t); dof_v = dof_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    rigid_v = rigid.view(rigid.shape[0], 13)
    cf_t = gym.acquire_net_contact_force_tensor(sim); cf = gymtorch.wrap_tensor(cf_t).view(-1, 3)

    # FIX
    root_v[0, 0:3] = torch.tensor([0.0, 0.0, root_z], device="cuda:0")
    root_v[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
    root_v[0, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_all))
    for i in range(ndof): dof_v[i, 0] = target_pos[i]; dof_v[i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # PD
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, np.float32); dg_np = np.zeros(ndof, np.float32)
    for i, n in enumerate(dof_names):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(target_pos, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah)["effort"][i].item()
                            for i in range(ndof)], device="cuda:0")

    return sim, env, ah, dof_names, body_names, ndof, target_pos, default_t, pg, dg, tau_lim, \
           root_all, root_v, dof_all, dof_v, rigid_v, cf


def verify_clean(root_v, dof_v, target_pos, root_z, ndof):
    ok = abs(root_v[0, 2].item() - root_z) < 0.01
    ok = ok and root_v[0, 7:13].abs().max().item() < 1e-3
    ok = ok and dof_v[:, 1].abs().max().item() < 1e-3
    for i in range(ndof):
        if abs(dof_v[i, 0].item() - target_pos[i]) > 0.001:
            ok = False; break
    return ok


def box_bottom(rigid_v, body_names):
    L = body_names.index("left_ankle_roll_link"); R = body_names.index("right_ankle_roll_link")
    return rigid_v[L, 2].item() - 0.025, rigid_v[R, 2].item() - 0.025


# ==============================================================================
print("=" * 70)
print("Q1 FLIP DIAGNOSIS")
print("=" * 70)

conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
rc = conf.robot; rz0 = float(rc.init_state.pos[2]); print(f"config root_z={rz0}")

# ====== SEC 1: Script check ======
print("\n" + "=" * 70)
print("SEC 1: Script consistency")
for script in ["test_q1_standing.py", "record_q1_standing.py"]:
    with open(str(PROJECT_ROOT / script)) as f:
        c = f.read()
    ok = all(k in c for k in ["set_actor_root_state_tensor","set_dof_state_tensor","7:13"])
    print(f"  {script}: fix_present={ok}")

# ====== SEC 2: Before/After first simulate ======
print("\n" + "=" * 70)
print("SEC 2: Before/After first simulate")
gym = gymapi.acquire_gym()
sim, env, ah, dnames, bnames, ndof, tgt, dflt_t, pg, dg, tlim, \
    r_all, r_v, d_all, d_v, rig_v, cf = setup_all(gym, rc, rz0)

print("\n  --- BEFORE first simulate ---")
clean = verify_clean(r_v, d_v, tgt, rz0, ndof)
print(f"  STATE CLEAN: {clean}" if clean else f"  STATE DIRTY!")
print(f"  root_z={r_v[0,2].item():.4f} quat={r_v[0,3:7].cpu().numpy()} lv={r_v[0,7:10].cpu().numpy()} av={r_v[0,10:13].cpu().numpy()}")
print(f"  dof_vel_max={d_v[:,1].abs().max().item():.6f} dof_pos_err_max={max(abs(d_v[i,0].item()-tgt[i]) for i in range(ndof)):.6f}")

Lb, Rb = box_bottom(rig_v, bnames)
print(f"  foot_box_bottom: L={Lb:.4f} R={Rb:.4f} {'PENETRATING' if Lb<0 or Rb<0 else 'ABOVE'}")

body_z = rig_v[:,2].cpu().numpy()
print(f"  body_z range: [{body_z.min():.4f}, {body_z.max():.4f}]")

# DOF drive check
props = gym.get_actor_dof_properties(env, ah)
modes = {0:"NONE", 1:"POS", 2:"VEL", 3:"EFFORT"}
bad_modes = []
for i, n in enumerate(dnames):
    dm = int(props["driveMode"][i])
    if dm != 3: bad_modes.append((i, n, dm))
if bad_modes:
    for i,n,dm in bad_modes: print(f"  BAD DRIVE MODE: {n} = {modes.get(dm,str(dm))}")
else:
    print(f"  DOF drive: all EFFORT (mode 3) — OK")

# 1 step with tau=0
print("\n  --- AFTER 1 step (tau=0) ---")
zero_t = torch.zeros(ndof, device="cuda:0")
gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_t.contiguous()))
for _ in range(4): gym.simulate(sim)
gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

print(f"  root_z={r_v[0,2].item():.4f} lv={r_v[0,7:10].norm().item():.4f} av={r_v[0,10:13].norm().item():.4f}")
pgz1 = projected_gz(r_v[0,3].item(),r_v[0,4].item(),r_v[0,5].item(),r_v[0,6].item())
print(f"  pgz={pgz1:.4f} dof_vel_max={d_v[:,1].abs().max().item():.4f}")

cf_n = cf.norm(dim=1); vals, idxs = cf_n.topk(min(5, len(cf_n)))
for j in range(len(vals)):
    if vals[j] > 0.1:
        print(f"    contact: {bnames[idxs[j].item()]:35s} {vals[j].item():.0f} N")
gym.destroy_sim(sim)

# ====== SEC 3: 5 experiments ======
print("\n" + "=" * 70)
print("SEC 3: 5 Control Experiments (10 frames each)")

configs = [
    ("A: noPD +g  +ground", True,  -9.81, False),
    ("B: noPD -g  +ground", True,   0.0,  False),
    ("C: noPD +g  -ground", False, -9.81, False),
    ("D: PD   +g  -ground", False, -9.81, True),
    ("E: PD   +g  +ground", True,  -9.81, True),
]

for label, add_gr, grav_z, use_pd in configs:
    g2 = gymapi.acquire_gym()
    s2, e2, a2, dn2, bn2, nd2, tgt2, df2, pg2, dg2, tl2, \
        ra2, rv2, da2, dv2, rig2, cf2 = setup_all(g2, rc, rz0, gravity_z=grav_z, add_ground=add_gr)
    clean2 = verify_clean(rv2, dv2, tgt2, rz0, nd2)
    Lb2,Rb2 = box_bottom(rig2, bn2)

    Lf2 = bn2.index("left_ankle_roll_link"); Rf2 = bn2.index("right_ankle_roll_link")
    desc = f"{label} [clean={clean2} Lbox={Lb2:.3f} Rbox={Rb2:.3f}]"
    print(f"\n  {desc}")

    for f in range(10):
        g2.refresh_actor_root_state_tensor(s2); g2.refresh_dof_state_tensor(s2); g2.refresh_net_contact_force_tensor(s2)
        dp2 = dv2[:,0]; dv2_ = dv2[:,1]
        tau2 = (pg2*(df2-dp2)-dg2*dv2_).clamp(-tl2,tl2) if use_pd else torch.zeros(nd2, device="cuda:0")
        g2.set_dof_actuation_force_tensor(s2, gymtorch.unwrap_tensor(tau2.contiguous()))
        for _ in range(4): g2.simulate(s2)
        g2.fetch_results(s2, True)
        g2.refresh_actor_root_state_tensor(s2); g2.refresh_dof_state_tensor(s2); g2.refresh_net_contact_force_tensor(s2)

        rz_f = rv2[0,2].item(); lv_f = rv2[0,7:10].norm().item(); av_f = rv2[0,10:13].norm().item()
        cf_n2 = cf2.norm(dim=1); top_i2 = cf_n2.argmax().item(); top_n2 = bn2[top_i2]
        if f < 5:
            print(f"    f{f}: rz={rz_f:+.2f} lv={lv_f:.1f} av={av_f:.1f} Lfz={cf2[Lf2,2].item():.0f} Rfz={cf2[Rf2,2].item():.0f} tau={abs(tau2.cpu().numpy()).max():.0f} cf_top={top_n2}:{cf_n2[top_i2].item():.0f}")

        if abs(rz_f) > 100:
            print(f"    f{f}: EXPLODED!")
            break
    g2.destroy_sim(s2)

# ====== SEC 4-6: root_z scan ======
print("\n" + "=" * 70)
print("SEC 4-6: root_z scan (no PD, 3 frames)")
print(f"  {'rz':>6s} {'Lbox_bot':>9s} {'Rbox_bot':>9s} {'Lfz_f0':>9s} {'Rfz_f0':>9s} {'lv_f0':>7s} {'av_f0':>7s}")

for rz_test in [0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]:
    g3 = gymapi.acquire_gym()
    s3, _, _, dn3, bn3, nd3, tgt3, _, _, _, _, \
        ra3, rv3, da3, dv3, rig3, cf3 = setup_all(g3, rc, rz_test)
    Lf3 = bn3.index("left_ankle_roll_link"); Rf3 = bn3.index("right_ankle_roll_link")
    Lb3,Rb3 = box_bottom(rig3, bn3)

    lv0, av0, lfz0, rfz0 = 0,0,0,0
    for f in range(3):
        g3.set_dof_actuation_force_tensor(s3, gymtorch.unwrap_tensor(torch.zeros(nd3, device="cuda:0").contiguous()))
        for _ in range(4): g3.simulate(s3)
        g3.fetch_results(s3, True)
        g3.refresh_actor_root_state_tensor(s3); g3.refresh_net_contact_force_tensor(s3)
        if f == 0:
            lv0 = rv3[0,7:10].norm().item(); av0 = rv3[0,10:13].norm().item()
            lfz0 = cf3[Lf3,2].item(); rfz0 = cf3[Rf3,2].item()

    print(f"  {rz_test:+.2f} {Lb3:+9.4f} {Rb3:+9.4f} {lfz0:+9.0f} {rfz0:+9.0f} {lv0:+7.2f} {av0:+7.2f}")
    g3.destroy_sim(s3)

# ====== SEC 7: Conclusions ======
print("\n" + "=" * 70)
print("SEC 7: PRELIMINARY CONCLUSIONS")
print("=" * 70)
print("""
  Check output above for:
  1. SEC 2 "STATE CLEAN" — init is correct
  2. SEC 2 "AFTER 1 step" — where does first velocity come from
  3. SEC 3 Exp A vs B vs C — isolate contact/gravity/PD
  4. SEC 2 "DOF drive" — all must be EFFORT (mode 3)
  5. SEC 2 "foot_box_bottom" — penetration depth
  6. SEC 4-6 root_z scan — find box-bottom≈0 setting
""")

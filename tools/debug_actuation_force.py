"""
Debug actuation force tensor — isolate noPD anomaly.
Hypothesis: not calling set_dof_actuation_force_tensor leaves stale GPU buffer.
"""
import sys, numpy as np, csv, pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def pgz(qx, qy, qz, qw): return 1.0 - 2.0*(qx*qx + qy*qy)

# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot
rz = float(rc.init_state.pos[2])
defaults = dict(rc.init_state.default_joint_angles)

print("=" * 70)
print("ACTUATION FORCE TENSOR DEBUG")
print("  root_z=%s" % rz)
print("=" * 70)


def run_case(label, grav_z, with_ground, control_mode, n_steps, filename):
    """control_mode: 'noPD_no_write', 'noPD_zero_write', 'PD'"""
    gym = gymapi.acquire_gym()

    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)

    # ---- Static box ground ----
    if with_ground:
        bo = gymapi.AssetOptions(); bo.fix_base_link = True
        ba = gym.create_box(sim, 20.0, 20.0, 0.05, bo)
        gnd_ah = gym.create_actor(env, ba, gymapi.Transform(p=gymapi.Vec3(0, 0, -0.025)), "ground_box", 0, -1, 0)
        gp = gym.get_actor_rigid_shape_properties(env, gnd_ah)
        for i in range(len(gp)): gp[i].contact_offset = 0.001; gp[i].rest_offset = 0.0; gp[i].restitution = 0.0; gp[i].friction = 0.8
        gym.set_actor_rigid_shape_properties(env, gnd_ah, gp)

    # ---- Q1 actor ----
    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao, k, getattr(ac, k))
    ao.default_dof_drive_mode = 3
    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"),
                           "q1_22dof_rl_collision.urdf", ao)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dnames)

    ah_q1 = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, rz)), "q1", -1, 0, 0)

    tgt = [float(defaults.get(n, 0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah_q1, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah_q1, dof_st, gymapi.STATE_ALL)

    qp = gym.get_actor_rigid_shape_properties(env, ah_q1)
    for i in range(len(qp)): qp[i].contact_offset = 0.001; qp[i].rest_offset = 0.0; qp[i].restitution = 0.0; qp[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah_q1, qp)

    gym.prepare_sim(sim)

    # ---- Tensors + state fix ----
    rt = gym.acquire_actor_root_state_tensor(sim)
    r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim)
    d_all = gymtorch.wrap_tensor(dt); d_flat = d_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    n_actors = gym.get_actor_count(env)
    q1_idx = 0
    for ai in range(n_actors):
        if "q1" in gym.get_actor_name(env, ai).lower(): q1_idx = ai; break

    r_v[q1_idx, 0:3] = torch.tensor([0.0, 0.0, rz], device="cuda:0")
    r_v[q1_idx, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda:0"); r_v[q1_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    for i in range(nd): d_flat[i, 0] = tgt[i]; d_flat[i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    assert r_v[q1_idx, 7:13].abs().max().item() < 1e-3
    assert d_flat[:nd, 1].abs().max().item() < 1e-3
    print("\n  [%s] STATE_CLEAN q1_idx=%d" % (label, q1_idx))

    # ---- PD gains ----
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dnames):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(tgt, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah_q1)["effort"][i].item() for i in range(nd)], device="cuda:0")
    total_mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, ah_q1))

    Lf = bnames.index("left_ankle_roll_link"); Rf = bnames.index("right_ankle_roll_link")

    # ---- CREATE TORQUE TENSOR ----
    total_dofs = gym.get_sim_dof_count(sim)
    torque_tensor = torch.zeros(total_dofs, device="cuda:0")
    print("  total_dofs=%d  Q1_dofs=%d  q1_slice=[0:%d]" % (total_dofs, nd, nd))

    # ---- PRE-SIM: write zero torque for zero_write/PD cases ----
    wrote_before = False
    if control_mode != 'noPD_no_write':
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_tensor.contiguous()))
        wrote_before = True
        print("  PRE-SIM: wrote zero torque tensor (norm=%.6f)" % torque_tensor.norm().item())

    # ---- Run ----
    csv_rows = []; pkl_data = []; lv_prev = np.zeros(3)

    for fno in range(n_steps + 1):
        if fno == 0:
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        else:
            # Write torque BEFORE simulate
            wrote_this_frame = False
            if control_mode == 'noPD_zero_write':
                torque_tensor.zero_()
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_tensor.contiguous()))
                wrote_this_frame = True
            elif control_mode == 'PD':
                torque_tensor.zero_()
                dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
                tau = (pg * (default_t - dz) - dg * dvel).clamp(-tau_lim, tau_lim)
                torque_tensor[:nd] = tau
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_tensor.contiguous()))
                wrote_this_frame = True
            # control_mode == 'noPD_no_write': DO NOTHING

            for _ in range(4): gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v = rigid.view(rigid.shape[0], 13)
        tsim = fno * 0.02

        rz_f = r_v[q1_idx, 2].item()
        lv = r_v[q1_idx, 7:10].cpu().numpy(); av = r_v[q1_idx, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[q1_idx, 3:7].cpu().numpy().tolist()
        pgz_v = pgz(qx, qy, qz_, qw)
        lv_n = np.linalg.norm(lv); av_n = np.linalg.norm(av)

        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"; top_v = cf_np[top_i]

        dv = lv - lv_prev; app_f = total_mass * dv / 0.02; app_fn = np.linalg.norm(app_f)
        lv_prev = lv.copy()

        tau_norm = torque_tensor.norm().item()
        tau_max = torque_tensor.abs().max().item()
        tau_nz = (torque_tensor.abs() > 1e-6).sum().item()

        row = {
            'frame': fno, 't': tsim, 'rz': rz_f, 'pgz': pgz_v,
            'lv_norm': lv_n, 'av_norm': av_n,
            'top_cf_name': top_n, 'top_cf_val': top_v,
            'apparent_force_norm': app_fn,
            'tau_tensor_norm': tau_norm, 'tau_tensor_max': tau_max, 'tau_tensor_nz': tau_nz,
            'wrote_actuation': wrote_before if fno == 0 else (control_mode != 'noPD_no_write'),
            'control_mode': control_mode,
        }
        csv_rows.append(row)

        if fno < 10:
            wrote = row['wrote_actuation']
            print("  f%d: rz=%.3f lv=%.2f av=%.2f pgz=%.3f cf=%s:%.0f appF=%.0f tau_norm=%.3f wrote=%s" %
                  (fno, rz_f, lv_n, av_n, pgz_v, top_n, top_v, app_fn, tau_norm, wrote))

    # Save CSV
    with open(str(OUTPUT_DIR / (filename + ".csv")), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    f1 = csv_rows[1] if len(csv_rows) > 1 else {}
    flipped = f1.get('lv_norm', 0) > 5.0
    print("  => f1_rz=%.3f f1_lv=%.2f f1_av=%.2f f1_cf=%s:%.0f flipped=%s" %
          (f1.get('rz',0), f1.get('lv_norm',0), f1.get('av_norm',0),
           f1.get('top_cf_name','?'), f1.get('top_cf_val',0), flipped))
    gym.destroy_sim(sim)
    return csv_rows


# ====== RUN 6 CASES ======
results = {}

for label, grav_z, with_ground, control_mode, fname in [
    ("1_noPD_no_write+g+gnd", -9.81, True, "noPD_no_write", "act_force_case1_noPD_no_write"),
    ("2_noPD_zero_write+g+gnd", -9.81, True, "noPD_zero_write", "act_force_case2_noPD_zero_write"),
    ("3_noPD_zero_write+zerog+gnd", 0.0, True, "noPD_zero_write", "act_force_case3_zerog_zero_write"),
    ("4_noPD_no_write+zerog+gnd", 0.0, True, "noPD_no_write", "act_force_case4_zerog_no_write"),
    ("5_PD+g+gnd", -9.81, True, "PD", "act_force_case5_PD"),
    ("6_noPD_zero_write+g+nognd", -9.81, False, "noPD_zero_write", "act_force_case6_zero_write_nognd"),
]:
    rows = run_case(label, grav_z, with_ground, control_mode, 50, fname)
    results[label] = rows


# ====== TABLE 1 ======
print("\n" + "=" * 100)
print("TABLE 1: Case Comparison")
print("=" * 100)
hdr = "  %-35s %6s %5s %15s %6s %7s %7s %7s %7s %18s %8s %8s %6s"
print(hdr % ("case", "gnd", "grav", "ctrl_mode", "wrote", "f1_rz", "f1_lv", "f1_av",
             "f1_pgz", "cf_body", "cf_N", "appF", "flip"))
for label, rows in results.items():
    if rows and len(rows) > 1:
        r = rows[1]
        gnd = "no" if "nognd" in label else "yes"
        grav = "off" if "zerog" in label else "on"
        cm = r.get('control_mode', '?')
        wr = str(r.get('wrote_actuation', '?'))
        fl = "YES" if r.get('lv_norm', 0) > 5 else "no"
        print(hdr % (label, gnd, grav, cm, wr, r['rz'], r['lv_norm'], r['av_norm'],
                     r['pgz'], r.get('top_cf_name','?'), r.get('top_cf_val',0),
                     r.get('apparent_force_norm',0), fl))


# ====== TABLE 2 ======
print("\n" + "=" * 100)
print("TABLE 2: Conclusions")
print("=" * 100)

# Check Case 1 vs Case 2
c1 = results.get("1_noPD_no_write+g+gnd", [])
c2 = results.get("2_noPD_zero_write+g+gnd", [])
c5 = results.get("5_PD+g+gnd", [])

f1_1 = c1[1] if len(c1) > 1 else {}
f1_2 = c2[1] if len(c2) > 1 else {}
f1_5 = c5[1] if len(c5) > 1 else {}

c1_flipped = f1_1.get('lv_norm', 0) > 5
c2_flipped = f1_2.get('lv_norm', 0) > 5
c5_flipped = f1_5.get('lv_norm', 0) > 5

print("  1. noPD_no_write anomalous: %s (lv=%.1f)" % ("YES" if c1_flipped else "no", f1_1.get('lv_norm', 0)))
print("  2. noPD_zero_write normal:  %s (lv=%.1f)" % ("YES" if not c2_flipped else "NO", f1_2.get('lv_norm', 0)))
print("  3. PD normal:               %s (lv=%.1f)" % ("YES" if not c5_flipped else "NO", f1_5.get('lv_norm', 0)))

if c1_flipped and not c2_flipped:
    print("\n  *** ROOT CAUSE: noPD_no_write does NOT call set_dof_actuation_force_tensor ***")
    print("  *** The actuation force buffer contains uninitialized/stale values. ***")
    print("  *** Fix: ALWAYS call set_dof_actuation_force_tensor with explicit zeros in noPD mode. ***")
elif c2_flipped:
    print("\n  *** noPD_zero_write STILL flips — root cause is NOT actuation force buffer ***")
    print("  *** Need to continue investigating other causes (collision, state, physics) ***")
else:
    print("\n  *** Both noPD_zero_write and PD are normal — root cause confirmed as actuation buffer ***")

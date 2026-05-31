"""
Q1 PD Standing Compare — prepared_pose vs yaml_default.
"""
import sys, os, csv, pickle
from pathlib import Path
import numpy as np, cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def pgz(qx, qy, qz, qw): return 1.0 - 2.0*(qx*qx + qy*qy)

def rpy(qx, qy, qz, qw):
    import numpy as np
    r = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    p = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
    y = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return np.degrees(r), np.degrees(p), np.degrees(y)

def to_bgr(raw, h, w):
    arr = np.asarray(raw)
    if arr.ndim == 1:
        if arr.size == h*w:
            packed = arr.astype(np.uint32).reshape(h,w)
            rgba = np.zeros((h,w,4), dtype=np.uint8)
            rgba[...,0]=(packed>>16)&0xFF; rgba[...,1]=(packed>>8)&0xFF; rgba[...,2]=packed&0xFF; rgba[...,3]=255
            arr = rgba
        else: arr = arr.reshape(h, w, 4)
    elif arr.ndim == 2 and arr.shape[1] == w*4: arr = arr.reshape(h, w, 4)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return cv2.cvtColor(arr[...,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.zeros((h,w,3), dtype=np.uint8)

def overlay(img, d, label):
    f = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        label,
        "f=%d t=%.2f" % (d.get('frame',0), d.get('t',0)),
        "rz=%.3f pgz=%.3f" % (d.get('rz',0), d.get('pgz',1)),
        "lv=%.2f av=%.2f" % (d.get('lv_norm',0), d.get('av_norm',0)),
        "tau=%.1f sat=%.0f%%" % (d.get('tau_max',0), d.get('tau_sat',0)*100),
        "cf=%s:%.0f" % (d.get('top_cf_name','?'), d.get('top_cf_val',0)),
    ]
    for i, l in enumerate(lines):
        cv2.putText(img, l, (10, 20+i*22), f, 0.5, (0,255,0), 1)


# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot
rz_cfg = float(rc.init_state.pos[2])
urdf_name = "q1_22dof_rl_collision.urdf"
defaults_dict = dict(rc.init_state.default_joint_angles)

print("=" * 70)
print("Q1 PD STANDING COMPARE")
print("  yaml: q1_22dof_rl_collision.yaml  urdf: %s  root_z=%.3f" % (urdf_name, rz_cfg))
print("=" * 70)


def run_pd_case(label, target_mode, n_steps, fname):
    """target_mode: 'prepared_pose' or 'yaml_default'"""
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    if hasattr(sp, 'enable_camera_sensors'): sp.enable_camera_sensors = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
    gym.add_ground(sim, pp)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)

    # Load Q1
    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao, k, getattr(ac, k))
    ao.default_dof_drive_mode = 3
    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf_name, ao)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dnames)

    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, rz_cfg)), "q1", -1, 0, 0)

    # Print basic info
    print("\n  [%s]" % label)
    print("  urdf=%s  yaml=%s" % (urdf_name, "q1_22dof_rl_collision.yaml"))

    # Verify DOF drive mode
    dof_props = gym.get_actor_dof_properties(env, ah)
    modes_ok = all(int(dof_props["driveMode"][i]) == 3 for i in range(nd))
    print("  driveMode all EFFORT: %s" % modes_ok)

    # ---- STEP 1: set DOF BEFORE prepare_sim ----
    tgt_default = [float(defaults_dict.get(n, 0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt_default[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    # ---- STEP 2: shape props ----
    qp = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(qp)): qp[i].contact_offset = 0.001; qp[i].rest_offset = 0.0; qp[i].restitution = 0.0; qp[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah, qp)

    cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
    cam_h = gym.create_camera_sensor(env, cam)

    total_mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, ah))
    print("  total_mass=%.1f kg  shapes=%d" % (total_mass, len(qp)))

    # ---- STEP 3: prepare_sim ----
    gym.prepare_sim(sim)

    # ---- STEP 4: acquire tensors ----
    rt = gym.acquire_actor_root_state_tensor(sim)
    r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim)
    d_all = gymtorch.wrap_tensor(dt); d_flat = d_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # Record prepared pose (post-prepare_sim, before any fix)
    prepared_dof_pos = d_flat[:nd, 0].clone().cpu().numpy()

    print("  prepared knee=[%.4f,%.4f] hip_pitch=[%.4f,%.4f] ankleP=[%.4f,%.4f]" %
          (prepared_dof_pos[3], prepared_dof_pos[9],
           prepared_dof_pos[0], prepared_dof_pos[6],
           prepared_dof_pos[4], prepared_dof_pos[10]))
    print("  default  knee=[%.4f,%.4f] hip_pitch=[%.4f,%.4f] ankleP=[%.4f,%.4f]" %
          (tgt_default[3], tgt_default[9], tgt_default[0], tgt_default[6], tgt_default[4], tgt_default[10]))

    # ---- STEP 5: only zero velocities ----
    r_v[0, 7:13] = 0.0; d_flat[:nd, 1] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # ---- Determine target ----
    if target_mode == 'prepared_pose':
        target_dof = prepared_dof_pos.copy()
    else:
        target_dof = np.array(tgt_default)

    print("  target knee=[%.4f,%.4f] diff_from_current=[%.4f,%.4f]" %
          (target_dof[3], target_dof[9],
           d_flat[3,0].item() - target_dof[3], d_flat[9,0].item() - target_dof[9]))

    # Print diff between prepared and default
    print("\n  prepared vs yaml_default diff (top 8):")
    diffs = [(abs(prepared_dof_pos[i] - tgt_default[i]), dnames[i], prepared_dof_pos[i], tgt_default[i]) for i in range(nd)]
    diffs.sort(reverse=True)
    for d, name, prep, yml in diffs[:8]:
        print("    %-35s prep=%.4f yml=%.4f diff=%.4f" % (name, prep, yml, d))

    # ---- PD gains ----
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dnames):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(target_dof, device="cuda:0")
    effort_limits = np.array([dof_props["effort"][i].item() for i in range(nd)])
    tau_lim = torch.tensor(effort_limits, device="cuda:0")

    Lf = bnames.index("left_ankle_roll_link"); Rf = bnames.index("right_ankle_roll_link")
    torque_t = torch.zeros(nd, device="cuda:0")

    # ---- Video ----
    writer = cv2.VideoWriter(str(OUTPUT_DIR / (fname + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    csv_rows = []

    def capture(fno, tsim):
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v = rigid.view(rigid.shape[0], 13)
        dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
        rz_f = r_v[0, 2].item(); lv = r_v[0, 7:10].cpu().numpy(); av = r_v[0, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[0, 3:7].cpu().numpy().tolist()
        pgz_v = pgz(qx, qy, qz_, qw); roll_v, pitch_v, yaw_v = rpy(qx, qy, qz_, qw)
        lv_n = np.linalg.norm(lv); av_n = np.linalg.norm(av)
        tau_np = torque_t.cpu().numpy()
        tau_max = abs(tau_np).max(); tau_sat = (abs(tau_np) > 0.95 * effort_limits).mean()
        sat_joints = [dnames[i] for i in range(nd) if abs(tau_np[i]) > 0.95 * effort_limits[i]]
        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"; top_v = cf_np[top_i]
        nfc = sum(cf_np[i] for i in range(len(bnames)) if 'ankle_roll' not in bnames[i] and cf_np[i] > 1)
        bz = rigid_v[:,2].cpu().numpy(); bz_min = bz.min(); bz_min_idx = bz.argmin()
        bz_min_name = bnames[bz_min_idx] if bz_min_idx < len(bnames) else "?"
        dof_err = max(abs(dz[i].item() - target_dof[i]) for i in range(nd))
        dvel_max = dvel.abs().max().item()
        row = {
            'frame': fno, 't': tsim,
            'rz': rz_f, 'pgz': pgz_v, 'roll': roll_v, 'pitch': pitch_v, 'yaw': yaw_v,
            'lv_norm': lv_n, 'av_norm': av_n,
            'dof_pos_err_max': dof_err, 'dof_vel_max': dvel_max,
            'tau_max': tau_max, 'tau_sat': tau_sat,
            'saturated_joints': ",".join(sat_joints),
            'L_fz': cf_np[Lf], 'R_fz': cf_np[Rf],
            'non_foot_cf_sum': nfc,
            'top_cf_name': top_n, 'top_cf_val': top_v,
            'bz_min': bz_min, 'bz_min_name': bz_min_name,
        }
        csv_rows.append(row)
        return row

    # Frame 0
    gym.set_camera_location(cam_h, env, gymapi.Vec3(2.5, 2.5, 1.5), gymapi.Vec3(0, 0, 0.4))
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
    r0 = capture(0, 0.0)
    fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
    overlay(fr, r0, label)
    writer.write(fr)

    stand_ok = True; fell_t = None; min_pgz = 1.0; max_av = 0.0; max_tau = 0.0; max_sat = 0.0; nf_total = 0
    first_sat_joint = None

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        torque_t.zero_()
        dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
        tau_v = (pg * (default_t - dz) - dg * dvel).clamp(-tau_lim, tau_lim)
        torque_t[:] = tau_v
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))

        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        fno = step + 1; tsim = fno * 0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        row = capture(fno, tsim)
        min_pgz = min(min_pgz, row['pgz'])
        max_av = max(max_av, row['av_norm'])
        max_tau = max(max_tau, row['tau_max'])
        max_sat = max(max_sat, row['tau_sat'])
        nf_total += (1 if row['non_foot_cf_sum'] > 0 else 0)
        if first_sat_joint is None and row['saturated_joints']:
            first_sat_joint = row['saturated_joints']

        if stand_ok:
            if row['pgz'] < 0.5: stand_ok = False; fell_t = tsim
            if row['rz'] < 0.15: stand_ok = False; fell_t = tsim if fell_t is None else fell_t

        if fno % 25 == 0:
            print("  f%4d t=%.1f rz=%.3f pgz=%.3f lv=%.2f av=%.2f tau=%.1f sat=%.1f%% cf=%s:%.0f nfc=%s" %
                  (fno, tsim, row['rz'], row['pgz'], row['lv_norm'], row['av_norm'],
                   row['tau_max'], row['tau_sat']*100, row['top_cf_name'], row['top_cf_val'],
                   "YES" if row['non_foot_cf_sum'] > 0 else "no"))

        if fno % 4 == 0:
            rx = r_v[0, 0].item(); ry = r_v[0, 1].item(); rz2 = r_v[0, 2].item()
            gym.set_camera_location(cam_h, env,
                gymapi.Vec3(rx+2.5, ry+2.5, max(rz2+1.2, 1.5)), gymapi.Vec3(rx, ry, rz2+0.3))
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
            overlay(fr, row, label)
            writer.write(fr)

    writer.release()
    with open(str(OUTPUT_DIR / (fname + ".csv")), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    final_rz = csv_rows[-1]['rz'] if csv_rows else 0
    print("  => stand=%s fell_t=%s min_pgz=%.3f max_av=%.1f max_tau=%.1f nf=%d first_sat=%s" %
          (stand_ok, fell_t, min_pgz, max_av, max_tau, nf_total,
           first_sat_joint[:40] if first_sat_joint else "none"))
    gym.destroy_sim(sim)

    return {
        'label': label, 'target': target_mode,
        'stand': stand_ok, 'fell_t': fell_t,
        'final_rz': final_rz, 'min_pgz': min_pgz, 'max_av': max_av,
        'max_tau': max_tau, 'max_sat': max_sat, 'nf_total': nf_total,
        'first_sat': first_sat_joint,
        'prepared_pos': prepared_dof_pos, 'default_pos': np.array(tgt_default),
        'dof_names': dnames,
    }


# Run both cases
res_A = run_pd_case("A_prepared_pose", "prepared_pose", 250, "q1_pd_A_prepared_pose")  # 5s
res_B = run_pd_case("B_yaml_default", "yaml_default", 250, "q1_pd_B_yaml_default")

# ====== TABLE 1 ======
print("\n" + "=" * 80)
print("TABLE 1: PD Standing Results")
print("=" * 80)
print("  %-20s %15s %6s %10s %10s %10s %10s %10s %10s %8s %30s" %
      ("case", "target", "stand", "fell_t", "final_rz", "min_pgz", "max_av", "max_tau", "max_sat", "nf", "first_saturated"))

for r in [res_A, res_B]:
    print("  %-20s %15s %6s %10s %10.3f %10.3f %10.1f %10.1f %9.1f%% %8d %30s" %
          (r['label'], r['target'], r['stand'], r['fell_t'],
           r['final_rz'], r['min_pgz'], r['max_av'], r['max_tau'],
           r['max_sat']*100, r['nf_total'],
           r['first_sat'][:30] if r['first_sat'] else "none"))

# ====== TABLE 2 ======
print("\n" + "=" * 80)
print("TABLE 2: prepared_dof_pos vs default_joint_angles")
print("=" * 80)
diffs = [(abs(res_A['prepared_pos'][i] - res_A['default_pos'][i]),
          res_A['dof_names'][i], res_A['prepared_pos'][i], res_A['default_pos'][i])
         for i in range(len(res_A['dof_names']))]
diffs.sort(reverse=True)
print("  %-35s %10s %10s %10s" % ("joint", "prepared", "yaml", "diff"))
for d, name, prep, yml in diffs:
    print("  %-35s %+10.4f %+10.4f %+10.4f" % (name, prep, yml, d))

# ====== TABLE 3 ======
print("\n" + "=" * 80)
print("TABLE 3: Conclusions")
print("=" * 80)
print("  1. PD hold prepared pose:  %s (fell_t=%s)" % ("STOOD" if res_A['stand'] else "FELL", res_A['fell_t']))
print("  2. PD hold yaml default:   %s (fell_t=%s)" % ("STOOD" if res_B['stand'] else "FELL", res_B['fell_t']))

if res_A['stand'] and not res_B['stand']:
    print("  3. YAML default_joint_angles NOT suitable for Q1.")
    print("     prepared_pose is stable, yaml_default causes fall.")
    print("     -> Update default_joint_angles to prepared_pose values.")
elif not res_A['stand'] and not res_B['stand']:
    print("  3. Neither pose stands. PD gains insufficient or torque limit bottleneck.")
    print("     First saturated joint: %s" % (res_A.get('first_sat','?')))
    if res_A['max_sat'] > 0.9:
        print("     Torque saturation >= 90%% — PD gains too low or effort limits too low.")
elif res_A['stand'] and res_B['stand']:
    print("  3. Both poses stand. Current PD/default pose is sufficient.")

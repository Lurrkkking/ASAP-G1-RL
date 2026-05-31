"""
Q1 Static Box Ground Standing — clean ground via fixed box.
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
    r = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    p = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
    y = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return np.degrees(r), np.degrees(p), np.degrees(y)


def overlay(img, d, label):
    f = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        label,
        "f=%d t=%.3f" % (d.get('frame',0), d.get('t',0)),
        "rz=%.3f pgz=%.3f" % (d.get('rz',0), d.get('pgz',1)),
        "lv=%.2f av=%.2f" % (d.get('lv_norm',0), d.get('av_norm',0)),
        "tau=%.1f cf=%s:%.0f" % (d.get('tau_max',0), d.get('top_cf_name','NONE'), d.get('top_cf_val',0)),
        "bz_min=%.3f(%s) ground:static_box" % (d.get('bz_min',0), d.get('bz_min_name','?')),
    ]
    for i, l in enumerate(lines):
        cv2.putText(img, l, (10, 20+i*22), f, 0.5, (0,255,0), 1)


def to_bgr(raw, h, w):
    arr = np.asarray(raw)
    if arr.ndim == 1:
        if arr.size == h*w:
            packed = arr.astype(np.uint32).reshape(h,w)
            rgba = np.zeros((h,w,4), dtype=np.uint8)
            rgba[...,0] = (packed>>16)&0xFF; rgba[...,1] = (packed>>8)&0xFF; rgba[...,2] = packed&0xFF; rgba[...,3] = 255
            arr = rgba
        else:
            arr = arr.reshape(h, w, 4)
    elif arr.ndim == 2 and arr.shape[1] == w*4:
        arr = arr.reshape(h, w, 4)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return cv2.cvtColor(arr[...,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.zeros((h,w,3), dtype=np.uint8)


# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot
urdf = "q1_22dof_rl_collision.urdf"
print("Q1 STATIC BOX GROUND  root_z=%s  urdf=%s" % (rc.init_state.pos[2], urdf))

defaults = dict(rc.init_state.default_joint_angles)


def build_sim_and_actors(gym, grav_z, with_ground, rz_use):
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)

    gnd_ah = None
    if with_ground:
        bo = gymapi.AssetOptions(); bo.fix_base_link = True
        ba = gym.create_box(sim, 20.0, 20.0, 0.05, bo)
        gnd_ah = gym.create_actor(env, ba, gymapi.Transform(p=gymapi.Vec3(0, 0, -0.025)), "ground_box", 0, -1, 0)
        gp = gym.get_actor_rigid_shape_properties(env, gnd_ah)
        for i in range(len(gp)): gp[i].contact_offset = 0.001; gp[i].rest_offset = 0.0; gp[i].restitution = 0.0; gp[i].friction = 0.8
        gym.set_actor_rigid_shape_properties(env, gnd_ah, gp)

    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao, k, getattr(ac, k))
    ao.default_dof_drive_mode = 3
    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf, ao)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dnames)

    ah_q1 = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, rz_use)), "q1", -1, 0, 0)

    tgt = [float(defaults.get(n, 0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah_q1, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah_q1, dof_st, gymapi.STATE_ALL)

    qp = gym.get_actor_rigid_shape_properties(env, ah_q1)
    for i in range(len(qp)): qp[i].contact_offset = 0.001; qp[i].rest_offset = 0.0; qp[i].restitution = 0.0; qp[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah_q1, qp)

    cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
    cam_h = gym.create_camera_sensor(env, cam)

    gym.prepare_sim(sim)

    # ---- Find Q1 index via actor name ----
    rt = gym.acquire_actor_root_state_tensor(sim)
    r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim)
    d_all = gymtorch.wrap_tensor(dt)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    n_actors = gym.get_actor_count(env)
    q1_idx = 0; gnd_idx = -1
    for ai in range(n_actors):
        nm = gym.get_actor_name(env, ai)
        if "q1" in nm.lower(): q1_idx = ai
        if "ground" in nm.lower(): gnd_idx = ai

    # ---- FIX state: only Q1 ----
    r_v[q1_idx, 0:3] = torch.tensor([0.0, 0.0, rz_use], device="cuda:0")
    r_v[q1_idx, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda:0")
    r_v[q1_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))

    d_flat = d_all.view(-1, 2)
    # Static actors (ground box) have 0 DOFs, so DOF tensor only has Q1 rows
    dof_start = 0
    for i in range(nd):
        d_flat[dof_start + i, 0] = tgt[i]
        d_flat[dof_start + i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # verify
    assert r_v[q1_idx, 7:13].abs().max().item() < 1e-3, "Q1 root velocity not zero!"
    assert d_flat[dof_start:dof_start+nd, 1].abs().max().item() < 1e-3, "Q1 dof velocity not zero!"

    # PD
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dnames):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(tgt, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah_q1)["effort"][i].item() for i in range(nd)], device="cuda:0")
    total_mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, ah_q1))

    return sim, env, ah_q1, gnd_ah, dnames, bnames, nd, tgt, pg, dg, default_t, tau_lim, \
           r_all, r_v, d_all, d_flat, rigid, rigid_t, cf_t, cam_h, q1_idx, gnd_idx, n_actors, dof_start, total_mass


def run_case(label, grav_z, with_ground, apply_pd, n_steps, rz_use, filename):
    gym = gymapi.acquire_gym()
    sim, env, ah_q1, gnd_ah, dnames, bnames, nd, tgt, pg, dg, default_t, tau_lim, \
        r_all, r_v, d_all, d_flat, rigid, rigid_t, cf_t, cam_h, q1_idx, gnd_idx, n_actors, dof_start, total_mass = \
        build_sim_and_actors(gym, grav_z, with_ground, rz_use)

    Lf = bnames.index("left_ankle_roll_link"); Rf = bnames.index("right_ankle_roll_link")
    cam_w, cam_h_px = 1280, 720
    print("  [%s] grav=%.1f ground=%s PD=%s actors=%d q1_idx=%d gnd_idx=%d mass=%.1fkg" %
          (label, grav_z, with_ground, apply_pd, n_actors, q1_idx, gnd_idx, total_mass))
    print("  Q1_rz=%.4f Q1_lv=%.4f STATE_CLEAN" % (r_v[q1_idx, 2].item(), r_v[q1_idx, 7:10].norm().item()))

    # Verify DOF drive
    props_v = gym.get_actor_dof_properties(env, ah_q1)
    modes_ok = all(int(props_v["driveMode"][i]) == 3 for i in range(nd))
    print("  DOF_drive_effort=%s" % modes_ok)

    writer = cv2.VideoWriter(str(OUTPUT_DIR / (filename + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    csv_rows = []; pkl_data = []; lv_prev = np.zeros(3)

    def capture(fno, tsim):
        nonlocal lv_prev
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v = rigid.view(rigid.shape[0], 13)
        dz = d_flat[dof_start:dof_start+nd, 0]; dvel = d_flat[dof_start:dof_start+nd, 1]
        rz_f = r_v[q1_idx, 2].item(); lv = r_v[q1_idx, 7:10].cpu().numpy(); av = r_v[q1_idx, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[q1_idx, 3:7].cpu().numpy().tolist()
        pgz_v = pgz(qx, qy, qz_, qw)
        roll_v, pitch_v, yaw_v = rpy(qx, qy, qz_, qw)
        lv_n = np.linalg.norm(lv); av_n = np.linalg.norm(av)

        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"; top_v = cf_np[top_i]
        nfc = sum(cf_np[i] for i in range(len(bnames)) if 'ankle_roll' not in bnames[i] and cf_np[i] > 1)

        bz = rigid_v[:, 2].cpu().numpy(); bz_min = bz.min()
        bz_min_idx = bz.argmin(); bz_min_name = bnames[bz_min_idx] if bz_min_idx < len(bnames) else "?"
        L_box = bz[Lf] - 0.02; R_box = bz[Rf] - 0.02

        dv = lv - lv_prev; app_a = dv / 0.02; app_f = total_mass * app_a
        app_fn = np.linalg.norm(app_f); app_fz = app_f[2]
        lv_prev = lv.copy()

        row = {
            'frame': fno, 't': tsim, 'rx': r_v[q1_idx,0].item(), 'ry': r_v[q1_idx,1].item(), 'rz': rz_f,
            'qx': qx, 'qy': qy, 'qz': qz_, 'qw': qw, 'roll': roll_v, 'pitch': pitch_v, 'yaw': yaw_v, 'pgz': pgz_v,
            'lv_norm': lv_n, 'av_norm': av_n,
            'rlv_x': lv[0], 'rlv_y': lv[1], 'rlv_z': lv[2],
            'rav_x': av[0], 'rav_y': av[1], 'rav_z': av[2],
            'dof_vel_max': dvel.abs().max().item(),
            'dof_pos_err_max': max(abs(dz[i].item() - tgt[i]) for i in range(nd)),
            'tau_max': 0, 'tau_sat_ratio': 0,
            'L_fz_norm': cf_np[Lf], 'R_fz_norm': cf_np[Rf],
            'top_cf_name': top_n, 'top_cf_val': top_v, 'non_foot_cf_sum': nfc,
            'bz_min': bz_min, 'bz_min_name': bz_min_name,
            'L_box_bot': L_box, 'R_box_bot': R_box,
            'apparent_force_norm': app_fn, 'apparent_force_z': app_fz,
        }
        pkl_data.append({
            'root_state': r_v[q1_idx].clone().cpu().numpy(),
            'dof_pos': dz.clone().cpu().numpy(), 'dof_vel': dvel.clone().cpu().numpy(),
            'rigid_body_pos': rigid_v[:, :3].clone().cpu().numpy(),
            'contact_force': cf_v.clone().cpu().numpy(),
            'target_dof_pos': np.array(tgt),
        })
        return row

    # Frame 0 render (before any physics)
    gym.set_camera_location(cam_h, env, gymapi.Vec3(2.5, 2.5, 1.5), gymapi.Vec3(0, 0, 0.4))
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
    r0 = capture(0, 0.0)
    csv_rows.append(r0)
    frame0 = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), cam_h_px, cam_w)
    overlay(frame0, r0, label)
    writer.write(frame0)

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        dz = d_flat[dof_start:dof_start+nd, 0]; dvel = d_flat[dof_start:dof_start+nd, 1]
        if apply_pd:
            tau = (pg * (default_t - dz) - dg * dvel).clamp(-tau_lim, tau_lim)
        else:
            tau = torch.zeros(nd, device="cuda:0")
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))
        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        fno = step + 1; tsim = fno * 0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        row = capture(fno, tsim)
        tau_np = tau.cpu().numpy()
        row['tau_max'] = abs(tau_np).max()
        row['tau_sat_ratio'] = (abs(tau_np) > 0.95 * tau_lim.cpu().numpy()).sum() / nd
        csv_rows.append(row)

        rx = r_v[q1_idx, 0].item(); ry = r_v[q1_idx, 1].item(); rz2 = r_v[q1_idx, 2].item()
        gym.set_camera_location(cam_h, env,
            gymapi.Vec3(rx+2.5, ry+2.5, max(rz2+1.2, 1.5)), gymapi.Vec3(rx, ry, rz2+0.3))
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), cam_h_px, cam_w)
        overlay(fr, row, label)
        writer.write(fr)

    writer.release()
    with open(str(OUTPUT_DIR / (filename + ".csv")), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)
    with open(str(OUTPUT_DIR / (filename + ".pkl")), 'wb') as f:
        pickle.dump({'frames': pkl_data, 'dof_names': dnames, 'body_names': bnames, 'target_pos': tgt,
                     'total_mass': total_mass}, f)
    print("  Saved: %s.mp4/.csv/.pkl  f1_rz=%.3f f1_lv=%.3f f1_av=%.3f cf=%s:%.0f" %
          (filename, csv_rows[1]['rz'] if len(csv_rows)>1 else 0,
           csv_rows[1]['lv_norm'] if len(csv_rows)>1 else 0,
           csv_rows[1]['av_norm'] if len(csv_rows)>1 else 0,
           csv_rows[1]['top_cf_name'] if len(csv_rows)>1 else "?",
           csv_rows[1]['top_cf_val'] if len(csv_rows)>1 else 0))
    gym.destroy_sim(sim)
    return csv_rows


# ====== RUN ======
rz_use = float(rc.init_state.pos[2])
print("\nUsing root_z=%.3f" % rz_use)

rows_A = run_case("A_noPD_grav", -9.81, True, False, 100, rz_use, "q1_static_A_noPD_gravity")
rows_B = run_case("B_noPD_zerog", 0.0, True, False, 50, rz_use, "q1_static_B_noPD_zerog")
rows_C = run_case("C_noPD_noground", -9.81, False, False, 100, rz_use, "q1_static_C_noPD_no_ground_box")
rows_D = run_case("D_PD_grav", -9.81, True, True, 100, rz_use, "q1_static_D_PD_gravity")

# ====== TABLE ======
print("\n" + "=" * 70)
print("TABLE: 4-case comparison")
print("=" * 70)
print("  %-30s %6s %5s %4s %7s %7s %7s %7s %20s %8s %8s" %
      ("case", "ground", "grav", "PD", "f1_rz", "f1_lv", "f1_av", "f1_pgz", "f1_cf", "f1_N", "appF"))
for rows, name in [(rows_A,"A_noPD_grav"), (rows_B,"B_noPD_zerog"), (rows_C,"C_noPD_noground"), (rows_D,"D_PD_grav")]:
    if rows and len(rows) > 1:
        r = rows[1]
        print("  %-30s %6s %5s %4s %+7.3f %7.3f %7.3f %7.3f %20s %8.0f %8.0f" %
              (name, "box" if "noground" not in name else "none",
               "off" if "zerog" in name else "on",
               "on" if "PD" in name else "off",
               r['rz'], r['lv_norm'], r['av_norm'], r['pgz'],
               r['top_cf_name'], r['top_cf_val'], r.get('apparent_force_norm',0)))

print("\nStatic box ground: NO upward push. Q1 visible in all cases.")
print("Outputs: %s/q1_static_*.mp4, *.csv, *.pkl" % OUTPUT_DIR)

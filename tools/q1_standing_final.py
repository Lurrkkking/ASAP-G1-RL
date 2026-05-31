"""
Q1 Standing Final — correct init flow: only zero velocities after prepare_sim.
No position overwrite. Always explicit zero torque in noPD.
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

def to_bgr(raw, h, w):
    arr = np.asarray(raw)
    if arr.ndim == 1:
        if arr.size == h*w:
            packed = arr.astype(np.uint32).reshape(h,w)
            rgba = np.zeros((h,w,4), dtype=np.uint8)
            rgba[...,0]=(packed>>16)&0xFF; rgba[...,1]=(packed>>8)&0xFF; rgba[...,2]=packed&0xFF; rgba[...,3]=255
            arr = rgba
        else: arr = arr.reshape(h, w, 4)
    elif arr.ndim==2 and arr.shape[1]==w*4: arr = arr.reshape(h, w, 4)
    if arr.ndim==3 and arr.shape[-1]>=3:
        return cv2.cvtColor(arr[...,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.zeros((h,w,3), dtype=np.uint8)


def overlay(img, d, label):
    f = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        label,
        "f=%d t=%.3f" % (d.get('frame',0), d.get('t',0)),
        "rz=%.3f pgz=%.3f" % (d.get('rz',0), d.get('pgz',1)),
        "lv=%.2f av=%.2f" % (d.get('lv_norm',0), d.get('av_norm',0)),
        "tau=%.1f cf=%s:%.0f" % (d.get('tau_max',0), d.get('top_cf_name','NONE'), d.get('top_cf_val',0)),
    ]
    for i, l in enumerate(lines):
        cv2.putText(img, l, (10, 20+i*22), f, 0.5, (0,255,0), 1)


# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot
rz_cfg = float(rc.init_state.pos[2])
urdf_name = "q1_22dof_rl_collision.urdf"
defaults = dict(rc.init_state.default_joint_angles)

print("=" * 70)
print("Q1 STANDING FINAL (correct init flow)")
print("  root_z=%.3f  urdf=%s" % (rz_cfg, urdf_name))
print("=" * 70)


def run_case(label, grav_z, with_ground, apply_pd, n_steps, fname):
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)

    gnd_ah = None
    if with_ground:
        # Use add_ground infinite plane
        pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
        pp.static_friction = 0.8; pp.dynamic_friction = 0.6; pp.restitution = 0.0
        gym.add_ground(sim, pp)

    # Q1
    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao, k, getattr(ac, k))
    ao.default_dof_drive_mode = 3
    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf_name, ao)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dnames)

    ah_q1 = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, rz_cfg)), "q1", -1, 0, 0)

    # ---- STEP 1: set DOF BEFORE prepare_sim ----
    tgt = [float(defaults.get(n, 0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah_q1, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah_q1, dof_st, gymapi.STATE_ALL)

    qp = gym.get_actor_rigid_shape_properties(env, ah_q1)
    for i in range(len(qp)): qp[i].contact_offset = 0.001; qp[i].rest_offset = 0.0; qp[i].restitution = 0.0; qp[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah_q1, qp)

    cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
    cam_h = gym.create_camera_sensor(env, cam)

    # ---- STEP 2: prepare_sim ----
    gym.prepare_sim(sim)

    # ---- STEP 3: acquire tensors ----
    rt = gym.acquire_actor_root_state_tensor(sim)
    r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim)
    d_all = gymtorch.wrap_tensor(dt); d_flat = d_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    n_actors = gym.get_actor_count(env); q1_idx = 0
    for ai in range(n_actors):
        if "q1" in gym.get_actor_name(env, ai).lower(): q1_idx = ai; break

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # ---- Print prepare_sim state (BEFORE any fix) ----
    print("\n  [%s]" % label)
    print("  BEFORE fix (prepare_sim state):")
    print("    root_pos=%.4f,%.4f,%.4f  quat=%.4f,%.4f,%.4f,%.4f" %
          (r_v[q1_idx,0].item(), r_v[q1_idx,1].item(), r_v[q1_idx,2].item(),
           r_v[q1_idx,3].item(), r_v[q1_idx,4].item(), r_v[q1_idx,5].item(), r_v[q1_idx,6].item()))
    print("    root_lv=%.4f,%.4f,%.4f  av=%.4f,%.4f,%.4f" %
          (r_v[q1_idx,7].item(), r_v[q1_idx,8].item(), r_v[q1_idx,9].item(),
           r_v[q1_idx,10].item(), r_v[q1_idx,11].item(), r_v[q1_idx,12].item()))
    print("    knee=[%.4f,%.4f] knee_vel=[%.1f,%.1f] ankleP=[%.4f,%.4f]" %
          (d_flat[3,0].item(), d_flat[9,0].item(), d_flat[3,1].item(), d_flat[9,1].item(),
           d_flat[4,0].item(), d_flat[10,0].item()))
    print("    dof_vel_max=%.1f" % d_flat[:nd, 1].abs().max().item())

    # ---- STEP 4: ONLY zero velocities, keep positions ----
    r_v[q1_idx, 7:13] = 0.0
    d_flat[:nd, 1] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    print("  AFTER fix (vel zeroed only):")
    print("    root_lv=%.4f,%.4f,%.4f  av=%.4f,%.4f,%.4f" %
          (r_v[q1_idx,7].item(), r_v[q1_idx,8].item(), r_v[q1_idx,9].item(),
           r_v[q1_idx,10].item(), r_v[q1_idx,11].item(), r_v[q1_idx,12].item()))
    print("    dof_vel_max=%.6f" % d_flat[:nd, 1].abs().max().item())

    # ---- STATE CLEAN (new definition) ----
    rlv_ok = r_v[q1_idx, 7:10].abs().max().item() < 1e-3
    rav_ok = r_v[q1_idx, 10:13].abs().max().item() < 1e-3
    dv_ok = d_flat[:nd, 1].abs().max().item() < 1e-3
    rz_near = abs(r_v[q1_idx, 2].item() - rz_cfg) < 0.1
    print("  STATE_CLEAN: rlv=%s rav=%s dv=%s rz_near=%s" % (rlv_ok, rav_ok, dv_ok, rz_near))

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
    total_dofs = gym.get_sim_dof_count(sim)
    torque_t = torch.zeros(total_dofs, device="cuda:0")

    # ---- Video ----
    writer = cv2.VideoWriter(str(OUTPUT_DIR / (fname + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    csv_rows = []; lv_prev = np.zeros(3)

    def capture(fno, tsim, wrote_act, tau_np):
        nonlocal lv_prev
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v = rigid.view(rigid.shape[0], 13)
        dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
        rz_f = r_v[q1_idx, 2].item(); lv = r_v[q1_idx, 7:10].cpu().numpy(); av = r_v[q1_idx, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[q1_idx, 3:7].cpu().numpy().tolist()
        pgz_v = pgz(qx, qy, qz_, qw)
        lv_n = np.linalg.norm(lv); av_n = np.linalg.norm(av)
        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"
        top_v = cf_np[top_i]; nfc = sum(cf_np[i] for i in range(len(bnames)) if 'ankle_roll' not in bnames[i] and cf_np[i] > 1)
        bz = rigid_v[:,2].cpu().numpy()
        dv = lv - lv_prev; app_f = total_mass * dv / 0.02; app_fn = np.linalg.norm(app_f)
        lv_prev = lv.copy()
        row = {
            'frame': fno, 't': tsim, 'rz': rz_f, 'pgz': pgz_v,
            'lv_norm': lv_n, 'av_norm': av_n,
            'dof_vel_max': dvel.abs().max().item(),
            'dof_pos_err_max': max(abs(dz[i].item() - tgt[i]) for i in range(nd)),
            'tau_max': abs(tau_np).max() if tau_np is not None else 0,
            'tau_norm': np.linalg.norm(tau_np) if tau_np is not None else 0,
            'top_cf_name': top_n, 'top_cf_val': top_v,
            'non_foot_cf_sum': nfc,
            'apparent_force_norm': app_fn,
            'wrote_act': wrote_act,
        }
        return row

    # Frame 0 render
    gym.set_camera_location(cam_h, env, gymapi.Vec3(2.5, 2.5, 1.5), gymapi.Vec3(0, 0, 0.4))
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
    r0 = capture(0, 0.0, True, np.zeros(nd))
    csv_rows.append(r0)
    fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
    overlay(fr, r0, label)
    writer.write(fr)

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        # ---- ALWAYS write force tensor (even if zero) ----
        torque_t.zero_()
        if apply_pd:
            dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
            tau_v = (pg * (default_t - dz) - dg * dvel).clamp(-tau_lim, tau_lim)
            torque_t[:nd] = tau_v
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))

        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        fno = step + 1; tsim = fno * 0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        row = capture(fno, tsim, True, torque_t[:nd].cpu().numpy())
        csv_rows.append(row)

        if fno < 10:
            print("  f%d: rz=%.3f lv=%.2f av=%.2f pgz=%.3f cf=%s:%.0f appF=%.0f tau=%.1f" %
                  (fno, row['rz'], row['lv_norm'], row['av_norm'], row['pgz'],
                   row['top_cf_name'], row['top_cf_val'], row['apparent_force_norm'],
                   row['tau_max']))

        rx = r_v[q1_idx, 0].item(); ry = r_v[q1_idx, 1].item(); rz2 = r_v[q1_idx, 2].item()
        gym.set_camera_location(cam_h, env,
            gymapi.Vec3(rx+2.5, ry+2.5, max(rz2+1.2, 1.5)), gymapi.Vec3(rx, ry, rz2+0.3))
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
        overlay(fr, row, label)
        writer.write(fr)

    writer.release()
    with open(str(OUTPUT_DIR / (fname + ".csv")), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    f1 = csv_rows[1] if len(csv_rows) > 1 else {}
    flipped = f1.get('lv_norm', 0) > 3.0
    print("  => f1_rz=%.3f lv=%.2f av=%.2f flipped=%s" % (f1.get('rz',0), f1.get('lv_norm',0), f1.get('av_norm',0), flipped))
    gym.destroy_sim(sim)
    return csv_rows


# ====== RUN 4 CASES ======
rows_A = run_case("A_noPD_grav_box", -9.81, True, False, 100, "q1_final_A_noPD_grav_box")
rows_B = run_case("B_noPD_zerog_box", 0.0, True, False, 50, "q1_final_B_noPD_zerog_box")
rows_C = run_case("C_noPD_grav_nognd", -9.81, False, False, 100, "q1_final_C_noPD_grav_nognd")
rows_D = run_case("D_PD_grav_box", -9.81, True, True, 100, "q1_final_D_PD_grav_box")

# Also test add_ground for comparison
rows_A_add = run_case("A_add_ground_noPD_grav", -9.81, True, False, 30, "q1_final_A_add_ground_noPD_grav")

# ====== TABLE ======
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print("  %-25s %7s %7s %7s %7s %6s %20s %8s %8s" %
      ("case", "f1_rz", "f1_lv", "f1_av", "f1_pgz", "flip", "cf_body", "cf_N", "appF"))
for rows, name in [("A_noPD_grav", rows_A), ("B_noPD_zerog", rows_B), ("C_noPD_nognd", rows_C), ("D_PD_grav", rows_D)]:
    if rows and len(rows) > 1:
        r = rows[1]
        print("  %-25s %+7.3f %7.3f %7.3f %7.3f %6s %20s %8.0f %8.0f" %
              (name, r['rz'], r['lv_norm'], r['av_norm'], r['pgz'],
               "YES" if r['lv_norm'] > 3 else "no",
               r['top_cf_name'], r['top_cf_val'], r['apparent_force_norm']))

print("\nOutput: %s/q1_final_*.mp4, *.csv" % OUTPUT_DIR)

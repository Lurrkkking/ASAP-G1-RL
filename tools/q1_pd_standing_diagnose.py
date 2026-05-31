"""
Q1 PD Standing Diagnose — fixed-base sanity, gain sweep, COM, joint limits.
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
print("Q1 PD STANDING DIAGNOSE")
print("  yaml: q1_22dof_rl_collision.yaml  urdf: %s  root_z=%.3f" % (urdf_name, rz_cfg))
print("=" * 70)


GAIN_SETS = {
    "G0_current": {
        "kp": {"hip_pitch":60,"hip_roll":60,"hip_yaw":60,"knee":80,"ankle_pitch":30,"ankle_roll":20,"waist_yaw":50,"waist_roll":50,"shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
        "kd": {"hip_pitch":2.0,"hip_roll":2.0,"hip_yaw":2.0,"knee":3.0,"ankle_pitch":1.0,"ankle_roll":0.8,"waist_yaw":2.0,"waist_roll":2.0,"shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
    },
    "G1_medium": {
        "kp": {"hip_pitch":80,"hip_roll":80,"hip_yaw":80,"knee":120,"ankle_pitch":40,"ankle_roll":30,"waist_yaw":50,"waist_roll":50,"shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
        "kd": {"hip_pitch":3.0,"hip_roll":3.0,"hip_yaw":3.0,"knee":5.0,"ankle_pitch":2.0,"ankle_roll":1.5,"waist_yaw":2.0,"waist_roll":2.0,"shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
    },
    "G2_strong": {
        "kp": {"hip_pitch":100,"hip_roll":100,"hip_yaw":100,"knee":160,"ankle_pitch":50,"ankle_roll":35,"waist_yaw":60,"waist_roll":60,"shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
        "kd": {"hip_pitch":4.0,"hip_roll":4.0,"hip_yaw":4.0,"knee":7.0,"ankle_pitch":3.0,"ankle_roll":2.0,"waist_yaw":2.5,"waist_roll":2.5,"shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
    },
    "G3_very_strong": {
        "kp": {"hip_pitch":120,"hip_roll":120,"hip_yaw":120,"knee":200,"ankle_pitch":60,"ankle_roll":40,"waist_yaw":70,"waist_roll":70,"shoulder_pitch":25,"shoulder_roll":25,"shoulder_yaw":15,"elbow":20},
        "kd": {"hip_pitch":5.0,"hip_roll":5.0,"hip_yaw":5.0,"knee":9.0,"ankle_pitch":4.0,"ankle_roll":2.5,"waist_yaw":3.0,"waist_roll":3.0,"shoulder_pitch":1.0,"shoulder_roll":1.0,"shoulder_yaw":0.5,"elbow":0.8},
    },
}


def build_gains(dnames, gset):
    nd = len(dnames)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dnames):
        for kbase in gset["kp"]:
            if kbase in n:
                pg_np[i] = gset["kp"][kbase]; dg_np[i] = gset["kd"][kbase]; break
    return torch.tensor(pg_np, device="cuda:0"), torch.tensor(dg_np, device="cuda:0")


def run_pd_trial(gname, fix_base, target_mode, n_steps, save_video):
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

    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao, k, getattr(ac, k))
    ao.default_dof_drive_mode = 3
    ao.fix_base_link = fix_base
    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf_name, ao)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dnames)

    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, rz_cfg)), "q1", -1, 0, 0)

    tgt_default = [float(defaults_dict.get(n, 0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt_default[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    qp = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(qp)): qp[i].contact_offset = 0.001; qp[i].rest_offset = 0.0; qp[i].restitution = 0.0; qp[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah, qp)

    total_mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, ah))
    dof_props = gym.get_actor_dof_properties(env, ah)
    effort_limits = np.array([dof_props["effort"][i].item() for i in range(nd)])

    if save_video:
        cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
        cam_h = gym.create_camera_sensor(env, cam)

    gym.prepare_sim(sim)

    rt = gym.acquire_actor_root_state_tensor(sim)
    r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim)
    d_all = gymtorch.wrap_tensor(dt); d_flat = d_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    prepared_dof_pos = d_flat[:nd, 0].clone().cpu().numpy()
    r_v[0, 7:13] = 0.0; d_flat[:nd, 1] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    if target_mode == 'prepared_pose':
        target_dof = prepared_dof_pos.copy()
    else:
        target_dof = np.array(tgt_default)

    default_t = torch.tensor(target_dof, device="cuda:0")
    tau_lim = torch.tensor(effort_limits, device="cuda:0")
    gset = GAIN_SETS[gname]
    pg, dg = build_gains(dnames, gset)
    torque_t = torch.zeros(nd, device="cuda:0")

    Lf = bnames.index("left_ankle_roll_link"); Rf = bnames.index("right_ankle_roll_link")

    # ---- COM / support polygon (init) ----
    rigid_v = rigid.view(rigid.shape[0], 13)
    body_masses = np.array([p.mass for p in gym.get_actor_rigid_body_properties(env, ah)])
    body_positions = rigid_v[:, :3].cpu().numpy()
    com = np.average(body_positions, axis=0, weights=body_masses)
    L_foot_center = rigid_v[Lf, :3].cpu().numpy()
    R_foot_center = rigid_v[Rf, :3].cpu().numpy()
    foot_half = np.array([0.09, 0.04])  # box half-size x,y
    support_x_min = min(L_foot_center[0]-foot_half[0], R_foot_center[0]-foot_half[0])
    support_x_max = max(L_foot_center[0]+foot_half[0], R_foot_center[0]+foot_half[0])
    support_y_min = min(L_foot_center[1]-foot_half[1], R_foot_center[1]-foot_half[1])
    support_y_max = max(L_foot_center[1]+foot_half[1], R_foot_center[1]+foot_half[1])
    com_in_support = (support_x_min <= com[0] <= support_x_max and support_y_min <= com[1] <= support_y_max)
    com_margin_x = min(com[0]-support_x_min, support_x_max-com[0])
    com_margin_y = min(com[1]-support_y_min, support_y_max-com[1])

    # Print init info
    if save_video:
        print("\n  [%s fix_base=%s target=%s]" % (gname, fix_base, target_mode))
        print("  prepared knee=%.4f,%.4f hipP=%.4f,%.4f ankleP=%.4f,%.4f" %
              (prepared_dof_pos[3], prepared_dof_pos[9], prepared_dof_pos[0], prepared_dof_pos[6],
               prepared_dof_pos[4], prepared_dof_pos[10]))
        print("  target   knee=%.4f,%.4f hipP=%.4f,%.4f ankleP=%.4f,%.4f" %
              (target_dof[3], target_dof[9], target_dof[0], target_dof[6], target_dof[4], target_dof[10]))
        print("  COM=[%.3f,%.3f,%.3f] mass=%.1f support_x=[%.3f,%.3f] y=[%.3f,%.3f] COM_in=%s margin=[%.3f,%.3f]" %
              (com[0], com[1], com[2], total_mass, support_x_min, support_x_max,
               support_y_min, support_y_max, com_in_support, com_margin_x, com_margin_y))

    # ---- Video ----
    label = "%s_%s_%s" % (gname, "fix" if fix_base else "free", target_mode[:6])
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(OUTPUT_DIR / ("q1_pd_diag_%s.mp4" % label)),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    csv_rows = []

    def capture(fno, tsim):
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v2 = rigid.view(rigid.shape[0], 13)
        dz = d_flat[:nd, 0]; dvel = d_flat[:nd, 1]
        rz_f = r_v[0, 2].item(); lv = r_v[0, 7:10].cpu().numpy(); av = r_v[0, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[0, 3:7].cpu().numpy().tolist()
        pgz_v = pgz(qx, qy, qz_, qw); roll_v, pitch_v, yaw_v = rpy(qx, qy, qz_, qw)
        tau_np = torque_t.cpu().numpy()
        tau_max = abs(tau_np).max(); tau_sat = (abs(tau_np) > 0.95 * effort_limits).mean()
        sat_joints = [dnames[i] for i in range(nd) if abs(tau_np[i]) > 0.95 * effort_limits[i]]
        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"
        nfc = sum(cf_np[i] for i in range(len(bnames)) if 'ankle_roll' not in bnames[i] and cf_np[i] > 1)
        bz = rigid_v2[:,2].cpu().numpy(); bz_min = bz.min()
        bz_min_idx = bz.argmin(); bz_min_name = bnames[bz_min_idx] if bz_min_idx < len(bnames) else "?"
        dof_err = max(abs(dz[i].item() - target_dof[i]) for i in range(nd))
        dvel_max = dvel.abs().max().item()
        row = {
            'frame': fno, 't': tsim, 'rz': rz_f, 'pgz': pgz_v, 'roll': roll_v, 'pitch': pitch_v, 'yaw': yaw_v,
            'lv_norm': np.linalg.norm(lv), 'av_norm': np.linalg.norm(av),
            'dof_pos_err_max': dof_err, 'dof_vel_max': dvel_max,
            'tau_max': tau_max, 'tau_sat': tau_sat,
            'saturated_joints': ",".join(sat_joints),
            'L_fz': cf_np[Lf], 'R_fz': cf_np[Rf], 'non_foot_cf_sum': nfc,
            'top_cf_name': top_n, 'top_cf_val': cf_np[top_i],
            'bz_min': bz_min, 'bz_min_name': bz_min_name,
        }
        csv_rows.append(row)
        return row

    # Frame 0
    if save_video:
        gym.set_camera_location(cam_h, env, gymapi.Vec3(2.5, 2.5, 1.5), gymapi.Vec3(0, 0, 0.4))
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        r0 = capture(0, 0.0)
        fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
        overlay(fr, r0, label)
        writer.write(fr)

    stand_ok = True; fell_t = None; min_pgz = 1.0; max_av = 0.0; max_tau_v = 0.0; max_sat_v = 0.0
    nf_total = 0; first_sat = None; max_dof_err = 0.0; max_dvel = 0.0
    knee_min = [999, 999]; knee_max = [-999, -999]
    knee_limits_low = [rc.dof_pos_lower_limit_list[3], rc.dof_pos_lower_limit_list[9]]
    knee_limits_high = [rc.dof_pos_upper_limit_list[3], rc.dof_pos_upper_limit_list[9]]

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        torque_t.zero_()
        dz_v = d_flat[:nd, 0]; dvel_v = d_flat[:nd, 1]
        tau_v = (pg * (default_t - dz_v) - dg * dvel_v).clamp(-tau_lim, tau_lim)
        torque_t[:] = tau_v
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))

        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        fno = step + 1; tsim = fno * 0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        row = capture(fno, tsim)

        min_pgz = min(min_pgz, row['pgz']); max_av = max(max_av, row['av_norm'])
        max_tau_v = max(max_tau_v, row['tau_max']); max_sat_v = max(max_sat_v, row['tau_sat'])
        max_dof_err = max(max_dof_err, row['dof_pos_err_max']); max_dvel = max(max_dvel, row['dof_vel_max'])
        nf_total += (1 if row['non_foot_cf_sum'] > 0 else 0)
        if first_sat is None and row['saturated_joints']: first_sat = row['saturated_joints']

        for ki, kidx in enumerate([3, 9]):
            knee_min[ki] = min(knee_min[ki], dz_v[kidx].item())
            knee_max[ki] = max(knee_max[ki], dz_v[kidx].item())

        if stand_ok:
            if row['pgz'] < 0.5: stand_ok = False; fell_t = tsim
            if row['rz'] < 0.15: stand_ok = False; fell_t = tsim if fell_t is None else fell_t

        if save_video and fno % 4 == 0:
            rx = r_v[0, 0].item(); ry = r_v[0, 1].item(); rz2 = r_v[0, 2].item()
            gym.set_camera_location(cam_h, env, gymapi.Vec3(rx+2.5, ry+2.5, max(rz2+1.2, 1.5)), gymapi.Vec3(rx, ry, rz2+0.3))
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
            overlay(fr, row, label)
            writer.write(fr)

    if save_video:
        writer.release()
        with open(str(OUTPUT_DIR / ("q1_pd_diag_%s.csv" % label)), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    knee_over_limit = (knee_min[0] < knee_limits_low[0]*1.01 or knee_min[1] < knee_limits_low[1]*1.01 or
                       knee_max[0] > knee_limits_high[0]*0.99 or knee_max[1] > knee_limits_high[1]*0.99)

    gym.destroy_sim(sim)

    return {
        'gname': gname, 'fix_base': fix_base, 'target': target_mode,
        'stand': stand_ok, 'fell_t': fell_t, 'stand_time': fell_t if fell_t else 5.0,
        'min_pgz': min_pgz, 'max_av': max_av, 'max_tau': max_tau_v, 'max_sat': max_sat_v,
        'max_dof_err': max_dof_err, 'max_dvel': max_dvel,
        'nf_total': nf_total, 'first_sat': first_sat,
        'knee_min': knee_min, 'knee_max': knee_max,
        'knee_limits_low': knee_limits_low, 'knee_limits_high': knee_limits_high,
        'knee_over_limit': knee_over_limit,
        'com': com, 'com_in_support': com_in_support, 'com_margin_x': com_margin_x, 'com_margin_y': com_margin_y,
        'support': (support_x_min, support_x_max, support_y_min, support_y_max),
        'prepared_pos': prepared_dof_pos, 'default_pos': np.array(tgt_default),
        'dof_names': dnames,
    }


# ====== TASK 2: Fixed-base sanity ======
print("\n" + "=" * 70)
print("TASK 2: Fixed-base PD sanity check")
print("=" * 70)

fb_results = []
for gname in ["G0_current", "G1_medium", "G2_strong", "G3_very_strong"]:
    r = run_pd_trial(gname, True, "prepared_pose", 250, gname == "G0_current")
    fb_results.append(r)
    print("  %s: max_err=%.3f max_vel=%.1f max_tau=%.1f sat=%.0f%% oscillate=%s" %
          (gname, r['max_dof_err'], r['max_dvel'], r['max_tau'],
           r['max_sat']*100, r['max_dof_err'] > 0.1))

# ====== TASK 3: Free-base gain sweep ======
print("\n" + "=" * 70)
print("TASK 3: Free-base prepared pose gain sweep")
print("=" * 70)

free_results = []
for gname in ["G0_current", "G1_medium", "G2_strong", "G3_very_strong"]:
    r = run_pd_trial(gname, False, "prepared_pose", 250, True)
    free_results.append(r)
    print("  %s: stand=%s fell_t=%s pgz_min=%.2f tau=%.1f sat=%.0f%% nf=%d first_sat=%s" %
          (gname, r['stand'], r['fell_t'], r['min_pgz'], r['max_tau'],
           r['max_sat']*100, r['nf_total'],
           r['first_sat'][:40] if r['first_sat'] else "none"))

# ====== TASK 6: yaml default vs prepared (if best gains found) ======
print("\n" + "=" * 70)
print("TASK 6: yaml default target with best gains")
print("=" * 70)
best_gname = max(free_results, key=lambda r: r['stand_time'] if not r['stand'] else 999)['gname']
r_def = run_pd_trial(best_gname, False, "yaml_default", 250, True)
print("  %s+yaml: stand=%s fell_t=%s pgz_min=%.2f tau=%.1f" %
      (best_gname, r_def['stand'], r_def['fell_t'], r_def['min_pgz'], r_def['max_tau']))

# ====== TABLES ======
r0 = free_results[0]
print("\n" + "=" * 70)
print("TABLE 1: Fixed-base PD sanity")
print("=" * 70)
print("  %-15s %10s %10s %10s %10s %10s" % ("gains", "max_err", "max_vel", "max_tau", "sat%", "oscillate"))
for r in fb_results:
    print("  %-15s %10.4f %10.1f %10.1f %9.0f%% %10s" %
          (r['gname'], r['max_dof_err'], r['max_dvel'], r['max_tau'], r['max_sat']*100, r['max_dof_err'] > 0.1))

print("\n" + "=" * 70)
print("TABLE 2: Free-base prepared pose gain sweep")
print("=" * 70)
print("  %-15s %8s %10s %10s %10s %10s %10s %10s %30s" %
      ("gains", "stand", "fell_t", "min_pgz", "max_av", "max_tau", "sat%", "nf", "first_sat"))
for r in free_results:
    print("  %-15s %8s %10s %10.3f %10.1f %10.1f %9.0f%% %10d %30s" %
          (r['gname'], r['stand'], r['fell_t'], r['min_pgz'], r['max_av'],
           r['max_tau'], r['max_sat']*100, r['nf_total'],
           r['first_sat'][:30] if r['first_sat'] else "none"))

print("\n" + "=" * 70)
print("TABLE 3: COM / Support Polygon")
print("=" * 70)
print("  COM=[%.3f, %.3f, %.3f]" % (r0['com'][0], r0['com'][1], r0['com'][2]))
sx = r0['support']
print("  Support X=[%.3f, %.3f] Y=[%.3f, %.3f]" % (sx[0], sx[1], sx[2], sx[3]))
print("  COM in support: %s  margin=[%.3f, %.3f]" % (r0['com_in_support'], r0['com_margin_x'], r0['com_margin_y']))

print("\n" + "=" * 70)
print("TABLE 4: Joint Limit Check")
print("=" * 70)
for ki, kidx in enumerate([3, 9]):
    side = "L" if ki == 0 else "R"
    print("  knee_%s: range=[%.3f, %.3f] limits=[%.3f, %.3f] over=%s" %
          (side, r0['knee_min'][ki], r0['knee_max'][ki],
           r0['knee_limits_low'][ki], r0['knee_limits_high'][ki], r0['knee_over_limit']))

print("\n" + "=" * 70)
print("TABLE 5: prepared vs yaml default")
print("=" * 70)
diffs = [(abs(r0['prepared_pos'][i] - r0['default_pos'][i]), r0['dof_names'][i],
          r0['prepared_pos'][i], r0['default_pos'][i]) for i in range(len(r0['dof_names']))]
diffs.sort(reverse=True)
print("  %-35s %10s %10s %10s" % ("joint", "prepared", "yaml", "diff"))
for d, name, prep, yml in diffs[:10]:
    print("  %-35s %+10.4f %+10.4f %+10.4f" % (name, prep, yml, d))

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
fb_stable = all(r['max_dof_err'] < 0.1 for r in fb_results)
print("  1. Fixed-base PD stable: %s" % fb_stable)
best_free = max(free_results, key=lambda r: r['stand_time'])
print("  2. Best free-base: %s stand_time=%.1fs" % (best_free['gname'], best_free['stand_time']))
print("  3. COM in support: %s margin=[%.3f,%.3f]" % (r0['com_in_support'], r0['com_margin_x'], r0['com_margin_y']))
print("  4. Knee over limit: %s" % r0['knee_over_limit'])
print("  5. yaml default vs prepared: max diff=%.4f (%s)" % (diffs[0][0], diffs[0][1]))
print("  6. Self-collision currently: self_collisions=%s (in create_actor)" % rc.asset.self_collisions)

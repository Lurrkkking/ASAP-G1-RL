"""
G1 vs Q1 Standing Benchmark — identical flow, find root difference.
Usage: python3 compare_g1_q1_standing.py [--robot g1|q1] [--case A|B|C|D|E]
Default: runs all cases for both robots.
"""
import sys, os, csv, pickle
from pathlib import Path
import numpy as np, cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs" / "g1_q1_compare"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch

# ====== ROBOT CONFIGS ======
ROBOTS = {
    "g1": {
        "yaml": "humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml",
        "urdf": "g1_29dof_anneal_23dof.urdf",
        "urdf_dir": "g1",
        "root_z_override": None,  # use yaml value
    },
    "q1": {
        "yaml": "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml",
        "urdf": "q1_22dof_rl_collision.urdf",
        "urdf_dir": "q1",
        "root_z_override": None,
    },
}


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
    elif arr.ndim == 2 and arr.shape[1] == w*4: arr = arr.reshape(h, w, 4)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return cv2.cvtColor(arr[...,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.zeros((h,w,3), dtype=np.uint8)

def overlay(img, d, label):
    f = cv2.FONT_HERSHEY_SIMPLEX
    lines = [label,
             "f=%d t=%.2f rz=%.3f pgz=%.3f" % (d.get('frame',0), d.get('t',0), d.get('rz',0), d.get('pgz',1)),
             "av=%.1f tau=%.1f sat=%.0f%%" % (d.get('av_norm',0), d.get('tau_max',0), d.get('tau_sat',0)*100),
             "cf=%s:%.0f" % (d.get('top_cf_name','?'), d.get('top_cf_val',0))]
    for i,l in enumerate(lines):
        cv2.putText(img, l, (10,20+i*22), f, 0.5, (0,255,0), 1)


def run_benchmark(robot_name, case_label, grav_z, with_ground, apply_pd, target_mode, effort_scale, n_steps):
    """Unified benchmark for G1 or Q1."""
    rcfg = ROBOTS[robot_name]
    yaml_path = str(PROJECT_ROOT / rcfg["yaml"])
    conf = OmegaConf.load(yaml_path)
    rc = conf.robot
    root_z = float(rc.init_state.pos[2]) if rcfg["root_z_override"] is None else rcfg["root_z_override"]
    defaults_dict = dict(rc.init_state.default_joint_angles)

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    if hasattr(sp, 'enable_camera_sensors'): sp.enable_camera_sensors = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    if with_ground:
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

    asset_root = str(PROJECT_ROOT / "humanoidverse/data/robots" / rcfg["urdf_dir"])
    asset = gym.load_asset(sim, asset_root, rcfg["urdf"], ao)
    dn = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dn)

    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, root_z)), robot_name, -1, 0, 0)

    tgt_yaml = [float(defaults_dict.get(n, 0)) for n in dn]
    ds = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"] = tgt_yaml[i]; ds[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, ds, gymapi.STATE_ALL)

    qp = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    gym.set_actor_rigid_shape_properties(env, ah, qp)

    cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
    cam_h = gym.create_camera_sensor(env, cam)

    total_mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, ah))
    body_masses = [p.mass for p in gym.get_actor_rigid_body_properties(env, ah)]

    gym.prepare_sim(sim)

    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim == 2 else r_all.view(-1, 13)
    dt = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); df = d_all.view(-1, 2)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = gym.acquire_net_contact_force_tensor(sim)

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # Print asset comparison info (only on first run)
    if case_label == "A":
        rigid_v = rigid.view(rigid.shape[0], 13)
        body_positions = rigid_v[:, :3].cpu().numpy()
        com_init = np.average(body_positions, axis=0, weights=np.array(body_masses))
        print("  [%s] ndof=%d nbodies=%d mass=%.1fkg root_z=%.3f prepared_knee=[%.3f,%.3f] COM=[%.3f,%.3f,%.3f]" %
              (robot_name, nd, len(bnames), total_mass, root_z,
               df[3,0].item() if nd>3 else 0, df[9,0].item() if nd>9 else 0,
               com_init[0], com_init[1], com_init[2]))

    prepared = df[:nd, 0].clone().cpu().numpy()
    r_v[0, 7:13] = 0.0; df[:nd, 1] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    # Target
    if target_mode == 'prepared_pose':
        target = prepared
    else:
        target = np.array(tgt_yaml)

    # Gains
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dn):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(target, device="cuda:0")

    effort_orig = np.array([gym.get_actor_dof_properties(env, ah)["effort"][i].item() for i in range(nd)])
    effort_use = effort_orig * effort_scale
    tau_lim = torch.tensor(effort_use, device="cuda:0")
    torque_t = torch.zeros(nd, device="cuda:0")

    Lf = bnames.index("left_ankle_roll_link") if "left_ankle_roll_link" in bnames else 0
    Rf = bnames.index("right_ankle_roll_link") if "right_ankle_roll_link" in bnames else 1

    label_full = "%s_%s_%s" % (robot_name, case_label, target_mode[:6])
    writer = cv2.VideoWriter(str(OUTPUT_DIR / ("%s.mp4" % label_full)),
                             cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    csv_rows = []

    def capture(fno, tsim):
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1, 3)
        rigid_v2 = rigid.view(rigid.shape[0], 13)
        dz = df[:nd, 0]; dvel = df[:nd, 1]
        rz_f = r_v[0, 2].item(); lv = r_v[0, 7:10].cpu().numpy(); av = r_v[0, 10:13].cpu().numpy()
        qx, qy, qz_, qw = r_v[0, 3:7].cpu().numpy(); pgz_v = pgz(qx, qy, qz_, qw)
        tau_np = torque_t.cpu().numpy()
        tau_cmd_raw = (pg*(default_t-dz)-dg*dvel).cpu().numpy()
        tau_max_c = abs(tau_np).max(); tau_sat = (abs(tau_np) > 0.95*effort_use).mean()
        cf_np = cf_v.norm(dim=1).cpu().numpy()
        top_i = cf_np.argmax(); top_n = bnames[top_i] if top_i < len(bnames) else "?"
        row = {
            'frame': fno, 't': tsim, 'rz': rz_f, 'pgz': pgz_v,
            'lv_norm': np.linalg.norm(lv), 'av_norm': np.linalg.norm(av),
            'dof_vel_max': dvel.abs().max().item(),
            'tau_max': tau_max_c, 'tau_cmd_max': abs(tau_cmd_raw).max(), 'tau_sat': tau_sat,
            'L_fz': cf_np[Lf], 'R_fz': cf_np[Rf],
            'top_cf_name': top_n, 'top_cf_val': cf_np[top_i],
            'kneeL': dz[3].item() if nd > 3 else 0, 'kneeR': dz[9].item() if nd > 9 else 0,
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
    overlay(fr, r0, label_full); writer.write(fr)

    stand_ok = True; fell_t = None; min_pgz = 1.0; max_av_v = 0.0; max_tau_c = 0.0; max_sat_v = 0.0
    nf_tot = 0; knee_min = [99, 99]; knee_hit_n = 0

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        torque_t.zero_()
        if apply_pd:
            tau_v = (pg*(default_t-df[:nd,0]) - dg*df[:nd,1]).clamp(-tau_lim, tau_lim)
        else:
            tau_v = torch.zeros(nd, device="cuda:0")
        torque_t[:] = tau_v
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        fno = step + 1; tsim = fno * 0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        row = capture(fno, tsim)
        min_pgz = min(min_pgz, row['pgz']); max_av_v = max(max_av_v, row['av_norm'])
        max_tau_c = max(max_tau_c, row['tau_max']); max_sat_v = max(max_sat_v, row['tau_sat'])
        if nd > 3: knee_min[0] = min(knee_min[0], df[3,0].item())
        if nd > 9: knee_min[1] = min(knee_min[1], df[9,0].item())
        if nd > 3 and df[3,0].item() < 0.005: knee_hit_n += 1
        if nd > 9 and df[9,0].item() < 0.005: knee_hit_n += 1
        nf_tot += (1 if row['top_cf_name'] not in ['left_ankle_roll_link','right_ankle_roll_link','pelvis'] and row['top_cf_val'] > 5 else 0)
        if stand_ok:
            if row['pgz'] < 0.5: stand_ok = False; fell_t = tsim
            if row['rz'] < 0.15: stand_ok = False; fell_t = tsim if fell_t is None else fell_t

        if fno % 4 == 0:
            rx = r_v[0,0].item(); ry = r_v[0,1].item(); rz2 = r_v[0,2].item()
            gym.set_camera_location(cam_h, env, gymapi.Vec3(rx+2.5,ry+2.5,max(rz2+1.2,1.5)), gymapi.Vec3(rx,ry,rz2+0.3))
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            fr = to_bgr(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), 720, 1280)
            overlay(fr, row, label_full); writer.write(fr)

    writer.release()
    stand_t = fell_t if fell_t else (n_steps*0.02)
    gym.destroy_sim(sim)

    if case_label == "A":
        print("  => %s: stand=%s t=%.1fs pgz_min=%.2f tau_max=%.0f sat=%.0f%%" %
              (label_full, stand_ok, stand_t, min_pgz, max_tau_c, max_sat_v*100))

    return {
        'robot': robot_name, 'case': case_label, 'target': target_mode,
        'stand': stand_ok, 'stand_time': stand_t, 'min_pgz': min_pgz,
        'max_av': max_av_v, 'max_tau': max_tau_c, 'max_sat': max_sat_v,
        'knee_min': knee_min, 'knee_hit': knee_hit_n, 'nf': nf_tot,
        'total_mass': total_mass, 'ndof': nd, 'nbodies': len(bnames),
        'prepared_knee': (prepared[3] if nd > 3 else 0, prepared[9] if nd > 9 else 0),
        'yaml_knee': (tgt_yaml[3] if nd > 3 else 0, tgt_yaml[9] if nd > 9 else 0),
    }


# ====== RUN ALL ======
print("=" * 70)
print("G1 vs Q1 STANDING BENCHMARK")
print("=" * 70)

all_results = []

for robot in ["g1", "q1"]:
    print("\n--- %s ---" % robot.upper())

    # Case A: noPD + gravity + ground
    rA = run_benchmark(robot, "A_noPD", -9.81, True, False, "prepared_pose", 1.0, 100)
    all_results.append(rA)

    # Case B: zero-g + ground
    rB = run_benchmark(robot, "B_zerog", 0.0, True, False, "prepared_pose", 1.0, 50)
    all_results.append(rB)

    # Case C: PD hold prepared pose
    rC = run_benchmark(robot, "C_PD_prepared", -9.81, True, True, "prepared_pose", 1.0, 250)
    all_results.append(rC)

    # Case D: PD hold yaml default
    rD = run_benchmark(robot, "D_PD_yaml", -9.81, True, True, "yaml_default", 1.0, 250)
    all_results.append(rD)

    # Case E: Q1 only — effort x5
    if robot == "q1":
        rE = run_benchmark(robot, "E_PD_eff5x", -9.81, True, True, "prepared_pose", 5.0, 250)
        all_results.append(rE)


# ====== TABLE 1 ======
print("\n" + "=" * 90)
print("TABLE 1: G1 vs Q1 Standing Results")
print("=" * 90)
print("  %-3s %-18s %8s %8s %8s %8s %8s %8s %8s %8s %8s" %
      ("bot", "case", "stand", "t(s)", "min_pgz", "max_av", "max_tau", "sat%", "knee_min", "knee_hit", "nf"))
for r in all_results:
    km = min(r['knee_min']) if r['knee_min'][0] < 99 else 0
    print("  %-3s %-18s %8s %8.1f %8.3f %8.1f %8.0f %7.0f%% %8.2f %8d %8d" %
          (r['robot'], r['case'], r['stand'], r['stand_time'], r['min_pgz'],
           r['max_av'], r['max_tau'], r['max_sat']*100, km, r['knee_hit'], r['nf']))


# ====== TABLE 2: key differences ======
print("\n" + "=" * 90)
print("TABLE 2: Key G1 vs Q1 Differences")
print("=" * 90)

g1_info = next(r for r in all_results if r['robot']=='g1' and r['case']=='A_noPD')
q1_info = next(r for r in all_results if r['robot']=='q1' and r['case']=='A_noPD')

print("  %-30s %15s %15s" % ("metric", "G1", "Q1"))
print("  %-30s %15d %15d" % ("num_dofs", g1_info['ndof'], q1_info['ndof']))
print("  %-30s %15d %15d" % ("num_bodies", g1_info['nbodies'], q1_info['nbodies']))
print("  %-30s %15.1f %15.1f" % ("total_mass_kg", g1_info['total_mass'], q1_info['total_mass']))
print("  %-30s %15.3f %15.3f" % ("prepared_knee_L", g1_info['prepared_knee'][0], q1_info['prepared_knee'][0]))
print("  %-30s %15.3f %15.3f" % ("prepared_knee_R", g1_info['prepared_knee'][1], q1_info['prepared_knee'][1]))
print("  %-30s %15.3f %15.3f" % ("yaml_knee_L", g1_info['yaml_knee'][0], q1_info['yaml_knee'][0]))
print("  %-30s %15.3f %15.3f" % ("yaml_knee_R", g1_info['yaml_knee'][1], q1_info['yaml_knee'][1]))

print("\nDone. Videos: %s/" % OUTPUT_DIR)

"""
Q1 Truth Debug Rollout — video + CSV + PKL, frame-aligned, 4 cases.
"""
import sys, os, hashlib, json, csv, pickle, time
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def quat_to_rpy(qx, qy, qz, qw):
    r = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    p = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(qw*qy-qx*qz)), np.sqrt(1-2*(qw*qy-qx*qz)))
    y = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return np.degrees(r), np.degrees(p), np.degrees(y)


def projected_gz(qx, qy, qz, qw):
    return 1.0 - 2.0*(qx*qx + qy*qy)


def draw_overlay(frame, data, case_label):
    """Draw status overlay on video frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"{case_label}",
        f"frame={data.get('frame',0)}  t={data.get('t',0):.3f}s",
        f"rz={data.get('rz',0):.3f}  pgz={data.get('pgz',1):.3f}",
        f"lv={data.get('lv_norm',0):.2f}  av={data.get('av_norm',0):.2f}",
        f"tau_max={data.get('tau_max',0):.1f}  contact={data.get('top_cf_name','NONE')}:{data.get('top_cf_val',0):.0f}",
        f"body_z_min={data.get('bz_min',0):.3f} ({data.get('bz_min_name','?')})",
    ]
    y0 = 20
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + i * 22), font, 0.5, (0, 255, 0), 1)


def run_rollout(case_name, yaml_path, urdf_name, root_z, gravity_z, add_ground,
                apply_pd, n_steps, video_path, csv_path, pkl_path):
    """Run a rollout, save video+CSV+PKL."""

    # ======== Asset identification ========
    conf = OmegaConf.load(yaml_path)
    rc = conf.robot
    urdf_full = str(PROJECT_ROOT / "humanoidverse/data/robots/q1" / urdf_name)
    yaml_hash = sha256(yaml_path)
    urdf_hash = sha256(urdf_full)

    print(f"\n{'='*70}")
    print(f"CASE: {case_name}")
    print(f"  yaml: {yaml_path}  sha256={yaml_hash}")
    print(f"  urdf: {urdf_full}  sha256={urdf_hash}")
    print(f"  root_z={root_z}  gravity={gravity_z}  ground={add_ground}  PD={apply_pd}")

    # Verify we're using the right URDF
    if "rl_collision" not in urdf_name:
        print("  ERROR: not using rl_collision URDF!")
        return None

    # ======== Build sim ========
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, gravity_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    if add_ground:
        pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
        pp.static_friction = 1.0; pp.dynamic_friction = 1.0; pp.restitution = 0.0
        gym.add_ground(sim, pp)

    ac = rc.asset; opts = gymapi.AssetOptions()
    opts.collapse_fixed_joints = True; opts.replace_cylinder_with_capsule = True
    opts.flip_visual_attachments = False
    opts.fix_base_link = False; opts.density = ac.density
    opts.angular_damping = 0.0; opts.linear_damping = 0.0
    opts.max_angular_velocity = 1000.0; opts.max_linear_velocity = 1000.0
    opts.armature = 0.001; opts.thickness = 0.01
    opts.default_dof_drive_mode = 3

    asset = gym.load_asset(sim, str(PROJECT_ROOT / "humanoidverse/data/robots/q1"), urdf_name, opts)
    dnames = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset)
    nd = len(dnames); nb = len(bnames)

    print(f"  num_dof={nd}  num_bodies={nb}")
    print(f"  dof_names[:5]={dnames[:5]}")
    print(f"  body_names[:5]={bnames[:5]}")

    # ======== Create actor ========
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)
    start_pose = gymapi.Transform(
        p=gymapi.Vec3(0, 0, root_z),
        r=gymapi.Quat(0, 0, 0, 1))
    ah = gym.create_actor(env, asset, start_pose, "q1", -1, 0, 0)

    # Set DOF BEFORE prepare_sim
    defaults = dict(rc.init_state.default_joint_angles)
    tgt = [float(defaults.get(n, 0.0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"] = tgt[i]; dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    # Camera
    cam = gymapi.CameraProperties(); cam.width = 1280; cam.height = 720
    cam_h = gym.create_camera_sensor(env, cam)

    gym.prepare_sim(sim)

    # ======== SET contact_offset IN ROLLOUT ========
    props = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(props)):
        props[i].contact_offset = 0.001
        props[i].rest_offset = 0.0
        props[i].restitution = 0.0
        props[i].friction = 0.8
    gym.set_actor_rigid_shape_properties(env, ah, props)

    # Verify contact_offset applied
    props_v = gym.get_actor_rigid_shape_properties(env, ah)
    co_ok = all(abs(sp.contact_offset - 0.001) < 0.001 for sp in props_v)
    ro_ok = all(abs(sp.rest_offset) < 0.001 for sp in props_v)
    print(f"  contact_offset=0.001: {'OK' if co_ok else 'FAIL'}  rest_offset=0: {'OK' if ro_ok else 'FAIL'}")

    # Print shape details
    for i, sp in enumerate(props_v):
        bname = "?"
        # Try to map shape to body
        print(f"    shape[{i}]: co={sp.contact_offset:.4f} ro={sp.rest_offset:.4f} "
              f"friction={sp.friction:.2f} rest={sp.restitution:.2f}")

    # ======== Tensors + STATE FIX ========
    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt); r_v = r_all.view(-1, 13)
    dt_t = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt_t); d_v = d_all.view(-1, 2)
    cf_t = gym.acquire_net_contact_force_tensor(sim)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim)

    r_v[0, 0:3] = torch.tensor([0.0, 0.0, root_z], device="cuda:0")
    r_v[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
    r_v[0, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    for i in range(nd): d_v[i, 0] = tgt[i]; d_v[i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    # ======== VERIFY CLEAN STATE ========
    cf = gymtorch.wrap_tensor(cf_t).view(-1, 3)
    rigid = gymtorch.wrap_tensor(rigid_t); rigid_v = rigid.view(rigid.shape[0], 13)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

    rz_init = r_v[0, 2].item()
    rlv_max = r_v[0, 7:10].abs().max().item()
    rav_max = r_v[0, 10:13].abs().max().item()
    dv_max = d_v[:, 1].abs().max().item()
    dp_err_max = max(abs(d_v[i, 0].item() - tgt[i]) for i in range(nd))
    qx, qy, qz_, qw = r_v[0, 3].item(), r_v[0, 4].item(), r_v[0, 5].item(), r_v[0, 6].item()
    pgz_init = projected_gz(qx, qy, qz_, qw)
    bz = rigid_v[:, 2].cpu().numpy()
    Lf_idx = bnames.index("left_ankle_roll_link"); Rf_idx = bnames.index("right_ankle_roll_link")
    L_box = bz[Lf_idx] - 0.02; R_box = bz[Rf_idx] - 0.02

    clean = rlv_max < 1e-6 and rav_max < 1e-6 and dv_max < 1e-6 and dp_err_max < 1e-6
    print(f"  STATE CLEAN: {clean}")
    print(f"    rz={rz_init:.4f} pgz={pgz_init:.4f} rlv_max={rlv_max:.6f} rav_max={rav_max:.6f}")
    print(f"    dv_max={dv_max:.6f} dp_err_max={dp_err_max:.6f}")
    print(f"    body_z=[{bz.min():.4f},{bz.max():.4f}] foot_box_bot=[{L_box:.4f},{R_box:.4f}]")

    if not clean:
        print("  ERROR: STATE NOT CLEAN!")
        return None

    # ======== PD gains ========
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(nd, np.float32); dg_np = np.zeros(nd, np.float32)
    for i, n in enumerate(dnames):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(tgt, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah)["effort"][i].item()
                            for i in range(nd)], device="cuda:0")

    # ======== Video writer ========
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 50, (1280, 720))

    # ======== Data storage ========
    csv_rows = []
    pkl_frames = []

    def capture_frame(frame_no, t_sim):
        nonlocal cf_t
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        dp = d_v[:, 0]; dv_ = d_v[:, 1]
        tau_current = (pg * (default_t - dp) - dg * dv_).clamp(-tau_lim, tau_lim) if apply_pd else torch.zeros(nd, device="cuda:0")

        rz = r_v[0, 2].item()
        qx_, qy_, qz__, qw_ = r_v[0, 3].item(), r_v[0, 4].item(), r_v[0, 5].item(), r_v[0, 6].item()
        pgz = projected_gz(qx_, qy_, qz__, qw_)
        r, p, y = quat_to_rpy(qx_, qy_, qz__, qw_)
        lv = r_v[0, 7:10].norm().item(); av = r_v[0, 10:13].norm().item()
        tau_np = tau_current.cpu().numpy(); tau_max = abs(tau_np).max()
        tau_sat = (abs(tau_np) > 0.95 * tau_lim.cpu().numpy()).sum() / nd

        bz_ = rigid_v[:, 2].cpu().numpy()
        bz_min_idx = bz_.argmin(); bz_min = bz_.min(); bz_min_name = bnames[bz_min_idx]

        cf_np = cf.norm(dim=1).cpu().numpy()
        L_fz_norm = cf_np[Lf_idx]; R_fz_norm = cf_np[Rf_idx]
        top_i = cf_np.argmax(); top_cf_name = bnames[top_i]; top_cf_val = cf_np[top_i]

        # Non-foot contact sum
        non_foot_sum = sum(cf_np[i] for i in range(len(bnames))
                           if 'ankle_roll' not in bnames[i] and cf_np[i] > 1.0)

        L_box_bot = bz_[Lf_idx] - 0.02; R_box_bot = bz_[Rf_idx] - 0.02

        row = {
            'frame': frame_no, 't': t_sim,
            'rx': r_v[0,0].item(), 'ry': r_v[0,1].item(), 'rz': rz,
            'qx': qx_, 'qy': qy_, 'qz': qz__, 'qw': qw_,
            'rlv_x': r_v[0,7].item(), 'rlv_y': r_v[0,8].item(), 'rlv_z': r_v[0,9].item(),
            'rav_x': r_v[0,10].item(), 'rav_y': r_v[0,11].item(), 'rav_z': r_v[0,12].item(),
            'roll': r, 'pitch': p, 'yaw': y, 'pgz': pgz,
            'lv_norm': lv, 'av_norm': av,
            'dof_vel_max': dv_.abs().max().item(),
            'dof_pos_err_max': max(abs(dp[i].item() - tgt[i]) for i in range(nd)),
            'tau_max': tau_max, 'tau_sat_ratio': tau_sat,
            'L_fz_norm': L_fz_norm, 'R_fz_norm': R_fz_norm,
            'top_cf_name': top_cf_name, 'top_cf_val': top_cf_val,
            'non_foot_cf_sum': non_foot_sum,
            'bz_min': bz_min, 'bz_min_name': bz_min_name,
            'L_box_bot': L_box_bot, 'R_box_bot': R_box_bot,
        }
        csv_rows.append(row)

        pkl_frames.append({
            'root_state': r_v[0].clone().cpu().numpy(),
            'dof_pos': dp.clone().cpu().numpy(),
            'dof_vel': dv_.clone().cpu().numpy(),
            'rigid_body_pos': rigid_v[:, :3].clone().cpu().numpy(),
            'contact_force': cf.clone().cpu().numpy(),
            'torque': tau_np.copy(),
            'target_dof_pos': np.array(tgt),
        })

        return row, tau_current

    # ======== Render frame 0 (before any simulate) ========
    gym.set_camera_location(cam_h, env, gymapi.Vec3(2.5, 2.5, 1.5), gymapi.Vec3(0, 0, 0.4))
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    img_raw = gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR)

    def write_frame_from_raw(img_raw, overlay_data, case_label_text):
        arr = np.array(img_raw); h, w = 720, 1280
        if arr.ndim == 1:
            if arr.size == h*w: packed=arr.astype(np.uint32).reshape(h,w)
            else: arr = arr.reshape(h, w, 4)
            rgba = np.zeros((h,w,4), dtype=np.uint8)
            if arr.ndim == 2:
                rgba[...,0]=(packed>>16)&0xFF; rgba[...,1]=(packed>>8)&0xFF; rgba[...,2]=packed&0xFF; rgba[...,3]=255
            else:
                rgba = arr
            arr = rgba
        elif arr.ndim == 2 and arr.shape[1] == w*4: arr = arr.reshape(h, w, 4)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            bgr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            draw_overlay(bgr, overlay_data, case_label_text)
            writer.write(bgr)

    pre_row, _ = capture_frame(0, 0.0)
    write_frame_from_raw(img_raw, pre_row, case_name)

    # ======== Sim loop ========
    print(f"  Simulating {n_steps} steps ({n_steps*0.02:.1f}s)...")

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)

        dp_v = d_v[:, 0]; dv_v = d_v[:, 1]
        if apply_pd:
            tau = (pg * (default_t - dp_v) - dg * dv_v).clamp(-tau_lim, tau_lim)
        else:
            tau = torch.zeros(nd, device="cuda:0")
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)

        frame_no = step + 1
        t_sim = frame_no * 0.02

        row, _ = capture_frame(frame_no, t_sim)

        # Render
        gym.refresh_actor_root_state_tensor(sim)
        rx = r_v[0, 0].item(); ry = r_v[0, 1].item(); rz2 = r_v[0, 2].item()
        gym.set_camera_location(cam_h, env,
            gymapi.Vec3(rx + 2.5, ry + 2.5, max(rz2 + 1.2, 1.5)),
            gymapi.Vec3(rx, ry, rz2 + 0.3))
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        write_frame_from_raw(gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR), row, case_name)

    writer.release()

    # ======== Save CSV ========
    with open(csv_path, 'w', newline='') as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader(); w.writerows(csv_rows)

    # ======== Save PKL ========
    with open(pkl_path, 'wb') as f:
        pickle.dump({'frames': pkl_frames, 'dof_names': dnames, 'body_names': bnames,
                     'target_pos': tgt, 'stiffness': pg_np, 'damping': dg_np}, f)

    print(f"  Saved: {video_path}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {pkl_path}")

    gym.destroy_sim(sim)

    # Return first 6 frames for comparison table
    return {
        'case': case_name,
        'frames_0_5': [{k: row[k] for k in ['frame','t','rz','pgz','lv_norm','av_norm',
                          'tau_max','top_cf_name','top_cf_val','bz_min']}
                       for row in csv_rows[:6]],
        'yaml_hash': yaml_hash, 'urdf_hash': urdf_hash,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    yaml_path = str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml")
    urdf_name = "q1_22dof_rl_collision.urdf"
    root_z = float(OmegaConf.load(yaml_path).robot.init_state.pos[2])

    print("=" * 70)
    print("Q1 TRUTH ROLLOUT DEBUG")
    print(f"  yaml: {yaml_path}")
    print(f"  urdf: {urdf_name}")
    print(f"  root_z: {root_z}")
    print("=" * 70)

    results = {}

    # Case A: noPD + gravity + ground
    rA = run_rollout(
        "A_noPD_g_ground", yaml_path, urdf_name, root_z,
        -9.81, True, False, 100,
        OUTPUT_DIR / "q1_truth_A_noPD.mp4",
        OUTPUT_DIR / "q1_truth_A_noPD.csv",
        OUTPUT_DIR / "q1_truth_A_noPD.pkl")
    results['A'] = rA

    # Case B: noPD + gravity + NO ground
    rB = run_rollout(
        "B_noPD_g_noground", yaml_path, urdf_name, root_z,
        -9.81, False, False, 100,
        OUTPUT_DIR / "q1_truth_B_noground.mp4",
        OUTPUT_DIR / "q1_truth_B_noground.csv",
        OUTPUT_DIR / "q1_truth_B_noground.pkl")
    results['B'] = rB

    # Case C: noPD + zero gravity + ground
    rC = run_rollout(
        "C_noPD_zerog_ground", yaml_path, urdf_name, root_z,
        0.0, True, False, 100,
        OUTPUT_DIR / "q1_truth_C_zerog.mp4",
        OUTPUT_DIR / "q1_truth_C_zerog.csv",
        OUTPUT_DIR / "q1_truth_C_zerog.pkl")
    results['C'] = rC

    # Case D: PD + gravity + ground
    rD = run_rollout(
        "D_PD_g_ground", yaml_path, urdf_name, root_z,
        -9.81, True, True, 100,
        OUTPUT_DIR / "q1_truth_D_PD.mp4",
        OUTPUT_DIR / "q1_truth_D_PD.csv",
        OUTPUT_DIR / "q1_truth_D_PD.pkl")
    results['D'] = rD

    # ======== Comparison Tables ========
    print("\n" + "=" * 70)
    print("TABLE 1: Case Comparison (Frame 1 data)")
    print("=" * 70)
    print(f"  {'case':>6s} {'ground':>7s} {'grav':>6s} {'PD':>5s} {'rz_f1':>7s} {'lv_f1':>7s} "
          f"{'av_f1':>7s} {'pgz_f1':>7s} {'top_cf':>25s} {'cf_val':>8s}")
    for cid, r in results.items():
        if r and len(r['frames_0_5']) > 1:
            f1 = r['frames_0_5'][1]
            print(f"  {cid:>6s} {'yes' if 'noground' not in r['case'] else 'no':>7s} "
                  f"{'on' if 'zerog' not in r['case'] else 'off':>6s} "
                  f"{'on' if 'PD' in r['case'] else 'off':>5s} "
                  f"{f1['rz']:+7.3f} {f1['lv_norm']:+7.3f} {f1['av_norm']:+7.3f} "
                  f"{f1['pgz']:+7.3f} {f1['top_cf_name']:>25s} {f1['top_cf_val']:+8.0f}")

    print(f"\nTABLE 2: Frames 0-5 for Case A (noPD + ground)")
    print(f"  {'f':>3s} {'rz':>7s} {'lv':>7s} {'av':>7s} {'pgz':>7s} {'top_cf':>25s} {'cf':>8s} {'tau':>7s} {'bz_min':>7s}")
    if results['A']:
        for row in results['A']['frames_0_5'][:6]:
            print(f"  {row['frame']:3d} {row['rz']:+7.3f} {row['lv_norm']:+7.3f} {row['av_norm']:+7.3f} "
                  f"{row['pgz']:+7.3f} {row['top_cf_name']:>25s} {row['top_cf_val']:+8.0f} "
                  f"{row['tau_max']:+7.1f} {row['bz_min']:+7.3f}")

    print(f"\nTABLE 2b: Frames 0-5 for Case B (noPD NO ground)")
    if results['B']:
        for row in results['B']['frames_0_5'][:6]:
            print(f"  {row['frame']:3d} {row['rz']:+7.3f} {row['lv_norm']:+7.3f} {row['av_norm']:+7.3f} "
                  f"{row['pgz']:+7.3f} {row['top_cf_name']:>25s} {row['top_cf_val']:+8.0f} "
                  f"{row['tau_max']:+7.1f} {row['bz_min']:+7.3f}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Videos: q1_truth_A_noPD.mp4, q1_truth_B_noground.mp4, q1_truth_C_zerog.mp4, q1_truth_D_PD.mp4")


if __name__ == "__main__":
    main()

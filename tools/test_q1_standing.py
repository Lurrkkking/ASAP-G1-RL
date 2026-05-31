"""
Q1 Standing Test v5 — clean init: DOF before prepare_sim, then fix ALL state after.
"""
import sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def quat_to_rpy(qx, qy, qz, qw):
    sinr = 2.0*(qw*qx + qy*qz); cosr = 1.0 - 2.0*(qx*qx + qy*qy)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0*(qw*qy - qz*qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny = 2.0*(qw*qz + qx*qy); cosy = 1.0 - 2.0*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def projected_gz(qx, qy, qz, qw):
    return 1.0 - 2.0 * (qx*qx + qy*qy)


def main():
    conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
    rc = conf.robot

    root_z = float(rc.init_state.pos[2])
    print("=" * 80)
    print(f"Q1 Standing Test v5 (clean init after prepare_sim)")
    print(f"  root_z={root_z}  density={rc.asset.density}  action_scale={rc.control.action_scale}")
    print("=" * 80)

    # ---- setup ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    gym.add_ground(sim, gymapi.PlaneParams())

    ac = rc.asset
    opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = ac.default_dof_drive_mode
    asset = gym.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof.urdf", opts)
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)
    ndof = len(dof_names)

    # ---- create actor ----
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    start_pose = gymapi.Transform(
        p=gymapi.Vec3(float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), root_z),
        r=gymapi.Quat(float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
                      float(rc.init_state.rot[2]), float(rc.init_state.rot[3])))
    ah = gym.create_actor(env, asset, start_pose, "q1", -1, 0, 0)

    # ---- STEP 1: set DOF BEFORE prepare_sim (per-actor API) ----
    defaults = dict(rc.init_state.default_joint_angles)
    target_pos = [float(defaults.get(n, 0.0)) for n in dof_names]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(ndof):
        dof_st[i]["pos"] = target_pos[i]
        dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    # ---- STEP 2: prepare_sim ----
    gym.prepare_sim(sim)

    # ---- STEP 3: acquire all tensors ----
    rt = gym.acquire_actor_root_state_tensor(sim)
    root_all = gymtorch.wrap_tensor(rt)
    root_v = root_all.view(-1, 13)  # (num_actors, 13)

    dt_t = gym.acquire_dof_state_tensor(sim)
    dof_all = gymtorch.wrap_tensor(dt_t)
    dof_v = dof_all.view(-1, 2)     # (num_actors*ndof, 2) or (ndof, 2)

    rigid_t = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_t)
    num_rigid = rigid.shape[0]
    rigid_v = rigid.view(num_rigid, 13)

    cf_t = gym.acquire_net_contact_force_tensor(sim)
    cf = gymtorch.wrap_tensor(cf_t).view(-1, 3)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    # ---- STEP 4: print dirty state before fix ----
    robot_idx = 0  # single actor
    print(f"\n  BEFORE fix (dirty from prepare_sim):")
    print(f"    root_z={root_v[robot_idx,2].item():.4f}  "
          f"lv={root_v[robot_idx,7:10].cpu().numpy()}  "
          f"av={root_v[robot_idx,10:13].cpu().numpy()}")
    dof_vel_before = dof_v[:, 1].abs().max().item()
    dof_pos_err = max(abs(dof_v[i, 0].item() - target_pos[i]) for i in range(ndof))
    print(f"    max|dof_vel|={dof_vel_before:.1f}  max|dof_pos_err|={dof_pos_err:.4f}")

    # ---- STEP 5: FIX root state ----
    root_v[robot_idx, 0:3] = torch.tensor(
        [float(rc.init_state.pos[0]), float(rc.init_state.pos[1]), root_z],
        device="cuda:0")
    root_v[robot_idx, 3:7] = torch.tensor(
        [float(rc.init_state.rot[0]), float(rc.init_state.rot[1]),
         float(rc.init_state.rot[2]), float(rc.init_state.rot[3])],
        device="cuda:0")
    root_v[robot_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_all))

    # ---- STEP 6: FIX dof state ----
    for i in range(ndof):
        dof_v[i, 0] = target_pos[i]
        dof_v[i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_all))

    # ---- STEP 7: refresh and verify ----
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    root_z_actual = root_v[robot_idx, 2].item()
    root_quat = root_v[robot_idx, 3:7].cpu().numpy()
    root_lv = root_v[robot_idx, 7:10].cpu().numpy()
    root_av = root_v[robot_idx, 10:13].cpu().numpy()
    dof_vel_after = dof_v[:, 1].abs().max().item()
    dof_pos_err_after = max(abs(dof_v[i, 0].item() - target_pos[i]) for i in range(ndof))

    print(f"\n  AFTER fix (should be clean):")
    print(f"    root_z={root_z_actual:.4f}  quat={root_quat}  "
          f"lv={root_lv}  av={root_av}")
    print(f"    max|dof_vel|={dof_vel_after:.6f}  max|dof_pos_err|={dof_pos_err_after:.6f}")

    # ---- ASSERTIONS ----
    assert abs(root_z_actual - root_z) < 0.01, f"root_z wrong: {root_z_actual} vs {root_z}"
    assert np.abs(root_lv).max() < 1e-3, f"root_lin_vel not zero: {root_lv}"
    assert np.abs(root_av).max() < 1e-3, f"root_ang_vel not zero: {root_av}"
    assert dof_vel_after < 1e-3, f"dof_vel not zero: max={dof_vel_after}"
    print(f"\n  ASSERTIONS PASSED: clean initial state confirmed")

    # ---- Build PD gains ----
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, dtype=np.float32); dg_np = np.zeros(ndof, dtype=np.float32)
    for i, n in enumerate(dof_names):
        for k in stiff:
            if k in n: pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor(target_pos, device="cuda:0")
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah)["effort"][i].item()
                            for i in range(ndof)], device="cuda:0")

    # ---- Body indices ----
    Lf = body_names.index("left_ankle_roll_link")
    Rf = body_names.index("right_ankle_roll_link")
    print(f"\n  foot indices: L={Lf} R={Rf}")
    print(f"  init foot_z: L={rigid_v[Lf,2].item():.4f}  R={rigid_v[Rf,2].item():.4f}")

    # ---- Simulate with per-frame output for first 10 frames ----
    print(f"\n{'='*120}")
    print(f"FIRST 10 FRAMES")
    print(f"{'frame':>5s}  {'rz':>8s}  {'pgz':>8s}  {'lv_n':>7s}  {'av_n':>7s}  "
          f"{'|tau|':>7s}  {'|dv|':>7s}  {'L_fz':>8s}  {'R_fz':>8s}  "
          f"{'kneeL':>7s}  {'ankL':>7s}  {'root_quat':>30s}")
    print(f"{'-'*120}")

    fell_t = None
    for step in range(5000):  # max 25s
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if step % 4 == 0:
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)

            dp = dof_v[:, 0]; dv = dof_v[:, 1]
            tau = pg * (default_t - dp) - dg * dv
            tau = torch.clamp(tau, -tau_lim, tau_lim)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

            frame_no = step // 4
            t_sim = frame_no * 0.02

            if frame_no < 10:
                rz = root_v[robot_idx, 2].item()
                quat = root_v[robot_idx, 3:7].cpu().numpy()
                lv_n = root_v[robot_idx, 7:10].norm().item()
                av_n = root_v[robot_idx, 10:13].norm().item()
                pgz = projected_gz(quat[0], quat[1], quat[2], quat[3])
                tau_np = tau.cpu().numpy()
                dv_np = dv.cpu().numpy()
                Lfz = cf[Lf, 2].item(); Rfz = cf[Rf, 2].item()
                r, p, y = quat_to_rpy(quat[0], quat[1], quat[2], quat[3])
                print(f"{frame_no:5d}  {rz:+8.3f}  {pgz:+8.3f}  {lv_n:+7.3f}  {av_n:+7.3f}  "
                      f"{abs(tau_np).max():+7.1f}  {abs(dv_np).max():+7.1f}  {Lfz:+8.0f}  {Rfz:+8.0f}  "
                      f"{dp[3].item():+7.3f}  {dp[4].item():+7.3f}  "
                      f"[{quat[0]:+.3f} {quat[1]:+.3f} {quat[2]:+.3f} {quat[3]:+.3f}] rpy=[{r:+.0f} {p:+.0f} {y:+.0f}]")

            if pgz < 0.3 and fell_t is None:
                fell_t = t_sim

            if fell_t and frame_no > 25:
                break  # stop shortly after fall

    # ---- Final summary ----
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    final_rz = root_v[robot_idx, 2].item()
    final_quat = root_v[robot_idx, 3:7].cpu().numpy()
    final_pgz = projected_gz(final_quat[0], final_quat[1], final_quat[2], final_quat[3])

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Frame 0 root_z:           {root_z} (set) → {root_z_actual:.4f} (actual)")

    # Check frame 1
    # We need to track this during the loop
    print(f"  Clean init:               ASSERTIONS PASSED")
    print(f"  Fall time:                {fell_t}")
    print(f"  Final root_z:             {final_rz:.3f}")
    print(f"  Final pgz:                {final_pgz:.3f}")
    print(f"  Robot shot to 0.8m:       {'YES' if root_z_actual > 0.6 else 'NO'}")
    print(f"  Root velocity clean:      {'YES' if np.abs(root_lv).max() < 1e-3 else 'NO'}")
    print(f"  DOF velocity clean:       {'YES' if dof_vel_after < 1e-3 else 'NO'}")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

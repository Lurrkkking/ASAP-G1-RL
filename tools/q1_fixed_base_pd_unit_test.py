"""
Q1 Fixed-base PD Unit Test — sign, damping, joint limit, mapping.
"""
import sys, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def pgz(qx, qy, qz, qw): return 1.0 - 2.0*(qx*qx + qy*qy)

def setup_fixed_base(gym, grav_z=0.0):
    conf = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
    rc = conf.robot; rz = float(rc.init_state.pos[2])
    defaults = dict(rc.init_state.default_joint_angles)

    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
    # No ground — no contact interference for fixed-base tests
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode = 3
    ao.fix_base_link = True

    asset = gym.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),
                           "q1_22dof_rl_collision.urdf", ao)
    dnames = gym.get_asset_dof_names(asset); nd = len(dnames)

    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)

    tgt = [float(defaults.get(n,0)) for n in dnames]
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): dof_st[i]["pos"]=tgt[i]; dof_st[i]["vel"]=0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    gym.prepare_sim(sim)

    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt); df = d_all.view(-1,2)

    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    prepared = df[:nd,0].clone().cpu().numpy()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    effort_limits = np.array([gym.get_actor_dof_properties(env,ah)["effort"][i].item() for i in range(nd)])
    tau_lim = torch.tensor(effort_limits, device="cuda:0")

    # Build G0 gains
    stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dnames):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg=torch.tensor(pg_np,device="cuda:0"); dg=torch.tensor(dg_np,device="cuda:0")

    torque_t = torch.zeros(nd, device="cuda:0")

    return sim, env, ah, dnames, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, effort_limits


# ====== TASK 2: Zero-error static test ======
print("=" * 70)
print("TASK 2: Zero-error static test (gravity off, target=prepared)")
print("=" * 70)

for grav, label in [(0.0, "grav_off"), (-9.81, "grav_on")]:
    g0 = gymapi.acquire_gym()
    sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, _ = setup_fixed_base(g0, grav)
    default_t = torch.tensor(prepared, device="cuda:0")

    max_err = 0; max_vel = 0; max_tau = 0; knee_min = [99,99]; knee_max = [-99,-99]
    for step in range(100):  # 2s
        g0.refresh_dof_state_tensor(sim)
        dz=df[:nd,0]; dvel=df[:nd,1]
        tau_v = (pg*(default_t-dz) - dg*dvel).clamp(-tau_lim,tau_lim)
        torque_t[:] = tau_v
        g0.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): g0.simulate(sim)
        g0.fetch_results(sim, True)
        g0.refresh_dof_state_tensor(sim)
        err = (dz-default_t).abs().max().item(); vel = dvel.abs().max().item()
        tau_m = abs(tau_v.cpu().numpy()).max()
        max_err = max(max_err, err); max_vel = max(max_vel, vel); max_tau = max(max_tau, tau_m)
        for ki, kidx in enumerate([3,9]):
            knee_min[ki] = min(knee_min[ki], dz[kidx].item())
            knee_max[ki] = max(knee_max[ki], dz[kidx].item())
        if step == 0:
            print("  %s step0: err_max=%.6f vel_max=%.6f tau_max=%.6f knee=[%.4f,%.4f]" %
                  (label, err, vel, tau_m, dz[3].item(), dz[9].item()))

    print("  %s: err=%.6f vel=%.4f tau=%.4f knee_range=[%.3f,%.3f]/[%.3f,%.3f] %s" %
          (label, max_err, max_vel, max_tau, knee_min[0], knee_max[0], knee_min[1], knee_max[1],
           "OSCILLATES" if max_err > 0.001 else "STABLE"))
    g0.destroy_sim(sim)

# ====== TASK 3: Single-joint sign test ======
print("\n" + "=" * 70)
print("TASK 3: Single-joint perturbation sign test")
print("=" * 70)

test_joints = [("left_knee_joint", 3), ("right_knee_joint", 9),
               ("left_hip_pitch_joint", 0), ("right_hip_pitch_joint", 6),
               ("left_ankle_pitch_joint", 4), ("right_ankle_pitch_joint", 10)]

sign_results = []
for jname, jidx in test_joints:
    for delta, dlabel in [(+0.05, "+0.05"), (-0.05, "-0.05")]:
        g1 = gymapi.acquire_gym()
        sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, _ = setup_fixed_base(g1, 0.0)
        target = prepared.copy()
        target[jidx] += delta
        default_t = torch.tensor(target, device="cuda:0")

        initial_err = df[jidx,0].item() - target[jidx]
        final_pos = df[jidx,0].item()

        for step in range(50):  # 1s
            g1.refresh_dof_state_tensor(sim)
            dz=df[:nd,0]; dvel=df[:nd,1]
            tau_v = (pg*(default_t-dz) - dg*dvel).clamp(-tau_lim,tau_lim)
            torque_t[:] = tau_v
            g1.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
            for _ in range(4): g1.simulate(sim)
            g1.fetch_results(sim, True)
            g1.refresh_dof_state_tensor(sim)
            final_pos = dz[jidx].item()

        err_start = prepared[jidx] - target[jidx]
        err_end = final_pos - target[jidx]
        moved_toward = abs(err_end) < abs(err_start)
        pass_fail = "PASS" if moved_toward else "FAIL"
        sign_results.append((jname, dlabel, prepared[jidx], target[jidx], final_pos, err_start, err_end, moved_toward, pass_fail))
        print("  %-25s %s: start=%.4f tgt=%.4f final=%.4f err0=%.4f errF=%.4f %s" %
              (jname, dlabel, prepared[jidx], target[jidx], final_pos, err_start, err_end, pass_fail))
        g1.destroy_sim(sim)

# ====== TASK 4: Damping sign test ======
print("\n" + "=" * 70)
print("TASK 4: Damping sign test (inject +1 rad/s at knee)")
print("=" * 70)

for formula_label, dsign in [("D1: -kd*vel", -1.0), ("D2: +kd*vel", +1.0)]:
    g2 = gymapi.acquire_gym()
    sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, _ = setup_fixed_base(g2, 0.0)
    default_t = torch.tensor(prepared, device="cuda:0")

    # Inject velocity at left knee
    df[3,1] = 1.0  # +1 rad/s
    g2.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    g2.refresh_dof_state_tensor(sim)
    vel_start = df[3,1].item()

    for step in range(20):  # 0.4s
        g2.refresh_dof_state_tensor(sim)
        dz=df[:nd,0]; dvel=df[:nd,1]
        tau_v = pg*(default_t-dz) + dsign * dg*dvel  # test both signs
        tau_v = tau_v.clamp(-tau_lim,tau_lim)
        torque_t[:] = tau_v
        g2.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): g2.simulate(sim)
        g2.fetch_results(sim, True)
        g2.refresh_dof_state_tensor(sim)

    vel_end = df[3,1].item()
    decayed = abs(vel_end) < abs(vel_start)
    print("  %s: vel %.4f -> %.4f decayed=%s %s" % (formula_label, vel_start, vel_end, decayed, "PASS" if decayed else "FAIL"))
    g2.destroy_sim(sim)

# ====== TASK 5: Target consistency ======
print("\n" + "=" * 70)
print("TASK 5: Target consistency check")
print("=" * 70)

g3 = gymapi.acquire_gym()
sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, _ = setup_fixed_base(g3, 0.0)
max_diff = max(abs(df[i,0].item() - prepared[i]) for i in range(nd))
print("  max|initial_pos - prepared_target| = %.6f" % max_diff)
for name, idx in [("left_knee_joint",3),("right_knee_joint",9),
                   ("left_hip_pitch_joint",0),("left_ankle_pitch_joint",4)]:
    print("  %s: prepared=%.6f target=%.6f init_pos=%.6f err=%.6f" %
          (name, prepared[idx], prepared[idx], df[idx,0].item(), df[idx,0].item()-prepared[idx]))
g3.destroy_sim(sim)

# ====== TASK 6: Knee limit sweep ======
print("\n" + "=" * 70)
print("TASK 6: Knee joint limit behavior (slow target sweep)")
print("=" * 70)

for knee_idx, kname in [(3, "left_knee"), (9, "right_knee")]:
    g4 = gymapi.acquire_gym()
    sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, effort_limits = setup_fixed_base(g4, 0.0)
    limit_low = float(0.0)  # Q1 knee lower limit
    limit_high = float(2.4435)

    for target_val in [prepared[knee_idx] - 0.1, prepared[knee_idx] - 0.2, limit_low + 0.05]:
        target = prepared.copy()
        target[knee_idx] = target_val
        default_t = torch.tensor(target, device="cuda:0")

        hit_limit = False; max_vel_hit = 0.0
        for step in range(50):
            g4.refresh_dof_state_tensor(sim)
            dz=df[:nd,0]; dvel=df[:nd,1]
            tau_v = (pg*(default_t-dz) - dg*dvel).clamp(-tau_lim,tau_lim)
            torque_t[:] = tau_v
            g4.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
            for _ in range(4): g4.simulate(sim)
            g4.fetch_results(sim, True)
            g4.refresh_dof_state_tensor(sim)
            if dz[knee_idx].item() < limit_low + 0.001:
                hit_limit = True
                max_vel_hit = max(max_vel_hit, abs(dvel[knee_idx].item()))

        final_knee = df[knee_idx,0].item()
        print("  %s target=%.3f final=%.3f hit_limit=%s max_vel=%.2f" %
              (kname, target_val, final_knee, hit_limit, max_vel_hit))
    g4.destroy_sim(sim)

# ====== TASK 7: Damping sweep ======
print("\n" + "=" * 70)
print("TASK 7: Damping sweep (gravity on, fixed base)")
print("=" * 70)

for kd_scale, label in [(1.0, "kd×1"), (2.0, "kd×2"), (4.0, "kd×4"), (8.0, "kd×8")]:
    g5 = gymapi.acquire_gym()
    sim, env, ah, dn, nd, prepared, tgt, pg, dg, tau_lim, df, d_all, r_v, r_all, torque_t, effort_limits = setup_fixed_base(g5, -9.81)
    dg_scaled = dg * kd_scale
    default_t = torch.tensor(prepared, device="cuda:0")

    max_err, max_vel, max_tau, knee_min, knee_max = 0,0,0,[99,99],[-99,-99]
    hit_limit = False
    for step in range(100):
        g5.refresh_dof_state_tensor(sim)
        dz=df[:nd,0]; dvel=df[:nd,1]
        tau_v = (pg*(default_t-dz) - dg_scaled*dvel).clamp(-tau_lim,tau_lim)
        torque_t[:] = tau_v
        g5.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): g5.simulate(sim)
        g5.fetch_results(sim, True)
        g5.refresh_dof_state_tensor(sim)
        max_err = max(max_err, (dz-default_t).abs().max().item())
        max_vel = max(max_vel, dvel.abs().max().item())
        max_tau = max(max_tau, abs(tau_v.cpu().numpy()).max())
        for ki,kidx in enumerate([3,9]):
            knee_min[ki] = min(knee_min[ki], dz[kidx].item())
            knee_max[ki] = max(knee_max[ki], dz[kidx].item())
            if dz[kidx].item() < 0.005: hit_limit = True

    print("  %s: err=%.3f vel=%.2f tau=%.1f knee=[%.3f,%.3f]/[%.3f,%.3f] hit_limit=%s" %
          (label, max_err, max_vel, max_tau, knee_min[0], knee_max[0], knee_min[1], knee_max[1], hit_limit))
    g5.destroy_sim(sim)

# ====== SUMMARY TABLES ======
print("\n" + "=" * 70)
print("TABLE 1: Zero-error static")
print("=" * 70)
# already printed above

print("\n" + "=" * 70)
print("TABLE 2: Single-joint sign test")
print("=" * 70)
print("  %-25s %6s %10s %10s %10s %10s %6s" % ("joint", "delta", "prepared", "target", "final", "err_end", "PASS?"))
fails = 0
for (jname, dlabel, prep, tgt2, final, err0, errF, moved, pf) in sign_results:
    if not moved: fails += 1
    print("  %-25s %6s %10.4f %10.4f %10.4f %10.4f %6s" % (jname, dlabel, prep, tgt2, final, errF, pf))

print("\n  Total sign test failures: %d/%d" % (fails, len(sign_results)))

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("  1. PD position sign: %s" % ("PASS" if fails == 0 else "FAIL (%d failures)" % fails))
print("  2. Damping sign: check TABLE 3 above (D1 should decay)")
print("  3. Torque mapping: %s" % ("PASS" if fails == 0 else "CHECK"))
print("  4. Target consistency: max|target-initial|=%.6f %s" % (max_diff, "PASS" if max_diff < 1e-6 else "CHECK"))
print("  5. Fixed-base oscillates with gravity off: check TASK 2 grav_off output")
print("  6. If grav_off stable but grav_on unstable: damping/torque insufficient")
print("  7. Damping sweep: check TABLE above for kd×1 vs kd×8 comparison")

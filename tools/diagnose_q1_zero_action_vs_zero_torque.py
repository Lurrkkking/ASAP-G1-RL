"""
Q1 Zero-Action vs Zero-Torque Isolation — find root rotation source.
"""
import sys, csv, numpy as np
from pathlib import Path
PROJECT_ROOT = Path("/root/autodl-tmp/ASAP")
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from isaacgym import gymapi, gymtorch
import torch
from omegaconf import OmegaConf

# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot; rz = float(rc.init_state.pos[2])
defaults_dict = dict(rc.init_state.default_joint_angles)

def build_and_reset(ground=True):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    s = g.create_sim(0,0,gymapi.SIM_PHYSX,sp)
    if ground:
        pp=gymapi.PlaneParams(); pp.normal=gymapi.Vec3(0,0,1); pp.static_friction=0.8; pp.dynamic_friction=0.6
        g.add_ground(s, pp)
    e = g.create_env(s, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

    ac=rc.asset; ao=gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode=3
    asset=g.load_asset(s, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof_rl_collision.urdf", ao)
    dn=g.get_asset_dof_names(asset); bn=g.get_actor_rigid_body_names(e) if False else None
    nd=len(dn)
    ah=g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)
    # Set DOF BEFORE prepare_sim
    tgt=[float(defaults_dict.get(n,0)) for n in dn]
    ds=g.get_actor_dof_states(e, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"]=tgt[i]; ds[i]["vel"]=0.0
    g.set_actor_dof_states(e, ah, ds, gymapi.STATE_ALL)
    # Shape props
    qp=g.get_actor_rigid_shape_properties(e, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    g.set_actor_rigid_shape_properties(e, ah, qp)
    # prepare_sim
    g.prepare_sim(s)
    # Acquire
    rt=g.acquire_actor_root_state_tensor(s); r_all=gymtorch.wrap_tensor(rt)
    r_v=r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt=g.acquire_dof_state_tensor(s); d_all=gymtorch.wrap_tensor(dt); df=d_all.view(-1,2)
    rigid_t=g.acquire_rigid_body_state_tensor(s); rigid=gymtorch.wrap_tensor(rigid_t)
    cf_t=g.acquire_net_contact_force_tensor(s); cf=gymtorch.wrap_tensor(cf_t).view(-1,3)
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)
    # Only zero velocities, keep positions
    prepared=df[:nd,0].clone()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    g.set_actor_root_state_tensor(s, gymtorch.unwrap_tensor(r_all))
    g.set_dof_state_tensor(s, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

    # PD gains
    stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dn):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg=torch.tensor(pg_np,device="cuda:0",dtype=torch.float32)
    dg=torch.tensor(dg_np,device="cuda:0",dtype=torch.float32)
    eff=np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])
    tau_lim=torch.tensor(eff,device="cuda:0",dtype=torch.float32)
    bn=g.get_actor_rigid_body_names(e, ah)
    body_masses=np.array([p.mass for p in g.get_actor_rigid_body_properties(e,ah)])

    return g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim, rigid_t, cf_t, body_masses, r_all, d_all


def quat_to_yaw(qx,qy,qz,qw):
    return np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

def compute_angular_momentum(rigid_v, body_masses):
    """Approximate total angular momentum around z-axis."""
    com = np.average(rigid_v[:,:3], axis=0, weights=body_masses)
    Lz = 0.0
    for i in range(len(body_masses)):
        r = rigid_v[i,:2] - com[:2]
        v = rigid_v[i,7:9]
        Lz += body_masses[i] * (r[0]*v[1] - r[1]*v[0])
    return Lz


# ====== RUN 6 CASES ======
print("=" * 70)
print("ZERO-ACTION vs ZERO-TORQUE ISOLATION")
print("=" * 70)

cases = {
    "A_zero_action_gnd":   ("action", True),
    "B_zero_action_nognd": ("action", False),
    "C_zero_torque_gnd":   ("torque", True),
    "D_zero_torque_nognd": ("torque", False),
    "E_hold_current_gnd":  ("hold", True),
    "F_hold_current_nognd":("hold", False),
}

all_results = []
for label, (mode, ground) in cases.items():
    g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim, rigid_t, cf_t, body_masses, r_all, d_all = build_and_reset(ground=ground)
    Lf=bn.index("left_ankle_roll_link"); Rf=bn.index("right_ankle_roll_link")

    # Capture prepared pose for 'hold' mode
    current_target = prepared.clone()

    # Verify init velocities
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

    rlv0 = r_v[0,7:10].norm().item()
    rav0 = r_v[0,10:13].norm().item()
    dv0 = df[:nd,1].abs().max().item()

    last_yaw = None; yaw_cum = 0; yaw_rates = []; lz_vals = []
    tau_norms = []; action_norms = []

    print("\n%s: mode=%s ground=%s  init: rlv=%.6f rav=%.6f dv=%.6f knee=[%.3f,%.3f]" %
          (label, mode, ground, rlv0, rav0, dv0, df[3,0].item(), df[9,0].item()))

    for step in range(300):  # 6 seconds
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

        dp_v=df[:nd,0]; dv_v=df[:nd,1]

        if mode == "action":
            # Standard PD: target = default_dof_pos (= yaml default, knee=0.30 if not overridden)
            target_pos = torch.tensor(tgt, device="cuda:0", dtype=torch.float32)
            torque = (pg*(target_pos-dp_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        elif mode == "torque":
            # ZERO torque directly
            torque = torch.zeros(nd, device="cuda:0")
            target_pos = dp_v.clone()  # unused
        else:  # "hold"
            # Hold current position
            target_pos = current_target
            torque = (pg*(target_pos-dp_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)

        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)

        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

        # Yaw
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy()
        yaw=quat_to_yaw(qx,qy,qz_,qw)
        if last_yaw is not None:
            dy=yaw-last_yaw
            if dy>np.pi: dy-=2*np.pi
            elif dy<-np.pi: dy+=2*np.pi
            yaw_cum+=dy; yaw_rates.append(dy/0.02)
        last_yaw=yaw

        # Angular momentum
        rigid_v=gymtorch.wrap_tensor(rigid_t).view(-1,13).cpu().numpy()
        lz_vals.append(compute_angular_momentum(rigid_v, body_masses))

        tau_norms.append(torque.norm().item())
        action_norms.append(0 if mode in ("torque","hold") else 1)

        if step == 0:
            print("  f0: tau_norm=%.4f target_knee=[%.3f,%.3f] curr=[%.3f,%.3f]" %
                  (torque.norm().item(),
                   target_pos[3].item(), target_pos[9].item(),
                   dp_v[3].item(), dp_v[9].item()))

    yaw_deg = np.degrees(yaw_cum)
    yaw_rate = np.degrees(np.mean(np.abs(yaw_rates))) if yaw_rates else 0
    lz0 = lz_vals[0]; lz_end = lz_vals[-1]
    tau_mean = np.mean(tau_norms) if tau_norms else 0

    all_results.append({
        "label":label, "mode":mode, "ground":ground,
        "rlv0":rlv0, "rav0":rav0, "dv0":dv0,
        "yaw_deg":yaw_deg, "yaw_rate":yaw_rate,
        "lz0":lz0, "lz_end":lz_end, "lz_delta":lz_end-lz0,
        "tau_mean":tau_mean,
        "knee_prepared":(prepared[3].item(), prepared[9].item()),
    })
    print("  => yaw=%.1f deg  rate=%.1f deg/s  tau_mean=%.4f  Lz: %.4f->%.4f" %
          (yaw_deg, yaw_rate, tau_mean, lz0, lz_end))
    g.destroy_sim(s)


# ====== TABLE ======
print("\n" + "=" * 90)
print("TABLE: Zero-Action vs Zero-Torque")
print("=" * 90)
print("  %-22s %7s %7s %9s %9s %9s %8s %8s %8s %8s %8s" %
      ("case","mode","ground","rlv0","rav0","dv0","yaw_deg","yaw_rate","Lz_delta","tau_mean","knee_prep"))
for r in all_results:
    print("  %-22s %7s %7s %9.6f %9.6f %9.6f %+8.1f %8.1f %+8.4f %8.4f [%.2f,%.2f]" %
          (r["label"],r["mode"],"yes" if r["ground"] else "no",
           r["rlv0"],r["rav0"],r["dv0"],r["yaw_deg"],r["yaw_rate"],
           r["lz_delta"],r["tau_mean"],r["knee_prepared"][0],r["knee_prepared"][1]))

print("\nCONCLUSIONS:")
# Check key comparisons
a = next(r for r in all_results if r["label"]=="A_zero_action_gnd")
c = next(r for r in all_results if r["label"]=="C_zero_torque_gnd")
d = next(r for r in all_results if r["label"]=="D_zero_torque_nognd")
e = next(r for r in all_results if r["label"]=="E_hold_current_gnd")

print("  A (zero action+gnd) yaw=%.1f vs C (zero torque+gnd) yaw=%.1f: %s" %
      (a["yaw_deg"], c["yaw_deg"],
       "action PD target IS the problem" if abs(a["yaw_deg"])>2*abs(c["yaw_deg"]) else "similar"))
print("  D (zero torque+no gnd) yaw=%.1f: %s" %
      (d["yaw_deg"], "STILL ROTATES — phantom force or state injection" if abs(d["yaw_deg"])>5 else "near zero"))
print("  E (hold current) yaw=%.1f: %s" %
      (e["yaw_deg"], "target=current works" if abs(e["yaw_deg"])<5 else "still rotates"))

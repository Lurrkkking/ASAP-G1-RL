"""
Q1 Actuator Softening — find PD params that don't induce yaw.
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

conf = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot; rz = float(rc.init_state.pos[2])
defaults_dict = dict(rc.init_state.default_joint_angles)
orig_stiff = dict(rc.control.stiffness); orig_damp = dict(rc.control.damping)

def quat_to_yaw(qx,qy,qz,qw):
    return np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

def build_and_reset(kp_scale=1.0, kd_scale=1.0, action_scale_mult=1.0):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    s = g.create_sim(0,0,gymapi.SIM_PHYSX,sp)
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
    dn=g.get_asset_dof_names(asset); nd=len(dn); bn=g.get_actor_rigid_body_names(e) if False else None
    ah=g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)
    tgt=[float(defaults_dict.get(n,0)) for n in dn]
    ds=g.get_actor_dof_states(e, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"]=tgt[i]; ds[i]["vel"]=0.0
    g.set_actor_dof_states(e, ah, ds, gymapi.STATE_ALL)
    qp=g.get_actor_rigid_shape_properties(e, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    g.set_actor_rigid_shape_properties(e, ah, qp)
    g.prepare_sim(s)
    rt=g.acquire_actor_root_state_tensor(s); r_all=gymtorch.wrap_tensor(rt)
    r_v=r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt=g.acquire_dof_state_tensor(s); d_all=gymtorch.wrap_tensor(dt); df=d_all.view(-1,2)
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    prepared=df[:nd,0].clone()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    g.set_actor_root_state_tensor(s, gymtorch.unwrap_tensor(r_all))
    g.set_dof_state_tensor(s, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    # Custom PD gains
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dn):
        for k in orig_stiff:
            if k in n: pg_np[i]=orig_stiff[k]*kp_scale; dg_np[i]=orig_damp[k]*kd_scale; break
    pg=torch.tensor(pg_np,device="cuda:0",dtype=torch.float32)
    dg=torch.tensor(dg_np,device="cuda:0",dtype=torch.float32)
    eff=np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])
    tau_lim=torch.tensor(eff,device="cuda:0",dtype=torch.float32)
    bn=g.get_actor_rigid_body_names(e, ah)
    return g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim


# ====== TASK 1: Target mode comparison ======
print("=" * 70)
print("TASK 1: Target mode comparison (knee yaml=0.30, prepared=0.42)")
print("=" * 70)

for mode_label, mode in [("G_target_every_frame","every_frame"), ("A_zero_action","fixed_yaml"), ("E_hold_prepared","fixed_prepared")]:
    g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim = build_and_reset()
    yaw_cum=0; last_yaw=None; tau_mean=0; tau_max=0

    if mode == "fixed_yaml":
        target = torch.tensor(tgt, device="cuda:0", dtype=torch.float32)
    elif mode == "fixed_prepared":
        target = prepared.clone()
    else:
        target = None  # set per-frame

    for step in range(200):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        dp_v=df[:nd,0]; dv_v=df[:nd,1]
        if mode == "every_frame":
            target = dp_v.clone()
        torque = (pg*(target-dp_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        g.refresh_actor_root_state_tensor(s)
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy(); yaw=quat_to_yaw(qx,qy,qz_,qw)
        if last_yaw is not None:
            dy=yaw-last_yaw
            if dy>np.pi: dy-=2*np.pi
            elif dy<-np.pi: dy+=2*np.pi
            yaw_cum+=dy
        last_yaw=yaw
        tau_np=torque.cpu().numpy(); tau_mean+=abs(tau_np).max(); tau_max=max(tau_max,abs(tau_np).max())

    tau_mean/=200
    yaw_deg=np.degrees(yaw_cum)
    print("  %-25s yaw=%+7.1f deg  tau_mean=%7.1f tau_max=%7.1f" % (mode_label, yaw_deg, tau_mean, tau_max))
    g.destroy_sim(s)


# ====== TASK 2: Actuator softening sweep ======
print("\n" + "=" * 70)
print("TASK 2: Actuator Softening Sweep (zero action)")
print("=" * 70)

actuator_sets = {
    "A0_current": (1.0, 1.0),
    "A1_half": (0.5, 0.5),
    "A2_quarter": (0.25, 0.5),
    "A3_knee10_hip5": (0.1, 0.2),
}

for label, (kp_s, kd_s) in actuator_sets.items():
    g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim = build_and_reset(kp_scale=kp_s, kd_scale=kd_s)
    yaw_cum=0; last_yaw=None; tau_mean=0

    for step in range(200):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        df_v=df[:nd,0]; dv_v=df[:nd,1]
        target=torch.tensor(tgt,device="cuda:0",dtype=torch.float32)
        torque=(pg*(target-df_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        g.refresh_actor_root_state_tensor(s)
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy(); yaw=quat_to_yaw(qx,qy,qz_,qw)
        if last_yaw is not None:
            dy=yaw-last_yaw
            if dy>np.pi: dy-=2*np.pi
            elif dy<-np.pi: dy+=2*np.pi
            yaw_cum+=dy
        last_yaw=yaw
        tau_mean+=abs(torque.cpu().numpy()).max()
    tau_mean/=200
    yaw_deg=np.degrees(yaw_cum)
    print("  %-15s kpx%.1f/kdx%.1f: yaw=%+7.1f deg  tau_mean=%7.1f" % (label, kp_s, kd_s, yaw_deg, tau_mean))
    g.destroy_sim(s)


# ====== TASK 3: Action mask ablation ======
print("\n" + "=" * 70)
print("TASK 3: Action Mask Ablation (yaw/roll joints zeroed)")
print("=" * 70)

# yaw/roll indices: waist_yaw(13), hip_yaw_L(2), hip_yaw_R(8), hip_roll_L(1), hip_roll_R(7), ankle_roll_L(5), ankle_roll_R(11)
mask_sets = {
    "M0_full": [],
    "M1_zero_waist_yaw": [13],
    "M2_zero_hip_yaw": [2,8],
    "M3_zero_yaw_all": [2,8,13],
    "M4_zero_yaw_roll": [2,8,13,1,7,5,11],
}

for label, mask in mask_sets.items():
    g, s, e, ah, dn, bn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim = build_and_reset()
    yaw_cum=0; last_yaw=None; tau_mean=0

    for step in range(200):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        df_v=df[:nd,0]; dv_v=df[:nd,1]
        target=torch.tensor(tgt,device="cuda:0",dtype=torch.float32)
        torque_full=(pg*(target-df_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        # Zero torque on masked joints
        for j in mask:
            torque_full[j]=0.0
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque_full.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        g.refresh_actor_root_state_tensor(s)
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy(); yaw=quat_to_yaw(qx,qy,qz_,qw)
        if last_yaw is not None:
            dy=yaw-last_yaw; dy=np.clip(dy,-np.pi,np.pi) if abs(dy)>np.pi else dy
            yaw_cum+=dy
        last_yaw=yaw
        tau_mean+=abs(torque_full.cpu().numpy()).max()
    tau_mean/=200
    print("  %-20s yaw=%+7.1f deg  tau_mean=%7.1f" % (label, np.degrees(yaw_cum), tau_mean))
    g.destroy_sim(s)


# ====== SUMMARY TABLE ======
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("  Task 1: 'target every frame' should give tau≈0, minimal yaw")
print("  Task 2: Lower kp/kd should reduce tau and yaw")
print("  Task 3: Masking yaw/roll joints shows which channels cause rotation")
print("  Recommended: use the softest actuator that doesn't explode for PPO smoke")

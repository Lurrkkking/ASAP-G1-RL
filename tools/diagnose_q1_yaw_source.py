"""
Q1 Yaw Source Diagnosis — where does rotation come from?
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

# ====== Load policy if available ======
ckpt_path = sorted((PROJECT_ROOT / "logs/TEST_Q1_loco").glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)[-1] if list((PROJECT_ROOT / "logs/TEST_Q1_loco").glob("*/model_*.pt")) else None
ckpt = None; actor = None
if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["actor_model_state_dict"]
    actor_in = sd["actor_module.module.0.weight"].shape[1]
    actor = torch.nn.Sequential(
        torch.nn.Linear(actor_in,512), torch.nn.ELU(),
        torch.nn.Linear(512,256), torch.nn.ELU(),
        torch.nn.Linear(256,128), torch.nn.ELU(),
        torch.nn.Linear(128,22),
    )
    actor.load_state_dict({k.replace("actor_module.module.",""):v for k,v in sd.items() if "std" not in k})
    actor.to("cuda:0"); actor.eval()
    print("Loaded policy from %s" % ckpt_path.name)
else:
    print("No checkpoint found at %s" % ckpt_path)

# ====== Config ======
cfg = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = cfg.robot; rz = float(rc.init_state.pos[2])
defaults_dict = dict(rc.init_state.default_joint_angles)

yaw_joints = [2,8,13,1,7,5,11]  # hip_yaw_LR, waist_yaw, hip_roll_LR, ankle_roll_LR

def build_sim(ground=True, friction=0.8):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    s = g.create_sim(0,0,gymapi.SIM_PHYSX,sp)
    if ground:
        pp=gymapi.PlaneParams(); pp.normal=gymapi.Vec3(0,0,1)
        pp.static_friction=friction; pp.dynamic_friction=friction
        g.add_ground(s, pp)
    e = g.create_env(s, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    ac=rc.asset; ao=gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode=3
    asset=g.load_asset(s, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof_rl_collision.urdf", ao)
    dn=g.get_asset_dof_names(asset); nd=len(dn)
    ah=g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)
    tgt=[float(defaults_dict.get(n,0)) for n in dn]
    ds=g.get_actor_dof_states(e, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"]=tgt[i]; ds[i]["vel"]=0.0
    g.set_actor_dof_states(e, ah, ds, gymapi.STATE_ALL)
    qp=g.get_actor_rigid_shape_properties(e, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=friction
    g.set_actor_rigid_shape_properties(e, ah, qp)
    g.prepare_sim(s)
    rt=g.acquire_actor_root_state_tensor(s); r_all=gymtorch.wrap_tensor(rt)
    r_v=r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt=g.acquire_dof_state_tensor(s); d_all=gymtorch.wrap_tensor(dt); df=d_all.view(-1,2)

    # Init state: use prepared pose (set DOF before prepare_sim → snap after)
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    prepared=df[:nd,0].clone().cpu().numpy()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    g.set_actor_root_state_tensor(s, gymtorch.unwrap_tensor(r_all))
    g.set_dof_state_tensor(s, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)

    eff=np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])
    stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dn):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg=torch.tensor(pg_np,device="cuda:0",dtype=torch.float32)
    dg=torch.tensor(dg_np,device="cuda:0",dtype=torch.float32)
    tau_lim=torch.tensor(eff,device="cuda:0",dtype=torch.float32)
    return g, s, e, ah, dn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim


def run_case(label, n_frames, action_source, ground, friction, zero_yaw_joints):
    g, s, e, ah, dn, nd, tgt, prepared, r_v, df, pg, dg, tau_lim = build_sim(ground=ground, friction=friction)

    bn = g.get_actor_rigid_body_names(e, ah)
    cf_t = g.acquire_net_contact_force_tensor(s); rigid_t = g.acquire_rigid_body_state_tensor(s)
    Lf=bn.index("left_ankle_roll_link"); Rf=bn.index("right_ankle_roll_link")
    body_masses = np.array([p.mass for p in g.get_actor_rigid_body_properties(e,ah)])

    last_action=np.zeros(nd); last_yaw=None; yaw_cum=0
    yaw_rates=[]; mz_all=[]; slip_L=[]; slip_R=[]; action_means=[]; tau_means=[]

    for step in range(n_frames):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s); g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)
        rs=r_v[0].cpu().numpy(); dp_v=df[:nd,0]; dv_v=df[:nd,1]
        rigid = gymtorch.wrap_tensor(rigid_t).view(-1,13)
        cf = gymtorch.wrap_tensor(cf_t).view(-1,3)

        # Get action
        if action_source == "zero":
            mean_a = torch.zeros(nd, device="cuda:0")
        else:
            qx,qy,qz_,qw=rs[3:7]
            pg_vec=np.array([2*(qx*qz_-qw*qy),2*(qy*qz_+qw*qx),1-2*(qx*qx+qy*qy)])
            obs_vec=np.concatenate([rs[7:10]*2.0, rs[10:13]*0.25, pg_vec, [0,0,0], dp_v.cpu().numpy(), dv_v.cpu().numpy(), last_action])
            with torch.no_grad():
                mean_a = actor(torch.tensor(obs_vec,dtype=torch.float32,device="cuda:0").unsqueeze(0)).squeeze(0)

        # Zero yaw-related joints if requested
        if zero_yaw_joints:
            for j in yaw_joints:
                mean_a[j] = 0.0

        # PD control
        action_scaled = mean_a * 0.25
        target_pos = action_scaled.float() + torch.tensor(tgt,device="cuda:0",dtype=torch.float32)
        torque = (pg*(target_pos-dp_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        last_action = mean_a.cpu().numpy().copy()
        g.refresh_actor_root_state_tensor(s); g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

        # ---- Compute yaw ----
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy()
        yaw = np.arctan2(2*(qw*qz_+qx*qy), 1-2*(qy*qy+qz_*qz_))
        if last_yaw is not None:
            dy = yaw - last_yaw
            if dy > np.pi: dy -= 2*np.pi
            elif dy < -np.pi: dy += 2*np.pi
            yaw_cum += dy; yaw_rates.append(dy/0.02)

        # ---- Contact yaw moment ----
        cf_v = gymtorch.wrap_tensor(cf_t).view(-1,3).cpu().numpy()
        rigid_v = gymtorch.wrap_tensor(rigid_t).view(-1,13).cpu().numpy()
        com = np.average(rigid_v[:,:3], axis=0, weights=body_masses)
        r_L = rigid_v[Lf,:2] - com[:2]; r_R = rigid_v[Rf,:2] - com[:2]
        FL = cf_v[Lf]; FR = cf_v[Rf]
        Mz = (r_L[0]*FL[1] - r_L[1]*FL[0]) + (r_R[0]*FR[1] - r_R[1]*FR[0])
        mz_all.append(Mz)

        # ---- Foot slip (rigid body velocity of ankle link) ----
        if np.linalg.norm(FL) > 10:
            slip_L.append(np.linalg.norm(rigid_v[Lf,7:9]))
        if np.linalg.norm(FR) > 10:
            slip_R.append(np.linalg.norm(rigid_v[Rf,7:9]))

        action_means.append(mean_a.cpu().numpy().copy())
        tau_means.append(torque.cpu().numpy().copy())
        last_yaw = yaw

    result = {
        "label": label,
        "yaw_cum_deg": np.degrees(yaw_cum),
        "yaw_rate_mean": np.mean(np.abs(yaw_rates)) if yaw_rates else 0,
        "yaw_rate_deg_s": np.degrees(np.mean(yaw_rates)) if yaw_rates else 0,
        "Mz_mean": np.mean(mz_all), "Mz_std": np.std(mz_all),
        "slip_L_mean": np.mean(slip_L) if slip_L else 0,
        "slip_R_mean": np.mean(slip_R) if slip_R else 0,
        "action_means": np.mean(action_means, axis=0) if action_means else np.zeros(nd),
        "tau_means": np.mean(tau_means, axis=0) if tau_means else np.zeros(nd),
    }
    g.destroy_sim(s)
    return result, dn


# ====== RUN CASES ======
print("=" * 70)
print("YAW SOURCE DIAGNOSIS")
print("=" * 70)

results = []
for label, src, gnd, fric, zero_yaw in [
    ("A_zero_gnd", "zero", True, 0.8, False),
    ("B_policy_gnd", "policy", True, 0.8, False),
    ("C_policy_no_yaw", "policy", True, 0.8, True),
    ("D_policy_no_gnd", "policy", False, 0.8, False),
    ("E_policy_low_fric", "policy", True, 0.01, False),
]:
    r, dn = run_case(label, 300, src, gnd, fric, zero_yaw)
    results.append(r)
    print("\n%s:" % label)
    print("  yaw_change=%.1f deg  yaw_rate=%.2f deg/s  Mz_mean=%.1f Nm" %
          (r["yaw_cum_deg"], r["yaw_rate_deg_s"], r["Mz_mean"]))
    print("  slip_L=%.4f m/s  slip_R=%.4f m/s" % (r["slip_L_mean"], r["slip_R_mean"]))

    # Action bias on yaw-related joints
    if src == "policy":
        am = r["action_means"]
        tm = r["tau_means"]
        for j in yaw_joints:
            print("  %-25s action=%+7.4f tau=%+7.1f" % (dn[j], am[j], tm[j]))


# ====== TABLE ======
print("\n" + "=" * 80)
print("TABLE: Yaw Source Comparison")
print("=" * 80)
print("  %-20s %8s %8s %8s %10s %10s %10s %10s %10s" %
      ("case","ground","friction","zero_yaw","yaw_deg","yaw_rate","Mz_mean","slip_L","slip_R"))
for r in results:
    gnd = "yes" if "no_gnd" not in r["label"] else "no"
    fric = "0.01" if "low_fric" in r["label"] else "0.8"
    zy = "yes" if "no_yaw" in r["label"] else "no"
    print("  %-20s %8s %8s %8s %+9.1f %9.2f %+9.1f %9.4f %9.4f" %
          (r["label"], gnd, fric, zy, r["yaw_cum_deg"], r["yaw_rate_deg_s"],
           r["Mz_mean"], r["slip_L_mean"], r["slip_R_mean"]))

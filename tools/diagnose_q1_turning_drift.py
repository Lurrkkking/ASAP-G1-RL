"""
Q1 Turning/Drift Diagnosis — zero-action, symmetry, mapping, contact.
"""
import sys, csv, numpy as np
from pathlib import Path
PROJECT_ROOT = Path("/root/autodl-tmp/ASAP")
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from isaacgym import gymapi, gymtorch
import torch


# ====== TASK 1: Zero-action vs policy rollout ======
print("=" * 70)
print("TASK 1: Zero-action vs Policy Rollout — drift check")
print("=" * 70)

from omegaconf import OmegaConf
conf = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot; rz = float(rc.init_state.pos[2])
defaults = dict(rc.init_state.default_joint_angles)

def build_env_and_reset():
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
    dn=g.get_asset_dof_names(asset); nd=len(dn)
    ah=g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)
    tgt=[float(defaults.get(n,0)) for n in dn]
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

    # Only zero velocities, reset DOF pos to yaml default (matching training reset)
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    for i in range(nd): df[i,0]=tgt[i]
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

    return g, s, e, ah, dn, nd, tgt, r_v, df, pg, dg, tau_lim


for case_label, action_mode in [("A_zero_action", "zero"), ("C_policy", "policy")]:
    g, s, e, ah, dn, nd, tgt, r_v, df, pg, dg, tau_lim = build_env_and_reset()
    eff_check=np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])

    # Load policy if needed
    actor = None
    if action_mode == "policy":
        ckpt = torch.load(str(PROJECT_ROOT/"logs/TEST_Q1_loco/20260529_164937-Q1_Stand_AntiSlip-legged_base-q1_22dof/model_1200.pt"), map_location="cpu")
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

    init_yaw = np.arctan2(2*(r_v[0,6].item()*r_v[0,5].item()+r_v[0,3].item()*r_v[0,4].item()),
                           1-2*(r_v[0,4].item()**2+r_v[0,5].item()**2))
    yaw_cum = 0; last_yaw = init_yaw; lv_sum=np.zeros(2); last_action=np.zeros(22)

    print("\n%s:" % case_label)
    print("  init_yaw=%.3f knee=[%.3f,%.3f] hip_roll=[%+.4f,%+.4f]" %
          (np.degrees(init_yaw), df[3,0].item(), df[9,0].item(), df[1,0].item(), df[7,0].item()))
    print("  %-4s %7s %7s %7s %7s %7s %7s %7s" % ("f","yaw_deg","yaw_rate","lv_xy","a_mean","kneeL","kneeR","hipR_LR"))

    for step in range(500):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        rs=r_v[0].cpu().numpy(); dp_v=df[:nd,0]; dv_v=df[:nd,1]

        if action_mode == "zero":
            mean_a = torch.zeros(nd, device="cuda:0")
        else:
            qx,qy,qz_,qw=rs[3:7]
            pg_vec=np.array([2*(qx*qz_-qw*qy),2*(qy*qz_+qw*qx),1-2*(qx*qx+qy*qy)])
            obs_vec=np.concatenate([rs[7:10]*2.0, rs[10:13]*0.25, pg_vec,
                                     [0,0,0], dp_v.cpu().numpy(), dv_v.cpu().numpy(), last_action])
            with torch.no_grad():
                mean_a = actor(torch.tensor(obs_vec,dtype=torch.float32,device="cuda:0").unsqueeze(0)).squeeze(0)

        action_scaled = mean_a * 0.25
        target_pos = action_scaled.float() + torch.tensor(tgt,device="cuda:0",dtype=torch.float32)
        torque = (pg*(target_pos-dp_v.float())-dg*dv_v.float()).clamp(-tau_lim,tau_lim)
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        last_action = mean_a.cpu().numpy().copy()

        g.refresh_actor_root_state_tensor(s)
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy()
        yaw = np.arctan2(2*(qw*qz_+qx*qy), 1-2*(qy*qy+qz_*qz_))
        dy = yaw - last_yaw
        if dy > np.pi: dy -= 2*np.pi
        elif dy < -np.pi: dy += 2*np.pi
        yaw_cum += dy; last_yaw = yaw
        lv_xy = np.linalg.norm(r_v[0,7:9].cpu().numpy())

        if step % 25 == 0:
            yaw_rate = dy / 0.02
            print("  %-4d %+7.1f %+7.1f %7.3f %7.3f %+7.3f %+7.3f %+7.4f" %
                  (step, np.degrees(yaw), np.degrees(yaw_rate), lv_xy,
                   mean_a.abs().mean().item() if action_mode!="zero" else 0,
                   df[3,0].item(), df[9,0].item(), df[1,0].item()-df[7,0].item()))

    print("  => total_yaw=%.1f deg  yaw_rate_mean=%.2f deg/s" %
          (np.degrees(yaw_cum), np.degrees(yaw_cum/(500*0.02))))
    g.destroy_sim(s)


# ====== TASK 4: Action-DOF mapping check ======
print("\n" + "=" * 70)
print("TASK 4: Action-DOF Mapping")
print("=" * 70)

g, s, e, ah, dn, nd, tgt, r_v, df, pg, dg, tau_lim = build_env_and_reset()
print("  %-3s %-35s %-6s %-8s %-8s %-12s %-12s" % ("idx","dof_name","side","lower","upper","default","effort"))
for i,n in enumerate(dn):
    side = "L" if "left" in n else ("R" if "right" in n else "-")
    lo = rc.dof_pos_lower_limit_list[i]; hi = rc.dof_pos_upper_limit_list[i]
    print("  %-3d %-35s %-6s %+8.3f %+8.3f %+12.4f %+12.1f" %
          (i, n, side, lo, hi, tgt[i], g.get_actor_dof_properties(e,ah)["effort"][i].item()))

# Check G1 23dof vs Q1 22dof
print("\n  Q1 has 22 DOF (no waist_pitch). G1 has 23 DOF.")
print("  Q1 waist order: roll, yaw (G1: yaw, roll, pitch)")
q1_waist = [n for n in dn if "waist" in n]
print("  Q1 waist joints:", q1_waist)
g.destroy_sim(s)


# ====== TASK 5: Single-joint action pulse test ======
print("\n" + "=" * 70)
print("TASK 5: Single-joint Action Pulse Test")
print("=" * 70)

pulse_joints = [
    ("left_hip_yaw_joint", 2), ("right_hip_yaw_joint", 8),
    ("left_hip_roll_joint", 1), ("right_hip_roll_joint", 7),
    ("left_ankle_roll_joint", 5), ("right_ankle_roll_joint", 11),
    ("left_knee_joint", 3), ("right_knee_joint", 9),
]

for jname, jidx in pulse_joints:
    g, s, e, ah, dn, nd, tgt, r_v, df, pg, dg, tau_lim = build_env_and_reset()
    tau = torch.zeros(nd, device="cuda:0")
    action = torch.zeros(nd, device="cuda:0")
    action[jidx] = 0.2
    target_pos = action*0.25 + torch.tensor(tgt,device="cuda:0",dtype=torch.float32)

    moved_joints = set()
    for step in range(20):
        g.refresh_dof_state_tensor(s)
        torque = (pg*(target_pos-df[:nd,0].float())-dg*df[:nd,1].float()).clamp(-tau_lim,tau_lim)
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        g.refresh_dof_state_tensor(s)
        # Check which DOFs moved more than 0.005
        for i in range(nd):
            if abs(df[i,0].item()-tgt[i]) > 0.005:
                moved_joints.add(i)

    intended = dn[jidx]
    actual = ", ".join([dn[i] for i in sorted(moved_joints)[:4]]) if moved_joints else "NONE"
    direction = df[jidx,0].item() - tgt[jidx]
    expected_sign = "+" if 0.2 > 0 else "-"
    actual_sign = "+" if direction > 0.001 else ("-" if direction < -0.001 else "0")
    ok = intended in actual and actual_sign == "+"
    print("  %-30s pulse=+0.2 moved=%s sign=%s %s" % (jname, actual, actual_sign, "PASS" if ok else "FAIL"))
    g.destroy_sim(s)


# ====== TASK 6: Left-right symmetry at init ======
print("\n" + "=" * 70)
print("TASK 6: Init Pose Left-Right Symmetry")
print("=" * 70)

g, s, e, ah, dn, nd, tgt, r_v, df, pg, dg, tau_lim = build_env_and_reset()
pairs = [("hip_pitch_joint", 0, 6), ("hip_roll_joint", 1, 7), ("hip_yaw_joint", 2, 8),
         ("knee_joint", 3, 9), ("ankle_pitch_joint", 4, 10), ("ankle_roll_joint", 5, 11)]
print("  %-20s %10s %10s %10s %10s" % ("joint","left","right","diff","expected"))
for name, li, ri in pairs:
    lv = df[li,0].item(); rv = df[ri,0].item()
    # hip_roll: left positive = outward, right positive = outward → opposite signs
    # hip_yaw: left positive = outward, right positive = outward → opposite signs
    # ankle_roll: left positive = outward, right positive = outward → opposite signs
    # Others: symmetric
    if "roll" in name or "yaw" in name:
        expected = "opposite"
        ok = abs(lv + rv) < 0.01  # should sum to ~0
    else:
        expected = "equal"
        ok = abs(lv - rv) < 0.01
    print("  %-20s %+10.4f %+10.4f %+10.4f %10s %s" %
          (name, lv, rv, lv-rv, expected, "PASS" if ok else "FAIL"))
g.destroy_sim(s)


# ====== TASK 7: Check yaw/heading setup ======
print("\n" + "=" * 70)
print("TASK 7: Yaw/Heading Config Check")
print("=" * 70)
print("  IsaacGym quat order: (x,y,z,w)")
print("  Q1 standing command: lin_vel_x=0, lin_vel_y=0, yaw_vel=0")
print("  No heading command used in stand smoke")
print("  tracking_ang_vel tracks yaw_vel command via ang_vel_z in body frame")
print("  Body frame angular velocity: av_z positive = CCW about z (yaw left)")

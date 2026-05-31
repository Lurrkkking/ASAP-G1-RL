"""
Load Q1 checkpoint, rollout, print velocity tracking.
Isaac Gym before torch.
"""
import sys, numpy as np
from pathlib import Path
PROJECT_ROOT = Path("/root/autodl-tmp/ASAP")
sys.path.insert(0, str(PROJECT_ROOT))

from isaacgym import gymapi, gymtorch
import torch

ckpt_path = PROJECT_ROOT / "logs/TEST_Q1_loco/20260529_173050-Q1_Stand_FixReset-legged_base-q1_22dof/model_200.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu")
sd = ckpt["actor_model_state_dict"]

# Build MLP from state dict
actor_in = sd["actor_module.module.0.weight"].shape[1]
actor = torch.nn.Sequential(
    torch.nn.Linear(actor_in, 512), torch.nn.ELU(),
    torch.nn.Linear(512, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 22),
)
actor.load_state_dict({k.replace("actor_module.module.",""): v for k,v in sd.items() if "std" not in k})
actor.to("cuda:0"); actor.eval()
print("actor ok, input_dim=%d" % actor_in)

# Build standalone sim (same as q1_standing_final.py approach)
from omegaconf import OmegaConf
conf = OmegaConf.load(str(PROJECT_ROOT/"humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot; rz = float(rc.init_state.pos[2])
defaults = dict(rc.init_state.default_joint_angles)

g = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
s = g.create_sim(0,0,gymapi.SIM_PHYSX,sp)
pp=gymapi.PlaneParams(); pp.normal=gymapi.Vec3(0,0,1); pp.static_friction=0.8; pp.dynamic_friction=0.6
g.add_ground(s, pp)
e = g.create_env(s, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

ac = rc.asset; ao = gymapi.AssetOptions()
for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
          "fix_base_link","density","angular_damping","linear_damping",
          "max_angular_velocity","max_linear_velocity","armature","thickness"]:
    setattr(ao,k,getattr(ac,k))
ao.default_dof_drive_mode=3
asset = g.load_asset(s, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof_rl_collision.urdf", ao)
dn = g.get_asset_dof_names(asset); nd = len(dn)
ah = g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)
tgt = [float(defaults.get(n,0)) for n in dn]
ds = g.get_actor_dof_states(e, ah, gymapi.STATE_ALL)
for i in range(nd): ds[i]["pos"]=tgt[i]; ds[i]["vel"]=0.0
g.set_actor_dof_states(e, ah, ds, gymapi.STATE_ALL)
qp = g.get_actor_rigid_shape_properties(e, ah)
for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
g.set_actor_rigid_shape_properties(e, ah, qp)

g.prepare_sim(s)
rt = g.acquire_actor_root_state_tensor(s); r_all = gymtorch.wrap_tensor(rt)
r_v = r_all if r_all.ndim==2 else r_all.view(-1,13)
dt = g.acquire_dof_state_tensor(s); d_all = gymtorch.wrap_tensor(dt); df = d_all.view(-1,2)
g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)

prepared = df[:nd,0].clone().cpu().numpy()
r_v[0,7:13]=0.0; df[:nd,1]=0.0
# ALSO reset DOF pos to yaml default (matching training env _reset_dofs behavior)
for i in range(nd): df[i,0] = tgt[i]
g.set_actor_root_state_tensor(s, gymtorch.unwrap_tensor(r_all))
g.set_dof_state_tensor(s, gymtorch.unwrap_tensor(d_all))
g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)

print("init rz=%.3f knee=[%.3f,%.3f]" % (r_v[0,2].item(), df[3,0].item(), df[9,0].item()))

# Pre-compute PD gains and limits
eff = np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])
stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
for i,n in enumerate(dn):
    for k in stiff:
        if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
pg=torch.tensor(pg_np,device="cuda:0",dtype=torch.float32)
dg=torch.tensor(dg_np,device="cuda:0",dtype=torch.float32)
tau_lim_t = torch.tensor(eff, device="cuda:0", dtype=torch.float32)

# Build obs (simplified: same dims as training)
# actor_obs: base_ang_vel(3) + proj_grav(3) + cmd_lin_vel(2) + cmd_ang_vel(1) + dof_pos(22) + dof_vel(22) + actions(22) = 75
# If withlinvel adds base_lin_vel(3): 78
def get_obs():
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    rs = r_v[0].cpu().numpy()
    dp = df[:nd,0].cpu().numpy(); dv = df[:nd,1].cpu().numpy()
    base_ang_vel = rs[10:13]
    base_lin_vel = rs[7:10]
    # proj_grav from quat
    qx,qy,qz_,qw = rs[3:7]
    proj_grav = np.array([2*(qx*qz_ - qw*qy), 2*(qy*qz_ + qw*qx), 1-2*(qx*qx + qy*qy)])
    cmd = np.array([0.0, 0.0, 0.0])  # zero cmd
    actions = np.zeros(22)  # last action (zeros for first step)
    # actor_in=78 means with base_lin_vel
    obs_vec = np.concatenate([base_ang_vel, proj_grav, base_lin_vel, cmd[:2], [cmd[2]], dp, dv, actions])
    return torch.tensor(obs_vec, dtype=torch.float32, device="cuda:0").unsqueeze(0)

print("\nRollout — tracking velocity vs zero command:")
print("%-4s %7s %7s %7s %7s %7s %7s %7s %7s" % ("f","rz","pgz","lv_x","lv_y","av_z","a_norm","kneeL","kneeR"))

all_lv_xy = []; all_av_z = []
last_action = torch.zeros(22, device="cuda:0")
for step in range(500):
    # Get obs and predict action
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    rs = r_v[0].cpu().numpy()
    dp_v = df[:nd,0]; dv_v = df[:nd,1]
    base_ang_vel = r_v[0,10:13].cpu().numpy()
    base_lin_vel = r_v[0,7:10].cpu().numpy()
    qx,qy,qz_,qw = rs[3:7]
    proj_grav = np.array([2*(qx*qz_-qw*qy), 2*(qy*qz_+qw*qx), 1-2*(qx*qx+qy*qy)])
    cmd = np.array([0, 0, 0])
    # Training obs order: base_lin_vel(3), base_ang_vel(3), proj_grav(3), cmd_lin(2), cmd_ang(1), dof_pos(22), dof_vel(22), actions(22)
    # Scales: base_lin_vel=2.0, base_ang_vel=0.25
    obs_vec = np.concatenate([
        base_lin_vel * 2.0, base_ang_vel * 0.25, proj_grav,
        cmd[:2], [cmd[2]], dp_v.cpu().numpy(), dv_v.cpu().numpy(), last_action.cpu().numpy()])
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device="cuda:0").unsqueeze(0)

    with torch.no_grad():
        mean = actor(obs_t).squeeze(0)

    # Apply action via PD (same as training: action_scale=0.25, control_type=P)
    action_scaled = mean * 0.25
    # Training uses default_dof_pos (yaml default, knee=0.30) as PD base, NOT prepared pose
    default_dof = torch.tensor(tgt, device="cuda:0", dtype=torch.float32)
    target_pos = action_scaled.float() + default_dof
    torque = (pg*(target_pos - dp_v.float()) - dg*dv_v.float()).clamp(-tau_lim_t, tau_lim_t).to(torch.float32)
    g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque.contiguous()))
    for _ in range(4): g.simulate(s)
    g.fetch_results(s, True)

    last_action = mean.detach()

    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    lv_x = r_v[0,7].item(); lv_y = r_v[0,8].item(); av_z = r_v[0,12].item()
    qx,qy,qz_,qw = r_v[0,3:7].cpu().numpy(); pgz_v = 1-2*(qx*qx+qy*qy)
    all_lv_xy.append(np.linalg.norm([lv_x, lv_y])); all_av_z.append(av_z)

    if step % 25 == 0:
        print("%-4d %+7.3f %7.3f %+7.3f %+7.3f %+7.3f %7.3f %+7.3f %+7.3f" %
              (step, r_v[0,2].item(), pgz_v, lv_x, lv_y, av_z, mean.abs().mean().item(), df[3,0].item(), df[9,0].item()))

# Summary
all_lv = np.array(all_lv_xy); all_av = np.array(all_av_z)
print("\n=== VELOCITY SUMMARY ===")
print("Mean |lin_vel_xy|: %.4f m/s (target=0)" % all_lv.mean())
print("Max  |lin_vel_xy|: %.4f m/s" % all_lv.max())
print("Mean ang_vel_z:   %.4f rad/s (target=0)" % all_av.mean())
print("Std  ang_vel_z:   %.4f rad/s" % all_av.std())
print("Max  |ang_vel_z|: %.4f rad/s" % abs(all_av).max())
print("total_rotation:   %.1f degrees over 10s" % (np.degrees(all_av.sum() * 0.02)))

g.destroy_sim(s)

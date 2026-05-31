"""
G1 vs Q1 PD + Gravity Compensation — numerical gravity torque via mass matrix.
"""
import sys, os, csv
from pathlib import Path
import numpy as np, cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs" / "pd_gravity_comp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


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
             "f=%d rz=%.3f pgz=%.3f" % (d.get('frame',0), d.get('rz',0), d.get('pgz',1)),
             "av=%.1f tau_pd=%.0f tau_g=%.0f tau_tot=%.0f sat=%.0f%%" %
             (d.get('av_norm',0), d.get('tau_pd',0), d.get('tau_g',0), d.get('tau_max',0), d.get('tau_sat',0)*100),
             "cf=%s:%.0f" % (d.get('top_cf_name','?'), d.get('top_cf_val',0))]
    for i,l in enumerate(lines):
        cv2.putText(img, l, (10,20+i*22), f, 0.5, (0,255,0), 1)


ROBOTS = {
    "g1": {"yaml": "humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml",
           "urdf": "g1_29dof_anneal_23dof.urdf", "urdf_dir": "g1"},
    "q1": {"yaml": "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml",
           "urdf": "q1_22dof_rl_collision.urdf", "urdf_dir": "q1"},
}


def compute_gravity_torque(gym, sim, env, ah, nd, prepared, robot_name, root_z):
    """
    Numerically compute gravity compensation torque at prepared pose.
    Run 1 step with zero torque under gravity, compute joint accel,
    use mass matrix: tau_g = M * a_g ≈ -M * a_measured.
    Sign: tau_g is the torque needed to CANCEL gravity (add this as feedforward).
    """
    dt = gym.acquire_actor_root_state_tensor(sim)
    # Re-set to prepared pose with zero velocity
    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim==2 else r_all.view(-1,13)
    doft = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(doft); df = d_all.view(-1,2)

    r_v[0,7:13]=0.0
    for i in range(nd): df[i,0]=prepared[i]; df[i,1]=0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    # Measure joint acceleration under gravity with zero torque
    v_before = df[:nd,1].clone().cpu().numpy()
    tau_zero = torch.zeros(nd, device="cuda:0")
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau_zero.contiguous()))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    v_after = df[:nd,1].clone().cpu().numpy()

    a_measured = (v_after - v_before) / 0.005  # one substep
    # tau_gravity = the torque that would produce this acceleration = M * a
    # For feedforward, we want to CANCEL gravity: tau_ff = +tau_gravity = M*a
    # Approximation: tau_g ≈ M * a
    # Since we don't have M easily, use rough estimate based on effort limits and observed accel

    # Simpler approach: compute from static equilibrium.
    # The gravity torque at a joint ≈ needed torque to hold against gravity.
    # We can approximate using the joint error under gravity with stiff PD.
    # Or use the fact that: tau_g ≈ effort_limit * (a_measured / a_max)

    # Actually, the most practical approach: use PD with very high gains to find equilibrium torque
    # Reset again
    for i in range(nd): df[i,0]=prepared[i]; df[i,1]=0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_dof_state_tensor(sim)

    # Apply gravity and measure the PD torque needed after 1 control step
    # This gives us a first-order estimate of gravity torque
    # Use stiff PD to find equilibrium quickly
    tau_gravity = np.zeros(nd)

    # Run several steps with stiff PD to converge to gravity-compensating torque
    for step in range(10):
        gym.refresh_dof_state_tensor(sim)
        err = prepared - df[:nd,0].cpu().numpy()
        vel = df[:nd,1].cpu().numpy()
        # Very stiff gains for convergence
        stiff_tau = 500.0 * err - 20.0 * vel
        stiff_tau = np.clip(stiff_tau, -500, 500)
        tau_t = torch.tensor(stiff_tau, device="cuda:0", dtype=torch.float32)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau_t.contiguous()))
        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)

    # The converged torque ≈ gravity compensation
    tau_gravity = stiff_tau.copy()
    return tau_gravity


def run_case(rname, case_label, grav_z, apply_pd, target_mode, alpha_ff, n_steps, tau_g_precomputed):
    rcfg = ROBOTS[rname]
    conf = OmegaConf.load(str(PROJECT_ROOT / rcfg["yaml"])); rc = conf.robot
    root_z = float(rc.init_state.pos[2])
    defaults_dict = dict(rc.init_state.default_joint_angles)

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, grav_z)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4; sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10; sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    if hasattr(sp,'enable_camera_sensors'): sp.enable_camera_sensors = True
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
    asset = gym.load_asset(sim, str(PROJECT_ROOT/"humanoidverse/data/robots"/rcfg["urdf_dir"]), rcfg["urdf"], ao)
    dn = gym.get_asset_dof_names(asset); bnames = gym.get_asset_rigid_body_names(asset); nd = len(dn)
    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z)), rname, -1, 0, 0)

    tgt_yaml = [float(defaults_dict.get(n,0)) for n in dn]
    ds = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"] = tgt_yaml[i]; ds[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, ds, gymapi.STATE_ALL)
    qp = gym.get_actor_rigid_shape_properties(env, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    gym.set_actor_rigid_shape_properties(env, ah, qp)

    cam = gymapi.CameraProperties(); cam.width=1280; cam.height=720
    cam_h = gym.create_camera_sensor(env, cam)

    gym.prepare_sim(sim)
    rt = gym.acquire_actor_root_state_tensor(sim); r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt_t = gym.acquire_dof_state_tensor(sim); d_all = gymtorch.wrap_tensor(dt_t); df = d_all.view(-1,2)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    prepared = df[:nd,0].clone().cpu().numpy()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    if target_mode == 'prepared_pose': target = prepared
    else: target = np.array(tgt_yaml)

    stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dn):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg=torch.tensor(pg_np,device="cuda:0"); dg=torch.tensor(dg_np,device="cuda:0")
    default_t=torch.tensor(target,device="cuda:0")
    effort_orig=np.array([gym.get_actor_dof_properties(env,ah)["effort"][i].item() for i in range(nd)])
    tau_lim=torch.tensor(effort_orig,device="cuda:0")
    torque_t=torch.zeros(nd,device="cuda:0")

    if tau_g_precomputed is not None:
        tau_g_t = torch.tensor(tau_g_precomputed, device="cuda:0")
    else:
        tau_g_t = torch.zeros(nd, device="cuda:0")

    Lf=bnames.index("left_ankle_roll_link") if "left_ankle_roll_link" in bnames else 0
    Rf=bnames.index("right_ankle_roll_link") if "right_ankle_roll_link" in bnames else 1

    label="%s_a=%.2f_%s" % (rname,alpha_ff,target_mode[:6])
    writer=cv2.VideoWriter(str(OUTPUT_DIR/("%s.mp4"%label)),cv2.VideoWriter_fourcc(*"mp4v"),50,(1280,720))
    csv_rows=[]

    def capture(fno,tsim):
        cf_v=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(-1,3)
        rigid_v=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(-1,13)
        dz=df[:nd,0]; dvel=df[:nd,1]
        rz_f=r_v[0,2].item(); lv=r_v[0,7:10].cpu().numpy(); av=r_v[0,10:13].cpu().numpy()
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy(); pgz_v=pgz(qx,qy,qz_,qw)
        tau_np=torque_t.cpu().numpy()
        tau_pd_raw=(pg*(default_t-dz)-dg*dvel).cpu().numpy()
        tau_max_c=abs(tau_np).max(); tau_sat=(abs(tau_np)>0.95*effort_orig).mean()
        cf_np=cf_v.norm(dim=1).cpu().numpy()
        top_i=cf_np.argmax(); top_n=bnames[top_i] if top_i<len(bnames) else "?"
        row={'frame':fno,'t':tsim,'rz':rz_f,'pgz':pgz_v,'lv_norm':np.linalg.norm(lv),'av_norm':np.linalg.norm(av),
             'tau_pd':abs(tau_pd_raw).max(),'tau_g':abs(tau_g_t.cpu().numpy()).max(),
             'tau_max':tau_max_c,'tau_sat':tau_sat,
             'top_cf_name':top_n,'top_cf_val':cf_np[top_i],
             'kneeL':dz[3].item() if nd>3 else 0,'kneeR':dz[9].item() if nd>9 else 0}
        csv_rows.append(row)
        return row

    gym.set_camera_location(cam_h,env,gymapi.Vec3(2.5,2.5,1.5),gymapi.Vec3(0,0,0.4))
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    r0=capture(0,0.0); fr=to_bgr(gym.get_camera_image(sim,env,cam_h,gymapi.IMAGE_COLOR),720,1280)
    overlay(fr,r0,label); writer.write(fr)

    stand_ok=True; fell_t=None; min_pgz=1.0; max_av_v=0.0; max_tau_c=0.0; max_sat_v=0.0
    knee_min=[99,99]; knee_hit_n=0

    for step in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        torque_t.zero_()
        if apply_pd:
            tau_pd_v = pg*(default_t-df[:nd,0]) - dg*df[:nd,1]
            tau_v = alpha_ff * tau_g_t + tau_pd_v
            tau_v = tau_v.clamp(-tau_lim, tau_lim)
        else:
            tau_v = torch.zeros(nd, device="cuda:0")
        torque_t[:] = tau_v
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim, True)
        fno=step+1; tsim=fno*0.02
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
        row=capture(fno,tsim)
        min_pgz=min(min_pgz,row['pgz']); max_av_v=max(max_av_v,row['av_norm'])
        max_tau_c=max(max_tau_c,row['tau_max']); max_sat_v=max(max_sat_v,row['tau_sat'])
        if nd>3: knee_min[0]=min(knee_min[0],df[3,0].item())
        if nd>9: knee_min[1]=min(knee_min[1],df[9,0].item())
        if nd>3 and df[3,0].item()<0.005: knee_hit_n+=1
        if nd>9 and df[9,0].item()<0.005: knee_hit_n+=1
        if stand_ok:
            if row['pgz']<0.5: stand_ok=False; fell_t=tsim
            if row['rz']<0.15: stand_ok=False; fell_t=tsim if fell_t is None else fell_t
        if fno%4==0:
            rx=r_v[0,0].item(); ry=r_v[0,1].item(); rz2=r_v[0,2].item()
            gym.set_camera_location(cam_h,env,gymapi.Vec3(rx+2.5,ry+2.5,max(rz2+1.2,1.5)),gymapi.Vec3(rx,ry,rz2+0.3))
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            fr=to_bgr(gym.get_camera_image(sim,env,cam_h,gymapi.IMAGE_COLOR),720,1280)
            overlay(fr,row,label); writer.write(fr)

    writer.release()
    stand_t=fell_t if fell_t else n_steps*0.02
    gym.destroy_sim(sim)
    return {'robot':rname,'alpha':alpha_ff,'target':target_mode,'stand':stand_ok,'stand_time':stand_t,
            'min_pgz':min_pgz,'max_av':max_av_v,'max_tau':max_tau_c,'max_sat':max_sat_v,
            'knee_min':knee_min,'knee_hit':knee_hit_n}


# ====== PRE-COMPUTE GRAVITY TORQUE FOR EACH ROBOT ======
print("="*70)
print("Computing gravity compensation torques...")
print("="*70)

tau_g_data = {}
for rname in ["g1","q1"]:
    rcfg = ROBOTS[rname]
    conf = OmegaConf.load(str(PROJECT_ROOT/rcfg["yaml"])); rc = conf.robot
    root_z = float(rc.init_state.pos[2])
    defaults_dict = dict(rc.init_state.default_joint_angles)

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
    pp=gymapi.PlaneParams(); pp.normal=gymapi.Vec3(0,0,1)
    pp.static_friction=0.8; pp.dynamic_friction=0.6; pp.restitution=0.0
    gym.add_ground(sim, pp)
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

    ac=rc.asset; ao=gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode=3
    asset=gym.load_asset(sim,str(PROJECT_ROOT/"humanoidverse/data/robots"/rcfg["urdf_dir"]),rcfg["urdf"],ao)
    dn=gym.get_asset_dof_names(asset); nd=len(dn)
    ah=gym.create_actor(env,asset,gymapi.Transform(p=gymapi.Vec3(0,0,root_z)),rname,-1,0,0)
    tgt_y=[float(defaults_dict.get(n,0)) for n in dn]
    ds=gym.get_actor_dof_states(env,ah,gymapi.STATE_ALL)
    for i in range(nd): ds[i]['pos']=tgt_y[i]; ds[i]['vel']=0.0
    gym.set_actor_dof_states(env,ah,ds,gymapi.STATE_ALL)
    qp=gym.get_actor_rigid_shape_properties(env,ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    gym.set_actor_rigid_shape_properties(env,ah,qp)
    gym.prepare_sim(sim)

    rt=gym.acquire_actor_root_state_tensor(sim); r_all=gymtorch.wrap_tensor(rt); r_v=r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt_t=gym.acquire_dof_state_tensor(sim); d_all=gymtorch.wrap_tensor(dt_t); df=d_all.view(-1,2)
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    prepared=df[:nd,0].clone().cpu().numpy()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(r_all))
    gym.set_dof_state_tensor(sim,gymtorch.unwrap_tensor(d_all))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)

    # Compute gravity torque via stiff PD convergence
    effort_orig=np.array([gym.get_actor_dof_properties(env,ah)["effort"][i].item() for i in range(nd)])
    tau_conv=np.zeros(nd)
    for step in range(30):
        gym.refresh_dof_state_tensor(sim)
        err=prepared-df[:nd,0].cpu().numpy(); vel=df[:nd,1].cpu().numpy()
        tau_raw=500.0*err-20.0*vel
        tau_raw=np.clip(tau_raw,-np.maximum(effort_orig*3,500),np.maximum(effort_orig*3,500))
        tau_conv=tau_raw.copy()
        tau_t=torch.tensor(tau_raw,device="cuda:0",dtype=torch.float32)
        gym.set_dof_actuation_force_tensor(sim,gymtorch.unwrap_tensor(tau_t.contiguous()))
        for _ in range(4): gym.simulate(sim)
        gym.fetch_results(sim,True)
        gym.refresh_dof_state_tensor(sim)

    tau_g_data[rname]=tau_conv.copy()
    print("  %s: gravity tau knee=[%+.1f,%+.1f] hipP=[%+.1f,%+.1f] ankleP=[%+.1f,%+.1f]" %
          (rname,tau_conv[3],tau_conv[9],tau_conv[0],tau_conv[6],tau_conv[4],tau_conv[10]))
    gym.destroy_sim(sim)


# ====== RUN SWEEP ======
print("\n"+"="*70)
print("Gravity compensation sweep: alpha x prepared_pose")
print("="*70)

all_res=[]
for rname in ["g1","q1"]:
    tau_g=tau_g_data[rname]
    for alpha in [0.0,0.25,0.5,0.75,1.0]:
        r=run_case(rname,"grav",-9.81,True,"prepared_pose",alpha,250,tau_g)
        all_res.append(r)
        print("  %s a=%.2f: stand=%s t=%.1f pgz_min=%.2f tau=%.0f sat=%.0f%% knee_min=%.2f hit=%d" %
              (rname,alpha,r['stand'],r['stand_time'],r['min_pgz'],r['max_tau'],r['max_sat']*100,
               min(r['knee_min']),r['knee_hit']))

# Also test sign: + vs -
for rname in ["g1","q1"]:
    tau_g=tau_g_data[rname]
    for sign,slabel in [(-1.0,"neg"),(1.0,"pos")]:
        r=run_case(rname,"sign_%s"%slabel,-9.81,True,"prepared_pose",sign,250,tau_g*sign if sign<0 else tau_g)
        all_res.append(r)


# ====== TABLE ======
print("\n"+"="*90)
print("TABLE 1: Gravity Compensation Sweep")
print("="*90)
print("  %-3s %6s %8s %8s %8s %8s %8s %8s %8s %8s" %
      ("bot","alpha","stand","t(s)","min_pgz","max_av","max_tau","sat%","knee_min","hit"))
for r in sorted(all_res,key=lambda x:(x['robot'],x['alpha'])):
    print("  %-3s %+5.2f %8s %8.1f %8.3f %8.1f %8.0f %7.0f%% %8.2f %8d" %
          (r['robot'],r['alpha'],r['stand'],r['stand_time'],r['min_pgz'],
           r['max_av'],r['max_tau'],r['max_sat']*100,min(r['knee_min']),r['knee_hit']))

print("\nOutput: %s/" % OUTPUT_DIR)

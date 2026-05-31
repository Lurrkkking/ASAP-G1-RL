"""
Q1 Effort Limit Sweep — verify if 36 Nm is the bottleneck.
"""
import sys, os, csv
from pathlib import Path
import numpy as np, cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    lines = [
        "%s  f=%d t=%.2f" % (label, d.get('frame',0), d.get('t',0)),
        "rz=%.3f pgz=%.3f" % (d.get('rz',0), d.get('pgz',1)),
        "av=%.1f tau=%.1f sat=%.0f%%" % (d.get('av_norm',0), d.get('tau_max',0), d.get('tau_sat',0)*100),
        "knee=[%.2f,%.2f] cf=%s:%.0f" % (d.get('kneeL',0), d.get('kneeR',0), d.get('top_cf','?'), d.get('top_cf_val',0)),
    ]
    for i, l in enumerate(lines):
        cv2.putText(img, l, (10, 20+i*22), f, 0.5, (0,255,0), 1)


# ====== CONFIG ======
conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml"))
rc = conf.robot; rz = float(rc.init_state.pos[2])
defaults_dict = dict(rc.init_state.default_joint_angles)

print("=" * 70)
print("Q1 EFFORT LIMIT SWEEP")
print("  root_z=%.3f  kp/kd=G0_current  target=prepared_pose" % rz)
print("=" * 70)


def run_effort_case(effort_scale, n_steps=250):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    if hasattr(sp,'enable_camera_sensors'): sp.enable_camera_sensors=True
    s = g.create_sim(0,0,gymapi.SIM_PHYSX,sp)
    pp = gymapi.PlaneParams(); pp.normal=gymapi.Vec3(0,0,1)
    pp.static_friction=0.8; pp.dynamic_friction=0.6; pp.restitution=0.0
    g.add_ground(s, pp)
    e = g.create_env(s, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)

    ac = rc.asset; ao = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode=3
    asset = g.load_asset(s, str(PROJECT_ROOT/"humanoidverse/data/robots/q1"), "q1_22dof_rl_collision.urdf", ao)
    dn = g.get_asset_dof_names(asset); bnames = g.get_asset_rigid_body_names(asset); nd = len(dn)

    ah = g.create_actor(e, asset, gymapi.Transform(p=gymapi.Vec3(0,0,rz)), "q1", -1, 0, 0)

    tgt_yaml = [float(defaults_dict.get(n,0)) for n in dn]
    ds = g.get_actor_dof_states(e, ah, gymapi.STATE_ALL)
    for i in range(nd): ds[i]["pos"]=tgt_yaml[i]; ds[i]["vel"]=0.0
    g.set_actor_dof_states(e, ah, ds, gymapi.STATE_ALL)
    qp = g.get_actor_rigid_shape_properties(e, ah)
    for i in range(len(qp)): qp[i].contact_offset=0.001; qp[i].rest_offset=0.0; qp[i].restitution=0.0; qp[i].friction=0.8
    g.set_actor_rigid_shape_properties(e, ah, qp)

    cam = gymapi.CameraProperties(); cam.width=1280; cam.height=720
    cam_h = g.create_camera_sensor(e, cam)

    g.prepare_sim(s)
    rt = g.acquire_actor_root_state_tensor(s); r_all = gymtorch.wrap_tensor(rt)
    r_v = r_all if r_all.ndim==2 else r_all.view(-1,13)
    dt = g.acquire_dof_state_tensor(s); d_all = gymtorch.wrap_tensor(dt); df = d_all.view(-1,2)
    rigid_t = g.acquire_rigid_body_state_tensor(s); rigid = gymtorch.wrap_tensor(rigid_t)
    cf_t = g.acquire_net_contact_force_tensor(s)
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

    prepared = df[:nd,0].clone().cpu().numpy()
    r_v[0,7:13]=0.0; df[:nd,1]=0.0
    g.set_actor_root_state_tensor(s, gymtorch.unwrap_tensor(r_all))
    g.set_dof_state_tensor(s, gymtorch.unwrap_tensor(d_all))
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s); g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)

    # Effort limits — temporary override
    effort_orig = np.array([g.get_actor_dof_properties(e,ah)["effort"][i].item() for i in range(nd)])
    effort_scaled = effort_orig * effort_scale

    # G0 gains
    stiff=dict(rc.control.stiffness); damp=dict(rc.control.damping)
    pg_np=np.zeros(nd,np.float32); dg_np=np.zeros(nd,np.float32)
    for i,n in enumerate(dn):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg=torch.tensor(pg_np,device="cuda:0"); dg=torch.tensor(dg_np,device="cuda:0")
    default_t=torch.tensor(prepared,device="cuda:0")
    tau_lim=torch.tensor(effort_scaled,device="cuda:0")
    torque_t=torch.zeros(nd,device="cuda:0")

    Lf=bnames.index("left_ankle_roll_link"); Rf=bnames.index("right_ankle_roll_link")
    knee_lim_low=0.0
    elabel = "eff_x%.1f" % effort_scale

    writer = cv2.VideoWriter(str(OUTPUT_DIR/("q1_eff_%s.mp4" % elabel.replace('.','_'))),
                             cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280,720))
    csv_rows=[]

    def capture(fno, tsim):
        cf_v=gymtorch.wrap_tensor(cf_t).view(-1,3)
        rigid_v=rigid.view(rigid.shape[0],13)
        dz=df[:nd,0]; dvel=df[:nd,1]
        rz_f=r_v[0,2].item(); lv=r_v[0,7:10].cpu().numpy(); av=r_v[0,10:13].cpu().numpy()
        qx,qy,qz_,qw=r_v[0,3:7].cpu().numpy(); pgz_v=pgz(qx,qy,qz_,qw)
        tau_np=torque_t.cpu().numpy()
        tau_cmd_raw=(pg*(default_t-dz)-dg*dvel).cpu().numpy()
        tau_max_clip=abs(tau_np).max(); tau_max_cmd=abs(tau_cmd_raw).max()
        tau_sat=(abs(tau_np)>0.95*effort_scaled).mean()
        sat_joints=[dn[i] for i in range(nd) if abs(tau_np[i])>0.95*effort_scaled[i]]
        cf_np=cf_v.norm(dim=1).cpu().numpy()
        top_i=cf_np.argmax(); top_n=bnames[top_i] if top_i<len(bnames) else "?"
        nfc=sum(cf_np[i] for i in range(len(bnames)) if 'ankle_roll' not in bnames[i] and cf_np[i]>1)
        bz=rigid_v[:,2].cpu().numpy()
        knee_hit=dz[3].item()<knee_lim_low+0.005 or dz[9].item()<knee_lim_low+0.005
        row={
            'frame':fno,'t':tsim,'rz':rz_f,'pgz':pgz_v,'lv_norm':np.linalg.norm(lv),'av_norm':np.linalg.norm(av),
            'dof_pos_err_max':(dz-default_t).abs().max().item(),'dof_vel_max':dvel.abs().max().item(),
            'tau_max':tau_max_clip,'tau_cmd_max':tau_max_cmd,'tau_sat':tau_sat,
            'saturated_joints':",".join(sat_joints),'L_fz':cf_np[Lf],'R_fz':cf_np[Rf],'non_foot_cf_sum':nfc,
            'top_cf':top_n,'top_cf_val':cf_np[top_i],
            'kneeL':dz[3].item(),'kneeR':dz[9].item(),'kneeL_vel':dvel[3].item(),'kneeR_vel':dvel[9].item(),
            'kneeL_tau':tau_np[3],'kneeR_tau':tau_np[9],
            'kneeL_tau_cmd':tau_cmd_raw[3],'kneeR_tau_cmd':tau_cmd_raw[9],
            'knee_hit':knee_hit,'bz_min':bz.min(),
        }
        csv_rows.append(row)
        return row

    # Frame 0
    g.set_camera_location(cam_h, e, gymapi.Vec3(2.5,2.5,1.5), gymapi.Vec3(0,0,0.4))
    g.step_graphics(s); g.render_all_camera_sensors(s)
    g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
    g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)
    r0=capture(0,0.0)
    fr=to_bgr(g.get_camera_image(s,e,cam_h,gymapi.IMAGE_COLOR),720,1280); overlay(fr,r0,elabel); writer.write(fr)

    stand_ok=True; fell_t=None; min_pgz=1.0; max_av_v=0.0; max_tau_c=0.0; max_tau_cl=0.0; max_sat_v=0.0
    max_dvel_v=0.0; nf_tot=0; first_sat=None; knee_min=[99,99]; knee_max=[-99,-99]; knee_hit_n=0

    for step in range(n_steps):
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)
        torque_t.zero_()
        tau_v=(pg*(default_t-df[:nd,0])-dg*df[:nd,1]).clamp(-tau_lim,tau_lim)
        torque_t[:]=tau_v
        g.set_dof_actuation_force_tensor(s, gymtorch.unwrap_tensor(torque_t.contiguous()))
        for _ in range(4): g.simulate(s)
        g.fetch_results(s, True)
        fno=step+1; tsim=fno*0.02
        g.refresh_actor_root_state_tensor(s); g.refresh_dof_state_tensor(s)
        g.refresh_rigid_body_state_tensor(s); g.refresh_net_contact_force_tensor(s)
        row=capture(fno,tsim)

        min_pgz=min(min_pgz,row['pgz']); max_av_v=max(max_av_v,row['av_norm'])
        max_tau_c=max(max_tau_c,row['tau_cmd_max']); max_tau_cl=max(max_tau_cl,row['tau_max'])
        max_sat_v=max(max_sat_v,row['tau_sat']); max_dvel_v=max(max_dvel_v,row['dof_vel_max'])
        nf_tot+=(1 if row['non_foot_cf_sum']>0 else 0)
        if first_sat is None and row['saturated_joints']: first_sat=row['saturated_joints']
        for ki,kidx in enumerate([3,9]):
            knee_min[ki]=min(knee_min[ki],df[kidx,0].item())
            knee_max[ki]=max(knee_max[ki],df[kidx,0].item())
        if row['knee_hit']: knee_hit_n+=1
        if stand_ok:
            if row['pgz']<0.5: stand_ok=False; fell_t=tsim
            if row['rz']<0.15: stand_ok=False; fell_t=tsim if fell_t is None else fell_t

        if fno%50==0:
            print("  f%4d t=%.1f rz=%.3f pgz=%.3f av=%.1f tau_cmd=%.1f tau_clip=%.1f sat=%.0f%% knee=[%.2f,%.2f] nf=%d" %
                  (fno,tsim,row['rz'],row['pgz'],row['av_norm'],row['tau_cmd_max'],row['tau_max'],
                   row['tau_sat']*100,row['kneeL'],row['kneeR'],nf_tot))
        if fno%4==0:
            rx=r_v[0,0].item(); ry=r_v[0,1].item(); rz2=r_v[0,2].item()
            g.set_camera_location(cam_h, e, gymapi.Vec3(rx+2.5,ry+2.5,max(rz2+1.2,1.5)), gymapi.Vec3(rx,ry,rz2+0.3))
            g.step_graphics(s); g.render_all_camera_sensors(s)
            fr=to_bgr(g.get_camera_image(s,e,cam_h,gymapi.IMAGE_COLOR),720,1280); overlay(fr,row,elabel); writer.write(fr)

    writer.release()
    with open(str(OUTPUT_DIR/("q1_eff_%s.csv" % elabel.replace('.','_'))),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    stand_t=fell_t if fell_t else 5.0
    print("  => %s: stand=%s t=%.1fs pgz_min=%.2f tau_cmd_max=%.0f tau_clip=%.0f sat=%.0f%% knee=[%.2f,%.2f] hit=%d" %
          (elabel,stand_ok,stand_t,min_pgz,max_tau_c,max_tau_cl,max_sat_v*100,knee_min[0],knee_min[1],knee_hit_n))
    g.destroy_sim(s)
    return {
        'scale':effort_scale,'stand':stand_ok,'stand_time':stand_t,'min_pgz':min_pgz,
        'max_av':max_av_v,'max_dvel':max_dvel_v,'max_tau_cmd':max_tau_c,'max_tau_clip':max_tau_cl,
        'max_sat':max_sat_v,'knee_min':knee_min,'knee_max':knee_max,'knee_hit_n':knee_hit_n,
        'nf_tot':nf_tot,'first_sat':first_sat,'effort_knee':effort_scaled[3],'effort_hip':effort_scaled[0],
        'effort_ankle':effort_scaled[4],
    }


# ====== RUN SWEEP ======
results=[]
for sc in [1.0,1.5,2.0,3.0,5.0]:
    results.append(run_effort_case(sc))

# If best stands, test yaml default
best_standing=next((r for r in results if r['stand']), None)
if best_standing:
    # Quick inline test of yaml default with same effort scale
    g2=gymapi.acquire_gym()
    sp=gymapi.SimParams(); sp.dt=0.005; sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,-9.81)
    sp.physx.solver_type=1; sp.physx.num_position_iterations=4; sp.physx.num_velocity_iterations=1
    sp.physx.num_threads=10; sp.physx.use_gpu=True; sp.use_gpu_pipeline=True
    s2=g2.create_sim(0,0,gymapi.SIM_PHYSX,sp); g2.add_ground(s2,gymapi.PlaneParams())
    e2=g2.create_env(s2,gymapi.Vec3(0,0,0),gymapi.Vec3(0,0,0),1)
    ac=rc.asset; ao=gymapi.AssetOptions()
    for k in ["collapse_fixed_joints","replace_cylinder_with_capsule","flip_visual_attachments",
              "fix_base_link","density","angular_damping","linear_damping",
              "max_angular_velocity","max_linear_velocity","armature","thickness"]:
        setattr(ao,k,getattr(ac,k))
    ao.default_dof_drive_mode=3
    asset2=g2.load_asset(s2,str(PROJECT_ROOT/"humanoidverse/data/robots/q1"),"q1_22dof_rl_collision.urdf",ao)
    dn2=g2.get_asset_dof_names(asset2); nd2=len(dn2)
    ah2=g2.create_actor(e2,asset2,gymapi.Transform(p=gymapi.Vec3(0,0,rz)),"q1",-1,0,0)
    tgt_y=[float(defaults_dict.get(n,0)) for n in dn2]
    ds2=g2.get_actor_dof_states(e2,ah2,gymapi.STATE_ALL)
    for i in range(nd2): ds2[i]["pos"]=tgt_y[i]; ds2[i]["vel"]=0.0
    g2.set_actor_dof_states(e2,ah2,ds2,gymapi.STATE_ALL)
    qp2=g2.get_actor_rigid_shape_properties(e2,ah2)
    for i in range(len(qp2)): qp2[i].contact_offset=0.001; qp2[i].rest_offset=0.0; qp2[i].restitution=0.0; qp2[i].friction=0.8
    g2.set_actor_rigid_shape_properties(e2,ah2,qp2)
    g2.prepare_sim(s2)
    rt2=g2.acquire_actor_root_state_tensor(s2); r_all2=gymtorch.wrap_tensor(rt2)
    r_v2=r_all2 if r_all2.ndim==2 else r_all2.view(-1,13)
    dt2=g2.acquire_dof_state_tensor(s2); d_all2=gymtorch.wrap_tensor(dt2); df2=d_all2.view(-1,2)
    g2.refresh_actor_root_state_tensor(s2); g2.refresh_dof_state_tensor(s2)
    r_v2[0,7:13]=0.0; df2[:nd2,1]=0.0
    g2.set_actor_root_state_tensor(s2,gymtorch.unwrap_tensor(r_all2))
    g2.set_dof_state_tensor(s2,gymtorch.unwrap_tensor(d_all2))
    g2.refresh_actor_root_state_tensor(s2); g2.refresh_dof_state_tensor(s2)
    eff2=np.array([g2.get_actor_dof_properties(e2,ah2)["effort"][i].item() for i in range(nd2)])*best_standing['scale']
    pg_np2=np.zeros(nd2,np.float32); dg_np2=np.zeros(nd2,np.float32)
    stiff2=dict(rc.control.stiffness); damp2=dict(rc.control.damping)
    for i,n in enumerate(dn2):
        for k in stiff2:
            if k in n: pg_np2[i]=stiff2[k]; dg_np2[i]=damp2[k]; break
    pg2=torch.tensor(pg_np2,device="cuda:0"); dg2=torch.tensor(dg_np2,device="cuda:0")
    default_y2=torch.tensor(np.array(tgt_y),device="cuda:0")
    tau_lim2=torch.tensor(eff2,device="cuda:0"); torque_t2=torch.zeros(nd2,device="cuda:0")
    stand_y2=True; fell_y2=None; min_pgz_y2=1.0; max_tau_y2=0.0
    for step in range(250):
        g2.refresh_dof_state_tensor(s2); g2.refresh_actor_root_state_tensor(s2)
        tau_v2=(pg2*(default_y2-df2[:nd2,0])-dg2*df2[:nd2,1]).clamp(-tau_lim2,tau_lim2)
        torque_t2[:]=tau_v2
        g2.set_dof_actuation_force_tensor(s2,gymtorch.unwrap_tensor(torque_t2.contiguous()))
        for _ in range(4): g2.simulate(s2)
        g2.fetch_results(s2,True)
        g2.refresh_actor_root_state_tensor(s2); g2.refresh_dof_state_tensor(s2)
        fno=step+1; tsim=fno*0.02
        rz_f=r_v2[0,2].item(); qx,qy,qz_,qw=r_v2[0,3:7].cpu().numpy(); pgz_f=1-2*(qx*qx+qy*qy)
        max_tau_y2=max(max_tau_y2,abs(tau_v2.cpu().numpy()).max()); min_pgz_y2=min(min_pgz_y2,pgz_f)
        if stand_y2 and (pgz_f<0.5 or rz_f<0.15): stand_y2=False; fell_y2=tsim
    g2.destroy_sim(s2)
    yaml_r={'stand':stand_y2,'fell_t':fell_y2,'min_pgz':min_pgz_y2,'max_tau':max_tau_y2}
    print("\n  YAML default (eff_x%.1f): stand=%s fell_t=%s pgz_min=%.2f" %
          (best_standing['scale'],yaml_r['stand'],yaml_r['fell_t'],yaml_r['min_pgz']))


# ====== TABLES ======
print("\n" + "=" * 80)
print("TABLE 1: Effort Limit Sweep (target=prepared_pose, G0 kp/kd)")
print("=" * 80)
print("  %-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %6s %8s %8s" %
      ("eff", "stand", "t(s)", "min_pgz", "av_max", "tau_cmd", "tau_clip", "sat%",
       "knee_eff", "hip_eff", "knee_min", "hit", "nf", "first_sat"))
for r in results:
    fs = r['first_sat'][:30] if r['first_sat'] else "none"
    print("  x%-6.1f %8s %8.1f %8.3f %8.1f %8.0f %8.0f %7.0f%% %8.0f %8.0f %8.2f %6d %8d %8s" %
          (r['scale'], r['stand'], r['stand_time'], r['min_pgz'], r['max_av'],
           r['max_tau_cmd'], r['max_tau_clip'], r['max_sat']*100,
           r['effort_knee'], r['effort_hip'], min(r['knee_min']), r['knee_hit_n'],
           r['nf_tot'], fs))

print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)
best = max(results, key=lambda r: r['stand_time'])
print("  1. 36 Nm bottleneck: %s" % ("YES" if not results[0]['stand'] and best['stand'] else "NO"))
print("  2. Best effort scale: x%.1f (knee=%.0fNm hip=%.0fNm)" %
      (best['scale'], best['effort_knee'], best['effort_hip']))
print("  3. Prepared pose stands: %s" % ("YES" if best['stand'] else "NO"))
print("  4. Stand time at best: %.1fs" % best['stand_time'])
print("  5. If x1.0 fails but x2.0+ succeeds: effort limit IS the bottleneck")
print("  6. If x5.0 still fails: check posture/contact/controller beyond torque")

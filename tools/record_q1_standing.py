"""
Q1 standing test — offscreen recording to MP4.
"""
import sys, os, numpy as np, cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from isaacgym import gymapi, gymtorch
import torch


def main():
    conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"))
    rc = conf.robot

    gym = gymapi.acquire_gym()

    sp = gymapi.SimParams()
    sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.physx.solver_type = 1; sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1; sp.physx.num_threads = 10
    sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0,0,1)
    pp.static_friction = 1.0; pp.dynamic_friction = 1.0
    gym.add_ground(sim, pp)

    # load asset
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

    # PD
    defaults = dict(rc.init_state.default_joint_angles)
    stiff = dict(rc.control.stiffness); damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, dtype=np.float32); dg_np = np.zeros(ndof, dtype=np.float32)
    for i, n in enumerate(dof_names):
        for k in stiff:
            if k in n: pg_np[i]=stiff[k]; dg_np[i]=damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0"); dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor([float(defaults.get(n,0)) for n in dof_names], device="cuda:0")

    # env
    root_z = float(rc.init_state.pos[2])
    env = gym.create_env(sim, gymapi.Vec3(0,0,0), gymapi.Vec3(0,0,0), 1)
    ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0,0,root_z)), "q1", -1, 0, 0)
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(ndof): dof_st[i]["pos"]=float(default_t[i].item()); dof_st[i]["vel"]=0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    # ---- offscreen camera ----
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280; cam_props.height = 720
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env,
                            gymapi.Vec3(2.5, 2.5, 1.5),
                            gymapi.Vec3(0.0, 0.0, 0.4))

    gym.prepare_sim(sim)

    # tensors
    rt = gym.acquire_actor_root_state_tensor(sim)
    root_all = gymtorch.wrap_tensor(rt).view(1,-1,13); robot = root_all[:,0,:]
    dt_t = gym.acquire_dof_state_tensor(sim)
    dof = gymtorch.wrap_tensor(dt_t).view(1,-1,2)
    cf_t = gym.acquire_net_contact_force_tensor(sim)
    cf = gymtorch.wrap_tensor(cf_t).view(1,-1,3)

    # ---- FIX: zero dirty velocities from prepare_sim ----
    robot[0, 0:3] = torch.tensor([0.0, 0.0, root_z], device="cuda:0")
    robot[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0")
    robot[0, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_all.reshape(-1, 13)))
    for i in range(ndof):
        dof[0, i, 0] = float(defaults.get(dof_names[i], 0))
        dof[0, i, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof.reshape(-1, 2)))
    gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim)
    tau_lim = torch.tensor([gym.get_actor_dof_properties(env,ah)["effort"][i].item() for i in range(ndof)], device="cuda:0")
    Lf = body_names.index("left_ankle_roll_link")
    Rf = body_names.index("right_ankle_roll_link")

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = str(PROJECT_ROOT / "q1_standing.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, 50, (1280, 720))
    print(f"Recording to {video_path}")
    print(f"Q1: {ndof} DOF, density={ac.density}, root_z={root_z}")
    print(f"Press Ctrl+C to stop early\n")

    step = 0
    try:
        for _ in range(2500):  # 12.5s at 200Hz sim = 2500 steps
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            if step % 4 == 0:
                gym.refresh_dof_state_tensor(sim)
                gym.refresh_actor_root_state_tensor(sim)
                gym.refresh_net_contact_force_tensor(sim)

                dp = dof[0,:,0]; dv = dof[0,:,1]
                tau = pg*(default_t-dp) - dg*dv
                tau = torch.clamp(tau, -tau_lim, tau_lim)
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

            # render every 4th sim step (50 fps video)
            if step % 4 == 0:
                # get latest robot position for camera tracking
                gym.refresh_actor_root_state_tensor(sim)
                rx = robot[0, 0].item()
                ry = robot[0, 1].item()
                rz = robot[0, 2].item()
                gym.set_camera_location(cam_handle, env,
                    gymapi.Vec3(rx + 2.5, ry + 2.5, max(rz + 1.2, 1.5)),
                    gymapi.Vec3(rx, ry, rz + 0.3))

                gym.step_graphics(sim)
                gym.render_all_camera_sensors(sim)
                color_raw = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
                if color_raw is not None and color_raw.size > 0:
                    img = np.asarray(color_raw)
                    h, w = cam_props.height, cam_props.width
                    # IsaacGym may return packed int32 or flat RGBA bytes
                    if img.ndim == 1:
                        # Packed int32: reshape to (h, w), then decode
                        if img.size == h * w:
                            packed = img.astype(np.uint32).reshape(h, w)
                            rgba = np.zeros((h, w, 4), dtype=np.uint8)
                            rgba[..., 0] = (packed >> 16) & 0xFF
                            rgba[..., 1] = (packed >> 8) & 0xFF
                            rgba[..., 2] = packed & 0xFF
                            rgba[..., 3] = 255
                            img = rgba
                        else:
                            img = img.reshape(h, w, 4)
                    elif img.ndim == 2 and img.shape[1] == w * 4:
                        img = img.reshape(h, w, 4)
                    elif img.ndim == 3 and img.shape[-1] == 4:
                        pass
                    if img.ndim == 3 and img.shape[-1] >= 3:
                        bgr = cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                        writer.write(bgr)

                cstep = step // 4
                if cstep % 50 == 0:
                    t = cstep * 0.02
                    rz = robot[0,2].item()
                    qw,qx,qy,qz = robot[0,6].item(),robot[0,3].item(),robot[0,4].item(),robot[0,5].item()
                    pgz = 1.0-2.0*(qx*qx+qy*qy)
                    Lfz = cf[0,Lf,2].item(); Rfz = cf[0,Rf,2].item()
                    print(f"t={t:5.1f}s  rz={rz:+8.3f}  pgz={pgz:+8.3f}  fz=[{Lfz:+7.0f},{Rfz:+7.0f}]  |tau|={abs(tau.cpu().numpy()).max():+5.1f}")

            step += 1
    except KeyboardInterrupt:
        print("\nStopped early.")

    writer.release()
    print(f"\nVideo saved: {video_path}")
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

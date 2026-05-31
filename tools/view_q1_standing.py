"""
Q1 standing test with IsaacGym viewer. Press ESC to quit.
"""
import sys, numpy as np
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
    sp.dt = 0.005
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = 10
    sp.physx.use_gpu = True
    sp.use_gpu_pipeline = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("sim failed")

    # ground
    pp = gymapi.PlaneParams()
    pp.normal = gymapi.Vec3(0, 0, 1)
    pp.static_friction = 1.0
    pp.dynamic_friction = 1.0
    gym.add_ground(sim, pp)

    # load asset
    ac = rc.asset
    opts = gymapi.AssetOptions()
    for k in ["collapse_fixed_joints", "replace_cylinder_with_capsule", "flip_visual_attachments",
              "fix_base_link", "density", "angular_damping", "linear_damping",
              "max_angular_velocity", "max_linear_velocity", "armature", "thickness"]:
        setattr(opts, k, getattr(ac, k))
    opts.default_dof_drive_mode = ac.default_dof_drive_mode

    asset_root = str(PROJECT_ROOT / "humanoidverse/data/robots/q1")
    asset = gym.load_asset(sim, asset_root, "q1_22dof.urdf", opts)
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)
    ndof = len(dof_names)
    print(f"Loaded Q1: {ndof} DOF, {len(body_names)} bodies")

    # build PD gains & default pose
    defaults = dict(rc.init_state.default_joint_angles)
    stiff = dict(rc.control.stiffness)
    damp = dict(rc.control.damping)
    pg_np = np.zeros(ndof, dtype=np.float32)
    dg_np = np.zeros(ndof, dtype=np.float32)
    for i, n in enumerate(dof_names):
        for k in stiff:
            if k in n:
                pg_np[i] = stiff[k]; dg_np[i] = damp[k]; break
    pg = torch.tensor(pg_np, device="cuda:0")
    dg = torch.tensor(dg_np, device="cuda:0")
    default_t = torch.tensor([float(defaults.get(n, 0)) for n in dof_names], device="cuda:0")

    # create env
    root_z = float(rc.init_state.pos[2])
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)
    pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, root_z))
    ah = gym.create_actor(env, asset, pose, "q1", -1, 0, 0)

    # set DOF before prepare_sim
    dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
    for i in range(ndof):
        dof_st[i]["pos"] = float(default_t[i].item())
        dof_st[i]["vel"] = 0.0
    gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

    gym.prepare_sim(sim)

    # ---- VIEWER ----
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    print("Viewer opened. Press ESC to quit, SPACE to pause.\n")

    # tensors
    rt = gym.acquire_actor_root_state_tensor(sim)
    root_all = gymtorch.wrap_tensor(rt).view(1, -1, 13)
    robot = root_all[:, 0, :]
    dt_t = gym.acquire_dof_state_tensor(sim)
    dof = gymtorch.wrap_tensor(dt_t).view(1, -1, 2)
    cf_t = gym.acquire_net_contact_force_tensor(sim)
    cf = gymtorch.wrap_tensor(cf_t).view(1, -1, 3)
    rigid_t = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_t)
    num_b = gym.get_actor_rigid_body_count(env, ah)
    npe = rigid.shape[0] // 1
    rpos = rigid.view(1, npe, 13)[..., :num_b, 0:3]

    tau_lim = torch.tensor([gym.get_actor_dof_properties(env, ah)["effort"][i].item()
                            for i in range(ndof)], device="cuda:0")

    Lf = body_names.index("left_ankle_roll_link")
    Rf = body_names.index("right_ankle_roll_link")

    # subscribe to keyboard
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_pause")

    paused = False
    step = 0

    while not gym.query_viewer_has_closed(viewer):
        # check events
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT" and evt.value > 0:
                gym.destroy_viewer(viewer)
                gym.destroy_sim(sim)
                return
            elif evt.action == "toggle_pause" and evt.value > 0:
                paused = not paused
                print(f"  {'PAUSED' if paused else 'RUNNING'}")

        if not paused:
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            if step % 4 == 0:
                gym.refresh_dof_state_tensor(sim)
                gym.refresh_actor_root_state_tensor(sim)
                gym.refresh_rigid_body_state_tensor(sim)
                gym.refresh_net_contact_force_tensor(sim)

                dp = dof[0, :, 0]; dv = dof[0, :, 1]
                tau = pg * (default_t - dp) - dg * dv
                tau = torch.clamp(tau, -tau_lim, tau_lim)
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau.contiguous()))

                cstep = step // 4
                if cstep % 50 == 0:
                    t = cstep * 0.02
                    rz = robot[0, 2].item()
                    qw, qx, qy, qz = robot[0, 6].item(), robot[0, 3].item(), robot[0, 4].item(), robot[0, 5].item()
                    pgz = 1.0 - 2.0 * (qx*qx + qy*qy)
                    Lfz = cf[0, Lf, 2].item()
                    Rfz = cf[0, Rf, 2].item()
                    tau_np = tau.cpu().numpy()
                    print(f"t={t:5.1f}s  rz={rz:+8.3f}  pgz={pgz:+8.3f}  "
                          f"L_fz={Lfz:+8.1f}  R_fz={Rfz:+8.1f}  |tau|={abs(tau_np).max():+6.1f}")

            step += 1

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

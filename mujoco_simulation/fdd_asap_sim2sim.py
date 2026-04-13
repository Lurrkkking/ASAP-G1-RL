import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime
import yaml
from scipy.spatial.transform import Rotation as R


def read_conf(config_file):
    cfg = SimpleNamespace()
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.cycle_time = config["cycle_time"]
    cfg.frame_stack = config["frame_stack"]

    cfg.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]

    cfg.clip_observations = config["clip_observations"]

    cfg.kps = np.array(config["kps"], dtype=np.float32)
    cfg.kds = np.array(config["kds"], dtype=np.float32)
    cfg.kp_scale = float(config.get("kp_scale", 1.0))
    cfg.kd_scale = float(config.get("kd_scale", 1.0))
    cfg.kps = cfg.kps * cfg.kp_scale
    cfg.kds = cfg.kds * cfg.kd_scale

    cfg.xml_path = config["xml_path"]
    cfg.num_actions = config["num_actions"]
    cfg.policy_path = config["policy_path"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.clip_actions = config["clip_actions"]
    cfg.action_scale = config["action_scale"]
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
    cfg.time_offset = float(config.get("time_offset", 0.0))
    cfg.phase_wrap = bool(config.get("phase_wrap", False))
    cfg.stop_at_motion_end = bool(config.get("stop_at_motion_end", True))
    cfg.action_filter_alpha = float(config.get("action_filter_alpha", 1.0))
    cfg.target_pos_rate_limit = float(config.get("target_pos_rate_limit", 0.0))
    cfg.safety_check_nonfinite = bool(config.get("safety_check_nonfinite", True))
    cfg.max_abs_qacc = float(config.get("max_abs_qacc", 5.0e4))
    cfg.solver_iterations = int(config.get("solver_iterations", 100))
    cfg.solver_ls_iterations = int(config.get("solver_ls_iterations", 50))

    return cfg


def get_mujoco_data(data):
    out = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(quat)
    # MuJoCo free-joint angular velocity is world-frame; convert to body local frame.
    base_angvel = r.apply(dq[3:6], inverse=True).astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

    out["mujoco_dof_pos"] = q[7:]
    out["mujoco_dof_vel"] = dq[6:]
    out["mujoco_base_angvel"] = base_angvel
    out["mujoco_gvec"] = gvec
    return out


def update_hist_obs(hist_dict, obs_sigle):
    slices = {
        "actions": slice(0, 23),
        "base_ang_vel": slice(23, 26),
        "dof_pos": slice(26, 49),
        "dof_vel": slice(49, 72),
        "projected_gravity": slice(72, 75),
        "ref_motion_phase": slice(75, 76),
    }

    for key, slc in slices.items():
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_sigle[0, slc], arr))
        hist_dict[key] = arr

    hist_obs = np.concatenate([hist_dict[key].reshape(1, -1) for key in hist_dict.keys()], axis=1).astype(np.float32)
    return hist_obs


def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]

    sim_time = cfg.time_offset + (counter + 1) * cfg.simulation_dt
    if cfg.phase_wrap:
        ref_motion_phase = (sim_time % cfg.cycle_time) / cfg.cycle_time
    else:
        phase_time = min(sim_time, cfg.cycle_time - 1e-6) if cfg.stop_at_motion_end else sim_time
        ref_motion_phase = np.clip(phase_time / cfg.cycle_time, 0, 1)
    num_obs_input = (cfg.frame_stack + 1) * cfg.num_single_obs

    obs_all = np.zeros([1, num_obs_input], dtype=np.float32)
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    obs_sigle[0, 0:23] = action
    obs_sigle[0, 23:26] = mujoco_base_angvel * cfg.obs_scale_base_ang_vel
    obs_sigle[0, 26:49] = (mujoco_dof_pos - cfg.default_dof_pos) * cfg.obs_scale_dof_pos
    obs_sigle[0, 49:72] = mujoco_dof_vel * cfg.obs_scale_dof_vel
    obs_sigle[0, 72:75] = mujoco_gvec * cfg.obs_scale_gvec
    obs_sigle[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion

    obs_all[0, 0:23] = obs_sigle[0, 0:23].copy()
    obs_all[0, 23:26] = obs_sigle[0, 23:26].copy()
    obs_all[0, 26:49] = obs_sigle[0, 26:49].copy()
    obs_all[0, 49:72] = obs_sigle[0, 49:72].copy()
    obs_all[0, 72:376] = hist_obs_c[0] * cfg.obs_scale_hist
    obs_all[0, 376:379] = obs_sigle[0, 72:75].copy()
    obs_all[0, 379] = obs_sigle[0, 75].copy()

    hist_obs_cat = update_hist_obs(hist_dict, obs_sigle)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    return obs_all, hist_obs_cat


def pd_control(target_pos, dof_pos, target_vel, dof_vel, cfg):
    return (target_pos - dof_pos) * cfg.kps + (target_vel - dof_vel) * cfg.kds


def run_mujoco(cfg, headless=False, video_out="", width=1280, height=720, video_fps=50,
               cam_follow_base=True, cam_distance=3.0, cam_azimuth=90.0, cam_elevation=-45.0,
               cam_lookat_z=0.82):
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt
    if cfg.solver_iterations > 0:
        model.opt.iterations = cfg.solver_iterations
    if hasattr(model.opt, "ls_iterations") and cfg.solver_ls_iterations > 0:
        model.opt.ls_iterations = cfg.solver_ls_iterations
    data.qpos[-cfg.num_actions:] = cfg.default_dof_pos
    mujoco.mj_step(model, data)

    viewer = None
    renderer = None
    frames = []
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = float(cam_distance)
    cam.azimuth = float(cam_azimuth)
    cam.elevation = float(cam_elevation)

    if video_out:
        renderer = mujoco.Renderer(model, height=height, width=width)
    elif not headless:
        try:
            import mujoco_viewer
        except Exception as exc:
            raise RuntimeError("mujoco_viewer not available; use --headless --video-out to render offscreen") from exc
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = float(cam_distance)
        viewer.cam.azimuth = float(cam_azimuth)
        viewer.cam.elevation = float(cam_elevation)
        viewer.cam.lookat[:] = np.array([0.0, -0.25, float(cam_lookat_z)])

    policy = onnxruntime.InferenceSession(cfg.policy_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    target_dof_pos = cfg.default_dof_pos.copy()
    action = np.zeros(cfg.num_actions, dtype=np.float32)
    raw_action = np.zeros(cfg.num_actions, dtype=np.float32)

    hist_dict = {
        "actions": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
        "base_ang_vel": np.zeros((cfg.frame_stack, 3), dtype=np.double),
        "dof_pos": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
        "dof_vel": np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
        "projected_gravity": np.zeros((cfg.frame_stack, 3), dtype=np.double),
        "ref_motion_phase": np.zeros((cfg.frame_stack, 1), dtype=np.double),
    }
    history_keys = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "ref_motion_phase"]
    hist_obs = [hist_dict[key].reshape(1, -1) for key in history_keys]
    hist_obs_c = np.concatenate(hist_obs, axis=1)

    counter = 0
    stop_reason = "duration_reached"
    sim_steps = int(cfg.simulation_duration / cfg.simulation_dt)
    render_every_steps = max(1, int(round(1.0 / max(cfg.simulation_dt * float(video_fps), 1e-9))))
    for _ in range(sim_steps):
        mj = get_mujoco_data(data)
        tau = pd_control(target_dof_pos, mj["mujoco_dof_pos"], np.zeros_like(cfg.kds), mj["mujoco_dof_vel"], cfg)
        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        if cfg.safety_check_nonfinite:
            if (not np.isfinite(data.qpos).all()) or (not np.isfinite(data.qvel).all()) or (not np.isfinite(data.qacc).all()):
                stop_reason = "numerical_instability_nonfinite"
                print(f"[WARN] Non-finite state detected at step={counter}; stop rollout")
                break
            max_abs_qacc = float(np.max(np.abs(data.qacc)))
            if max_abs_qacc > cfg.max_abs_qacc:
                stop_reason = "numerical_instability_qacc"
                print(
                    f"[WARN] Excessive qacc detected at step={counter}: "
                    f"max_abs_qacc={max_abs_qacc:.3e} > threshold={cfg.max_abs_qacc:.3e}; stop rollout"
                )
                break

        counter += 1
        if cfg.stop_at_motion_end and (cfg.time_offset + counter * cfg.simulation_dt) >= cfg.cycle_time:
            stop_reason = "motion_end"
            print(
                f"[INFO] Reached motion end at t={cfg.time_offset + counter * cfg.simulation_dt:.3f}s "
                f"(cycle_time={cfg.cycle_time:.3f}s); stop rollout"
            )
            break
        if counter % cfg.control_decimation == 0:
            obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mj, action, counter, cfg)
            raw_action = policy.run([output_name], {input_name: obs_buff})[0]
            raw_action = np.asarray(raw_action).reshape(-1).astype(np.float32)
            raw_action = np.clip(raw_action, -cfg.clip_actions, cfg.clip_actions)
            action = cfg.action_filter_alpha * raw_action + (1.0 - cfg.action_filter_alpha) * action
            target_from_action = action * cfg.action_scale + cfg.default_dof_pos
            if cfg.target_pos_rate_limit > 0:
                max_delta = cfg.target_pos_rate_limit * cfg.simulation_dt * cfg.control_decimation
                target_dof_pos = np.clip(target_from_action, target_dof_pos - max_delta, target_dof_pos + max_delta)
            else:
                target_dof_pos = target_from_action

        if cam_follow_base:
            base_x = float(data.qpos[0])
            base_y = float(data.qpos[1])
            cam.lookat[:] = np.array([base_x, base_y, float(cam_lookat_z)], dtype=np.float64)
            if viewer is not None:
                viewer.cam.lookat[:] = cam.lookat

        if renderer is not None:
            if counter % render_every_steps == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
        elif viewer is not None:
            viewer.render()

    if viewer is not None:
        viewer.close()

    if video_out:
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            raise RuntimeError("imageio is required to save mp4 in headless mode") from exc
        out = Path(video_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), frames, fps=int(video_fps))
        print(f"[INFO] captured_frames={len(frames)} render_stride={render_every_steps} sim_steps={sim_steps}")
        print(f"[DONE] saved video: {out.resolve()}")

    print(f"[INFO] stop_reason={stop_reason} steps_executed={counter}")


def _default_config_path():
    cwd_candidate = Path(os.getcwd()) / "g1_config" / "mujoco_config.yaml"
    if cwd_candidate.is_file():
        return str(cwd_candidate)
    script_candidate = Path(__file__).resolve().parent / "g1_config" / "mujoco_config.yaml"
    return str(script_candidate)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo ONNX sim2sim runner")
    parser.add_argument("--config", type=str, default=_default_config_path())
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--xml-path", type=str, default="")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video-out", type=str, default="")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cam-follow-base", action="store_true", default=True)
    parser.add_argument("--no-cam-follow-base", action="store_true")
    parser.add_argument("--cam-distance", type=float, default=3.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--cam-elevation", type=float, default=-45.0)
    parser.add_argument("--cam-lookat-z", type=float, default=0.82)
    args = parser.parse_args()

    if args.no_cam_follow_base:
        args.cam_follow_base = False

    cfg = read_conf(args.config)

    cfg_dir = Path(args.config).resolve().parent
    if not os.path.isabs(cfg.xml_path):
        cfg.xml_path = str((cfg_dir / cfg.xml_path).resolve())
    if not os.path.isabs(cfg.policy_path):
        cfg.policy_path = str((cfg_dir / cfg.policy_path).resolve())

    if args.xml_path:
        cfg.xml_path = args.xml_path
    if args.policy_path:
        cfg.policy_path = args.policy_path
    if args.duration > 0:
        cfg.simulation_duration = float(args.duration)

    if not os.path.isfile(cfg.xml_path):
        raise FileNotFoundError(f"xml_path not found: {cfg.xml_path}")
    if not os.path.isfile(cfg.policy_path):
        raise FileNotFoundError(f"policy_path not found: {cfg.policy_path}")

    print(f"[INFO] config={args.config}")
    print(f"[INFO] xml_path={cfg.xml_path}")
    print(f"[INFO] policy_path={cfg.policy_path}")
    print(
        f"[INFO] duration={cfg.simulation_duration}s dt={cfg.simulation_dt} decim={cfg.control_decimation} "
        f"cycle_time={cfg.cycle_time}s time_offset={cfg.time_offset}s phase_wrap={int(cfg.phase_wrap)} "
        f"stop_at_motion_end={int(cfg.stop_at_motion_end)} kp_scale={cfg.kp_scale:.3f} kd_scale={cfg.kd_scale:.3f} "
        f"action_filter_alpha={cfg.action_filter_alpha:.3f} target_pos_rate_limit={cfg.target_pos_rate_limit:.3f} "
        f"solver_iterations={cfg.solver_iterations} solver_ls_iterations={cfg.solver_ls_iterations}"
    )

    run_mujoco(
        cfg,
        headless=args.headless,
        video_out=args.video_out,
        width=args.width,
        height=args.height,
        video_fps=args.video_fps,
        cam_follow_base=args.cam_follow_base,
        cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_lookat_z=args.cam_lookat_z,
    )
    print("-----done------")


if __name__ == "__main__":
    main()

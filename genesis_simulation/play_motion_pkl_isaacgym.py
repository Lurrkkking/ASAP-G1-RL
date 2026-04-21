import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.append("/root/autodl-tmp/ASAP")

# Isaac Gym's gymtorch extension needs the ninja executable on PATH. The
# conda env may only have the Python ninja package installed.
if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"
try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

from humanoidverse.simulator.isaacgym.isaacgym import IsaacGym
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
import torch


def build_config(args):
    robot_base_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/robot/robot_base.yaml")
    cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml")
    sim_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/simulator/isaacgym.yaml")
    terrain_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/terrain/terrain_locomotion_plane.yaml")
    domain_rand_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/domain_rand/domain_rand_base.yaml")
    rewards_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml")
    env_cfg = OmegaConf.load("/root/autodl-tmp/ASAP/humanoidverse/config/env/motion_tracking.yaml")

    robot_cfg = OmegaConf.merge(robot_base_cfg.robot, cfg.robot)

    merged = OmegaConf.create(
        {
            "simulator": sim_cfg.simulator,
            "robot": robot_cfg,
            "terrain": terrain_cfg.terrain,
            "domain_rand": domain_rand_cfg.domain_rand,
            "rewards": rewards_cfg.rewards,
            "termination_scales": env_cfg.env.config.termination_scales,
            "num_envs": 1,
            "headless": bool(args.headless),
            "save_rendering_dir": str(Path(args.out).resolve().parent) if args.out else None,
            "auto_record": bool(args.out),
            "auto_record_num_frames": -1,
            "offscreen_record": bool(args.out),
            "offscreen_record_width": args.width,
            "offscreen_record_height": args.height,
            "offscreen_record_fps": args.record_fps,
            "experiment_name": Path(args.out).stem if args.out else "motion_playback",
        }
    )

    merged.robot.asset.asset_root = "/root/autodl-tmp/ASAP/humanoidverse/data/robots"
    merged.robot.asset.urdf_file = "g1/g1_29dof_anneal_23dof.urdf"
    merged.robot.motion.motion_file = args.motion_file
    merged.robot.motion.asset.assetRoot = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1"
    for key in list(merged.domain_rand.keys()):
        if isinstance(merged.domain_rand[key], bool):
            merged.domain_rand[key] = False
    return merged


def build_base_init_state(robot_cfg):
    pos = robot_cfg.init_state.pos
    rot = robot_cfg.init_state.rot
    lin_vel = robot_cfg.init_state.lin_vel
    ang_vel = robot_cfg.init_state.ang_vel
    return torch.tensor(list(pos) + list(rot) + list(lin_vel) + list(ang_vel), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Play a motion pkl in Isaac Gym without training.")
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--record-fps", type=int, default=50)
    parser.add_argument("--out", default="", help="Output mp4 path; default saves under /root/autodl-tmp/ASAP/logs_eval")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.out:
        motion_stem = Path(args.motion_file).stem
        args.out = f"/root/autodl-tmp/ASAP/logs_eval/{motion_stem}.mp4"

    cfg = build_config(args)
    device = args.device if torch.cuda.is_available() else "cpu"

    sim = IsaacGym(cfg, device=device)
    # Offscreen camera sensors still need a graphics device. We do not create
    # an interactive viewer, so this remains non-interactive while recording.
    sim.set_headless(False if args.out else args.headless)
    sim.setup()
    sim.setup_terrain("plane")
    sim.load_assets()

    env_origins = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
    base_init_state = build_base_init_state(cfg.robot).to(sim.device)
    sim.create_envs(num_envs=1, env_origins=env_origins, base_init_state=base_init_state)
    sim.prepare_sim()

    motion_cfg = cfg.robot.motion
    ml = MotionLibRobot(motion_cfg, num_envs=1, device=device)
    ml.load_motions(random_sample=False)

    motion_id = torch.tensor([0], dtype=torch.long, device=device)
    motion_length = float(ml.get_motion_length(motion_id)[0].item())
    sim_dt = float(cfg.simulator.config.sim.control_decimation) / float(cfg.simulator.config.sim.fps)
    total_steps = max(1, int(round(motion_length / sim_dt)))
    cfg.auto_record_num_frames = total_steps * args.loops

    print(f"[INFO] motion_file={args.motion_file}")
    print(f"[INFO] motion_length={motion_length:.4f}s sim_dt={sim_dt:.4f}s total_steps={total_steps}")
    print(f"[INFO] mp4_out={args.out}")

    # IsaacGym names the output file internally; rename it at the end to the requested path.
    expected_out = Path(args.out).resolve()

    env_ids = torch.tensor([0], dtype=torch.long, device=sim.device)
    dof_ids = torch.arange(sim.num_dof, dtype=torch.long, device=sim.device)

    for loop_idx in range(args.loops):
        print(f"[INFO] loop {loop_idx + 1}/{args.loops}")
        for step in range(total_steps):
            t = min(args.start_time + step * sim_dt, motion_length)
            motion_t = torch.tensor([t], dtype=torch.float32, device=device)
            offset = torch.zeros((1, 3), dtype=torch.float32, device=device)
            res = ml.get_motion_state(motion_id, motion_t, offset=offset)

            root_pos = res["root_pos"].detach().to(sim.device)
            root_rot = res["root_rot"].detach().to(sim.device)  # xyzw for isaacgym
            root_vel = res["root_vel"].detach().to(sim.device)
            root_ang_vel = res["root_ang_vel"].detach().to(sim.device)
            dof_pos = res["dof_pos"].detach().to(sim.device)
            dof_vel = res["dof_vel"].detach().to(sim.device)

            sim.robot_root_states[env_ids, :3] = root_pos
            sim.robot_root_states[env_ids, 3:7] = root_rot
            sim.robot_root_states[env_ids, 7:10] = root_vel
            sim.robot_root_states[env_ids, 10:13] = root_ang_vel

            flat_dof_ids = env_ids[:, None] * sim.num_dof + dof_ids[None, :]
            sim.dof_state[flat_dof_ids.reshape(-1), 0] = dof_pos.reshape(-1)
            sim.dof_state[flat_dof_ids.reshape(-1), 1] = dof_vel.reshape(-1)

            sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)
            sim.set_dof_state_tensor(env_ids, sim.dof_state)
            sim.simulate_at_each_physics_step()
            sim.refresh_sim_tensors()
            sim.render(sync_frame_time=not args.headless)

    sim.finalize_recording()
    generated = Path(sim.offscreen_video_path) if sim.offscreen_video_path else None
    if generated is not None and generated.is_file() and sim.offscreen_recorded_frames > 0:
        expected_out.parent.mkdir(parents=True, exist_ok=True)
        if generated.resolve() != expected_out:
            generated.replace(expected_out)
        print(f"[DONE] wrote {expected_out}")
    else:
        print(
            "[WARN] playback finished but no valid mp4 was generated "
            f"(frames={sim.offscreen_recorded_frames})"
        )
    print("[DONE] playback finished")


if __name__ == "__main__":
    main()

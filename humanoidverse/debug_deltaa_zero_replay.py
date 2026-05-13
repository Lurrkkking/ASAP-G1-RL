import math
import os
import sys
from pathlib import Path
from typing import Dict, List

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict


DEFAULT_SOURCE_CONFIG_PATH = (
    "/root/autodl-tmp/ASAP/logs/DeltaA_Sanity/"
    "20260511_204929-openloop_gym2gym_cr7_readtest-delta_a-g1_29dof_anneal_23dof/"
    "config.yaml"
)
DEFAULT_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_gym_rollout_with_action.pkl"
)
DEFAULT_NUM_STEPS = 134
REPO_ROOT = Path(__file__).resolve().parents[1]


def disable_domain_rand(cfg):
    if not hasattr(cfg, "domain_rand"):
        return
    for key in list(cfg.domain_rand.keys()):
        value = cfg.domain_rand[key]
        if isinstance(value, bool) and (
            key.startswith("randomize") or key.startswith("push") or key.endswith("noise")
        ):
            cfg.domain_rand[key] = False


def disable_terminations(cfg):
    termination = cfg.env.config.termination
    for key in list(termination.keys()):
        if isinstance(termination[key], bool):
            termination[key] = False


def load_base_config(override_config: OmegaConf) -> OmegaConf:
    debug_cfg = override_config.get("debug_deltaa", OmegaConf.create())
    source_config_path = str(debug_cfg.get("source_config_path", DEFAULT_SOURCE_CONFIG_PATH))
    if not Path(source_config_path).exists():
        raise FileNotFoundError(f"source_config_path does not exist: {source_config_path}")
    cfg = OmegaConf.load(source_config_path)

    motion_path = str(debug_cfg.get("motion_file", DEFAULT_MOTION_PATH))
    num_steps = int(debug_cfg.get("num_steps", DEFAULT_NUM_STEPS))

    with open_dict(cfg):
        cfg.headless = True
        cfg.num_envs = 1
        cfg.checkpoint = None
        cfg.use_wandb = False
        cfg.env.config.headless = True
        cfg.env.config.num_envs = 1
        cfg.env.config.max_episode_length_s = max(float(cfg.env.config.max_episode_length_s), float(num_steps + 10))
        cfg.env.config.noise_to_initial_level = 0
        cfg.env.config.enforce_randomize_motion_start_eval = False
        cfg.env.config.randomize_motion_start_train = False
        cfg.env.config.save_motion = False
        cfg.env.config.robot.motion.motion_file = motion_path
        cfg.env.config.debug_deltaa_num_steps = num_steps
        cfg.env.config.debug_deltaa_motion_file = motion_path
    disable_domain_rand(cfg)
    disable_terminations(cfg)
    return cfg


def tensor_to_np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def get_device(config):
    import torch

    if config.get("device", None):
        return config.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def quat_angle_error(q_ref, q_sim):
    import torch
    from isaac_utils.rotations import quat_conjugate, quat_mul

    q_diff = quat_mul(q_ref, quat_conjugate(q_sim, w_last=True), w_last=True)
    xyz = q_diff[..., :3]
    w = torch.clamp(q_diff[..., 3], -1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, torch.abs(w))
    return angle


def compute_motion_state(env, motion_time: float):
    import torch

    motion_times = torch.full((env.num_envs,), float(motion_time), dtype=torch.float32, device=env.device)
    return env._motion_lib.get_motion_state(env.motion_ids, motion_times, offset=env.env_origins)


def set_manual_state(env, motion_time: float, action_time: float):
    import torch
    from isaac_utils.rotations import quat_rotate_inverse

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    motion_res = compute_motion_state(env, motion_time)

    env.simulator.robot_root_states[env_ids, :3] = motion_res["root_pos"]
    env.simulator.robot_root_states[env_ids, 3:7] = motion_res["root_rot"]
    env.simulator.robot_root_states[env_ids, 7:10] = motion_res["root_vel"]
    env.simulator.robot_root_states[env_ids, 10:13] = motion_res["root_ang_vel"]
    env.simulator.dof_pos[env_ids] = motion_res["dof_pos"]
    env.simulator.dof_vel[env_ids] = motion_res["dof_vel"]

    env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
    env.simulator.set_dof_state_tensor(env_ids, env.simulator.dof_state)
    env._refresh_sim_tensors()

    env.motion_start_times[:] = float(action_time - env.dt)
    env.episode_length_buf.zero_()
    env.last_episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.actions.zero_()
    env.actions_after_delay.zero_()
    env.last_actions.zero_()
    env.last_dof_pos[:] = env.simulator.dof_pos
    env.last_dof_vel[:] = env.simulator.dof_vel
    env.last_root_vel[:] = env.simulator.robot_root_states[:, 7:13]
    env.feet_air_time.zero_()
    env.last_contacts.zero_()
    env.last_contacts_filt.zero_()
    env.feet_air_max_height.zero_()
    env.need_to_refresh_envs.zero_()
    env.history_handler.reset(env_ids)
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.simulator.robot_root_states[:, 7:10], w_last=True)
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.simulator.robot_root_states[:, 10:13], w_last=True)
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec, w_last=True)
    return motion_res


def compute_error_row(env, target_motion_time: float, step_idx: int, mode: str) -> Dict[str, float]:
    import torch

    motion_res = compute_motion_state(env, target_motion_time)

    root_pos_error = torch.linalg.norm(motion_res["root_pos"][0] - env.simulator.robot_root_states[0, :3])
    root_rot_error = quat_angle_error(
        motion_res["root_rot"][0:1],
        env.simulator.robot_root_states[0:1, 3:7],
    )[0]
    dof_pos_error = torch.linalg.norm(motion_res["dof_pos"][0] - env.simulator.dof_pos[0])
    dof_vel_error = torch.linalg.norm(motion_res["dof_vel"][0] - env.simulator.dof_vel[0])

    body_pos_err = motion_res["rg_pos"][0] - env.simulator._rigid_body_pos[0]
    body_pos_mpjpe = torch.linalg.norm(body_pos_err, dim=-1).mean()

    body_rot_err = quat_angle_error(motion_res["rb_rot"][0], env.simulator._rigid_body_rot[0]).mean()
    body_vel_error = torch.linalg.norm(
        motion_res["body_vel"][0] - env.simulator._rigid_body_vel[0], dim=-1
    ).mean()
    body_ang_vel_error = torch.linalg.norm(
        motion_res["body_ang_vel"][0] - env.simulator._rigid_body_ang_vel[0], dim=-1
    ).mean()

    total_diff_norm = torch.sqrt(
        root_pos_error.square()
        + root_rot_error.square()
        + dof_pos_error.square()
        + dof_vel_error.square()
        + body_pos_mpjpe.square()
        + body_rot_err.square()
        + body_vel_error.square()
        + body_ang_vel_error.square()
    )

    row = {
        "mode": mode,
        "step": step_idx,
        "target_motion_time": float(target_motion_time),
        "root_pos_error": float(root_pos_error.detach().cpu().item()),
        "root_rot_error": float(root_rot_error.detach().cpu().item()),
        "dof_pos_error": float(dof_pos_error.detach().cpu().item()),
        "dof_vel_error": float(dof_vel_error.detach().cpu().item()),
        "body_pos_mpjpe": float(body_pos_mpjpe.detach().cpu().item()),
        "body_rot_error": float(body_rot_err.detach().cpu().item()),
        "body_vel_error": float(body_vel_error.detach().cpu().item()),
        "body_ang_vel_error": float(body_ang_vel_error.detach().cpu().item()),
        "total_diff_norm": float(total_diff_norm.detach().cpu().item()),
    }
    if hasattr(env, "dif_global_body_pos"):
        row["existing_body_pos_mean"] = float(
            torch.linalg.norm(env.dif_global_body_pos[0], dim=-1).mean().detach().cpu().item()
        )
    else:
        row["existing_body_pos_mean"] = float("nan")
    return row


def print_rows(title: str, rows: List[Dict[str, float]], limit: int = None):
    if limit is not None:
        rows = rows[:limit]
    print(f"\n[{title}]")
    columns = [
        "step",
        "target_motion_time",
        "root_pos_error",
        "root_rot_error",
        "dof_pos_error",
        "dof_vel_error",
        "body_pos_mpjpe",
        "body_rot_error",
        "body_vel_error",
        "body_ang_vel_error",
        "total_diff_norm",
    ]
    header = " ".join(f"{col:>18}" for col in columns)
    print(header)
    for row in rows:
        print(
            " ".join(
                f"{row[col]:18.6f}" if col != "step" else f"{int(row[col]):18d}"
                for col in columns
            )
        )


def print_summary(title: str, rows: List[Dict[str, float]]):
    print(f"\n[{title}_SUMMARY]")
    metrics = [
        "root_pos_error",
        "root_rot_error",
        "dof_pos_error",
        "dof_vel_error",
        "body_pos_mpjpe",
        "body_rot_error",
        "body_vel_error",
        "body_ang_vel_error",
        "total_diff_norm",
    ]
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        print(
            f"{metric:>24} mean={values.mean():.6f} max={values.max():.6f} "
            f"min={values.min():.6f}"
        )


def print_env_info(env, num_probe_steps: int = 5):
    import torch

    probe_times = torch.arange(num_probe_steps, device=env.device, dtype=torch.float32) * env.dt
    motion_ids = env.motion_ids[:1].repeat(num_probe_steps)
    motion_len = env._motion_lib._motion_lengths[motion_ids]
    num_frames = env._motion_lib._motion_num_frames[motion_ids]
    motion_dt = env._motion_lib._motion_dt[motion_ids]
    frame_idx0, _, _ = env._motion_lib._calc_frame_blend(probe_times, motion_len, num_frames, motion_dt)
    f0l = frame_idx0 + env._motion_lib.length_starts[motion_ids]
    motion_actions = env._motion_lib._motion_actions[f0l]

    print("\n[ENV_INFO]")
    print(f"motion_file={env.config.debug_deltaa_motion_file}")
    print(f"dt={float(env.dt):.6f}")
    print(f"sim_dt={float(env.sim_dt):.6f}")
    print(f"control_decimation={int(env.config.simulator.config.sim.control_decimation)}")
    print(f"action_scale={float(env.config.robot.control.action_scale):.6f}")
    print(f"action_clip_value={float(env.config.robot.control.action_clip_value):.6f}")
    print(f"p_gains[:8]={tensor_to_np(env.p_gains[:8]).tolist()}")
    print(f"d_gains[:8]={tensor_to_np(env.d_gains[:8]).tolist()}")
    print(f"motion_times_first5={tensor_to_np(probe_times).tolist()}")
    print(f"frame_idx_first5={tensor_to_np(frame_idx0).astype(int).tolist()}")
    print(f"motion_action_first5_dim8={tensor_to_np(motion_actions[:, :8]).tolist()}")


def run_continuous_zero_delta_replay(env, num_steps: int) -> List[Dict[str, float]]:
    import torch

    set_manual_state(env, motion_time=0.0, action_time=0.0)
    rows = []
    zero_action = torch.zeros((env.num_envs, env.dim_actions), dtype=torch.float32, device=env.device)
    for step_idx in range(num_steps):
        _, _, dones, _ = env.step({"actions": zero_action})
        target_motion_time = float((step_idx + 1) * env.dt)
        rows.append(compute_error_row(env, target_motion_time, step_idx, "continuous"))
        if bool(dones[0].item()):
            print(f"[continuous_zero_delta_replay] done=True at step={step_idx}, continuing because terminations are disabled.")
    return rows


def run_one_step_reset_zero_delta_replay(env, num_steps: int) -> List[Dict[str, float]]:
    import torch

    rows = []
    zero_action = torch.zeros((env.num_envs, env.dim_actions), dtype=torch.float32, device=env.device)
    for step_idx in range(num_steps):
        action_time = float(step_idx * env.dt)
        set_manual_state(env, motion_time=action_time, action_time=action_time)
        env.step({"actions": zero_action})
        target_motion_time = float((step_idx + 1) * env.dt)
        rows.append(compute_error_row(env, target_motion_time, step_idx, "one_step_reset"))
    return rows


def run_pre_step_reset_error(env, num_steps: int) -> List[Dict[str, float]]:
    rows = []
    for step_idx in range(num_steps):
        motion_time = float(step_idx * env.dt)
        set_manual_state(env, motion_time=motion_time, action_time=motion_time)
        rows.append(compute_error_row(env, motion_time, step_idx, "pre_step_reset"))
    return rows


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    simulator_type = None
    config = load_base_config(override_config)
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        raise NotImplementedError("This diagnostic currently supports IsaacGym-style execution only.")
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch
    from humanoidverse.utils import config_utils as _config_utils  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
    import logging

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "debug_deltaa_zero_replay.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(REPO_ROOT)

    pre_process_config(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs

    device = get_device(config)
    env = instantiate(config.env, device=device)

    _ = env.reset_all()
    env.motion_start_times.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.episode_length_buf.zero_()
    env.last_episode_length_buf.zero_()

    print_env_info(env)

    max_pairs = int(env._motion_lib.get_motion_num_steps(env.motion_ids)[0].item()) - 1
    num_steps = min(int(config.env.config.debug_deltaa_num_steps), max_pairs)
    print(f"\n[RUN_INFO]\nnum_steps={num_steps} max_pairs={max_pairs}")

    continuous_rows = run_continuous_zero_delta_replay(env, num_steps)
    print_rows("continuous_zero_delta_replay", continuous_rows)
    print_summary("continuous_zero_delta_replay", continuous_rows)

    pre_step_rows = run_pre_step_reset_error(env, num_steps)
    print_rows("pre_step_reset_error", pre_step_rows)
    print_summary("pre_step_reset_error", pre_step_rows)

    one_step_rows = run_one_step_reset_zero_delta_replay(env, num_steps)
    print_rows("one_step_reset_zero_delta_replay", one_step_rows)
    print_summary("one_step_reset_zero_delta_replay", one_step_rows)


if __name__ == "__main__":
    main()

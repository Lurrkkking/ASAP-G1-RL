import csv
import json
import os
from pathlib import Path
from types import MethodType

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.config_utils import *  # noqa: F401,F403


TERM_REASON_NAMES = {
    0: "none",
    1: "timeout_no_contact",
    2: "wrong_body_contact",
    3: "ball_ground",
    4: "post_contact_window",
    5: "ball_too_high",
    6: "ball_too_far",
    7: "base_env",
}


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _vec(prefix, tensor):
    arr = tensor.detach().float().cpu().numpy().reshape(-1)
    return {f"{prefix}_{axis}": float(arr[i]) for i, axis in enumerate(("x", "y", "z"))}


def _contact_debug(env):
    import torch

    robot_body_forces = torch.norm(env.simulator.contact_forces[:, : env.num_bodies], dim=-1)
    all_body_forces = torch.norm(env.simulator.contact_forces, dim=-1)
    robot_body_pos = env.simulator._rigid_body_pos[:, : env.num_bodies]
    body_dist_to_ball = torch.norm(robot_body_pos - env.ball_pos.unsqueeze(1), dim=-1)
    near_ball = body_dist_to_ball < env.contact_dist_threshold
    contact_like = (robot_body_forces > env.contact_force_threshold) & near_ball
    min_body_dist, min_body_id = torch.min(body_dist_to_ball, dim=1)
    candidate_indices = getattr(
        env,
        "right_foot_candidate_body_indices",
        torch.tensor([env.right_foot_body_idx], dtype=torch.long, device=env.device),
    )
    candidate_body_dist = body_dist_to_ball[:, candidate_indices]
    min_candidate_dist, min_candidate_local_id = torch.min(candidate_body_dist, dim=1)
    min_candidate_body_id = candidate_indices[min_candidate_local_id]

    non_target_contact = contact_like.clone()
    non_target_contact[:, candidate_indices] = False
    wrong_contact_force = robot_body_forces.masked_fill(~non_target_contact, -1.0)
    wrong_force_max, wrong_contact_body_id = torch.max(wrong_contact_force, dim=1)
    max_contact_lambda = torch.max(robot_body_forces.masked_fill(~contact_like, 0.0), dim=1).values
    ball_force_norm = all_body_forces[0, env.ball_body_env_idx]
    right_foot_dist = body_dist_to_ball[0, env.right_foot_body_idx]
    ball_radius = float(getattr(env.simulator, "ball_radius", 0.0))
    min_body_id_int = int(_scalar(min_body_id[0]))

    return {
        "right_foot_contact_force_norm": _scalar(robot_body_forces[0, env.right_foot_body_idx]),
        "right_foot_dist_to_ball": _scalar(right_foot_dist),
        "right_foot_surface_gap_est": _scalar(right_foot_dist) - ball_radius,
        "right_foot_dist_margin_to_gate": _scalar(right_foot_dist) - float(env.contact_dist_threshold),
        "right_foot_near_ball": int(_scalar(near_ball[0, env.right_foot_body_idx])),
        "right_foot_contact_like": int(_scalar(contact_like[0, env.right_foot_body_idx])),
        "contact_force_threshold": float(env.contact_force_threshold),
        "contact_dist_threshold": float(env.contact_dist_threshold),
        "ball_radius": ball_radius,
        "ball_contact_force_norm": _scalar(ball_force_norm),
        "ball_contact_force_threshold": float(getattr(env, "ball_contact_force_threshold", env.contact_force_threshold)),
        "ball_has_contact": int(
            _scalar(ball_force_norm)
            > float(getattr(env, "ball_contact_force_threshold", env.contact_force_threshold))
        ),
        "ball_body_env_idx": int(env.ball_body_env_idx),
        "min_candidate_foot_body_dist_to_ball": _scalar(min_candidate_dist[0]),
        "min_candidate_foot_body_dist_margin_to_gate": _scalar(min_candidate_dist[0]) - float(env.contact_dist_threshold),
        "closest_candidate_foot_body_id": int(_scalar(min_candidate_body_id[0])),
        "closest_candidate_foot_body_name": env.body_names[int(_scalar(min_candidate_body_id[0]))],
        "candidate_foot_near_ball": int(_scalar(min_candidate_dist[0] < env.contact_dist_threshold)),
        "candidate_ball_contact_like": int(
            (
                _scalar(ball_force_norm)
                > float(getattr(env, "ball_contact_force_threshold", env.contact_force_threshold))
            )
            and bool(_scalar(min_candidate_dist[0] < env.contact_dist_threshold))
        ),
        "min_robot_body_dist_to_ball": _scalar(min_body_dist[0]),
        "min_robot_body_dist_margin_to_gate": _scalar(min_body_dist[0]) - float(env.contact_dist_threshold),
        "closest_robot_body_id": min_body_id_int,
        "closest_robot_body_name": env.body_names[min_body_id_int],
        "closest_robot_body_contact_force_norm": _scalar(robot_body_forces[0, min_body_id_int]),
        "max_contact_like_lambda": _scalar(max_contact_lambda[0]),
        "max_robot_contact_force_norm": _scalar(robot_body_forces[0].max()),
        "wrong_contact_force_max": _scalar(wrong_force_max[0]),
        "wrong_contact_body_id": _scalar(wrong_contact_body_id[0]),
    }


def _record_row(env, step, reset_triggered):
    import torch

    ball_pos = env.ball_pos[0]
    ball_lin_vel = env.ball_lin_vel[0]
    right_foot_pos = env.simulator._rigid_body_pos[0, env.right_foot_body_idx]
    right_foot_vel = env.simulator._rigid_body_vel[0, env.right_foot_body_idx]
    base_pos = env.simulator.robot_root_states[0, 0:3]
    dist = torch.norm(ball_pos - right_foot_pos)
    term_reason = int(_scalar(env.term_reason[0]))

    row = {
        "step": int(step),
        "task_phase": int(_scalar(env.task_phase[0])),
        "has_first_contact": int(_scalar(env.has_first_contact[0])),
        "first_contact_is_target": int(_scalar(env.first_contact_is_target[0])),
        "first_contact_body_id": int(_scalar(env.first_contact_body_id[0])),
        "first_contact_time": int(_scalar(env.first_contact_time[0])),
        "term_reason": term_reason,
        "term_reason_name": TERM_REASON_NAMES.get(term_reason, f"unknown_{term_reason}"),
        "reset_triggered": int(bool(reset_triggered)),
        "ball_to_right_foot_distance": _scalar(dist),
        "episode_length": int(_scalar(env.episode_length_buf[0])),
    }
    row.update(_vec("ball_pos", ball_pos))
    row.update(_vec("ball_lin_vel", ball_lin_vel))
    row.update(_vec("right_foot_pos", right_foot_pos))
    row.update(_vec("right_foot_vel", right_foot_vel))
    row.update(_vec("base_pos", base_pos))
    row.update(_contact_debug(env))
    return row


def _scripted_kick_action(env, step):
    import torch

    action = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    names = list(env.dof_names)

    def set_joint(name, target_offset_rad):
        if name in names:
            action[:, names.index(name)] = target_offset_rad / float(env.config.robot.control.action_scale)

    ball_rel_foot = (env.ball_pos[0] - env.simulator._rigid_body_pos[0, env.right_foot_body_idx]).detach()
    lateral_offset = float(ball_rel_foot[1].clamp(-0.08, 0.08).cpu())
    forward_offset = float(ball_rel_foot[0].clamp(0.08, 0.25).cpu())

    # Fast juggling-style upward tap: first chamber the leg under the falling
    # ball, then snap the knee/ankle upward before the ball reaches the ground.
    set_joint("right_hip_roll_joint", -1.5 * lateral_offset)
    set_joint("right_hip_yaw_joint", 0.4 * lateral_offset)

    if step < 2:
        phase = step / 1.0
        set_joint("right_hip_pitch_joint", 0.25 + 0.30 * phase + 0.5 * forward_offset)
        set_joint("right_knee_joint", 0.05 - 0.20 * phase)
        set_joint("right_ankle_pitch_joint", 0.20 + 0.35 * phase)
    elif step < 8:
        phase = (step - 2) / 5.0
        set_joint("right_hip_pitch_joint", 0.55 + 0.40 * phase + 0.3 * forward_offset)
        set_joint("right_knee_joint", -0.15 - 0.10 * phase)
        set_joint("right_ankle_pitch_joint", 0.55 + 0.20 * phase)
    elif step < 18:
        phase = (step - 8) / 9.0
        set_joint("right_hip_pitch_joint", 0.95 - 0.45 * phase)
        set_joint("right_knee_joint", -0.25 + 0.25 * phase)
        set_joint("right_ankle_pitch_joint", 0.75 - 0.35 * phase)
    else:
        decay = max(0.0, 1.0 - (step - 18) / 18.0)
        set_joint("right_hip_pitch_joint", 0.15 * decay)
        set_joint("right_knee_joint", 0.10 * decay)
        set_joint("right_ankle_pitch_joint", 0.20 * decay)

    return action


def _make_action(env, mode, step):
    import torch

    if mode == "zero_action":
        return torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    if mode == "scripted_kick":
        return _scripted_kick_action(env, step)
    raise ValueError(f"Unknown debug mode: {mode}")


def _save_table(rows, out_dir, mode):
    csv_path = out_dir / f"{mode}_rollout.csv"
    npz_path = out_dir / f"{mode}_rollout.npz"
    if not rows:
        raise RuntimeError("No rollout rows were recorded.")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric = {}
    for key in rows[0].keys():
        if isinstance(rows[0][key], str):
            continue
        numeric[key] = np.asarray([row[key] for row in rows])
    np.savez(npz_path, **numeric)
    return csv_path, npz_path


def _save_plots(rows, out_dir, mode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = np.asarray([r["step"] for r in rows])
    series = {
        "ball_z": np.asarray([r["ball_pos_z"] for r in rows]),
        "right_foot_z": np.asarray([r["right_foot_pos_z"] for r in rows]),
        "ball_to_right_foot_distance": np.asarray([r["ball_to_right_foot_distance"] for r in rows]),
        "ball_vz": np.asarray([r["ball_lin_vel_z"] for r in rows]),
    }

    pngs = []
    for name, values in series.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(step, values)
        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{mode}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        pngs.append(path)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, (name, values) in zip(axes, series.items()):
        ax.plot(step, values)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    overview = out_dir / f"{mode}_overview.png"
    fig.tight_layout()
    fig.savefig(overview, dpi=160)
    plt.close(fig)
    pngs.append(overview)
    return pngs


def _summarize(rows, mode):
    first_contact_rows = [r for r in rows if r["has_first_contact"]]
    first = first_contact_rows[0] if first_contact_rows else None
    closest = min(rows, key=lambda r: r["ball_to_right_foot_distance"])
    final = rows[-1]
    summary = {
        "mode": mode,
        "steps_recorded": len(rows),
        "first_contact": bool(first),
        "first_contact_step": None if first is None else int(first["step"]),
        "first_contact_is_target": None if first is None else bool(first["first_contact_is_target"]),
        "min_ball_to_right_foot_distance": float(closest["ball_to_right_foot_distance"]),
        "closest_step": int(closest["step"]),
        "ball_max_z": float(max(r["ball_pos_z"] for r in rows)),
        "final_term_reason": int(final["term_reason"]),
        "final_term_reason_name": final["term_reason_name"],
        "reset_triggered": bool(any(r["reset_triggered"] for r in rows)),
        "closest_state": closest,
    }
    return summary


def _print_summary(summary):
    print("\n================ HitBall Debug Summary ================")
    print(f"mode: {summary['mode']}")
    print(f"steps_recorded: {summary['steps_recorded']}")
    print(f"first_contact: {summary['first_contact']}")
    print(f"first_contact_step: {summary['first_contact_step']}")
    print(f"first_contact_is_target: {summary['first_contact_is_target']}")
    print(f"min_ball_to_right_foot_distance: {summary['min_ball_to_right_foot_distance']:.6f}")
    print(f"ball_max_z: {summary['ball_max_z']:.6f}")
    print(
        "final_term_reason: "
        f"{summary['final_term_reason']} ({summary['final_term_reason_name']})"
    )
    if not summary["first_contact"]:
        c = summary["closest_state"]
        print("closest_no_contact_state:")
        keys = [
            "step",
            "task_phase",
            "ball_to_right_foot_distance",
            "ball_contact_force_norm",
            "ball_contact_force_threshold",
            "ball_has_contact",
            "right_foot_contact_force_norm",
            "right_foot_surface_gap_est",
            "right_foot_dist_margin_to_gate",
            "right_foot_near_ball",
            "right_foot_contact_like",
            "min_candidate_foot_body_dist_to_ball",
            "closest_candidate_foot_body_id",
            "closest_candidate_foot_body_name",
            "candidate_foot_near_ball",
            "candidate_ball_contact_like",
            "min_robot_body_dist_to_ball",
            "closest_robot_body_id",
            "closest_robot_body_name",
            "closest_robot_body_contact_force_norm",
            "ball_pos_x",
            "ball_pos_y",
            "ball_pos_z",
            "right_foot_pos_x",
            "right_foot_pos_y",
            "right_foot_pos_z",
            "ball_lin_vel_z",
            "term_reason",
            "term_reason_name",
            "reset_triggered",
        ]
        for key in keys:
            print(f"  {key}: {c[key]}")
    print("=======================================================\n")


@hydra.main(config_path="humanoidverse/config", config_name="base", version_base="1.1")
def main(config):
    if config.simulator["_target_"].split(".")[-1] == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config

    mode = str(config.get("debug_mode", "zero_action"))
    steps = int(config.get("debug_steps", 160))
    out_dir = Path(str(config.get("debug_out_dir", "logs/hitball_debug")))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=os.environ.get("LOGURU_LEVEL", "WARNING").upper())

    config.num_envs = 1
    config.headless = True
    config.use_wandb = False
    config.auto_load_latest = False
    config.checkpoint = None
    config.offscreen_record = False
    config.auto_record = False
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_debug")

    pre_process_config(config)
    env = instantiate(config=config.env, device=config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    env.reset_all()

    rows = []
    reset_intercept = {"triggered": False, "env_ids": []}
    original_reset = env.reset_envs_idx

    def no_auto_reset(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) > 0:
            reset_intercept["triggered"] = True
            reset_intercept["env_ids"] = [int(x) for x in env_ids.detach().cpu().tolist()]
        return None

    env.reset_envs_idx = MethodType(no_auto_reset, env)

    try:
        for step in range(steps):
            reset_intercept["triggered"] = False
            action = _make_action(env, mode, step)
            env.step({"actions": action})
            reset_triggered = bool(reset_intercept["triggered"] or _scalar(env.reset_buf[0]))
            rows.append(_record_row(env, step, reset_triggered))
            if reset_triggered:
                break
    finally:
        env.reset_envs_idx = original_reset

    csv_path, npz_path = _save_table(rows, out_dir, mode)
    pngs = _save_plots(rows, out_dir, mode)
    summary = _summarize(rows, mode)
    summary_path = out_dir / f"{mode}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"csv: {csv_path}")
    print(f"npz: {npz_path}")
    print(f"summary_json: {summary_path}")
    for png in pngs:
        print(f"png: {png}")


if __name__ == "__main__":
    main()

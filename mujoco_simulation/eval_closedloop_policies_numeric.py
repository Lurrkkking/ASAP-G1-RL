#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_simulation.fdd_asap_sim2sim import ANKLE_INDICES, LOWER_BODY_INDICES, create_history, get_obs, pd_control

DEFAULT_MOTION_FILE = REPO_ROOT / "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"
DEFAULT_MUJOCO_CFG = REPO_ROOT / "mujoco_simulation/g1_config/mujoco_config.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "mujoco_simulation/out/eval_closedloop_numeric"
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]
BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
]
HIP_INDICES = (0, 1, 2, 6, 7, 8)
KNEE_INDICES = (3, 9)
UPPER_BODY_INDICES = tuple(range(12, 23))
EARLY_WINDOWS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 70)]


def read_conf(config_file: str):
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
    cfg.num_actions = config["num_actions"]
    cfg.control_decimation = config["control_decimation"]
    cfg.clip_actions = float(config["clip_actions"])
    cfg.action_scale = float(config["action_scale"])
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
    cfg.solver_iterations = int(config.get("solver_iterations", 100))
    cfg.solver_ls_iterations = int(config.get("solver_ls_iterations", 50))
    cfg.time_offset = float(config.get("time_offset", 0.0))
    cfg.phase_wrap = bool(config.get("phase_wrap", False))
    cfg.stop_at_motion_end = bool(config.get("stop_at_motion_end", True))
    cfg.xml_path = str((REPO_ROOT / "mujoco_simulation/g1_urdf/g1_29dof_anneal_23dof.xml").resolve())
    return cfg


def parse_case(spec: str):
    name, policy_path, scale = spec.split("|")
    return {"name": name, "policy_path": Path(policy_path), "scale": float(scale)}


def load_motion(path: Path):
    data = joblib.load(path)
    key = next(iter(data.keys()))
    return key, data[key]


def load_onnx(path: Path, policy_name: str):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"policy_name={policy_name}")
    print(f"policy_path={path}")
    print(f"input_name={inp.name}")
    print(f"input_shape={inp.shape}")
    print(f"output_name={out.name}")
    print(f"output_shape={out.shape}")
    if list(inp.shape) != [1, 380] or list(out.shape) != [1, 23]:
        raise RuntimeError(f"Unexpected ONNX signature for {policy_name}: {inp.shape} -> {out.shape}")
    return session, inp.name, out.name


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float32)
    return q[[1, 2, 3, 0]].copy()


def projected_gravity(root_rot_xyzw):
    return R.from_quat(root_rot_xyzw).apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float32)


def body_state(model, data):
    body_pos = []
    body_quat = []
    for name in BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_pos.append(data.xpos[body_id].astype(np.float32, copy=True))
        body_quat.append(wxyz_to_xyzw(data.xquat[body_id]))
    return np.stack(body_pos, axis=0), np.stack(body_quat, axis=0)


def root_state(data):
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    return {
        "root_pos": q[:3].astype(np.float32, copy=True),
        "root_rot": wxyz_to_xyzw(q[3:7]),
        "root_lin_vel": dq[:3].astype(np.float32, copy=True),
        "root_ang_vel": dq[3:6].astype(np.float32, copy=True),
    }


def foot_contacts(model, data):
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_sole_box")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_sole_box")
    left_flag = 0
    right_flag = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == left or c.geom2 == left:
            left_flag = 1
        if c.geom1 == right or c.geom2 == right:
            right_flag = 1
    return left_flag, right_flag


def mean_abs(x, idx=None):
    arr = np.asarray(x, dtype=np.float32)
    if idx is not None:
        arr = arr[list(idx)]
    return float(np.mean(np.abs(arr)))


def summarize_window(rows, start, end):
    chunk = rows[start:end]
    return {
        "root_height_min": float(min(r["root_height"] for r in chunk)),
        "projected_gravity_x_max_abs": float(max(abs(r["projected_gravity_x"]) for r in chunk)),
        "projected_gravity_y_max_abs": float(max(abs(r["projected_gravity_y"]) for r in chunk)),
        "action_mean_abs": float(np.mean([r["action_mean_abs"] for r in chunk])),
        "action_rate_mean_abs": float(np.mean([r["action_rate_mean_abs"] for r in chunk])),
        "ankle_action_mean_abs": float(np.mean([r["ankle_action_mean_abs"] for r in chunk])),
        "hip_action_mean_abs": float(np.mean([r["hip_action_mean_abs"] for r in chunk])),
        "ankle_action_diff_mean_abs_vs_baseline": float(np.mean([r["ankle_action_diff_mean_abs_vs_baseline"] for r in chunk])),
        "hip_action_diff_mean_abs_vs_baseline": float(np.mean([r["hip_action_diff_mean_abs_vs_baseline"] for r in chunk])),
        "lower_body_action_diff_mean_abs_vs_baseline": float(np.mean([r["lower_body_action_diff_mean_abs_vs_baseline"] for r in chunk])),
        "q_target_diff_mean_abs_vs_baseline": float(np.mean([r["q_target_diff_mean_abs_vs_baseline"] for r in chunk])),
    }


def rollout(case, baseline_session, baseline_input, baseline_output, motion, cfg, output_dir):
    session, input_name, output_name = load_onnx(case["policy_path"], case["name"])
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt
    if cfg.solver_iterations > 0:
        model.opt.iterations = cfg.solver_iterations
    if hasattr(model.opt, "ls_iterations") and cfg.solver_ls_iterations > 0:
        model.opt.ls_iterations = cfg.solver_ls_iterations
    data.qpos[-cfg.num_actions:] = cfg.default_dof_pos
    mujoco.mj_step(model, data)

    hist_dict, hist_obs_c = create_history(cfg)
    prev_case_action = np.zeros(cfg.num_actions, dtype=np.float32)
    prev_base_action = np.zeros(cfg.num_actions, dtype=np.float32)
    applied_action = np.zeros(cfg.num_actions, dtype=np.float32)
    rows = []
    per_joint_diffs = []
    num_steps = int(len(motion["dof"]))
    stop_reason = "motion_end"
    fall_step = None

    for step in range(num_steps):
        mj = {
            "mujoco_dof_pos": data.qpos[7:].copy().astype(np.float32),
            "mujoco_dof_vel": data.qvel[6:].copy().astype(np.float32),
            "mujoco_base_angvel": data.qvel[3:6].copy().astype(np.float32),
            "mujoco_gvec": projected_gravity(wxyz_to_xyzw(data.qpos[3:7])),
        }
        obs, hist_obs_c = get_obs(hist_obs_c, hist_dict, mj, applied_action, step, cfg)
        actor_obs = obs.astype(np.float32)
        case_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)
        base_action = baseline_session.run([baseline_output], {baseline_input: actor_obs})[0][0].astype(np.float32)
        case_action = np.clip(case_action, -cfg.clip_actions, cfg.clip_actions)
        base_action = np.clip(base_action, -cfg.clip_actions, cfg.clip_actions)
        q_target = case_action * cfg.action_scale + cfg.default_dof_pos
        base_q_target = base_action * cfg.action_scale + cfg.default_dof_pos
        case_rate = case_action - prev_case_action
        base_rate = base_action - prev_base_action
        action_diff = case_action - base_action
        q_target_diff = q_target - base_q_target

        data.ctrl[:] = pd_control(q_target, mj["mujoco_dof_pos"], np.zeros_like(cfg.kds), mj["mujoco_dof_vel"], cfg)
        data.ctrl[:] = np.clip(data.ctrl, -cfg.tau_limit, cfg.tau_limit)
        for _ in range(int(cfg.control_decimation)):
            mujoco.mj_step(model, data)

        root = root_state(data)
        gvec = projected_gravity(root["root_rot"])
        body_pos, body_quat = body_state(model, data)
        left_contact, right_contact = foot_contacts(model, data)
        root_height = float(root["root_pos"][2])
        fall = root_height < 0.45 or abs(float(gvec[0])) > 0.85 or abs(float(gvec[1])) > 0.85

        row = {
            "step": step,
            "time": float((step + 1) * cfg.control_decimation * cfg.simulation_dt),
            "phase": float(min((step + 1) * cfg.control_decimation * cfg.simulation_dt / cfg.cycle_time, 1.0)),
            "root_height": root_height,
            "projected_gravity_x": float(gvec[0]),
            "projected_gravity_y": float(gvec[1]),
            "dof_pos": data.qpos[7:].astype(np.float32).tolist(),
            "dof_vel": data.qvel[6:].astype(np.float32).tolist(),
            "policy_action": case_action.tolist(),
            "q_target": q_target.tolist(),
            "action_rate": case_rate.tolist(),
            "action_rate_mean_abs": float(np.mean(np.abs(case_rate))),
            "ctrl": data.ctrl.astype(np.float32).tolist(),
            "left_foot_contact": int(left_contact),
            "right_foot_contact": int(right_contact),
            "body_pos": body_pos.tolist(),
            "body_quat": body_quat.tolist(),
            "root_pos": root["root_pos"].tolist(),
            "root_rot": root["root_rot"].tolist(),
            "root_lin_vel": root["root_lin_vel"].tolist(),
            "root_ang_vel": root["root_ang_vel"].tolist(),
            "action_mean_abs": float(np.mean(np.abs(case_action))),
            "action_max_abs": float(np.max(np.abs(case_action))),
            "action_rate_max_abs": float(np.max(np.abs(case_rate))),
            "q_target_mean_abs": float(np.mean(np.abs(q_target))),
            "q_target_max_abs": float(np.max(np.abs(q_target))),
            "ctrl_mean_abs": float(np.mean(np.abs(data.ctrl))),
            "ctrl_max_abs": float(np.max(np.abs(data.ctrl))),
            "ankle_action_mean_abs": mean_abs(case_action, ANKLE_INDICES),
            "hip_action_mean_abs": mean_abs(case_action, HIP_INDICES),
            "action_diff_mean_abs_vs_baseline": float(np.mean(np.abs(action_diff))),
            "action_diff_max_abs_vs_baseline": float(np.max(np.abs(action_diff))),
            "q_target_diff_mean_abs_vs_baseline": float(np.mean(np.abs(q_target_diff))),
            "q_target_diff_max_abs_vs_baseline": float(np.max(np.abs(q_target_diff))),
            "lower_body_action_diff_mean_abs_vs_baseline": mean_abs(action_diff, LOWER_BODY_INDICES),
            "ankle_action_diff_mean_abs_vs_baseline": mean_abs(action_diff, ANKLE_INDICES),
            "hip_action_diff_mean_abs_vs_baseline": mean_abs(action_diff, HIP_INDICES),
            "knee_action_diff_mean_abs_vs_baseline": mean_abs(action_diff, KNEE_INDICES),
            "upper_body_action_diff_mean_abs_vs_baseline": mean_abs(action_diff, UPPER_BODY_INDICES),
            "action_rate_diff_mean_abs_vs_baseline": float(np.mean(np.abs(case_rate - base_rate))),
        }
        rows.append(row)
        per_joint_diffs.append(np.abs(action_diff))
        prev_case_action = case_action
        prev_base_action = base_action
        applied_action = case_action

        if fall:
            fall_step = step
            stop_reason = "fall"
            break

    summary = {
        "policy_name": case["name"],
        "policy_path": str(case["policy_path"]),
        "scale": float(case["scale"]),
        "num_steps": len(rows),
        "fall_step": fall_step,
        "root_height_min": float(min(r["root_height"] for r in rows)),
        "root_height_max": float(max(r["root_height"] for r in rows)),
        "root_height_final": float(rows[-1]["root_height"]),
        "projected_gravity_x_max_abs": float(max(abs(r["projected_gravity_x"]) for r in rows)),
        "projected_gravity_y_max_abs": float(max(abs(r["projected_gravity_y"]) for r in rows)),
        "action_mean_abs": float(np.mean([r["action_mean_abs"] for r in rows])),
        "action_max_abs": float(np.max([r["action_max_abs"] for r in rows])),
        "action_rate_mean_abs": float(np.mean([r["action_rate_mean_abs"] for r in rows])),
        "action_rate_max_abs": float(np.max([r["action_rate_max_abs"] for r in rows])),
        "q_target_mean_abs": float(np.mean([r["q_target_mean_abs"] for r in rows])),
        "q_target_max_abs": float(np.max([r["q_target_max_abs"] for r in rows])),
        "ctrl_mean_abs": float(np.mean([r["ctrl_mean_abs"] for r in rows])),
        "ctrl_max_abs": float(np.max([r["ctrl_max_abs"] for r in rows])),
        "left_foot_contact_rate": float(np.mean([r["left_foot_contact"] for r in rows])),
        "right_foot_contact_rate": float(np.mean([r["right_foot_contact"] for r in rows])),
        "action_diff_mean_abs_vs_baseline": float(np.mean([r["action_diff_mean_abs_vs_baseline"] for r in rows])),
        "action_diff_max_abs_vs_baseline": float(np.max([r["action_diff_max_abs_vs_baseline"] for r in rows])),
        "q_target_diff_mean_abs_vs_baseline": float(np.mean([r["q_target_diff_mean_abs_vs_baseline"] for r in rows])),
        "q_target_diff_max_abs_vs_baseline": float(np.max([r["q_target_diff_max_abs_vs_baseline"] for r in rows])),
        "lower_body_action_diff_mean_abs_vs_baseline": float(np.mean([r["lower_body_action_diff_mean_abs_vs_baseline"] for r in rows])),
        "ankle_action_diff_mean_abs_vs_baseline": float(np.mean([r["ankle_action_diff_mean_abs_vs_baseline"] for r in rows])),
        "hip_action_diff_mean_abs_vs_baseline": float(np.mean([r["hip_action_diff_mean_abs_vs_baseline"] for r in rows])),
        "knee_action_diff_mean_abs_vs_baseline": float(np.mean([r["knee_action_diff_mean_abs_vs_baseline"] for r in rows])),
        "upper_body_action_diff_mean_abs_vs_baseline": float(np.mean([r["upper_body_action_diff_mean_abs_vs_baseline"] for r in rows])),
        "action_rate_diff_mean_abs_vs_baseline": float(np.mean([r["action_rate_diff_mean_abs_vs_baseline"] for r in rows])),
        "per_joint_mean_abs_action_diff_sorted": sorted([{"joint": name, "mean_abs_diff": float(val)} for name, val in zip(JOINT_NAMES, np.mean(np.stack(per_joint_diffs, axis=0), axis=0))], key=lambda x: x["mean_abs_diff"], reverse=True),
        "stop_reason": stop_reason,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['name']}.json"
    csv_path = output_dir / f"{case['name']}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary, rows, json_path, csv_path


def recommend(summary_rows):
    by_name = {r["policy_name"]: r for r in summary_rows}
    s005 = by_name.get("scale005_model13100")
    s015 = by_name.get("scale015_model13100")
    if s005 and s015:
        if s005["fall_step"] is None and (s015["fall_step"] is not None or s015["ankle_action_diff_mean_abs_vs_baseline"] > s005["ankle_action_diff_mean_abs_vs_baseline"] * 1.2 or s015["hip_action_diff_mean_abs_vs_baseline"] > s005["hip_action_diff_mean_abs_vs_baseline"] * 1.2):
            s005["recommendation"] = "prefer scale=0.05"
            s015["recommendation"] = "more aggressive drift"
        elif s005["fall_step"] is None and s015["fall_step"] is None:
            s005["recommendation"] = "stable conservative candidate"
            s015["recommendation"] = "stable but check drift tradeoff"
        else:
            s005["recommendation"] = "unstable or marginal"
            s015["recommendation"] = "unstable or marginal"
    for row in summary_rows:
        row.setdefault("recommendation", "baseline reference" if row["scale"] == 0.0 else "candidate")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop MuJoCo numeric evaluation")
    parser.add_argument("--baseline-policy", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--mujoco-config", type=Path, default=DEFAULT_MUJOCO_CFG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()

    cfg = read_conf(str(args.mujoco_config))
    motion_key, motion = load_motion(args.motion_file)
    print(json.dumps({"motion_key": motion_key, "num_steps": int(len(motion["dof"])), "fps": int(motion["fps"]), "actual_control_dt": float(cfg.control_decimation * cfg.simulation_dt)}, ensure_ascii=True))

    baseline_session, baseline_input, baseline_output = load_onnx(args.baseline_policy, "baseline_model13000")
    baseline_summary, baseline_rows, baseline_json, baseline_csv = rollout({"name": "baseline_model13000", "policy_path": args.baseline_policy, "scale": 0.0}, baseline_session, baseline_input, baseline_output, motion, cfg, args.output_dir)
    print(f"saved_json={baseline_json}")
    print(f"saved_csv={baseline_csv}")

    outputs = [{"summary": baseline_summary, "rows": baseline_rows}]
    for spec in args.case:
        case = parse_case(spec)
        summary, rows, json_path, csv_path = rollout(case, baseline_session, baseline_input, baseline_output, motion, cfg, args.output_dir)
        outputs.append({"summary": summary, "rows": rows})
        print(f"saved_json={json_path}")
        print(f"saved_csv={csv_path}")

    summaries = [x["summary"] for x in outputs]
    stability_table = [{k: s[k] for k in ["policy_name", "num_steps", "fall_step", "root_height_min", "root_height_max", "root_height_final", "projected_gravity_x_max_abs", "projected_gravity_y_max_abs", "action_mean_abs", "action_max_abs", "action_rate_mean_abs", "action_rate_max_abs", "q_target_mean_abs", "q_target_max_abs", "ctrl_mean_abs", "ctrl_max_abs", "left_foot_contact_rate", "right_foot_contact_rate"]} for s in summaries]
    drift_table = [{k: s[k] for k in ["policy_name", "action_diff_mean_abs_vs_baseline", "action_diff_max_abs_vs_baseline", "q_target_diff_mean_abs_vs_baseline", "q_target_diff_max_abs_vs_baseline", "lower_body_action_diff_mean_abs_vs_baseline", "ankle_action_diff_mean_abs_vs_baseline", "hip_action_diff_mean_abs_vs_baseline", "knee_action_diff_mean_abs_vs_baseline", "upper_body_action_diff_mean_abs_vs_baseline", "action_rate_diff_mean_abs_vs_baseline"]} for s in summaries if s["policy_name"] != "baseline_model13000"]

    early_phase = []
    for out in outputs:
        name = out["summary"]["policy_name"]
        rows = out["rows"]
        for start, end in EARLY_WINDOWS:
            if end <= len(rows):
                row = summarize_window(rows, start, end)
                row["policy_name"] = name
                row["window"] = f"{start}-{end}"
                early_phase.append(row)
        full = summarize_window(rows, 0, len(rows))
        full["policy_name"] = name
        full["window"] = "full rollout"
        early_phase.append(full)

    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "policy_name": s["policy_name"],
            "scale": 0.0 if s["policy_name"] == "baseline_model13000" else (0.05 if s["policy_name"] == "scale005_model13100" else 0.15),
            "fall_step": s["fall_step"],
            "root_height_min": s["root_height_min"],
            "projected_gravity_x_max_abs": s["projected_gravity_x_max_abs"],
            "projected_gravity_y_max_abs": s["projected_gravity_y_max_abs"],
            "action_rate_mean_abs": s["action_rate_mean_abs"],
            "ankle_action_diff_mean_abs_vs_baseline": s.get("ankle_action_diff_mean_abs_vs_baseline", 0.0),
            "hip_action_diff_mean_abs_vs_baseline": s.get("hip_action_diff_mean_abs_vs_baseline", 0.0),
            "lower_body_action_diff_mean_abs_vs_baseline": s.get("lower_body_action_diff_mean_abs_vs_baseline", 0.0),
            "Eg_mpjpe_mm": None,
            "Empjpe_mm": None,
            "Eacc_mm_per_frame2": None,
            "Evel_mm_per_frame": None,
            "paper_mean_improve_%": None,
        })
    recommend(summary_rows)

    with open(args.output_dir / "summary_table.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(args.output_dir / "stability_table.json", "w", encoding="utf-8") as f:
        json.dump(stability_table, f, indent=2)
    with open(args.output_dir / "action_drift_table.json", "w", encoding="utf-8") as f:
        json.dump(drift_table, f, indent=2)
    with open(args.output_dir / "early_phase_drift_table.json", "w", encoding="utf-8") as f:
        json.dump(early_phase, f, indent=2)

    print("[STABILITY_TABLE]")
    print(json.dumps(stability_table, indent=2))
    print("[ACTION_DRIFT_TABLE]")
    print(json.dumps(drift_table, indent=2))
    print("[EARLY_PHASE_DRIFT_TABLE]")
    print(json.dumps(early_phase, indent=2))
    print("[PER_JOINT_ACTION_DIFF]")
    print(json.dumps([{"policy_name": s["policy_name"], "per_joint_mean_abs_action_diff_sorted": s["per_joint_mean_abs_action_diff_sorted"]} for s in summaries if s["policy_name"] != "baseline_model13000"], indent=2))
    print("[SUMMARY_TABLE]")
    print(json.dumps(summary_rows, indent=2))
    print("[NOTE] paper metric skipped due reference-body alignment path issues in current environment")


if __name__ == "__main__":
    main()

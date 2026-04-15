import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import genesis_simulation.run_onnx_motiontracking as mt


NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "20"))
MAX_STEPS_PER_EPISODE = int(os.environ.get("MAX_STEPS_PER_EPISODE", str(mt.NUM_STEPS)))
DATASET_OUT_NPZ = Path(os.environ.get("DATASET_OUT_NPZ", "/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/residual_dataset.npz"))
DEBUG_OUT_JSON = Path(os.environ.get("DEBUG_OUT_JSON", "/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/debug_sample.json"))
DEBUG_FRAMES = int(os.environ.get("DEBUG_FRAMES", "200"))


def _to_numpy_1d(x):
    x = mt.as_tensor_2d(x)
    return x[0].detach().cpu().numpy().astype(np.float32)


def _get_state(robot, motor_dofs):
    base_pos = _to_numpy_1d(robot.get_pos())
    quat_wxyz = mt.as_tensor_2d(robot.get_quat())
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]][0].detach().cpu().numpy().astype(np.float32)
    dof_pos = _to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
    dof_vel = _to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
    return np.concatenate([base_pos, quat_xyzw, dof_pos, dof_vel], axis=0).astype(np.float32)


def _reset_robot(robot, motor_dofs):
    applied_ref_init = mt.try_apply_reference_init(robot, motor_dofs)
    if applied_ref_init:
        return
    robot.set_pos(torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32, device=mt.SIM_DEVICE))
    robot.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=mt.SIM_DEVICE))
    robot.set_dofs_position(
        position=torch.tensor(mt.DEFAULT_DOF_POS, dtype=torch.float32, device=mt.SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )
    robot.set_dofs_velocity(
        velocity=torch.zeros(len(motor_dofs), dtype=torch.float32, device=mt.SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )


def _pad_episodes(episodes, state_dim, action_dim):
    lengths = np.asarray([ep["s"].shape[0] for ep in episodes], dtype=np.int32)
    num_eps = len(episodes)
    t_max = int(lengths.max()) if num_eps > 0 else 0

    s = np.zeros((num_eps, t_max, state_dim), dtype=np.float32)
    a = np.zeros((num_eps, t_max, action_dim), dtype=np.float32)
    s_next = np.zeros((num_eps, t_max, state_dim), dtype=np.float32)
    mask = np.zeros((num_eps, t_max), dtype=np.bool_)

    for i, ep in enumerate(episodes):
        t = ep["s"].shape[0]
        if t == 0:
            continue
        s[i, :t] = ep["s"]
        a[i, :t] = ep["a"]
        s_next[i, :t] = ep["s_next"]
        mask[i, :t] = True
    return s, a, s_next, mask, lengths


def main():
    mt.maybe_autofill_time_offset()

    if not Path(mt.URDF_PATH).is_file():
        raise FileNotFoundError(f"URDF not found: {mt.URDF_PATH}")
    if not Path(mt.ONNX_PATH).is_file():
        raise FileNotFoundError(f"ONNX not found: {mt.ONNX_PATH}")

    mt.gs.init(backend=mt.gs.gpu if "cuda" in mt.SIM_DEVICE else mt.gs.cpu)

    scene = mt.gs.Scene(
        sim_options=mt.gs.options.SimOptions(dt=mt.SIM_DT, substeps=1),
        rigid_options=mt.gs.options.RigidOptions(enable_self_collision=mt.USE_SELF_COLLISION),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(mt.gs.morphs.Plane(), material=mt.gs.materials.Rigid(friction=mt.FLOOR_FRICTION))
    robot = scene.add_entity(
        mt.gs.morphs.URDF(
            file=mt.URDF_PATH,
            merge_fixed_links=True,
            links_to_keep=mt.BODY_NAMES,
            pos=(0.0, 0.0, 0.8),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )
    scene.build()

    motor_dofs = [robot.get_joint(name).dof_idx_local for name in mt.RL_JOINT_NAMES]
    session = ort.InferenceSession(mt.ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out_shape = session.get_outputs()[0].shape
    policy_action_dim = int(out_shape[-1]) if (isinstance(out_shape, (list, tuple)) and isinstance(out_shape[-1], int)) else None

    base_session = None
    base_input_name = None
    base_output_name = None

    if policy_action_dim == 46:
        base_onnx = mt.BASE_ONNX_PATH
        if not base_onnx:
            cfg_path = mt._resolve_train_config_path()
            if cfg_path is not None:
                try:
                    cfg = mt._load_yaml(cfg_path)
                    base_ckpt = ((((cfg or {}).get("algo") or {}).get("config") or {}).get("policy_checkpoint"))
                    if base_ckpt:
                        base_ckpt = Path(str(base_ckpt))
                        cand = base_ckpt.parent / "exported" / (base_ckpt.stem + ".onnx")
                        if cand.is_file():
                            base_onnx = str(cand)
                except Exception:
                    pass

        if not base_onnx or (not Path(base_onnx).is_file()):
            raise FileNotFoundError(
                "Detected 46-dim ONNX in dataset collection, but no 23-dim base ONNX was found. "
                "Please set BASE_ONNX_PATH=/path/to/base_23.onnx"
            )

        base_session = ort.InferenceSession(base_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        base_input_name = base_session.get_inputs()[0].name
        base_output_name = base_session.get_outputs()[0].name
        base_out_shape = base_session.get_outputs()[0].shape
        base_dim = int(base_out_shape[-1]) if (isinstance(base_out_shape, (list, tuple)) and isinstance(base_out_shape[-1], int)) else None
        if base_dim != 23:
            raise ValueError(f"BASE_ONNX_PATH must output 23 dims, got {base_out_shape}")

    elif policy_action_dim not in (23,):
        raise ValueError(f"Unsupported ONNX action dim: {out_shape}. Expected 23 or 46.")

    print(f"[INFO] collect: policy_action_dim={policy_action_dim}, delta_max_scale={mt.DELTA_MAX_SCALE}")

    if mt.USE_IMPLICIT_PD:
        robot.set_dofs_kp(kp=torch.tensor(mt.KP, dtype=torch.float32, device=mt.SIM_DEVICE), dofs_idx_local=motor_dofs)
        robot.set_dofs_kv(kv=torch.tensor(mt.KD, dtype=torch.float32, device=mt.SIM_DEVICE), dofs_idx_local=motor_dofs)

    gravity_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=mt.SIM_DEVICE)
    state_dim = 3 + 4 + len(motor_dofs) + len(motor_dofs)
    action_dim = len(motor_dofs)
    episodes = []

    for ep in range(NUM_EPISODES):
        _reset_robot(robot, motor_dofs)
        scene.step()

        history_buffers = mt.make_history_buffers()
        last_actions_obs = np.zeros(action_dim, dtype=np.float32)
        last_policy_action = np.zeros(policy_action_dim if policy_action_dim is not None else action_dim, dtype=np.float32)
        last_base_action = np.zeros(action_dim, dtype=np.float32)

        ep_s = []
        ep_a = []
        ep_s_next = []

        for step in range(MAX_STEPS_PER_EPISODE):
            s_t = _get_state(robot, motor_dofs)
            dof_pos = s_t[7:7 + action_dim]
            dof_vel = s_t[7 + action_dim:]

            quat_xyzw = mt.as_tensor_2d(s_t[3:7])
            base_ang_vel_world = mt.as_tensor_2d(robot.get_ang())
            base_ang_vel_body = _to_numpy_1d(mt.quat_rotate_inverse(quat_xyzw, base_ang_vel_world))
            projected_gravity = _to_numpy_1d(mt.quat_rotate_inverse(quat_xyzw, gravity_world))

            t = mt.TIME_OFFSET + (step + 1) * (mt.SIM_DT * mt.CONTROL_DECIMATION)
            if mt.PHASE_WRAP:
                phase = np.float32((t % mt.MOTION_DURATION) / mt.MOTION_DURATION)
            else:
                if mt.STOP_AT_MOTION_END and t >= mt.MOTION_DURATION:
                    break
                phase = np.float32(min(t, mt.MOTION_DURATION - 1e-6) / mt.MOTION_DURATION)

            curr_features = mt.build_curr_features(
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                base_ang_vel_body=base_ang_vel_body,
                projected_gravity=projected_gravity,
                last_actions=last_actions_obs,
                phase=phase,
            )
            history_actor = mt.query_history_actor(history_buffers)
            actor_obs = mt.build_actor_obs(curr_features, history_actor)

            policy_action = session.run([output_name], {input_name: actor_obs[None, :]})[0][0].astype(np.float32)
            policy_action = np.clip(policy_action, -mt.ACTION_CLIP_VALUE, mt.ACTION_CLIP_VALUE)
            if mt.ACTION_FILTER_ALPHA < 0.999:
                policy_action = mt.ACTION_FILTER_ALPHA * policy_action + (1.0 - mt.ACTION_FILTER_ALPHA) * last_policy_action

            if policy_action_dim == 46:
                raw_delta = policy_action[:action_dim]
                raw_alpha = policy_action[action_dim:action_dim * 2]
                alpha = 1.0 / (1.0 + np.exp(-raw_alpha))
                delta = np.tanh(raw_delta) * mt.DELTA_MAX_SCALE * alpha

                base_action = base_session.run([base_output_name], {base_input_name: actor_obs[None, :]})[0][0].astype(np.float32)
                base_action = np.clip(base_action, -mt.ACTION_CLIP_VALUE, mt.ACTION_CLIP_VALUE)
                if mt.ACTION_FILTER_ALPHA < 0.999:
                    base_action = mt.ACTION_FILTER_ALPHA * base_action + (1.0 - mt.ACTION_FILTER_ALPHA) * last_base_action

                action = np.clip(base_action + delta, -mt.ACTION_CLIP_VALUE, mt.ACTION_CLIP_VALUE)
                next_actions_obs = base_action
                last_base_action = base_action.copy()
            else:
                action = policy_action
                next_actions_obs = action

            mt.update_history_buffers(history_buffers, curr_features)
            last_actions_obs = next_actions_obs.copy()
            last_policy_action = policy_action.copy()

            target_pos = (action * mt.ACTION_SCALE + mt.DEFAULT_DOF_POS).astype(np.float32)
            for _ in range(mt.CONTROL_DECIMATION):
                if mt.USE_IMPLICIT_PD:
                    robot.control_dofs_position(
                        position=torch.tensor(target_pos, dtype=torch.float32, device=mt.SIM_DEVICE),
                        dofs_idx_local=motor_dofs,
                    )
                else:
                    dof_pos_sub = _to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
                    dof_vel_sub = _to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
                    torques_req = mt.KP * (target_pos - dof_pos_sub) - mt.KD * dof_vel_sub
                    torques = np.clip(torques_req, -mt.TORQUE_LIMITS, mt.TORQUE_LIMITS)
                    robot.control_dofs_force(
                        torch.tensor(torques, dtype=torch.float32, device=mt.SIM_DEVICE),
                        dofs_idx_local=motor_dofs,
                    )
                scene.step()

            s_tp1 = _get_state(robot, motor_dofs)
            ep_s.append(s_t)
            ep_a.append(target_pos.copy())
            ep_s_next.append(s_tp1)

            if mt.STOP_ON_FALL and float(s_tp1[2]) < mt.FALL_ROOT_Z:
                break

        ep_pack = {
            "s": np.asarray(ep_s, dtype=np.float32),
            "a": np.asarray(ep_a, dtype=np.float32),
            "s_next": np.asarray(ep_s_next, dtype=np.float32),
        }
        episodes.append(ep_pack)
        print(f"[episode {ep}] transitions={ep_pack['s'].shape[0]}")

    s_pad, a_pad, s_next_pad, mask, episode_lengths = _pad_episodes(episodes, state_dim, action_dim)

    DATASET_OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET_OUT_NPZ,
        s=s_pad,
        a=a_pad,
        s_next=s_next_pad,
        mask=mask,
        episode_lengths=episode_lengths,
        state_dim=np.asarray([state_dim], dtype=np.int32),
        action_dim=np.asarray([action_dim], dtype=np.int32),
        quat_order=np.asarray(["xyzw"]),
        dof_names=np.asarray(mt.RL_JOINT_NAMES),
    )

    debug_frames = min(DEBUG_FRAMES, int(episode_lengths[0]) if len(episode_lengths) > 0 else 0)
    debug_dict = {
        "meta": {
            "episode_index": 0,
            "num_frames": debug_frames,
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "quat_order": "xyzw",
        },
        "s_t": s_pad[0, :debug_frames].tolist() if debug_frames > 0 else [],
        "a_t": a_pad[0, :debug_frames].tolist() if debug_frames > 0 else [],
        "s_tp1": s_next_pad[0, :debug_frames].tolist() if debug_frames > 0 else [],
    }
    DEBUG_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(debug_dict, f, ensure_ascii=False, indent=2)

    print(f"[DONE] dataset saved: {DATASET_OUT_NPZ}")
    print(f"[DONE] debug sample saved: {DEBUG_OUT_JSON}")


if __name__ == "__main__":
    main()

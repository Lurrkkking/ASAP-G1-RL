import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def set_seed(seed: int, torch_mod):
    random.seed(seed)
    np.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def load_eval_config(checkpoint: Path):
    cfg_candidates = [
        checkpoint.parent / "config.yaml",
        checkpoint.parent.parent / "config.yaml",
    ]
    cfg_path = None
    for p in cfg_candidates:
        if p.is_file():
            cfg_path = p
            break
    if cfg_path is None:
        raise FileNotFoundError(f"Could not find config.yaml near checkpoint: {checkpoint}")

    cfg = OmegaConf.load(cfg_path)
    if cfg.get("eval_overrides", None) is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
    return cfg, cfg_path


def disable_domain_rand(env_cfg):
    if not hasattr(env_cfg, "domain_rand"):
        return
    dr = env_cfg.domain_rand
    for k in list(dr.keys()):
        v = dr[k]
        if isinstance(v, bool):
            if k.startswith("randomize") or k.startswith("push") or k.endswith("noise"):
                dr[k] = False


def get_state(env):
    root = env.simulator.robot_root_states[0, :7].detach().cpu().numpy().astype(np.float32)
    dof_pos = env.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float32)
    dof_vel = env.simulator.dof_vel[0].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([root, dof_pos, dof_vel], axis=0).astype(np.float32)


def load_reference_npz(path: Path):
    with np.load(path, allow_pickle=False) as d:
        if "s" not in d:
            raise KeyError(f"Missing key in reference npz: s ({path})")
        s = d["s"].astype(np.float32)

        if s.ndim == 3:
            if "mask" in d:
                mask = d["mask"].astype(bool)
            elif "episode_lengths" in d:
                lengths = d["episode_lengths"].astype(np.int32)
                mask = np.zeros(s.shape[:2], dtype=bool)
                for i, l in enumerate(lengths):
                    mask[i, : int(l)] = True
            else:
                raise KeyError("Padded reference npz needs mask or episode_lengths")
            episode_lengths = np.sum(mask, axis=1).astype(np.int32)
        elif s.ndim == 2:
            mask = np.ones((s.shape[0],), dtype=bool)
            episode_lengths = np.asarray([s.shape[0]], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported shape for s: {s.shape}")
    return s, mask, episode_lengths


def zero_env_buffers(env, env_ids, torch_mod):
    env.actions[env_ids] = 0
    env.actions_after_delay[env_ids] = 0
    env.last_actions[env_ids] = 0
    env.last_dof_pos[env_ids] = 0
    env.last_dof_vel[env_ids] = 0
    env.last_root_vel[env_ids] = 0
    env.torques[env_ids] = 0
    env.feet_air_time[env_ids] = 0
    env.last_contacts[env_ids] = False
    env.last_contacts_filt[env_ids] = False
    env.feet_air_max_height[env_ids] = 0
    env.episode_length_buf[env_ids] = 0
    env.reset_buf[env_ids] = 0
    env.time_out_buf[env_ids] = False
    env.need_to_refresh_envs[env_ids] = False
    if hasattr(env, "action_queue"):
        env.action_queue[env_ids] = 0
    if hasattr(env, "closed_loop_actions"):
        env.closed_loop_actions[env_ids] = 0
    elif hasattr(env, "num_dof"):
        env.closed_loop_actions = torch_mod.zeros((env.num_envs, env.num_dof), device=env.device, dtype=env.simulator.dof_pos.dtype)
    if hasattr(env, "actions_closed_loop"):
        env.actions_closed_loop[env_ids] = 0
    elif hasattr(env, "num_dof"):
        env.actions_closed_loop = torch_mod.zeros((env.num_envs, env.num_dof), device=env.device, dtype=env.simulator.dof_pos.dtype)
    if hasattr(env, "raw_delta_a"):
        env.raw_delta_a[env_ids] = 0
    if hasattr(env, "alpha_t"):
        env.alpha_t[env_ids] = 0
    env.history_handler.reset(env_ids)


def set_env_state_from_s(env, s_vec, torch_mod):
    s_vec = np.asarray(s_vec, dtype=np.float32)
    expected_dim = 7 + 2 * env.num_dof
    if s_vec.shape[-1] != expected_dim:
        raise ValueError(f"State dim mismatch: expected {expected_dim}, got {s_vec.shape[-1]}")

    env_ids = torch_mod.tensor([0], device=env.device, dtype=torch_mod.long)
    dof_pos = torch_mod.from_numpy(s_vec[7 : 7 + env.num_dof]).to(device=env.device, dtype=env.simulator.dof_pos.dtype)
    dof_vel = torch_mod.from_numpy(s_vec[7 + env.num_dof : 7 + 2 * env.num_dof]).to(device=env.device, dtype=env.simulator.dof_vel.dtype)
    root = torch_mod.from_numpy(s_vec[:7]).to(device=env.device, dtype=env.simulator.robot_root_states.dtype)

    env.simulator.dof_pos[env_ids] = dof_pos.unsqueeze(0)
    env.simulator.dof_vel[env_ids] = dof_vel.unsqueeze(0)
    env.simulator.robot_root_states[env_ids, :7] = root.unsqueeze(0)
    if env.simulator.robot_root_states.shape[-1] > 7:
        env.simulator.robot_root_states[env_ids, 7:] = 0

    zero_env_buffers(env, env_ids, torch_mod)
    env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
    env.simulator.set_dof_state_tensor(env_ids, env.simulator.dof_state)

    env._refresh_sim_tensors()
    env._pre_compute_observations_callback()
    env._compute_observations()
    env._post_compute_observations_callback()

    clip_obs = env.config.normalization.clip_observations
    for obs_key, obs_val in env.obs_buf_dict.items():
        env.obs_buf_dict[obs_key] = torch_mod.clip(obs_val, -clip_obs, clip_obs)

    return env.obs_buf_dict


def main():
    parser = argparse.ArgumentParser(description="Collect Isaac one-step transitions from fixed reference states using a test ONNX policy")
    parser.add_argument("--ref-npz", type=str, required=True, help="Reference Isaac NPZ providing fixed states s")
    parser.add_argument("--checkpoint", type=str, required=True, help="PT checkpoint path for loading config")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX policy path")
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--debug-json", type=str, default="")
    parser.add_argument("--debug-frames", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=-1, help="For quick debug; -1 means all valid samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    ref_npz = Path(args.ref_npz)
    checkpoint = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    out_npz = Path(args.out_npz)
    debug_json = Path(args.debug_json) if args.debug_json else None

    if not ref_npz.is_file():
        raise FileNotFoundError(f"reference npz not found: {ref_npz}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"onnx not found: {onnx_path}")

    cfg, cfg_path = load_eval_config(checkpoint)

    simulator_target = str(cfg.simulator._target_)
    if simulator_target.endswith("IsaacGym"):
        import isaacgym  # noqa: F401

    import torch
    import humanoidverse.utils.config_utils  # noqa: F401
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

    set_seed(args.seed, torch)

    cfg, cfg_path = load_eval_config(checkpoint)
    with open_dict(cfg):
        cfg.headless = True
        cfg.num_envs = int(args.num_envs)
        cfg.seed = int(args.seed)
        cfg.device = args.device

        cfg.env.config.headless = True
        cfg.env.config.num_envs = int(args.num_envs)
        cfg.env.config.enforce_randomize_motion_start_eval = False
        if "resample_motion_when_training" in cfg.env.config:
            cfg.env.config.resample_motion_when_training = False
        if "noise_to_initial_level" in cfg.env.config:
            cfg.env.config.noise_to_initial_level = 0.0

        disable_domain_rand(cfg.env.config)

    pre_process_config(cfg)

    env = instantiate(cfg.env, device=args.device)
    env.set_is_evaluating()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    action_dim_env = int(env.num_dof)
    out_shape = session.get_outputs()[0].shape
    action_dim_onnx = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None

    base_session = None
    base_input_name = None
    base_output_name = None

    if action_dim_onnx == 46:
        base_ckpt = Path(str(cfg.algo.config.policy_checkpoint))
        base_onnx = base_ckpt.parent / "exported" / (base_ckpt.stem + ".onnx")
        if not base_onnx.is_file():
            raise FileNotFoundError(f"Base 23-dim ONNX not found: {base_onnx}")
        base_session = ort.InferenceSession(str(base_onnx), providers=providers)
        base_input_name = base_session.get_inputs()[0].name
        base_output_name = base_session.get_outputs()[0].name
    elif action_dim_onnx != action_dim_env:
        raise ValueError(
            f"ONNX action dim {action_dim_onnx} incompatible with env dof {action_dim_env}. "
            "Expected 23 or 46."
        )

    ref_s, ref_mask, ref_episode_lengths = load_reference_npz(ref_npz)
    if ref_s.shape[-1] != 7 + 2 * action_dim_env:
        raise ValueError(f"Reference state dim mismatch: expected {7 + 2 * action_dim_env}, got {ref_s.shape[-1]}")

    action_scale = float(env.config.robot.control.action_scale)
    default_dof_pos = env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32)

    print(f"[INFO] config={cfg_path}")
    print(f"[INFO] ref_npz={ref_npz}")
    print(f"[INFO] onnx={onnx_path}")
    print(f"[INFO] num_envs={env.num_envs}, action_dim={action_dim_env}, state_dim={7 + 2 * env.num_dof}")

    out_s = ref_s.copy()
    out_a = np.zeros(ref_s.shape[:-1] + (action_dim_env,), dtype=np.float32)
    out_s_next = np.zeros_like(ref_s, dtype=np.float32)

    total_valid = int(ref_mask.sum()) if ref_mask.ndim == 2 else int(ref_mask.shape[0])
    used = 0

    if ref_s.ndim == 3:
        num_eps, t_max = ref_s.shape[:2]
        for ei in range(num_eps):
            for ti in range(t_max):
                if not ref_mask[ei, ti]:
                    continue
                if args.max_samples > 0 and used >= args.max_samples:
                    break

                obs_dict = set_env_state_from_s(env, ref_s[ei, ti], torch)
                actor_obs = obs_dict["actor_obs"].detach().cpu().numpy().astype(np.float32)
                policy_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)

                if action_dim_onnx == 46:
                    raw_delta = policy_action[:action_dim_env]
                    raw_alpha = policy_action[action_dim_env : action_dim_env * 2]
                    alpha = 1.0 / (1.0 + np.exp(-raw_alpha))
                    delta_scale = float(getattr(cfg.env.config, "max_delta_scale", action_scale))
                    delta = np.tanh(raw_delta) * delta_scale * alpha
                    base_action = base_session.run([base_output_name], {base_input_name: actor_obs})[0][0].astype(np.float32)
                    env_action = np.clip(base_action + delta, -100.0, 100.0)
                    actions_closed_loop_t = torch.from_numpy(base_action[None, :]).to(args.device)
                    actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                else:
                    actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                    if getattr(cfg.env.config, "add_extra_action", False):
                        # Old PPODeltaA-style 23D path: policy output is interpreted as delta patch.
                        env_action = policy_action
                        actions_closed_loop_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                        _, _, _, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
                    else:
                        # Main-policy PPO-in-patched-env path: policy output is the base action.
                        env_action = policy_action
                        _, _, _, _ = env.step({"actions": actions_t})
                        actions_closed_loop_t = None

                if action_dim_onnx == 46:
                    _, _, _, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
                elif getattr(cfg.env.config, "add_extra_action", False):
                    pass
                else:
                    pass

                if hasattr(env, "executed_actions_total"):
                    target_pos = env.executed_actions_total[0].detach().cpu().numpy().astype(np.float32) + default_dof_pos
                else:
                    target_pos = env_action * action_scale + default_dof_pos

                out_a[ei, ti] = target_pos.astype(np.float32)
                out_s_next[ei, ti] = get_state(env)
                used += 1

            if args.max_samples > 0 and used >= args.max_samples:
                break
    else:
        for i in range(ref_s.shape[0]):
            if args.max_samples > 0 and used >= args.max_samples:
                break

            obs_dict = set_env_state_from_s(env, ref_s[i], torch)
            actor_obs = obs_dict["actor_obs"].detach().cpu().numpy().astype(np.float32)
            policy_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)

            if action_dim_onnx == 46:
                raw_delta = policy_action[:action_dim_env]
                raw_alpha = policy_action[action_dim_env : action_dim_env * 2]
                alpha = 1.0 / (1.0 + np.exp(-raw_alpha))
                delta_scale = float(getattr(cfg.env.config, "max_delta_scale", action_scale))
                delta = np.tanh(raw_delta) * delta_scale * alpha
                base_action = base_session.run([base_output_name], {base_input_name: actor_obs})[0][0].astype(np.float32)
                env_action = np.clip(base_action + delta, -100.0, 100.0)
                actions_closed_loop_t = torch.from_numpy(base_action[None, :]).to(args.device)
                actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)
            else:
                actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                if getattr(cfg.env.config, "add_extra_action", False):
                    env_action = policy_action
                    actions_closed_loop_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                    _, _, _, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
                else:
                    env_action = policy_action
                    _, _, _, _ = env.step({"actions": actions_t})
                    actions_closed_loop_t = None

            if action_dim_onnx == 46:
                _, _, _, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
            elif getattr(cfg.env.config, "add_extra_action", False):
                pass
            else:
                pass

            if hasattr(env, "executed_actions_total"):
                target_pos = env.executed_actions_total[0].detach().cpu().numpy().astype(np.float32) + default_dof_pos
            else:
                target_pos = env_action * action_scale + default_dof_pos

            out_a[i] = target_pos.astype(np.float32)
            out_s_next[i] = get_state(env)
            used += 1

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if ref_s.ndim == 3:
        np.savez_compressed(
            out_npz,
            s=out_s,
            a=out_a,
            s_next=out_s_next,
            mask=ref_mask,
            episode_lengths=ref_episode_lengths,
        )
    else:
        np.savez_compressed(
            out_npz,
            s=out_s,
            a=out_a,
            s_next=out_s_next,
        )

    if debug_json is not None and ref_s.ndim == 3:
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        valid_ts = np.flatnonzero(ref_mask[0])
        n = min(args.debug_frames, int(valid_ts.shape[0]))
        sample_idx = valid_ts[:n]
        sample = {
            "s": out_s[0, sample_idx].tolist(),
            "a": out_a[0, sample_idx].tolist(),
            "s_next": out_s_next[0, sample_idx].tolist(),
            "num_frames": int(n),
            "used_samples": int(used),
            "total_valid_samples": int(total_valid),
        }
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved: {out_npz}")
    if debug_json is not None:
        print(f"[DONE] debug: {debug_json}")
    print(f"[INFO] used_samples={used}")
    print(f"[INFO] total_valid_samples={total_valid}")


if __name__ == "__main__":
    main()

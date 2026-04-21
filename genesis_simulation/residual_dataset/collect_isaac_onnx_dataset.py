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


def pad_episodes(episodes, state_dim, action_dim):
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
    parser = argparse.ArgumentParser(description="Collect (s,a,s_next) in IsaacGym with ONNX policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="PT checkpoint path for loading config")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX policy path")
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--debug-json", type=str, default="")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--debug-frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    out_npz = Path(args.out_npz)
    debug_json = Path(args.debug_json) if args.debug_json else None

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"onnx not found: {onnx_path}")

    cfg, cfg_path = load_eval_config(checkpoint)

    # IsaacGym requires importing isaacgym before torch.
    simulator_target = str(cfg.simulator._target_)
    if simulator_target.endswith("IsaacGym"):
        import isaacgym  # noqa: F401

    import torch
    import humanoidverse.utils.config_utils  # register OmegaConf resolvers (eval/if/...)
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
    last_policy_action = np.zeros(action_dim_onnx if action_dim_onnx is not None else action_dim_env, dtype=np.float32)
    last_base_action = np.zeros(action_dim_env, dtype=np.float32)

    if action_dim_onnx == 46:
        policy_checkpoint = cfg.algo.config.get("policy_checkpoint", None)
        if policy_checkpoint in (None, "null", ""):
            policy_checkpoint = cfg.get("checkpoint", None)
        if policy_checkpoint in (None, "null", ""):
            raise ValueError(
                "46-dim ONNX collection requires a base 23-dim policy checkpoint. "
                "Set algo.config.policy_checkpoint or checkpoint in config.yaml."
            )
        base_ckpt = Path(str(policy_checkpoint))
        base_onnx = base_ckpt.parent / 'exported' / (base_ckpt.stem + '.onnx')
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

    action_scale = float(env.config.robot.control.action_scale)
    default_dof_pos = env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32)

    print(f"[INFO] config={cfg_path}")
    print(f"[INFO] onnx={onnx_path}")
    print(f"[INFO] num_envs={env.num_envs}, action_dim={action_dim_env}, state_dim={7 + 2 * env.num_dof}")

    episodes = []

    for ep in range(args.episodes):
        obs_dict = env.reset_all()

        ep_s = []
        ep_a = []
        ep_s_next = []

        for _ in range(args.max_steps):
            s_t = get_state(env)

            actor_obs = obs_dict["actor_obs"].detach().cpu().numpy().astype(np.float32)
            policy_action = session.run([output_name], {input_name: actor_obs})[0][0].astype(np.float32)

            if action_dim_onnx == 46:
                raw_delta = policy_action[:action_dim_env]
                raw_alpha = policy_action[action_dim_env: action_dim_env * 2]
                alpha = 1.0 / (1.0 + np.exp(-raw_alpha))
                delta = np.tanh(raw_delta) * float(getattr(cfg.env.config, 'max_delta_scale', action_scale)) * alpha

                base_action = base_session.run([base_output_name], {base_input_name: actor_obs})[0][0].astype(np.float32)
                action = np.clip(base_action + delta, -100.0, 100.0)
                actions_closed_loop_t = torch.from_numpy(base_action[None, :]).to(args.device)
                next_actions_obs = base_action
                last_base_action = base_action.copy()
                actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)
            else:
                action = policy_action
                actions_closed_loop_t = torch.from_numpy(policy_action[None, :]).to(args.device)
                next_actions_obs = action
                actions_t = torch.from_numpy(policy_action[None, :]).to(args.device)

            target_pos = action * action_scale + default_dof_pos
            obs_dict, _, reset_buf, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
            s_tp1 = get_state(env)

            ep_s.append(s_t)
            ep_a.append(target_pos.astype(np.float32))
            ep_s_next.append(s_tp1)

            if bool(reset_buf[0].item()):
                break

        ep_pack = {
            "s": np.asarray(ep_s, dtype=np.float32),
            "a": np.asarray(ep_a, dtype=np.float32),
            "s_next": np.asarray(ep_s_next, dtype=np.float32),
        }
        episodes.append(ep_pack)
        print(f"[EP {ep + 1:03d}] transitions={ep_pack['s'].shape[0]}")

    state_dim = 7 + 2 * env.num_dof
    s, a, s_next, mask, lengths = pad_episodes(episodes, state_dim=state_dim, action_dim=action_dim_env)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s,
        a=a,
        s_next=s_next,
        mask=mask,
        episode_lengths=lengths,
    )

    if debug_json is not None and len(episodes) > 0:
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        n = min(args.debug_frames, episodes[0]["s"].shape[0])
        sample = {
            "s": episodes[0]["s"][:n].tolist(),
            "a": episodes[0]["a"][:n].tolist(),
            "s_next": episodes[0]["s_next"][:n].tolist(),
            "num_frames": int(n),
        }
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved: {out_npz}")
    if debug_json is not None:
        print(f"[DONE] debug: {debug_json}")


if __name__ == "__main__":
    main()

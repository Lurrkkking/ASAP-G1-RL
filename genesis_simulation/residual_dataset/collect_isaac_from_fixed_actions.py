import argparse
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genesis_simulation.residual_dataset.collect_isaac_from_fixed_states import (  # noqa: E402
    disable_domain_rand,
    get_state,
    load_eval_config,
    set_env_state_from_s,
)


def main():
    parser = argparse.ArgumentParser(description="Collect Isaac one-step transitions from fixed states using provided patched actions")
    parser.add_argument("--patched-actions-npz", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="PT checkpoint path for loading env config")
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    patched_npz = Path(args.patched_actions_npz)
    checkpoint = Path(args.checkpoint)
    out_npz = Path(args.out_npz)
    if not patched_npz.is_file():
        raise FileNotFoundError(patched_npz)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cfg, cfg_path = load_eval_config(checkpoint)
    simulator_target = str(cfg.simulator._target_)
    if simulator_target.endswith("IsaacGym"):
        import isaacgym  # noqa: F401

    import torch
    import humanoidverse.utils.config_utils  # noqa: F401
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    with np.load(patched_npz, allow_pickle=False) as d:
        s = d["ref_s"].astype(np.float32)
        a_patch = d["ref_a_patch"].astype(np.float32)
        mask = d["ref_mask"].astype(bool)
        episode_lengths = d["ref_episode_lengths"].astype(np.int32)

    out_s_next = np.zeros_like(s, dtype=np.float32)
    used = 0
    for ei in range(s.shape[0]):
        for ti in range(s.shape[1]):
            if not mask[ei, ti]:
                continue
            if args.max_samples > 0 and used >= args.max_samples:
                break
            set_env_state_from_s(env, s[ei, ti], torch)
            target_pos = a_patch[ei, ti]
            env_action = (target_pos - env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32)) / float(env.config.robot.control.action_scale)
            actions_closed_loop_t = torch.from_numpy(env_action[None, :]).to(args.device)
            if getattr(env, "dim_actions", env_action.shape[0]) > env_action.shape[0]:
                actions_t = torch.zeros((1, int(env.dim_actions)), device=args.device, dtype=actions_closed_loop_t.dtype)
            else:
                actions_t = actions_closed_loop_t
            _, _, _, _ = env.step({"actions": actions_t, "actions_closed_loop": actions_closed_loop_t})
            out_s_next[ei, ti] = get_state(env)
            used += 1
        if args.max_samples > 0 and used >= args.max_samples:
            break

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s,
        a=a_patch,
        s_next=out_s_next,
        mask=mask,
        episode_lengths=episode_lengths,
    )
    print(f"[DONE] saved: {out_npz}")
    print(f"[INFO] used_samples={used}")
    print(f"[INFO] config={cfg_path}")


if __name__ == "__main__":
    main()

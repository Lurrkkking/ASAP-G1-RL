import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DeltaActionPatchMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = state_dim + action_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a_base):
        return self.net(torch.cat([s, a_base], dim=-1))


def main():
    parser = argparse.ArgumentParser(description="Apply trained delta-action patch model to fixed-state patch dataset")
    parser.add_argument("--dataset-npz", type=str, required=True)
    parser.add_argument("--patch-ckpt", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    dataset_npz = Path(args.dataset_npz)
    patch_ckpt = Path(args.patch_ckpt)
    out_npz = Path(args.out_npz)
    if not dataset_npz.is_file():
        raise FileNotFoundError(dataset_npz)
    if not patch_ckpt.is_file():
        raise FileNotFoundError(patch_ckpt)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    ckpt = torch.load(patch_ckpt, map_location=device)
    model = DeltaActionPatchMLP(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        depth=int(ckpt["depth"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    max_delta_scale = float(ckpt["max_delta_scale"])

    with np.load(dataset_npz, allow_pickle=False) as d:
        s = d["s"].astype(np.float32)
        a_base = d["a_base"].astype(np.float32)
        ref_s = d["ref_s"].astype(np.float32)
        ref_a_base = d["ref_a_base"].astype(np.float32)
        ref_mask = d["ref_mask"].astype(bool)
        ref_episode_lengths = d["ref_episode_lengths"].astype(np.int32)
        episode_index = d["episode_index"].astype(np.int32)
        timestep_index = d["timestep_index"].astype(np.int32)

    with torch.no_grad():
        s_t = torch.from_numpy(s).to(device)
        a_base_t = torch.from_numpy(a_base).to(device)
        raw_delta = model(s_t, a_base_t)
        delta = torch.tanh(raw_delta) * max_delta_scale
        a_patch = a_base_t + delta

    delta_np = delta.detach().cpu().numpy().astype(np.float32)
    a_patch_np = a_patch.detach().cpu().numpy().astype(np.float32)

    ref_delta = np.zeros_like(ref_a_base, dtype=np.float32)
    ref_a_patch = ref_a_base.copy()
    ref_a_patch[episode_index, timestep_index] = a_patch_np
    ref_delta[episode_index, timestep_index] = delta_np

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s,
        a_base=a_base,
        delta_a=delta_np,
        a_patch=a_patch_np,
        ref_s=ref_s,
        ref_a_base=ref_a_base,
        ref_a_patch=ref_a_patch,
        ref_delta_a=ref_delta,
        ref_mask=ref_mask,
        ref_episode_lengths=ref_episode_lengths,
        episode_index=episode_index,
        timestep_index=timestep_index,
    )
    print(f"[DONE] saved: {out_npz}")
    print(f"[INFO] num_samples={s.shape[0]}")
    print(f"[INFO] mean_abs_delta_a={float(np.mean(np.abs(delta_np))):.6e}")


if __name__ == "__main__":
    main()

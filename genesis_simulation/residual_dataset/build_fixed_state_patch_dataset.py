import argparse
from pathlib import Path

import numpy as np


def _load_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as d:
        for key in ("s", "a"):
            if key not in d:
                raise KeyError(f"Missing key in npz: {key}")
        s = d["s"].astype(np.float32)
        a = d["a"].astype(np.float32)
        if s.ndim == 3:
            if "mask" in d:
                mask = d["mask"].astype(bool)
            elif "episode_lengths" in d:
                lengths = d["episode_lengths"].astype(np.int64)
                mask = np.zeros(s.shape[:2], dtype=bool)
                for i, l in enumerate(lengths):
                    mask[i, : int(l)] = True
            else:
                raise KeyError("Padded NPZ needs either 'mask' or 'episode_lengths'")
            episode_lengths = np.sum(mask, axis=1).astype(np.int32)
        elif s.ndim == 2:
            mask = np.ones((s.shape[0],), dtype=bool)
            episode_lengths = np.asarray([s.shape[0]], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported s shape: {s.shape}")
    return s, a, mask, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Build flat fixed-state patch dataset from fixed-state Isaac NPZ")
    parser.add_argument("--fixed-state-npz", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    args = parser.parse_args()

    in_path = Path(args.fixed_state_npz)
    out_path = Path(args.out_npz)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    s, a_base, mask, episode_lengths = _load_npz(in_path)

    if s.ndim == 3:
        ep_idx, t_idx = np.nonzero(mask)
        s_flat = s[mask]
        a_base_flat = a_base[mask]
    else:
        ep_idx = np.zeros((s.shape[0],), dtype=np.int32)
        t_idx = np.arange(s.shape[0], dtype=np.int32)
        s_flat = s
        a_base_flat = a_base

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        s=s_flat.astype(np.float32),
        a_base=a_base_flat.astype(np.float32),
        ref_s=s.astype(np.float32),
        ref_a_base=a_base.astype(np.float32),
        ref_mask=mask,
        ref_episode_lengths=episode_lengths,
        episode_index=ep_idx.astype(np.int32),
        timestep_index=t_idx.astype(np.int32),
    )

    print(f"[DONE] saved: {out_path}")
    print(f"[INFO] num_samples={s_flat.shape[0]}")
    print(f"[INFO] state_dim={s_flat.shape[1]}")
    print(f"[INFO] action_dim={a_base_flat.shape[1]}")


if __name__ == "__main__":
    main()

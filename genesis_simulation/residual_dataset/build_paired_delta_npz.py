import argparse
from pathlib import Path

import numpy as np


def require_keys(d, keys, name):
    for k in keys:
        if k not in d:
            raise KeyError(f"{name} missing key: {k}")


def flatten_with_mask(s, a, y, mask):
    if s.ndim != 3 or a.ndim != 3 or y.ndim != 3 or mask.ndim != 2:
        raise ValueError("Expected padded arrays: s/a/y=[E,T,*], mask=[E,T]")
    return s[mask], a[mask], y[mask]


def pad_from_flat(s, a, y, episode_lengths):
    e = len(episode_lengths)
    t_max = int(np.max(episode_lengths)) if e > 0 else 0
    sd = s.shape[-1]
    ad = a.shape[-1]
    yd = y.shape[-1]

    s_pad = np.zeros((e, t_max, sd), dtype=np.float32)
    a_pad = np.zeros((e, t_max, ad), dtype=np.float32)
    y_pad = np.zeros((e, t_max, yd), dtype=np.float32)
    mask = np.zeros((e, t_max), dtype=np.bool_)

    cur = 0
    for i, l in enumerate(episode_lengths):
        l = int(l)
        if l <= 0:
            continue
        nxt = cur + l
        s_pad[i, :l] = s[cur:nxt]
        a_pad[i, :l] = a[cur:nxt]
        y_pad[i, :l] = y[cur:nxt]
        mask[i, :l] = True
        cur = nxt

    if cur != s.shape[0]:
        raise ValueError("episode_lengths does not match flattened sample count")

    return s_pad, a_pad, y_pad, mask


def main():
    parser = argparse.ArgumentParser(description="Build paired delta dataset: delta_target = s_next_isaac - s_next_genesis")
    parser.add_argument("--isaac-npz", type=str, required=True)
    parser.add_argument("--genesis-npz", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--flat", action="store_true", help="Save flattened [N,*] arrays instead of padded [E,T,*]")
    args = parser.parse_args()

    isaac_path = Path(args.isaac_npz)
    genesis_path = Path(args.genesis_npz)
    out_path = Path(args.out_npz)

    if not isaac_path.is_file():
        raise FileNotFoundError(f"isaac npz not found: {isaac_path}")
    if not genesis_path.is_file():
        raise FileNotFoundError(f"genesis npz not found: {genesis_path}")

    with np.load(isaac_path, allow_pickle=False) as di, np.load(genesis_path, allow_pickle=False) as dg:
        require_keys(di, ["s", "a", "s_next"], "isaac")
        require_keys(dg, ["s", "a", "s_next"], "genesis")

        si = di["s"].astype(np.float32)
        ai = di["a"].astype(np.float32)
        sni = di["s_next"].astype(np.float32)

        sg = dg["s"].astype(np.float32)
        ag = dg["a"].astype(np.float32)
        sng = dg["s_next"].astype(np.float32)

        if si.shape != sg.shape or ai.shape != ag.shape or sni.shape != sng.shape:
            raise ValueError(
                "Isaac and Genesis shapes do not match. "
                f"s: {si.shape} vs {sg.shape}, a: {ai.shape} vs {ag.shape}, s_next: {sni.shape} vs {sng.shape}"
            )

        if not np.allclose(si, sg, atol=1e-6, rtol=1e-6):
            max_diff = float(np.max(np.abs(si - sg)))
            raise ValueError(f"State anchors s are not aligned (max abs diff={max_diff:.6e}). Please collect paired data with same seeds/init.")
        if not np.allclose(ai, ag, atol=1e-6, rtol=1e-6):
            max_diff = float(np.max(np.abs(ai - ag)))
            raise ValueError(f"Action anchors a are not aligned (max abs diff={max_diff:.6e}). Please collect paired data with same policy/actions.")

        if "mask" in di and "mask" in dg:
            mi = di["mask"].astype(bool)
            mg = dg["mask"].astype(bool)
            if mi.shape != mg.shape:
                raise ValueError(f"mask shape mismatch: {mi.shape} vs {mg.shape}")
            if not np.array_equal(mi, mg):
                raise ValueError("mask mismatch between isaac and genesis")
            mask = mi
        else:
            if si.ndim == 3:
                if "episode_lengths" in di and "episode_lengths" in dg:
                    li = di["episode_lengths"].astype(np.int32)
                    lg = dg["episode_lengths"].astype(np.int32)
                    if li.shape != lg.shape or not np.array_equal(li, lg):
                        raise ValueError("episode_lengths mismatch between isaac and genesis")
                    e, t = si.shape[:2]
                    mask = np.zeros((e, t), dtype=np.bool_)
                    for i, l in enumerate(li):
                        mask[i, : int(l)] = True
                else:
                    raise ValueError("Padded dataset requires mask or episode_lengths in both npz files")
            else:
                mask = np.ones((si.shape[0],), dtype=np.bool_)

        delta_target = (sni - sng).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.flat:
            if si.ndim == 3:
                s_flat, a_flat, d_flat = flatten_with_mask(si, ai, delta_target, mask)
                np.savez_compressed(out_path, s=s_flat, a=a_flat, delta_target=d_flat)
                n = int(s_flat.shape[0])
            else:
                np.savez_compressed(out_path, s=si, a=ai, delta_target=delta_target)
                n = int(si.shape[0])
        else:
            if si.ndim == 3:
                if "episode_lengths" in di:
                    ep_len = di["episode_lengths"].astype(np.int32)
                else:
                    ep_len = np.sum(mask, axis=1).astype(np.int32)
                np.savez_compressed(
                    out_path,
                    s=si,
                    a=ai,
                    delta_target=delta_target,
                    mask=mask,
                    episode_lengths=ep_len,
                )
                n = int(np.sum(mask))
            else:
                np.savez_compressed(out_path, s=si, a=ai, delta_target=delta_target)
                n = int(si.shape[0])

        print(f"[DONE] saved: {out_path}")
        print(f"[INFO] samples={n}")
        print(f"[INFO] delta_target mean_abs={float(np.mean(np.abs(delta_target))):.6e}")


if __name__ == "__main__":
    main()

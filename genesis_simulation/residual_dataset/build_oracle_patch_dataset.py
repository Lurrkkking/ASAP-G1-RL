import argparse
from pathlib import Path

import numpy as np


def _sensitivity_from_weighted_jacobian(weighted_jacobian: np.ndarray) -> np.ndarray:
    # diag(J^T J) in the weighted state space gives per-joint local sensitivity.
    sens = np.sum(weighted_jacobian ** 2, axis=1)
    return sens.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build supervised oracle patch dataset from local-linear a_star output")
    parser.add_argument("--oracle-npz", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--min-eta", type=float, default=0.0)
    parser.add_argument("--min-abs-improvement", type=float, default=0.0)
    parser.add_argument("--topk-frac", type=float, default=1.0, help="keep top fraction by sample weight after thresholds")
    parser.add_argument("--weight-eps", type=float, default=1e-6)
    args = parser.parse_args()

    oracle_npz = Path(args.oracle_npz)
    out_npz = Path(args.out_npz)
    if not oracle_npz.is_file():
        raise FileNotFoundError(oracle_npz)

    with np.load(oracle_npz, allow_pickle=False) as d:
        required = ("s", "a_base", "delta_a_star", "weighted_jacobian", "eta", "improvement")
        for key in required:
            if key not in d:
                raise KeyError(f"Missing key in oracle npz: {key}")
        s = d["s"].astype(np.float32)
        a_base = d["a_base"].astype(np.float32)
        delta_a_star = d["delta_a_star"].astype(np.float32)
        weighted_jacobian = d["weighted_jacobian"].astype(np.float32)
        eta = d["eta"].astype(np.float32)
        improvement = d["improvement"].astype(np.float32)
        weights = d["weights"].astype(np.float32) if "weights" in d else None
        action_dof_indices = d["action_dof_indices"].astype(np.int64) if "action_dof_indices" in d else None

    keep = np.ones((s.shape[0],), dtype=bool)
    keep &= eta >= float(args.min_eta)
    keep &= improvement >= float(args.min_abs_improvement)

    sensitivity = _sensitivity_from_weighted_jacobian(weighted_jacobian)
    sample_weight = np.maximum(improvement, 0.0) * np.maximum(eta, 0.0)

    if args.topk_frac < 1.0:
        if not (0.0 < args.topk_frac <= 1.0):
            raise ValueError("--topk-frac must be in (0, 1]")
        keep_idx = np.nonzero(keep)[0]
        if keep_idx.size > 0:
            topk = max(1, int(np.ceil(keep_idx.size * args.topk_frac)))
            ranked = keep_idx[np.argsort(sample_weight[keep_idx])[::-1]]
            selected = ranked[:topk]
            keep = np.zeros_like(keep)
            keep[selected] = True

    if not np.any(keep):
        raise ValueError("No samples left after oracle filtering")

    sample_weight = sample_weight[keep]
    sample_weight = sample_weight / max(float(sample_weight.mean()), args.weight_eps)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "s": s[keep],
        "a_base": a_base[keep],
        "delta_a_star": delta_a_star[keep],
        "weighted_jacobian": weighted_jacobian[keep],
        "sensitivity": sensitivity[keep],
        "eta": eta[keep],
        "improvement": improvement[keep],
        "sample_weight": sample_weight.astype(np.float32),
        "keep_mask": keep,
        "min_eta": np.asarray(args.min_eta, dtype=np.float32),
        "min_abs_improvement": np.asarray(args.min_abs_improvement, dtype=np.float32),
        "topk_frac": np.asarray(args.topk_frac, dtype=np.float32),
    }
    if weights is not None:
        payload["state_weights"] = weights
    if action_dof_indices is not None:
        payload["action_dof_indices"] = action_dof_indices

    np.savez_compressed(out_npz, **payload)
    print(f"[DONE] saved: {out_npz}")
    print(f"[INFO] kept_samples={int(np.sum(keep))}/{len(keep)}")
    print(f"[INFO] kept_eta_mean={float(np.mean(eta[keep])):.6e}")
    print(f"[INFO] kept_improvement_mean={float(np.mean(improvement[keep])):.6e}")


if __name__ == "__main__":
    main()

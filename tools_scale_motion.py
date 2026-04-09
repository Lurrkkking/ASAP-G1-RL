#!/usr/bin/env python3
import argparse
from pathlib import Path
import copy
import joblib
import numpy as np


def scale_entry(entry: dict, scale: float, scale_smpl_joints: bool) -> dict:
    out = copy.deepcopy(entry)

    if "root_trans_offset" in out:
        arr = np.asarray(out["root_trans_offset"], dtype=np.float32)
        out["root_trans_offset"] = (arr * scale).astype(arr.dtype, copy=False)

    if scale_smpl_joints and "smpl_joints" in out:
        arr = np.asarray(out["smpl_joints"], dtype=np.float32)
        out["smpl_joints"] = (arr * scale).astype(arr.dtype, copy=False)

    return out


def main():
    parser = argparse.ArgumentParser(description="Scale global translation in motion pkl for size adaptation.")
    parser.add_argument("--in-file", required=True, help="Input motion .pkl path")
    parser.add_argument("--out-file", required=True, help="Output motion .pkl path")
    parser.add_argument("--scale", type=float, required=True, help="Scale factor, e.g. 0.92")
    parser.add_argument("--scale-smpl-joints", action="store_true", help="Also scale smpl_joints for visualization consistency")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    data = joblib.load(in_path)

    if isinstance(data, dict):
        scaled = {k: scale_entry(v, args.scale, args.scale_smpl_joints) for k, v in data.items()}
        n = len(scaled)
    elif isinstance(data, list):
        scaled = [scale_entry(v, args.scale, args.scale_smpl_joints) for v in data]
        n = len(scaled)
    else:
        raise TypeError(f"Unsupported top-level type: {type(data)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaled, out_path)

    print(f"[DONE] scaled {n} motion entries")
    print(f"[DONE] scale={args.scale}")
    print(f"[DONE] out={out_path}")


if __name__ == "__main__":
    main()

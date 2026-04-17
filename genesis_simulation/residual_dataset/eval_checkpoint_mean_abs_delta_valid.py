import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_NPZ = REPO_ROOT / "genesis_simulation" / "residual_dataset" / "isaac_26600_anchor.npz"
RL_PYTHON = Path("/root/miniconda3/envs/rl/bin/python")
GENESIS_PYTHON = Path("/root/autodl-tmp/env_genesis/bin/python")


def run_cmd(cmd, cwd, success_artifact: Path = None):
    print("[RUN]", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    env["PATH"] = f"/root/miniconda3/envs/rl/bin:{env.get('PATH', '')}"
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
    except subprocess.CalledProcessError as exc:
        if success_artifact is not None and success_artifact.is_file():
            print(
                f"[WARN] command exited with code {exc.returncode}, but artifact exists: {success_artifact}. "
                "Continuing."
            )
            return
        raise


def compute_mean_abs_delta_valid(paired_npz: Path) -> float:
    with np.load(paired_npz, allow_pickle=False) as d:
        if "delta_target" not in d:
            raise KeyError(f"Missing delta_target in {paired_npz}")
        delta = np.abs(d["delta_target"].astype(np.float32))
        if "mask" in d:
            mask = d["mask"].astype(bool)
            return float(np.mean(delta[mask]))
        return float(np.mean(delta))


def main():
    parser = argparse.ArgumentParser(
        description="Export a PT checkpoint to ONNX, run fixed-state evaluation, and print mean_abs_delta_valid."
    )
    parser.add_argument("checkpoint", type=str, help="PT checkpoint path")
    parser.add_argument(
        "--ref-npz",
        type=str,
        default=str(DEFAULT_REF_NPZ),
        help="Reference fixed-state Isaac NPZ",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional output directory for intermediate npz files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used by Isaac evaluation scripts",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Debug option; -1 means all valid samples",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    ref_npz = Path(args.ref_npz).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not ref_npz.is_file():
        raise FileNotFoundError(f"Reference NPZ not found: {ref_npz}")
    if not RL_PYTHON.is_file():
        raise FileNotFoundError(f"Expected python not found: {RL_PYTHON}")
    if not GENESIS_PYTHON.is_file():
        raise FileNotFoundError(f"Expected python not found: {GENESIS_PYTHON}")

    ckpt_stem = checkpoint.stem
    work_dir = Path(args.out_dir).resolve() if args.out_dir else checkpoint.parent / "fixed_state_eval"
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = checkpoint.parent / "exported" / f"{ckpt_stem}.onnx"
    isaac_npz = work_dir / f"{ckpt_stem}_on_26600states.npz"
    paired_npz = work_dir / f"paired_delta_{ckpt_stem}_on_26600states.npz"

    export_cmd = [
        str(RL_PYTHON),
        "humanoidverse/export_pt_to_onnx.py",
        f"+checkpoint={checkpoint}",
        f"+device={args.device}",
    ]
    run_cmd(export_cmd, REPO_ROOT, success_artifact=onnx_path)

    collect_cmd = [
        str(RL_PYTHON),
        "genesis_simulation/residual_dataset/collect_isaac_from_fixed_states.py",
        "--ref-npz",
        str(ref_npz),
        "--checkpoint",
        str(checkpoint),
        "--onnx",
        str(onnx_path),
        "--out-npz",
        str(isaac_npz),
        "--device",
        args.device,
    ]
    if args.max_samples > 0:
        collect_cmd.extend(["--max-samples", str(args.max_samples)])
    run_cmd(collect_cmd, REPO_ROOT)

    pair_cmd = [
        str(GENESIS_PYTHON),
        "genesis_simulation/residual_dataset/build_paired_delta_from_isaac_anchors.py",
        "--isaac-npz",
        str(isaac_npz),
        "--out-npz",
        str(paired_npz),
    ]
    if args.max_samples > 0:
        pair_cmd.extend(["--max-samples", str(args.max_samples)])
    run_cmd(pair_cmd, REPO_ROOT)

    metric = compute_mean_abs_delta_valid(paired_npz)
    print(f"\nmean_abs_delta_valid={metric:.9f}")
    print(f"paired_npz={paired_npz}")
    print(f"isaac_npz={isaac_npz}")
    print(f"onnx={onnx_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True, help="Path to ASAP raw SMPL npz")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gender", default="neutral")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    data = torch.load(input_path, map_location="cpu")
    if "smpl_params_global" not in data:
        raise KeyError("Missing 'smpl_params_global' in GVHMR result")

    smpl = data["smpl_params_global"]
    required = ["global_orient", "body_pose", "transl", "betas"]
    missing = [k for k in required if k not in smpl]
    if missing:
        raise KeyError(f"Missing SMPL keys: {missing}")

    global_orient = to_numpy(smpl["global_orient"]).astype(np.float32)
    body_pose = to_numpy(smpl["body_pose"]).astype(np.float32)
    trans = to_numpy(smpl["transl"]).astype(np.float32)
    betas = to_numpy(smpl["betas"]).astype(np.float32)

    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise ValueError(f"global_orient must be (T,3), got {global_orient.shape}")
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError(f"body_pose must be (T,63), got {body_pose.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"transl must be (T,3), got {trans.shape}")
    if betas.ndim != 2 or betas.shape[1] < 10:
        raise ValueError(f"betas must be (T,>=10), got {betas.shape}")
    if not (len(global_orient) == len(body_pose) == len(trans) == len(betas)):
        raise ValueError("Frame count mismatch in GVHMR SMPL params")

    poses = np.concatenate(
        [
            global_orient,
            body_pose,
            np.zeros((len(global_orient), 6), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    betas_out = betas[0, :16].astype(np.float32) if betas.shape[1] >= 16 else betas[0].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        poses=poses,
        trans=trans,
        betas=betas_out,
        gender=np.asarray(args.gender),
        mocap_framerate=np.int64(args.fps),
    )

    print(f"saved: {output_path}")
    print(f"frames: {len(poses)}")
    print(f"poses shape: {poses.shape}")
    print(f"trans shape: {trans.shape}")
    print(f"betas shape: {betas_out.shape}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot


def main():
    parser = argparse.ArgumentParser(description="Export motion reference state at a specific time offset to NPZ.")
    parser.add_argument("--config", required=True, help="Path to training config.yaml")
    parser.add_argument("--time-offset", type=float, required=True, help="Reference time offset in seconds")
    parser.add_argument("--motion-id", type=int, default=0, help="Motion index")
    parser.add_argument("--out", required=True, help="Output npz path")
    parser.add_argument("--device", default="cpu", help="Device for motion lib query, default cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    motion_cfg = cfg.robot.motion

    ml = MotionLibRobot(motion_cfg, num_envs=1, device=args.device)
    ml.load_motions(random_sample=False)

    mid = torch.tensor([args.motion_id], dtype=torch.long, device=args.device)
    t = torch.tensor([args.time_offset], dtype=torch.float32, device=args.device)
    offset = torch.zeros((1, 3), dtype=torch.float32, device=args.device)

    res = ml.get_motion_state(mid, t, offset=offset)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        time_offset=np.array([args.time_offset], dtype=np.float32),
        motion_id=np.array([args.motion_id], dtype=np.int32),
        root_pos=res["root_pos"][0].detach().cpu().numpy().astype(np.float32),
        root_rot_xyzw=res["root_rot"][0].detach().cpu().numpy().astype(np.float32),
        root_vel=res["root_vel"][0].detach().cpu().numpy().astype(np.float32),
        root_ang_vel=res["root_ang_vel"][0].detach().cpu().numpy().astype(np.float32),
        dof_pos=res["dof_pos"][0].detach().cpu().numpy().astype(np.float32),
        dof_vel=res["dof_vel"][0].detach().cpu().numpy().astype(np.float32),
    )

    print(f"[DONE] wrote {out}")
    print("[INFO] keys: root_pos, root_rot_xyzw, root_vel, root_ang_vel, dof_pos, dof_vel")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import joblib
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.data_process.fit_smpl_motion import process_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw SMPL npz motion")
    parser.add_argument("--output", required=True, help="Path to retargeted ASAP pkl")
    parser.add_argument("--motion-name", default=None, help="Outer dict key in output pkl")
    parser.add_argument("--robot", default="g1/g1_29dof_anneal_23dof")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    motion_name = args.motion_name or f"0-{input_path.stem}"

    config_dir = str((Path(__file__).resolve().parents[1] / "humanoidverse" / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="base", overrides=[f"+robot={args.robot}"])

    data = process_motion([motion_name], {motion_name: str(input_path)}, cfg)
    if motion_name not in data:
        raise RuntimeError(f"Failed to retarget motion: {motion_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_name: data[motion_name]}, output_path)
    print(f"saved: {output_path}")
    print(f"motion_name: {motion_name}")
    print(OmegaConf.to_yaml(cfg.robot.motion))


if __name__ == "__main__":
    main()

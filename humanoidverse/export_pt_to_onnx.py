import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger

from humanoidverse.utils.logging import HydraLoggerBridge
from utils.config_utils import *  # noqa: E402,F403


def _build_eval_config(override_config: OmegaConf) -> OmegaConf:
    if override_config.checkpoint is None:
        raise ValueError("Missing checkpoint. Please pass +checkpoint=/path/to/model_xxxx.pt")

    checkpoint = Path(override_config.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find training config.yaml near checkpoint: {checkpoint}"
        )

    logger.info(f"Loading training config file from {config_path}")
    train_config = OmegaConf.load(config_path)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

    return OmegaConf.merge(train_config, override_config)


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "export_onnx.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())
    config = _build_eval_config(override_config)

    checkpoint = Path(config.checkpoint)
    ckpt_num = checkpoint.stem.split("_")[-1]

    # Export path defaults to <checkpoint_dir>/exported/<checkpoint_name>.onnx
    default_export_dir = checkpoint.parent / "exported"
    default_onnx_name = checkpoint.with_suffix(".onnx").name
    export_dir = Path(config.get("export_dir", str(default_export_dir)))
    onnx_name = str(config.get("onnx_name", default_onnx_name))

    # Export script should run headless by default.
    config.headless = bool(config.get("headless", True))
    OmegaConf.update(config, "env.config.headless", config.headless, force_add=True)
    OmegaConf.update(
        config,
        "env.config.save_rendering_dir",
        str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"),
        force_add=True,
    )
    OmegaConf.update(config, "env.config.ckpt_dir", str(checkpoint.parent), force_add=True)

    simulator_type = config.simulator["_target_"].split(".")[-1]
    simulation_app = None
    if simulator_type == "IsaacSim":
        from omni.isaac.lab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser(description="Export policy checkpoint to ONNX.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    elif simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.inference_helpers import export_policy_as_onnx

    pre_process_config(config)
    device = config.device if config.get("device", None) else ("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_log_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    export_dir.mkdir(parents=True, exist_ok=True)
    example_obs_dict = algo.get_example_obs()
    export_policy_as_onnx(algo.inference_model, str(export_dir), onnx_name, example_obs_dict)
    logger.info(f"Exported policy as onnx to: {export_dir / onnx_name}")

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()

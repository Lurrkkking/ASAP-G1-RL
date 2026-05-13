import os
import sys
from pathlib import Path
from typing import Dict, List

if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import ninja

    os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import hydra
import joblib
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict


DEFAULT_SOURCE_CONFIG_PATH = (
    "/root/autodl-tmp/ASAP/logs/DeltaA_Sanity/"
    "20260511_204929-openloop_gym2gym_cr7_readtest-delta_a-g1_29dof_anneal_23dof/"
    "config.yaml"
)
DEFAULT_MOTION_PATH = (
    "/root/autodl-tmp/ASAP/humanoidverse/data/motions/"
    "g1_29dof_anneal_23dof/TairanTestbed/singles/"
    "0-CR7_level2_gym_rollout_with_action.pkl"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def tensor_to_np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def disable_domain_rand(cfg):
    if not hasattr(cfg, "domain_rand"):
        return
    for key in list(cfg.domain_rand.keys()):
        value = cfg.domain_rand[key]
        if isinstance(value, bool) and (
            key.startswith("randomize") or key.startswith("push") or key.endswith("noise")
        ):
            cfg.domain_rand[key] = False


def disable_terminations(cfg):
    termination = cfg.env.config.termination
    for key in list(termination.keys()):
        if isinstance(termination[key], bool):
            termination[key] = False


def load_base_config(override_config: OmegaConf) -> OmegaConf:
    diag_cfg = override_config.get("motion_lib_diag", OmegaConf.create())
    source_config_path = str(diag_cfg.get("source_config_path", DEFAULT_SOURCE_CONFIG_PATH))
    motion_path = str(diag_cfg.get("motion_file", DEFAULT_MOTION_PATH))
    if not Path(source_config_path).exists():
        raise FileNotFoundError(f"source_config_path does not exist: {source_config_path}")
    if not Path(motion_path).exists():
        raise FileNotFoundError(f"motion_file does not exist: {motion_path}")

    cfg = OmegaConf.load(source_config_path)
    with open_dict(cfg):
        cfg.headless = True
        cfg.num_envs = 1
        cfg.checkpoint = None
        cfg.use_wandb = False
        cfg.env.config.headless = True
        cfg.env.config.num_envs = 1
        cfg.env.config.noise_to_initial_level = 0
        cfg.env.config.enforce_randomize_motion_start_eval = False
        cfg.env.config.randomize_motion_start_train = False
        cfg.env.config.robot.motion.motion_file = motion_path
        cfg.env.config.save_motion = False
        cfg.env.config.motion_lib_diag_motion_file = motion_path
    disable_domain_rand(cfg)
    disable_terminations(cfg)
    return cfg


def get_device(config):
    import torch

    if config.get("device", None):
        return config.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_motion_dict(path: str):
    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Expected non-empty dict pkl, got {type(data).__name__}")
    key = next(iter(data.keys()))
    return key, data[key]


def get_motion_state(env, motion_time: float):
    import torch

    motion_times = torch.full((env.num_envs,), float(motion_time), dtype=torch.float32, device=env.device)
    return env._motion_lib.get_motion_state(env.motion_ids, motion_times, offset=env.env_origins)


def l2_error(a, b) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(diff))


def metric_summary(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"mean={arr.mean():.6f} max={arr.max():.6f} min={arr.min():.6f}"


def print_preview(rows: List[Dict[str, float]], limit: int = 5):
    print("\n[FRAME_PREVIEW]")
    columns = [
        "frame",
        "motion_time",
        "dof_pos_error",
        "dof_vel_error",
        "root_pos_error",
        "root_rot_error",
        "root_vel_error",
        "root_ang_vel_error",
        "body_vel_error",
        "body_ang_vel_error",
    ]
    print(" ".join(f"{col:>18}" for col in columns))
    for row in rows[:limit]:
        print(
            " ".join(
                f"{int(row[col]):18d}" if col == "frame" else f"{row[col]:18.6f}"
                for col in columns
            )
        )


def print_summary(rows: List[Dict[str, float]]):
    print("\n[SUMMARY]")
    metrics = [
        "dof_pos_error",
        "dof_vel_error",
        "root_pos_error",
        "root_rot_error",
        "root_vel_error",
        "root_ang_vel_error",
        "body_vel_error",
        "body_ang_vel_error",
    ]
    for metric in metrics:
        print(f"{metric:>24} {metric_summary([row[metric] for row in rows])}")


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    config = load_base_config(override_config)
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        raise NotImplementedError("This diagnostic currently supports IsaacGym-style execution only.")
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import logging
    from humanoidverse.utils import config_utils as _config_utils  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "motion_lib_vs_pkl_consistency.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(REPO_ROOT)

    pre_process_config(config)
    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs

    device = get_device(config)
    env = instantiate(config.env, device=device)

    motion_key, motion_dict = load_motion_dict(config.env.config.motion_lib_diag_motion_file)
    fps = float(motion_dict["fps"])
    dt = 1.0 / fps
    num_frames = int(motion_dict["dof"].shape[0])

    print("[INFO]")
    print(f"motion_key={motion_key}")
    print(f"motion_file={config.env.config.motion_lib_diag_motion_file}")
    print(f"motion_fps={fps:.6f}")
    print(f"motion_dt={dt:.6f}")
    print(f"num_frames={num_frames}")

    rows = []
    for t in range(num_frames):
        motion_time = t * dt
        motion_res = get_motion_state(env, motion_time)
        rows.append(
            {
                "frame": t,
                "motion_time": motion_time,
                "dof_pos_error": l2_error(tensor_to_np(motion_res["dof_pos"][0]), motion_dict["dof"][t]),
                "dof_vel_error": l2_error(tensor_to_np(motion_res["dof_vel"][0]), motion_dict["dof_vel"][t]),
                "root_pos_error": l2_error(tensor_to_np(motion_res["root_pos"][0]), motion_dict["root_trans_offset"][t]),
                "root_rot_error": l2_error(tensor_to_np(motion_res["root_rot"][0]), motion_dict["root_rot"][t]),
                "root_vel_error": l2_error(tensor_to_np(motion_res["root_vel"][0]), motion_dict["root_lin_vel"][t]),
                "root_ang_vel_error": l2_error(tensor_to_np(motion_res["root_ang_vel"][0]), motion_dict["root_ang_vel"][t]),
                "body_vel_error": l2_error(tensor_to_np(motion_res["body_vel"][0]), motion_dict["body_lin_vel"][t]),
                "body_ang_vel_error": l2_error(tensor_to_np(motion_res["body_ang_vel"][0]), motion_dict["body_ang_vel"][t]),
            }
        )

    print_preview(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()

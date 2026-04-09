import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _load_cfg(path):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(path)
    except Exception:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _to_ns(data)


def _as_dict(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "keys"):
        return {k: x[k] for k in x.keys()}
    return vars(x)


def _resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_template(s: str, cfg) -> str:
    if not isinstance(s, str):
        return str(s)
    out = s
    if "${robot.asset.robot_type}" in out:
        out = out.replace("${robot.asset.robot_type}", str(cfg.robot.asset.robot_type))
    return out


def _pd_from_config(cfg, joint_names):
    stiff = _as_dict(cfg.robot.control.stiffness)
    damp = _as_dict(cfg.robot.control.damping)
    kp = []
    kd = []
    for name in joint_names:
        found = False
        for key in stiff.keys():
            if key in name:
                kp.append(float(stiff[key]))
                kd.append(float(damp[key]))
                found = True
                break
        if not found:
            raise RuntimeError(f"PD gain missing for joint: {name}")
    return np.array(kp, dtype=np.float32), np.array(kd, dtype=np.float32)


def _print_pd_table(joint_names, kp, kd, kd_scale):
    print("\n=== PD Parameters (from training config) ===")
    print(f"KD_SCALE(for Genesis runner check) = {kd_scale:.6f}")
    print("idx | joint | kp | kd_train | kd_after_scale")
    for i, name in enumerate(joint_names):
        print(f"{i:02d} | {name} | {kp[i]:7.3f} | {kd[i]:8.4f} | {(kd[i] * kd_scale):8.4f}")


def _print_gym_mapping(cfg, joint_names):
    try:
        from isaacgym import gymapi, gymtorch
        import torch
    except Exception as e:
        print(f"\n[WARN] IsaacGym unavailable, skip Gym mapping. reason={e}")
        return None

    print("\n=== IsaacGym DOF Mapping ===")
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / float(cfg.simulator.config.sim.fps)
    sim_params.substeps = int(cfg.simulator.config.sim.substeps)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("[WARN] Failed to create IsaacGym sim, skip Gym mapping.")
        return None

    try:
        asset_opts = gymapi.AssetOptions()
        asset_opts.default_dof_drive_mode = int(cfg.robot.asset.default_dof_drive_mode)
        asset_opts.collapse_fixed_joints = bool(cfg.robot.asset.collapse_fixed_joints)
        asset_opts.replace_cylinder_with_capsule = bool(cfg.robot.asset.replace_cylinder_with_capsule)
        asset_opts.flip_visual_attachments = bool(cfg.robot.asset.flip_visual_attachments)
        asset_opts.fix_base_link = bool(cfg.robot.asset.fix_base_link)
        asset_opts.density = float(cfg.robot.asset.density)
        asset_opts.angular_damping = float(cfg.robot.asset.angular_damping)
        asset_opts.linear_damping = float(cfg.robot.asset.linear_damping)
        asset_opts.max_angular_velocity = float(cfg.robot.asset.max_angular_velocity)
        asset_opts.max_linear_velocity = float(cfg.robot.asset.max_linear_velocity)
        asset_opts.armature = float(cfg.robot.asset.armature)
        asset_opts.thickness = float(cfg.robot.asset.thickness)
        asset_opts.disable_gravity = bool(cfg.robot.asset.disable_gravity) if cfg.robot.asset.disable_gravity is not None else False

        repo_root = Path(__file__).resolve().parents[1]
        asset_root = _resolve_path(repo_root, str(cfg.robot.asset.asset_root))
        asset_file = _resolve_template(str(cfg.robot.asset.urdf_file), cfg)
        asset = gym.load_asset(sim, str(asset_root), asset_file, asset_opts)

        env = gym.create_env(sim, gymapi.Vec3(0.0, 0.0, 0.0), gymapi.Vec3(0.0, 0.0, 0.0), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*cfg.robot.init_state.pos)
        actor = gym.create_actor(env, asset, pose, str(cfg.robot.asset.robot_type), 0, int(cfg.robot.asset.self_collisions), 0)
        gym.prepare_sim(sim)

        dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
        gym.refresh_dof_state_tensor(sim)
        dof_pos = dof_state.view(1, -1, 2)[0, :, 0].detach().cpu().numpy()

        gym_names = gym.get_actor_dof_names(env, actor)
        name2idx = {n: i for i, n in enumerate(gym_names)}
        gym_idx = [name2idx[n] for n in joint_names]

        print(f"Gym dof count = {len(gym_names)}")
        print("rl_idx | joint | gym_idx | gym_init_pos")
        for i, name in enumerate(joint_names):
            gi = gym_idx[i]
            print(f"{i:02d} | {name} | {gi:02d} | {dof_pos[gi]: .6f}")
        return gym_idx
    finally:
        gym.destroy_sim(sim)


def _print_genesis_mapping(cfg, joint_names):
    try:
        import torch
        import genesis as gs
    except Exception as e:
        print(f"\n[WARN] Genesis unavailable, skip Genesis mapping. reason={e}")
        return None

    print("\n=== Genesis DOF Mapping ===")
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = _resolve_path(repo_root, os.path.join(str(cfg.robot.asset.asset_root), _resolve_template(str(cfg.robot.asset.urdf_file), cfg)))
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    backend = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(backend=backend)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / float(cfg.simulator.config.sim.fps), substeps=1),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            merge_fixed_links=True,
            links_to_keep=list(cfg.robot.body_names),
            pos=tuple(float(x) for x in cfg.robot.init_state.pos),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )
    scene.build()

    dof_ids = [robot.get_joint(name).dof_idx_local for name in joint_names]
    angles = _as_dict(cfg.robot.init_state.default_joint_angles)
    default_pos = [float(angles[name]) for name in joint_names]
    robot.set_dofs_position(position=torch.tensor(default_pos, dtype=torch.float32), dofs_idx_local=dof_ids)
    pos = robot.get_dofs_position(dof_ids).detach().cpu().numpy().reshape(-1)

    print("rl_idx | joint | genesis_dof_idx | genesis_pos_after_set")
    for i, name in enumerate(joint_names):
        print(f"{i:02d} | {name} | {dof_ids[i]:02d} | {pos[i]: .6f}")
    return dof_ids


def main():
    parser = argparse.ArgumentParser(description="Compare IsaacGym/Genesis DOF order and PD params for a checkpoint config.")
    parser.add_argument("--config", required=True, help="Path to training config.yaml (e.g., logs/.../config.yaml).")
    parser.add_argument("--kd-scale", type=float, default=1.5, help="KD scale used in genesis_simulation/run_onnx_motiontracking.py")
    parser.add_argument("--skip-gym", action="store_true", help="Skip IsaacGym mapping check.")
    parser.add_argument("--skip-genesis", action="store_true", help="Skip Genesis mapping check.")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    joint_names = list(cfg.robot.dof_names)

    kp, kd = _pd_from_config(cfg, joint_names)
    _print_pd_table(joint_names, kp, kd, args.kd_scale)

    print("\n=== Domain Rand PD Scale ===")
    print(f"randomize_pd_gain = {bool(cfg.domain_rand.randomize_pd_gain)}")
    print(f"kp_range = {list(cfg.domain_rand.kp_range)}")
    print(f"kd_range = {list(cfg.domain_rand.kd_range)}")
    if not bool(cfg.domain_rand.randomize_pd_gain):
        print("effective _kp_scale = 1.0, _kd_scale = 1.0 (no random scaling in training/eval env)")

    gym_idx = None if args.skip_gym else _print_gym_mapping(cfg, joint_names)
    genesis_idx = None if args.skip_genesis else _print_genesis_mapping(cfg, joint_names)

    if gym_idx is not None and genesis_idx is not None:
        mismatch = [(i, g, ge) for i, (g, ge) in enumerate(zip(gym_idx, genesis_idx)) if g != ge]
        print("\n=== Mapping Check (Gym idx vs Genesis idx under same RL order) ===")
        if len(mismatch) == 0:
            print("PASS: all 23 joints have identical index ordering.")
        else:
            print(f"FAIL: {len(mismatch)} mismatches")
            for i, g, ge in mismatch:
                print(f"rl_idx={i:02d}, gym_idx={g:02d}, genesis_idx={ge:02d}, joint={joint_names[i]}")


if __name__ == "__main__":
    main()

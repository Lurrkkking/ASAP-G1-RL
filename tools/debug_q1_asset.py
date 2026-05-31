"""
Minimal Q1 Isaac Gym asset debug script.
Loads Q1 URDF, prints dof/body names, validates against config.
No PPO, no motion_file, no training.
"""
import os
import sys
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from isaacgym import gymapi, gymtorch


def main():
    # 1. Load robot config
    config_path = PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof.yaml"
    config = OmegaConf.load(str(config_path))
    robot_cfg = config.robot

    print("=" * 60)
    print("Q1 Isaac Gym Asset Debug")
    print("=" * 60)

    # 2. URDF path
    asset_root = str(PROJECT_ROOT / "humanoidverse/data/robots/q1")
    urdf_file = "q1_22dof.urdf"
    asset_path = os.path.join(asset_root, urdf_file)
    print(f"\n[URDF] {asset_path}")
    print(f"[URDF exists] {os.path.exists(asset_path)}")

    # 3. Init Isaac Gym
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True

    device_type = "cuda" if sim_params.use_gpu_pipeline else "cpu"
    sim_device_id = 0
    sim = gym.create_sim(sim_device_id, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("ERROR: Failed to create sim")
        return
    print("[Sim] created OK")

    # 4. Asset options
    asset_options = gymapi.AssetOptions()
    asset_cfg = robot_cfg.asset
    asset_options.collapse_fixed_joints = asset_cfg.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = asset_cfg.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = asset_cfg.flip_visual_attachments
    asset_options.fix_base_link = asset_cfg.fix_base_link
    asset_options.density = asset_cfg.density
    asset_options.angular_damping = asset_cfg.angular_damping
    asset_options.linear_damping = asset_cfg.linear_damping
    asset_options.max_angular_velocity = asset_cfg.max_angular_velocity
    asset_options.max_linear_velocity = asset_cfg.max_linear_velocity
    asset_options.armature = asset_cfg.armature
    asset_options.thickness = asset_cfg.thickness
    asset_options.default_dof_drive_mode = asset_cfg.default_dof_drive_mode

    print(f"\n[AssetOptions] collapse_fixed_joints = {asset_options.collapse_fixed_joints}")
    print(f"[AssetOptions] default_dof_drive_mode = {asset_options.default_dof_drive_mode}")

    # 5. Load asset
    asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    print("[Asset] loaded OK")

    # 6. Get DOF info
    num_dof = gym.get_asset_dof_count(asset)
    dof_names_sim = gym.get_asset_dof_names(asset)
    print(f"\n[DOF] Isaac Gym count = {num_dof}")
    print(f"[DOF] YAML count      = {len(robot_cfg.dof_names)}")

    print("\n--- Simulator dof_names ---")
    for i, name in enumerate(dof_names_sim):
        print(f"  {i:2d}: {name}")

    yaml_dof_names = list(robot_cfg.dof_names)
    dof_match = (num_dof == len(yaml_dof_names)) and (dof_names_sim == yaml_dof_names)
    if dof_match:
        print("\n[DOF CHECK] PASS - simulator dof_names match YAML exactly")
    else:
        print("\n[DOF CHECK] FAIL")
        if num_dof != len(yaml_dof_names):
            print(f"  count mismatch: sim={num_dof} vs yaml={len(yaml_dof_names)}")
        else:
            for i, (s, y) in enumerate(zip(dof_names_sim, yaml_dof_names)):
                if s != y:
                    print(f"  index {i}: sim={s} vs yaml={y}")

    # 7. Get body info
    num_bodies = gym.get_asset_rigid_body_count(asset)
    body_names_sim = gym.get_asset_rigid_body_names(asset)
    print(f"\n[BODY] Isaac Gym count = {num_bodies}")
    print(f"[BODY] YAML count      = {len(robot_cfg.body_names)}")

    print("\n--- Simulator body_names ---")
    for i, name in enumerate(body_names_sim):
        print(f"  {i:2d}: {name}")

    yaml_body_names = list(robot_cfg.body_names)
    # Check if difference is only due to head_link (fixed joint)
    sim_body_set = set(body_names_sim)
    yaml_body_set = set(yaml_body_names)
    only_in_sim = sim_body_set - yaml_body_set
    only_in_yaml = yaml_body_set - sim_body_set

    if only_in_sim == {"head_link"} and not only_in_yaml:
        print(f"\n[BODY CHECK] PASS (with expected diff)")
        print(f"  collapse_fixed_joints=True merged head_link → torso_link.")
        print(f"  simulator has head_link; YAML excludes it (correct).")
        print(f"  simulator={num_bodies} bodies, YAML expects={len(yaml_body_names)}")
    elif only_in_sim or only_in_yaml:
        print(f"\n[BODY CHECK] DIFF")
        if only_in_sim:
            print(f"  Only in simulator: {only_in_sim}")
        if only_in_yaml:
            print(f"  Only in YAML: {only_in_yaml}")
        print(f"  Check collapse_fixed_joints setting.")
    else:
        if body_names_sim == yaml_body_names:
            print(f"\n[BODY CHECK] PASS - simulator body_names match YAML exactly")

    # 8. Build default_dof_pos, p_gains, d_gains (same logic as legged_robot_base.py)
    dof_names_list = dof_names_sim
    default_dof_pos = np.zeros(num_dof, dtype=np.float32)
    for i in range(num_dof):
        name = dof_names_list[i]
        angle = robot_cfg.init_state.default_joint_angles[name]
        default_dof_pos[i] = angle

    p_gains = np.zeros(num_dof, dtype=np.float32)
    d_gains = np.zeros(num_dof, dtype=np.float32)
    stiffness_dict = dict(robot_cfg.control.stiffness)
    damping_dict = dict(robot_cfg.control.damping)

    for i in range(num_dof):
        name = dof_names_list[i]
        found = False
        for key in stiffness_dict:
            if key in name:
                p_gains[i] = stiffness_dict[key]
                d_gains[i] = damping_dict[key]
                found = True
                break
        if not found:
            print(f"  WARNING: no PD gain for {name}")

    print(f"\n[Tensor shapes]")
    print(f"  default_dof_pos.shape = {default_dof_pos.shape}  ({list(default_dof_pos[:6])} ...)")
    print(f"  p_gains.shape         = {p_gains.shape}  ({list(p_gains[:6])} ...)")
    print(f"  d_gains.shape         = {d_gains.shape}  ({list(d_gains[:6])} ...)")

    print(f"\n[Per-joint gains]")
    for i in range(num_dof):
        name = dof_names_list[i]
        print(f"  {i:2d}: {name:40s}  kp={p_gains[i]:6.1f}  kd={d_gains[i]:5.2f}  default={default_dof_pos[i]:+.3f}")

    # 9. Find specific body indices
    # Create a minimal env to use find_actor_rigid_body_handle
    env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
    env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)
    actor_handle = gym.create_actor(env, asset, start_pose, "q1_22dof", 0, 1, 0)

    def get_body_index(name):
        try:
            return gym.find_actor_rigid_body_handle(env, actor_handle, name)
        except:
            return -1

    print(f"\n[Body indices]")
    left_foot_idx = get_body_index("left_ankle_roll_link")
    right_foot_idx = get_body_index("right_ankle_roll_link")
    print(f"  left_ankle_roll_link  = {left_foot_idx}")
    print(f"  right_ankle_roll_link = {right_foot_idx}")

    left_knee_idx = get_body_index("left_knee_link")
    right_knee_idx = get_body_index("right_knee_link")
    print(f"  left_knee_link        = {left_knee_idx}")
    print(f"  right_knee_link       = {right_knee_idx}")

    torso_idx = get_body_index("torso_link")
    pelvis_idx = get_body_index("pelvis")
    print(f"  torso_link (torso)    = {torso_idx}")
    print(f"  pelvis (base)         = {pelvis_idx}")

    # Check extend_config parents exist
    print(f"\n[Extend config parent checks]")
    for ext in robot_cfg.motion.extend_config:
        parent_idx = get_body_index(ext.parent_name)
        print(f"  extend '{ext.joint_name}' parent='{ext.parent_name}' body_idx={parent_idx}")

    # 10. Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  DOF names match:     {'YES' if dof_match else 'NO'}")
    if only_in_sim == {"head_link"} and not only_in_yaml:
        print("  BODY match:          YES (head_link expected diff)")
    elif body_names_sim == yaml_body_names:
        print("  BODY match:          YES")
    else:
        print(f"  BODY match:          NO (only_sim={only_in_sim}, only_yaml={only_in_yaml})")
    print(f"  default_dof_pos:     {default_dof_pos.shape}")
    print(f"  p_gains:             {p_gains.shape}")
    print(f"  d_gains:             {d_gains.shape}")
    print(f"  feet indices:        [{left_foot_idx}, {right_foot_idx}]")
    print(f"  knee indices:        [{left_knee_idx}, {right_knee_idx}]")
    print(f"  torso_index:         {torso_idx}")
    print(f"  pelvis_index:        {pelvis_idx}")
    print(f"  num_dof:             {num_dof}")
    print(f"  num_bodies:          {num_bodies}")

    if dof_match and left_foot_idx >= 0 and right_foot_idx >= 0 and left_knee_idx >= 0 and right_knee_idx >= 0:
        print(f"\n  >>> Ready for standing / locomotion minimal test <<<")
    else:
        print(f"\n  >>> ISSUES FOUND - fix before proceeding <<<")

    # Cleanup
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

"""
Q1 RL Env Zero-Action Rollout — uses real LeggedRobotLocomotion env to verify reset.
"""
import sys, os, csv
from pathlib import Path
import numpy as np, cv2, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

import hydra
from hydra import compose, initialize_config_dir


def pgz(qx, qy, qz, qw): return 1.0 - 2.0*(qx*qx + qy*qy)

def main():
    # Build config manually to avoid full Hydra dependency
    config_path = str(PROJECT_ROOT / "humanoidverse/config")
    robot_yaml = str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml")

    # Load base config components
    base_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/base.yaml"))
    structure = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/base/structure.yaml"))
    robot_conf = OmegaConf.load(robot_yaml)
    env_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/env/locomotion.yaml"))
    algo_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/algo/ppo.yaml"))
    obs_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/obs/loco/leggedloco_obs_singlestep_wolinvel.yaml"))
    rewards_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/rewards/loco/reward_g1_locomotion.yaml"))
    terrain_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/terrain/terrain_locomotion.yaml"))
    sim_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/simulator/isaacgym.yaml"))
    domain_conf = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/domain_rand/NO_domain_rand.yaml"))
    # Merge
    conf = OmegaConf.merge(base_conf, structure)
    conf.robot = robot_conf.robot
    conf.env = env_conf.env
    conf.algo = algo_conf.algo
    conf.obs = obs_conf.obs
    conf.rewards = rewards_conf.rewards
    conf.terrain = terrain_conf.terrain
    conf.simulator = sim_conf.simulator
    conf.domain_rand = domain_conf.domain_rand
    # normalization is already in legged_base.yaml

    # Inject robot/obs/algo into env.config
    conf.env.config.robot = conf.robot
    conf.env.config.obs = conf.obs
    conf.env.config.rewards = conf.rewards
    conf.env.config.terrain = conf.terrain
    conf.env.config.domain_rand = conf.domain_rand
    conf.env.config.simulator = conf.simulator
    conf.env.config.algo = conf.algo

    # Override for minimal test
    conf.num_envs = 4
    conf.headless = True
    conf.env.config.max_episode_length_s = 4.0
    conf.env_spacing = 5.0
    conf.terrain.mesh_type = "plane"  # flat ground
    conf.domain_rand.push_robots = False
    conf.domain_rand.randomize_friction = False
    conf.domain_rand.randomize_link_mass = False
    conf.domain_rand.randomize_pd_gain = False
    conf.domain_rand.randomize_ctrl_delay = False
    conf.simulator.config.sim.control_decimation = 4

    # Set obs dims (use G1-like defaults for smoke test)
    conf.robot.policy_obs_dim = 45
    conf.robot.critic_obs_dim = 238

    print("=" * 70)
    print("Q1 RL ENV ZERO-ACTION ROLLOUT")
    print("  robot: %s" % conf.robot.asset.robot_type)
    print("  urdf: %s" % conf.robot.asset.urdf_file)
    print("  num_envs=%d  action_scale=%.2f  control_type=%s" %
          (conf.num_envs, conf.robot.control.action_scale, conf.robot.control.control_type))
    print("  policy_obs_dim=%d  critic_obs_dim=%d" %
          (conf.robot.policy_obs_dim, conf.robot.critic_obs_dim))
    print("=" * 70)

    # Create env
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = instantiate(config=conf.env, device=device)

    # Reset all envs
    obs = env.reset_all()
    print("\n  Reset complete. dim_actions=%d  obs_keys=%s" %
          (env.dim_actions, list(obs.keys())))

    # Collect zero-action data
    all_data = []
    for step in range(100):  # 2 seconds at 50Hz
        actions = torch.zeros(env.num_envs, env.dim_actions, device=device)
        obs_dict, rew, reset, extras = env.step({"actions": actions})

        for ei in range(min(1, env.num_envs)):  # track env 0
            root_states = env.simulator.robot_root_states[ei].cpu().numpy()
            dof_pos = env.simulator.dof_pos[ei].cpu().numpy()
            dof_vel = env.simulator.dof_vel[ei].cpu().numpy()
            torques = env.torques[ei].cpu().numpy() if hasattr(env, 'torques') else np.zeros(env.dim_actions)
            qx, qy, qz_, qw = root_states[3:7]; pgz_v = pgz(qx, qy, qz_, qw)
            all_data.append({
                'frame': step, 'rz': root_states[2], 'pgz': pgz_v,
                'lv_norm': np.linalg.norm(root_states[7:10]),
                'av_norm': np.linalg.norm(root_states[10:13]),
                'tau_max': abs(torques).max(),
                'knee_L': dof_pos[3], 'knee_R': dof_pos[9],
            })
            if step < 10:
                print("  f%d: rz=%.3f pgz=%.3f lv=%.2f av=%.2f tau=%.1f knee=[%.2f,%.2f]" %
                      (step, root_states[2], pgz_v, np.linalg.norm(root_states[7:10]),
                       np.linalg.norm(root_states[10:13]), abs(torques).max(), dof_pos[3], dof_pos[9]))

    # Print summary
    if all_data:
        max_lv = max(d['lv_norm'] for d in all_data)
        max_av = max(d['av_norm'] for d in all_data)
        fell_frame = next((d['frame'] for d in all_data if d['pgz'] < 0.5), 100)

    print("\n  INIT STATE (env 0 after reset):")
    rs0 = env.simulator.robot_root_states[0].cpu().numpy()
    dp0 = env.simulator.dof_pos[0].cpu().numpy()
    dv0 = env.simulator.dof_vel[0].cpu().numpy()
    print("  root_pos=[%.3f,%.3f,%.3f] quat=[%.3f,%.3f,%.3f,%.3f]" % tuple(rs0[:7]))
    print("  root_vel=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]" % tuple(rs0[7:13]))
    print("  knee=[%.3f,%.3f] hipP=[%.3f,%.3f] ankleP=[%.3f,%.3f]" %
          (dp0[3], dp0[9], dp0[0], dp0[6], dp0[4], dp0[10]))
    print("  dof_vel_max=%.3f" % abs(dv0).max())

    print("\n  SUMMARY:")
    print("  max_lv=%.2f  max_av=%.2f  fell_at_frame=%d" % (max_lv, max_av, fell_frame))
    if max_lv < 5.0 and max_av < 20.0 and not np.isnan(max_lv):
        print("  RESULT: ENV RESET CLEAN — no explosion, natural fall")
    else:
        print("  RESULT: EXPLOSION DETECTED — check env reset flow!")

    env.simulator.gym.destroy_sim(env.simulator.sim)


if __name__ == "__main__":
    main()

"""
Q1 Stand Smoke Zero-Action Rollout — formal RL env, verify reset clean.
Isaac Gym must be imported before torch.
"""
import sys, os, csv, cv2, numpy as np
from pathlib import Path

# isaacgym BEFORE torch
from isaacgym import gymapi  # noqa
sys.path.insert(0, "/root/autodl-tmp/ASAP")

PROJECT_ROOT = Path("/root/autodl-tmp/ASAP")
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import torch
from omegaconf import OmegaConf


def pgz(qx, qy, qz, qw): return 1.0 - 2.0*(qx*qx + qy*qy)


def main():
    # Load config the same way train_agent.py does
    base = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/base.yaml"))
    structure = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/base/structure.yaml"))
    exp = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/exp/q1_stand_smoke.yaml"))

    # Merge: base → structure → exp overrides
    conf = OmegaConf.merge(base, structure)
    conf.algo = exp.algo
    conf.env = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/env/q1_stand_smoke.yaml")).env

    # Load all groups
    conf.robot = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/robot/q1/q1_22dof_rl_collision.yaml")).robot
    conf.obs = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/obs/loco/leggedloco_obs_singlestep_wolinvel.yaml")).obs
    conf.rewards = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/rewards/loco/reward_q1_stand_smoke.yaml")).rewards
    conf.terrain = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/terrain/terrain_locomotion.yaml")).terrain
    conf.simulator = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/simulator/isaacgym.yaml")).simulator
    conf.domain_rand = OmegaConf.load(str(PROJECT_ROOT / "humanoidverse/config/domain_rand/NO_domain_rand.yaml")).domain_rand

    # Override for minimal test
    conf.num_envs = 4
    conf.headless = True
    conf.env_spacing = 5.0
    conf.terrain.mesh_type = "plane"
    conf.simulator.config.sim.control_decimation = 4
    conf.robot.policy_obs_dim = 45
    conf.robot.critic_obs_dim = 238
    conf.save_rendering_dir = None
    conf.add_extra_action = False
    conf.env.config.save_rendering_dir = None

    # Inject into env.config (same as train_agent.py)
    conf.env.config.robot = conf.robot
    conf.env.config.obs = conf.obs
    conf.env.config.rewards = conf.rewards
    conf.env.config.terrain = conf.terrain
    conf.env.config.domain_rand = conf.domain_rand
    conf.env.config.simulator = conf.simulator
    conf.env.config.algo = conf.algo

    print("=" * 70)
    print("Q1 STAND SMOKE — Zero-Action RL Env Rollout")
    print("  robot=%s  num_envs=%d  max_ep_len=%.1fs" %
          (conf.robot.asset.robot_type, conf.num_envs, conf.env.config.max_episode_length_s))
    print("=" * 70)

    # Instantiate env — directly (not via Hydra, to avoid nested instantiation issues)
    from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
    device = "cuda:0"
    env = LeggedRobotLocomotion(config=conf, device=device)

    # Reset all
    obs = env.reset_all()
    print("\n  Reset OK. dim_actions=%d" % env.dim_actions)

    # Print init state
    rs0 = env.simulator.robot_root_states[0].cpu().numpy()
    dp0 = env.simulator.dof_pos[0].cpu().numpy()
    dv0 = env.simulator.dof_vel[0].cpu().numpy()
    default_dp = env.default_dof_pos[0].cpu().numpy()
    print("  Env 0 init:")
    print("    root_z=%.3f  quat=[%.3f,%.3f,%.3f,%.3f]" % (rs0[2], rs0[3], rs0[4], rs0[5], rs0[6]))
    print("    lv=[%.3f,%.3f,%.3f]  av=[%.3f,%.3f,%.3f]" % tuple(rs0[7:13]))
    print("    dof_vel_max=%.4f" % abs(dv0).max())
    print("    knee=[%.3f,%.3f] vs default=[%.3f,%.3f]" % (dp0[3], dp0[9], default_dp[3], default_dp[9]))
    print("    hipP=[%.3f,%.3f] ankleP=[%.3f,%.3f]" % (dp0[0], dp0[6], dp0[4], dp0[10]))

    # Run zero-action
    rows = []
    max_lv, max_av = 0, 0
    fell_frame = None

    for step in range(200):
        actions = torch.zeros(env.num_envs, env.dim_actions, device=device)
        obs_dict, rew, reset, extras = env.step({"actions": actions})

        rs = env.simulator.robot_root_states[0].cpu().numpy()
        dp = env.simulator.dof_pos[0].cpu().numpy()
        dv = env.simulator.dof_vel[0].cpu().numpy()
        torques = env.torques[0].cpu().numpy() if hasattr(env, 'torques') else np.zeros(env.dim_actions)
        qx, qy, qz_, qw = rs[3:7]; pgz_v = pgz(qx, qy, qz_, qw)
        lv_n = np.linalg.norm(rs[7:10]); av_n = np.linalg.norm(rs[10:13])
        max_lv = max(max_lv, lv_n); max_av = max(max_av, av_n)
        if fell_frame is None and pgz_v < 0.5:
            fell_frame = step

        if step < 10 or step % 50 == 0:
            print("  f%3d: rz=%.3f pgz=%.3f lv=%.2f av=%.2f tau=%.1f knee=[%.2f,%.2f]" %
                  (step, rs[2], pgz_v, lv_n, av_n, abs(torques).max(), dp[3], dp[9]))

        rows.append({'frame': step, 'rz': rs[2], 'pgz': pgz_v, 'lv_norm': lv_n,
                     'av_norm': av_n, 'tau_max': abs(torques).max(), 'knee_L': dp[3], 'knee_R': dp[9]})

    exit_ok = max_lv < 5.0 and max_av < 30.0 and not np.isnan(max_lv)
    print("\n  SUMMARY: max_lv=%.2f max_av=%.2f fell_at=%s exploded=%s" %
          (max_lv, max_av, fell_frame, not exit_ok))
    print("  RESULT: %s" % ("RL ENV RESET CLEAN" if exit_ok else "EXPLOSION — check env reset!"))

    if rows:
        with open(str(OUTPUT_DIR / "q1_stand_smoke_rollout.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    env.simulator.gym.destroy_sim(env.simulator.sim)


if __name__ == "__main__":
    main()

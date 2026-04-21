#!/bin/bash
set -euo pipefail

EXP_NAME="HitBall_G1_SingleHit_FromScratch"

echo "====================================================="
echo "从头训练 G1 single-hit 颠球启动任务：右脚第一次有效触球"
echo "task: HitBallTask"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  ++simulator._target_=humanoidverse.simulator.isaacgym.isaacgym_hitball.IsaacGym \
  +exp=hitball \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=hitball/reward_hitball \
  +obs=hitball/hitball_obs \
  project_name=TEST_HitBall \
  experiment_name=${EXP_NAME} \
  num_envs=1024 \
  headless=True \
  checkpoint=null \
  auto_load_latest=False \
  algo.config.num_mini_batches=4 \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.2

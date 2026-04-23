#!/bin/bash
set -euo pipefail

EXP_NAME="${1:-BallControl_G1}"
NUM_ENVS="${2:-1024}"
NUM_STEPS_PER_ENV="${3:-24}"
NUM_LEARNING_ITERATIONS="${4:-1000000}"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

echo "====================================================="
echo "启动 G1 ball control 训练"
echo "task: BallControlTask"
echo "experiment_name: ${EXP_NAME}"
echo "num_envs: ${NUM_ENVS}"
echo "num_steps_per_env: ${NUM_STEPS_PER_ENV}"
echo "num_learning_iterations: ${NUM_LEARNING_ITERATIONS}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  ++simulator._target_=humanoidverse.simulator.isaacgym.isaacgym_hitball.IsaacGym \
  +exp=ball_control \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=ball_control/reward_ball_control_minimal \
  +obs=ball_control/ball_control_obs \
  project_name=TEST_BallControl \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  use_wandb=False \
  checkpoint=null \
  auto_load_latest=False \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.8 \
  algo.config.num_steps_per_env="${NUM_STEPS_PER_ENV}" \
  algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}"

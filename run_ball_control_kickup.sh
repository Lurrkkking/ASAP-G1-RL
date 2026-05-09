#!/bin/bash
set -euo pipefail

EXP_NAME="${1:-BallControlKickup_G1}"
NUM_ENVS="${2:-1024}"
NUM_STEPS_PER_ENV="${3:-24}"
NUM_LEARNING_ITERATIONS="${4:-1000000}"
CHECKPOINT="${5:-/root/autodl-tmp/ASAP/logs/TEST_BallControlKickup/20260508_175154-BallControlKickup_G1-ball_control-g1_29dof_anneal_23dof/model_3000.pt}"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

echo "====================================================="
echo "启动 G1 ball control kickup 训练"
echo "task: BallControlTask"
echo "task_mode: kickup"
echo "experiment_name: ${EXP_NAME}"
echo "num_envs: ${NUM_ENVS}"
echo "num_steps_per_env: ${NUM_STEPS_PER_ENV}"
echo "num_learning_iterations: ${NUM_LEARNING_ITERATIONS}"
echo "checkpoint: ${CHECKPOINT}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  ++simulator._target_=humanoidverse.simulator.isaacgym.isaacgym_hitball.IsaacGym \
  +exp=ball_control \
  env=ball_control_kickup \
  +robot=g1/g1_29dof_anneal_23dof \
  ++robot.asset.self_collisions=0 \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=ball_control/reward_ball_control_kickup \
  +obs=ball_control/ball_control_obs \
  project_name=TEST_BallControlKickup \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  use_wandb=False \
  checkpoint="${CHECKPOINT}" \
  auto_load_latest=False \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.8 \
  algo.config.num_steps_per_env="${NUM_STEPS_PER_ENV}" \
  algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}"

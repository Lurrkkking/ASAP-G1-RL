#!/bin/bash
set -euo pipefail

EXP_NAME="${1:-HitBallCylinderV1_G1}"
NUM_ENVS="${2:-1024}"
NUM_STEPS_PER_ENV="${3:-24}"
NUM_LEARNING_ITERATIONS="${4:-1000000}"
CKPT="${5:-null}"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

echo "====================================================="
echo "启动 G1 hitball cylinder v1 训练"
echo "task: HitBallCylinderV1"
echo "experiment_name: ${EXP_NAME}"
echo "num_envs: ${NUM_ENVS}"
echo "num_steps_per_env: ${NUM_STEPS_PER_ENV}"
echo "num_learning_iterations: ${NUM_LEARNING_ITERATIONS}"
echo "checkpoint: ${CKPT}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym_hitball \
  +exp=hitball_cylinder_v1 \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  project_name=TEST_HitBallCylinderV1 \
  experiment_name="${EXP_NAME}" \
  num_envs="${NUM_ENVS}" \
  headless=True \
  use_wandb=False \
  checkpoint="${CKPT}" \
  auto_load_latest=False \
  ++algo.config.load_optimizer=False \
  ++algo.config.learn_sigma=False \
  algo.config.init_noise_std=0.5 \
  algo.config.num_steps_per_env="${NUM_STEPS_PER_ENV}" \
  algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}"

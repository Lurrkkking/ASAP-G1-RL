#!/bin/bash
set -euo pipefail

EXP_NAME="${1:-HitBallCylinderV1_G1}"
NUM_ENVS="${2:-1024}"
NUM_STEPS_PER_ENV="${3:-24}"
NUM_LEARNING_ITERATIONS="${4:-1000000}"
CKPT="${5:-/root/autodl-tmp/ASAP/logs/TEST_BallControlKickup/proper_height_and_body_stable_worth/model_1200.pt}"
WARMSTART_OUTPUT=""

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

if [[ "${CKPT}" != "null" && -f "${CKPT}" ]]; then
  CKPT_DIR="$(dirname "${CKPT}")"
  CKPT_BASE="$(basename "${CKPT}" .pt)"
  WARMSTART_OUTPUT="${CKPT_DIR}/${CKPT_BASE}_hitball_cylinder_v1_warmstart.pt"
  echo "检测到 checkpoint，先转换为 hitball_cylinder_v1 可加载的 warmstart ckpt"
  echo "source checkpoint: ${CKPT}"
  echo "warmstart checkpoint: ${WARMSTART_OUTPUT}"
  /root/miniconda3/envs/rl/bin/python /root/autodl-tmp/ASAP/tools/make_hitball_warmstart_ckpt.py \
    --source "${CKPT}" \
    --output "${WARMSTART_OUTPUT}" \
    --shared-actor-cols 75 \
    --shared-critic-cols 78 \
    --task-cols 33 \
    --new-input-init-std 0.0
  CKPT="${WARMSTART_OUTPUT}"
fi

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

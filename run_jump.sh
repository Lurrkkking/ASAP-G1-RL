#!/bin/bash

# ==========================================
# G1 Siuuu 续训脚本（强制使用指定 reward 文件）
# ==========================================

set -euo pipefail

# 1. checkpoint 路径
CKPT_FILE="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260407_155156-MotionTracking_CR7_V2/model_2900.pt"

# 2. motion 文件路径
MOTION_FILE="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass.pkl"

# 3. 新实验名
NEW_EXP_NAME="MotionTracking_CR7_V2_Jump_BendKnee"

echo "=========================================="
echo "Run from base config (NOT old run snapshot)"
echo "Checkpoint: ${CKPT_FILE}"
echo "Rewards file: humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml"
echo "=========================================="

# 4. 训练命令
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  "robot.motion.motion_file=${MOTION_FILE}" \
  "checkpoint=${CKPT_FILE}" \
  project_name=TEST_CR7_Siuuu \
  experiment_name=${NEW_EXP_NAME} \
  num_envs=4096 \
  headless=True \
  ++algo.config.learn.init_noise_std=1.0

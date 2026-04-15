#!/bin/bash
set -euo pipefail

# 1. 指向你提供的确切断点文件
CHECKPOINT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260412_123559-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_24500.pt"
MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

# 2. 建议改个名字，方便在 TensorBoard 对比续训后的效果
EXP_NAME="MotionTracking_CR7_Siuuu_Resume_V2_Boost"

echo "====================================================="
echo "执行断点续训：加载 24500.pt"
echo "策略：强制重置 Noise 为 1.0，冲击更高起跳高度"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=domain_rand_base \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_CR7_Siuuu \
  experiment_name=${EXP_NAME} \
  num_envs=12288 \
  headless=True \
  checkpoint=${CHECKPOINT} \
  auto_load_latest=False \
  algo.config.num_mini_batches=4 \
  algo.config.init_noise_std=1.0
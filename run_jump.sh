#!/bin/bash
set -euo pipefail

# 依然是那个 0.92 比例的定制版动作库
MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

# 精准指向你刚才断网前抢救下来的那颗“大脑”
CHECKPOINT_FILE="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260409_165618-MotionTracking_CR7_FullSystem_V2_Fresh_Resume-motion_tracking-g1_29dof_anneal_23dof/model_11700.pt"

# 加了 _Resume 后缀，这样你在 TensorBoard 里能清楚看到这是下半场
EXP_NAME="MotionTracking_CR7_FullSystem_V2_Fresh_Resume"

echo "====================================================="
echo "🚀 启动续训：从 model_3200.pt 恢复记忆"
echo "🛠️ 状态：继承 0.92 缩放版的探索状态，继续冲击起跳点"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_CR7_Siuuu \
  experiment_name=${EXP_NAME} \
  num_envs=4096 \
  headless=True \
  checkpoint=${CHECKPOINT_FILE}
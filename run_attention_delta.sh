#!/bin/bash
set -euo pipefail

# ===== 必填：闭环基础策略（23维动作）checkpoint =====
BASE_POLICY_CKPT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260412_123559-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_17900.pt"

# ===== 必填：动作参考 motion 文件 =====
MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

# ===== 可选：若要在已有 Attention-Delta checkpoint 上继续训练，填路径；否则留空 =====
DELTA_CKPT=""

# ===== 训练命名 =====
PROJECT_NAME="DeltaA_Finetune"
EXP_NAME="attention_delta_gate"

# ===== 训练超参数 =====
NUM_ENVS=4096
SAVE_INTERVAL=5
NUM_ITERS=1000
MAX_DELTA_SCALE=0.25
DEVICE="cuda:0"

# 可选：设置 CUDA 可见卡
# export CUDA_VISIBLE_DEVICES=0

echo "====================================================="
echo "启动 Attention-Delta 训练"
echo "BASE_POLICY_CKPT=${BASE_POLICY_CKPT}"
echo "MOTION_FILE=${MOTION_FILE}"
echo "DELTA_CKPT=${DELTA_CKPT}"
echo "====================================================="

CMD=(
  /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py
  +simulator=isaacgym
  +exp=train_delta_a_closed_loop
  +device=${DEVICE}
  +robot=g1/g1_29dof_anneal_23dof
  +terrain=terrain_locomotion_plane
  +obs=delta_a/train_policy_with_delta_a
  +domain_rand=NO_domain_rand_finetune_with_deltaA
  +rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning
  algo.config.policy_checkpoint=${BASE_POLICY_CKPT}
  robot.motion.motion_file=${MOTION_FILE}
  env.config.add_extra_action=True
  env.config.max_delta_scale=${MAX_DELTA_SCALE}
  num_envs=${NUM_ENVS}
  project_name=${PROJECT_NAME}
  experiment_name=${EXP_NAME}
  algo.config.save_interval=${SAVE_INTERVAL}
  algo.config.num_learning_iterations=${NUM_ITERS}
)

# 仅当 DELTA_CKPT 非空时追加 checkpoint（用于继续训练）
if [[ -n "${DELTA_CKPT}" ]]; then
  CMD+=(checkpoint=${DELTA_CKPT})
fi

HYDRA_FULL_ERROR=1 "${CMD[@]}"

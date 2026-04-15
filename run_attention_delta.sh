#!/bin/bash
set -euo pipefail

# ===== 1. 路径定义 (必须区分清楚) =====
# 🔴 这是原始的、纯净的 23 维模型 (用于参考大脑)
ORIGINAL_23DIM_CKPT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260411_104945-MotionTracking_CR7_FullSystem_V2_Fresh_8192-motion_tracking-g1_29dof_anneal_23dof/model_13000.pt"

# 🔴 这是手术后的、撑开到 46 维的模型 (用于主训练大脑起点)
SURGERY_46DIM_CKPT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260411_104945-MotionTracking_CR7_FullSystem_V2_Fresh_8192-motion_tracking-g1_29dof_anneal_23dof/model_13000_46dim_init.pt"

# 动作参考文件
MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

echo "启动 Step 3: 双脑协同微调训练 (防遗忘模式)"
echo "参考大脑 (23D): ${ORIGINAL_23DIM_CKPT}"
echo "主脑起点 (46D): ${SURGERY_46DIM_CKPT}"
echo "====================================================="

CMD=(
  /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py
  +simulator=isaacgym
  +exp=train_delta_a_closed_loop 
  +device=cuda:0
  +robot=g1/g1_29dof_anneal_23dof
  +terrain=terrain_locomotion_plane
  +obs=delta_a/train_policy_with_delta_a
  +domain_rand=NO_domain_rand_finetune_with_deltaA
  +rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning 
  
  # 🔴 关键配置 1: 主大脑空间设为 46
  robot.actions_dim=46 
  
  # 🔴 关键配置 2: 给参考大脑 (PPODeltaA) 传原始的 23 维模型
  ++algo.config.policy_checkpoint=${ORIGINAL_23DIM_CKPT}
  
  # 🔴 关键配置 3: 让主训练大脑加载手术后的 46 维起点
  checkpoint=${SURGERY_46DIM_CKPT}
  
  ++env.config.delta_ckpt_path=null
  ++env.config.use_gap_reward=True
  ++env.config.gap_model_path=/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_13000/best_residual_dynamics.pt
  ++env.config.gap_reward_scale=0.25
  ++env.config.gap_reward_sign=1.0
  ++env.config.add_extra_action=True
  ++env.config.max_delta_scale=0.06
  # DEBUG: set to 0.0 for one run to verify whether twitching comes from patch branch
  
  # ==========================================
  # 🛡️ 终极防御：三锁操作 (防止高频抽搐与主脑失忆)
  # ==========================================
  ++algo.config.learn_sigma=False       # 锁死探索噪声，不允许 PPO 增大随机抖动
  ++algo.config.init_noise_std=0.15        # 将初始噪声强制压制在极低水平 (对应补丁尺度)
  ++algo.config.actor_learning_rate=1e-5
  ++algo.config.critic_learning_rate=1e-5      # 降低学习率，防止前 23 维肌肉记忆被瞬间洗白
  # ==========================================

  robot.motion.motion_file=${MOTION_FILE}
  num_envs=4096
  project_name=Delta_Patch_Training
  experiment_name=Train_46dim_Patch_Stabilized
  algo.config.save_interval=100
  algo.config.num_learning_iterations=6000
  algo.config.num_steps_per_env=16
  algo.config.num_mini_batches=4
)

HYDRA_FULL_ERROR=1 "${CMD[@]}"
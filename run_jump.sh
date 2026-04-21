  #!/bin/bash
  set -euo pipefail

  MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

  EXP_NAME="MotionTracking_CR7_Siuuu_FromScratch_JumpHigh"

  echo "====================================================="
  echo "从头训练 C罗庆祝动作：重点优化起跳高度"
  echo "motion: ${MOTION_FILE}"
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
    num_envs=3072 \
    headless=True \
    checkpoint=null \
    auto_load_latest=False \
    algo.config.num_mini_batches=4 \
    ++algo.config.learn_sigma=False \
    algo.config.init_noise_std=0.8

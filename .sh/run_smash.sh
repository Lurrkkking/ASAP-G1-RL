#!/bin/bash
set -euo pipefail

MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-smash140_23dof_converted.pkl"
EXP_NAME="MotionTracking_smash140_fromscratch_v1"

echo "====================================================="
echo "从头训练 smash140：先学会起跳和落地"
echo "motion: ${MOTION_FILE}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real_smash \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=SMASH \
  experiment_name=${EXP_NAME} \
  num_envs=3072 \
  headless=True \
  checkpoint=null \
  auto_load_latest=False \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=False \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=False \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  algo.config.num_mini_batches=4 \
  algo.config.learn_sigma=False \
  robot.asset.self_collisions=0
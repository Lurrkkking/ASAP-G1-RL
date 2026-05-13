#!/bin/bash
set -euo pipefail

MOTION_FILE="/root/autodl-tmp/ASAP/outputs/kickball_example_asap_motion_anklefix.pkl"
REWARD_CFG="motion_tracking/reward_motion_tracking_kick_primitive_stable"
OBS_CFG="motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history"
EXP_NAME="MotionTracking_KickPrimitive_Own_Kickball"
CKPT=Null

echo "====================================================="
echo "训练 G1 原地颠球腿法 kick primitive motion tracking"
echo "motion: ${MOTION_FILE}"
echo "reward: ${REWARD_CFG}"
echo "obs: ${OBS_CFG}"
echo "checkpoint: ${CKPT}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking_kick \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=${REWARD_CFG} \
  +obs=${OBS_CFG} \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_KickPrimitive \
  experiment_name=${EXP_NAME} \
  num_envs=1024 \
  headless=True \
  checkpoint=${CKPT} \
  auto_load_latest=False \
  ++env.config.noise_to_initial_level=0.25 \
  ++env.config.randomize_motion_start_train=False \
  ++env.config.termination_scales.termination_motion_far_threshold=0.45 \
  algo.config.init_noise_std=0.8 \
  algo.config.num_steps_per_env=32 \
  algo.config.num_mini_batches=8

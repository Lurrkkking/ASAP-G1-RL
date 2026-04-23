#!/bin/bash
set -euo pipefail

MOTION_FILE="/root/autodl-tmp/ASAP/outputs/kickball_example_asap_motion_anklefix.pkl"
REWARD_CFG="motion_tracking/reward_motion_tracking_right_hitball"
OBS_CFG="motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history"
EXP_NAME="MotionTracking_KickPrimitive_RightHitBall_anklefixed"

echo "====================================================="
echo "训练 G1 原地颠球腿法 kick primitive motion tracking"
echo "motion: ${MOTION_FILE}"
echo "reward: ${REWARD_CFG}"
echo "obs: ${OBS_CFG}"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking_kick \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=domain_rand_base \
  +rewards=${REWARD_CFG} \
  +obs=${OBS_CFG} \
  "robot.motion.motion_file=${MOTION_FILE}" \
  project_name=TEST_KickPrimitive \
  experiment_name=${EXP_NAME} \
  num_envs=1024 \
  headless=True \
  checkpoint=null \
  auto_load_latest=False \
  algo.config.num_steps_per_env=96 \
  algo.config.num_mini_batches=4 

#!/bin/bash
set -euo pipefail

CKPT="/root/autodl-tmp/ASAP/logs/TEST_KickPrimitive/20260424_151051-MotionTracking_KickPrimitive_RightHitBall_anklefixed-motion_tracking-g1_29dof_anneal_23dof/model_4900_hitball_warmstart.pt"
EXP_NAME="HitBall_G1_SingleHit_Motion4900_AnkleFixed"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

echo "====================================================="
echo "从头训练 G1 single-hit 颠球启动任务：右脚第一次有效触球"
echo "task: HitBallTask"
echo "====================================================="

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  ++simulator._target_=humanoidverse.simulator.isaacgym.isaacgym_hitball.IsaacGym \
  +exp=hitball \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  ++robot.motion.motion_file=/root/autodl-tmp/ASAP/outputs/kickball_example_asap_motion_anklefix.pkl \
  +rewards=hitball/reward_hitball \
  +obs=hitball/hitball_obs \
  project_name=TEST2_HitBall \
  experiment_name=${EXP_NAME} \
  num_envs=1024 \
  headless=True \
  checkpoint=${CKPT} \
  auto_load_latest=False \
  ++algo.config.load_optimizer=False \
  algo.config.num_mini_batches=4 \
  ++algo.config.learn_sigma=True\
  algo.config.init_noise_std=0.35

#!/bin/bash
set -euo pipefail

MODE="${1:-zero_action}"
STEPS="${2:-160}"
OUT_DIR="${3:-logs/hitball_debug}"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"
if [[ "${OMP_NUM_THREADS:-}" == "0" ]]; then
  export OMP_NUM_THREADS=1
fi

HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python debug_hitball_rollout.py \
  +simulator=isaacgym \
  ++simulator._target_=humanoidverse.simulator.isaacgym.isaacgym_hitball.IsaacGym \
  +exp=hitball \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +domain_rand=NO_domain_rand \
  +rewards=hitball/reward_hitball \
  +obs=hitball/hitball_obs \
  project_name=TEST_HitBall \
  experiment_name=HitBall_Debug \
  num_envs=1 \
  headless=True \
  checkpoint=null \
  auto_load_latest=False \
  use_wandb=False \
  ++offscreen_record=False \
  ++auto_record=False \
  ++debug_mode="${MODE}" \
  ++debug_steps="${STEPS}" \
  ++debug_out_dir="${OUT_DIR}"

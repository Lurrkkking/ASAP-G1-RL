#!/bin/bash
set -euo pipefail

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

# Baseline 23D policy validated in residual-gap experiments.
ORIGINAL_23DIM_CKPT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600.pt"

# Optional 46D initialization for trainable PPO patch head.
SURGERY_46DIM_CKPT="/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600_46dim_init.pt"

# Residual model validated on the 26600 baseline.
GAP_MODEL_PATH_DEFAULT="/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_26600_posw16/best_residual_dynamics.pt"
FROZEN_PATCH_CKPT_DEFAULT="/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt"

MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

GAP_MODEL_PATH="${GAP_MODEL_PATH:-${GAP_MODEL_PATH_DEFAULT}}"
FROZEN_PATCH_CKPT="${FROZEN_PATCH_CKPT:-${FROZEN_PATCH_CKPT_DEFAULT}}"
GAP_REWARD_SCALE="${GAP_REWARD_SCALE:-0.20}"
GAP_REWARD_SIGN="${GAP_REWARD_SIGN:-1.0}"
MAX_DELTA_SCALE="${MAX_DELTA_SCALE:-0.06}"
REWARD_CONFIG="${REWARD_CONFIG:-motion_tracking/reward_motion_tracking_dm_2real_gapppo}"
PROJECT_NAME="${PROJECT_NAME:-Delta_Patch_Training}"
TRAIN_ACTION_MODE="${TRAIN_ACTION_MODE:-23}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Train_${TRAIN_ACTION_MODE}dim_Patch_26600_Gap26600}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_ITERS="${NUM_ITERS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-16}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"

if [[ ! -f "${ORIGINAL_23DIM_CKPT}" ]]; then
  echo "Missing 23D checkpoint: ${ORIGINAL_23DIM_CKPT}" >&2
  exit 1
fi

if [[ "${TRAIN_ACTION_MODE}" != "23" && "${TRAIN_ACTION_MODE}" != "46" ]]; then
  echo "Unsupported TRAIN_ACTION_MODE=${TRAIN_ACTION_MODE}; expected 23 or 46" >&2
  exit 1
fi

if [[ "${TRAIN_ACTION_MODE}" == "46" && ! -f "${SURGERY_46DIM_CKPT}" ]]; then
  echo "Missing 46D init checkpoint: ${SURGERY_46DIM_CKPT}" >&2
  echo "Create it first with: python genesis_simulation/residual_dataset/transplant.py" >&2
  exit 1
fi

if [[ ! -f "${FROZEN_PATCH_CKPT}" ]]; then
  echo "Missing frozen patch checkpoint: ${FROZEN_PATCH_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${GAP_MODEL_PATH}" ]]; then
  echo "Missing gap model: ${GAP_MODEL_PATH}" >&2
  exit 1
fi

echo "Launching PPO finetune with 26600 baseline"
echo "train mode      : ${TRAIN_ACTION_MODE}D"
echo "23D checkpoint : ${ORIGINAL_23DIM_CKPT}"
if [[ "${TRAIN_ACTION_MODE}" == "46" ]]; then
  echo "46D init       : ${SURGERY_46DIM_CKPT}"
fi
echo "frozen patch   : ${FROZEN_PATCH_CKPT}"
echo "gap model      : ${GAP_MODEL_PATH}"
echo "gap scale/sign : ${GAP_REWARD_SCALE} / ${GAP_REWARD_SIGN}"
echo "max delta scale: ${MAX_DELTA_SCALE}"
echo "reward config  : ${REWARD_CONFIG}"
echo "====================================================="

ACTION_DIM=23
ADD_EXTRA_ACTION=False
CHECKPOINT_ARG=()
if [[ "${TRAIN_ACTION_MODE}" == "46" ]]; then
  ACTION_DIM=46
  ADD_EXTRA_ACTION=True
  CHECKPOINT_ARG=("checkpoint=${SURGERY_46DIM_CKPT}")
fi

CMD=(
  /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py
  +simulator=isaacgym
  +exp=train_delta_a_closed_loop
  +device=cuda:0
  +robot=g1/g1_29dof_anneal_23dof
  +terrain=terrain_locomotion_plane
  +obs=delta_a/train_policy_with_delta_a
  +domain_rand=NO_domain_rand_finetune_with_deltaA
  +rewards=${REWARD_CONFIG}
  robot.actions_dim=${ACTION_DIM}
  ++algo.config.policy_checkpoint=${ORIGINAL_23DIM_CKPT}
  ++env.config.delta_ckpt_path=${FROZEN_PATCH_CKPT}
  ++env.config.use_gap_reward=True
  ++env.config.gap_model_path=${GAP_MODEL_PATH}
  ++env.config.gap_reward_scale=${GAP_REWARD_SCALE}
  ++env.config.gap_reward_sign=${GAP_REWARD_SIGN}
  ++env.config.add_extra_action=${ADD_EXTRA_ACTION}
  ++env.config.max_delta_scale=${MAX_DELTA_SCALE}
  ++algo.config.learn_sigma=False
  ++algo.config.init_noise_std=0.15
  ++algo.config.actor_learning_rate=1e-5
  ++algo.config.critic_learning_rate=1e-5
  robot.motion.motion_file=${MOTION_FILE}
  num_envs=${NUM_ENVS}
  project_name=${PROJECT_NAME}
  experiment_name=${EXPERIMENT_NAME}
  algo.config.save_interval=${SAVE_INTERVAL}
  algo.config.num_learning_iterations=${NUM_ITERS}
  algo.config.num_steps_per_env=${NUM_STEPS_PER_ENV}
  algo.config.num_mini_batches=${NUM_MINI_BATCHES}
  "${CHECKPOINT_ARG[@]}"
)

HYDRA_FULL_ERROR=1 "${CMD[@]}"

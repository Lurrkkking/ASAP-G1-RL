#!/bin/bash
set -euo pipefail

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"

FROZEN_PATCH_CKPT_DEFAULT="/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt"
GAP_MODEL_PATH_DEFAULT="/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_26600_posw16/best_residual_dynamics.pt"
MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

FROZEN_PATCH_CKPT="${FROZEN_PATCH_CKPT:-${FROZEN_PATCH_CKPT_DEFAULT}}"
GAP_MODEL_PATH="${GAP_MODEL_PATH:-${GAP_MODEL_PATH_DEFAULT}}"
USE_GAP_REWARD="${USE_GAP_REWARD:-False}"
GAP_REWARD_SCALE="${GAP_REWARD_SCALE:-0.20}"
GAP_REWARD_SIGN="${GAP_REWARD_SIGN:-1.0}"
MAX_DELTA_SCALE="${MAX_DELTA_SCALE:-0.06}"
FROZEN_PATCH_ALPHA_START="${FROZEN_PATCH_ALPHA_START:-0.0}"
FROZEN_PATCH_ALPHA_END="${FROZEN_PATCH_ALPHA_END:-0.2}"
FROZEN_PATCH_ALPHA_WARMUP_STEPS="${FROZEN_PATCH_ALPHA_WARMUP_STEPS:-62000}"
FROZEN_PATCH_ALPHA_DELAY_STEPS="${FROZEN_PATCH_ALPHA_DELAY_STEPS:-0}"
FROZEN_PATCH_ALPHA_SCHEDULE="${FROZEN_PATCH_ALPHA_SCHEDULE:-smoothstep}"
FROZEN_PATCH_MASK="${FROZEN_PATCH_MASK:-lower_waist}"
REWARD_CONFIG="${REWARD_CONFIG:-motion_tracking/reward_motion_tracking_dm_2real}"
PROJECT_NAME="${PROJECT_NAME:-TEST_CR7_Siuuu}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-MotionTracking_CR7_Siuuu_FrozenPatch_TabulaRasa}"
NUM_ENVS="${NUM_ENVS:-3072}"
NUM_ITERS="${NUM_ITERS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-16}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-False}"
RESET_POLICY_STD_ON_LOAD="${RESET_POLICY_STD_ON_LOAD:-False}"
POLICY_STD_ON_LOAD="${POLICY_STD_ON_LOAD:-0.8}"
INIT_NOISE_STD="${INIT_NOISE_STD:-0.8}"
DOMAIN_RAND_CONFIG="${DOMAIN_RAND_CONFIG:-domain_rand_base}"
HEADLESS="${HEADLESS:-True}"
AUTO_LOAD_LATEST="${AUTO_LOAD_LATEST:-False}"

if [[ ! -f "${FROZEN_PATCH_CKPT}" ]]; then
  echo "Missing frozen patch checkpoint: ${FROZEN_PATCH_CKPT}" >&2
  exit 1
fi

if [[ "${USE_GAP_REWARD}" == "True" || "${USE_GAP_REWARD}" == "true" ]]; then
  if [[ ! -f "${GAP_MODEL_PATH}" ]]; then
    echo "Missing gap model: ${GAP_MODEL_PATH}" >&2
    exit 1
  fi
fi

echo "Launching Tabula Rasa PPO in frozen-patch environment"
echo "frozen patch    : ${FROZEN_PATCH_CKPT}"
echo "gap model       : ${GAP_MODEL_PATH}"
echo "use gap reward  : ${USE_GAP_REWARD}"
echo "gap scale/sign  : ${GAP_REWARD_SCALE} / ${GAP_REWARD_SIGN}"
echo "max delta scale : ${MAX_DELTA_SCALE}"
echo "patch alpha     : ${FROZEN_PATCH_ALPHA_START} -> ${FROZEN_PATCH_ALPHA_END} (${FROZEN_PATCH_ALPHA_SCHEDULE}, warmup steps=${FROZEN_PATCH_ALPHA_WARMUP_STEPS}, delay=${FROZEN_PATCH_ALPHA_DELAY_STEPS})"
echo "patch mask      : ${FROZEN_PATCH_MASK}"
echo "reward config   : ${REWARD_CONFIG}"
echo "domain rand     : ${DOMAIN_RAND_CONFIG}"
echo "load optimizer  : ${LOAD_OPTIMIZER}"
echo "reset std/load  : ${RESET_POLICY_STD_ON_LOAD} / ${POLICY_STD_ON_LOAD}"
echo "====================================================="

CMD=(
  /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py
  +simulator=isaacgym
  +exp=motion_tracking
  env=delta_a_closed_loop
  +device=cuda:0
  +robot=g1/g1_29dof_anneal_23dof
  +terrain=terrain_locomotion_plane
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history
  +domain_rand=${DOMAIN_RAND_CONFIG}
  +rewards=${REWARD_CONFIG}
  robot.actions_dim=23
  checkpoint=null
  auto_load_latest=${AUTO_LOAD_LATEST}
  ++env.config.delta_ckpt_path=${FROZEN_PATCH_CKPT}
  ++env.config.use_gap_reward=${USE_GAP_REWARD}
  ++env.config.gap_model_path=${GAP_MODEL_PATH}
  ++env.config.gap_reward_scale=${GAP_REWARD_SCALE}
  ++env.config.gap_reward_sign=${GAP_REWARD_SIGN}
  ++env.config.add_extra_action=False
  ++env.config.max_delta_scale=${MAX_DELTA_SCALE}
  ++env.config.use_policy_action_as_base=True
  ++env.config.frozen_patch_alpha_start=${FROZEN_PATCH_ALPHA_START}
  ++env.config.frozen_patch_alpha_end=${FROZEN_PATCH_ALPHA_END}
  ++env.config.frozen_patch_alpha_warmup_steps=${FROZEN_PATCH_ALPHA_WARMUP_STEPS}
  ++env.config.frozen_patch_alpha_delay_steps=${FROZEN_PATCH_ALPHA_DELAY_STEPS}
  ++env.config.frozen_patch_alpha_schedule=${FROZEN_PATCH_ALPHA_SCHEDULE}
  ++env.config.frozen_patch_mask=${FROZEN_PATCH_MASK}
  ++algo.config.learn_sigma=False
  ++algo.config.init_noise_std=${INIT_NOISE_STD}
  ++algo.config.load_optimizer=${LOAD_OPTIMIZER}
  ++algo.config.reset_policy_std_on_load=${RESET_POLICY_STD_ON_LOAD}
  ++algo.config.policy_std_on_load=${POLICY_STD_ON_LOAD}
  robot.motion.motion_file=${MOTION_FILE}
  num_envs=${NUM_ENVS}
  headless=${HEADLESS}
  project_name=${PROJECT_NAME}
  experiment_name=${EXPERIMENT_NAME}
  algo.config.save_interval=${SAVE_INTERVAL}
  algo.config.num_learning_iterations=${NUM_ITERS}
  algo.config.num_steps_per_env=${NUM_STEPS_PER_ENV}
  algo.config.num_mini_batches=${NUM_MINI_BATCHES}
)

HYDRA_FULL_ERROR=1 "${CMD[@]}"

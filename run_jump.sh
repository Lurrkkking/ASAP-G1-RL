  #!/bin/bash
  set -euo pipefail

  MOTION_FILE="/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl"

  EXP_NAME="MotionTracking_CR7_FullSystem_V3_2048"

  echo "====================================================="
  echo "按 baseline13000group 配置启动 C 罗庆祝动作训练"
  echo "motion: ${MOTION_FILE}"
  echo "====================================================="

  HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +robot=g1/g1_29dof_anneal_23dof \
    +terrain=terrain_locomotion_plane \
    +domain_rand=domain_rand_base \
    +rewards=motion_tracking/reward_motion_tracking_cr7_baseline13000group_old \
    +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
    "robot.motion.motion_file=${MOTION_FILE}" \
    project_name=TEST_CR7_Siuuu \
    experiment_name=${EXP_NAME} \
    num_envs=2048 \
    headless=True \
    checkpoint=null \
    auto_load_latest=False \
    algo.config.num_mini_batches=4 \
    ++algo.config.learn_sigma=False \
    algo.config.init_noise_std=0.8 \
    domain_rand.push_interval_s=[5,10] \
    domain_rand.max_push_vel_xy=1.0 \
    domain_rand.base_com_range.x=[-0.03,0.03] \
    domain_rand.base_com_range.y=[-0.03,0.03] \
    domain_rand.base_com_range.z=[-0.05,0.05] \
    domain_rand.link_mass_range=[0.8,1.2] \
    domain_rand.kp_range=[0.75,1.25] \
    domain_rand.kd_range=[0.75,1.25] \
    domain_rand.friction_range=[0.5,1.25] \
    domain_rand.added_mass_range=[-2.0,2.0] \
    domain_rand.rfi_lim=0.1 \
    domain_rand.rfi_lim_range=[0.5,1.5] \
    domain_rand.ctrl_delay_step_range=[0,2]

# Codex Handoff

## Read First
This handoff keeps only the currently useful state for residual-gap, offline patch, and PPO patch integration work in ASAP.
Future Codex sessions should read this before changing:
- `genesis_simulation/residual_dataset/*`
- `humanoidverse/envs/delta_a/*`
- `run_attention_delta.sh`
- `run_frozen_patch_ppo.sh`

## Environment Notes
Use these environments consistently:
- Isaac / PPO / ONNX: `conda activate rl`
- Genesis-side pairing / oracle / dataset tools: `conda activate /root/autodl-tmp/env_genesis`

Run from repo root:
- `cd /root/autodl-tmp/ASAP`

## Trusted Conclusions
### 1. Residual gap modeling is useful as a one-step predictor
This remains true on the `26600` anchor data.

Safe summary:
- residual modeling improves one-step Isaac-vs-Genesis prediction overall
- strongest gains are on `root` and `dof_vel`
- `dof_pos` remains the weakest component

### 2. Existing PPO patch checkpoints are not validated successes
The fair comparison is fixed-state evaluation on the same baseline state set, not self-anchor rollout metrics.

Trusted fixed-state result on `3748` valid samples from `isaac_26600_anchor.npz`:
- baseline `26600 on 26600states`: `2.103550434`
- PPO patch `27000 on 26600states`: `2.921152353`
- PPO patch `27700 on 26600states`: `2.921092510`
- PPO patch `29500 on 26600states`: `2.920817614`

Safe conclusion:
- current PPO patch training is worse than baseline on the common fixed-state metric
- self-anchor improvement must not be treated as causal sim-gap reduction

### 3. Offline oracle-supervised patching is the first path that beats the cleaned baseline
Only corrected evaluator results should be trusted.

Trusted corrected fixed-state results on the cleaned benchmark:
- baseline `26600`: `2.020122528`
- oracle-supervised `p512`: `2.019357000`
- oracle-supervised `p1024`: `2.012954000`

Interpretation:
- `p512` is the first trusted offline patch that beats the cleaned baseline
- `p1024` is slightly better than `p512`
- both are still aggressive and nearly saturate the `0.10` action cap

### 4. Frozen-patch environment plus main-policy PPO now beats the cleaned baseline by a large margin
This is the current best validated direction.

Trusted fixed-state results for the clean `run_frozen_patch_ppo.sh` line:
- baseline `26600`: `2.020122528`
- `model_26650`: `2.017875671`
- `model_26900`: `1.989935994`
- `model_27100`: `1.940354466`
- `model_27300`: `1.960826874`
- `model_27550`: `1.953179717`
- `model_27750`: `1.962601423`
- `model_27950`: `1.912846565`
- `model_28200`: `1.865794301`
- `model_29200`: `1.866206646`
- `model_32500`: `1.813558698`

Interpretation:
- mounting the frozen offline patch in the environment and then fine-tuning the main PPO policy is now a validated success, not just an integration milestone
- best current gain is `0.206563830` better than the cleaned baseline, about `10.2%`
- compared with the offline `p1024` patch alone, `model_32500` gains an additional `0.199395302`
- the trend is not strictly monotonic, but the line is clearly learning something real after the confounders were removed
- this now exceeds the offline-patch-only gain and is the main active path

## Oracle Rollback Notes
Keep this section even if PPO becomes the active line, because future work may need to roll back to the oracle-supervised pipeline.

Key oracle datasets:
- `p512` oracle labels: `genesis_simulation/residual_dataset/action_star_local_linear_pilot512_lam10_delta010_v2.npz`
- `p512` filtered dataset: `genesis_simulation/residual_dataset/oracle_patch_dataset_pilot512_eta05_top50.npz`
- `p1024` oracle labels: `genesis_simulation/residual_dataset/action_star_local_linear_pilot1024_lam10_delta010_v2.npz`
- `p1024` filtered dataset: `genesis_simulation/residual_dataset/oracle_patch_dataset_pilot1024_eta05_top50.npz`

Key oracle-supervised checkpoints:
- `p512`: `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt`
- `p1024`: `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p1024_eta05_top50/best_delta_action_patch.pt`

Key oracle-supervised training facts:
- both runs used `max_delta_scale=0.10`
- both runs used `delta_mse_weight=1.0`, `jacobian_weight=1.0`, `patch_l2_weight=0.1`, `patch_l1_weight=0.01`
- `p512` best val loss: `0.003420586931`
- `p1024` best val loss: `0.003413439961`

Key corrected fixed-state results:
- baseline `26600`: `2.020122528`
- oracle-supervised `p512`: `2.019357000`
- oracle-supervised `p1024`: `2.012954000`
- extra gain of `p1024` over `p512`: `0.006403000`

Patch-magnitude caution:
- `p512`: `mean_abs_delta_a=0.057559010`, `p95_abs_delta_a=0.099046461`, `p99_abs_delta_a=0.099984467`, `max_abs_delta_a=0.100000001`
- `p1024`: `mean_abs_delta_a=0.056779899`, `p95_abs_delta_a=0.097883835`, `p99_abs_delta_a=0.099804811`, `max_abs_delta_a=0.099998951`

Rollback interpretation:
- if the new PPO-with-frozen-patch line regresses, the oracle-supervised path is still the last trusted pure offline fallback
- `p1024` is the better frozen checkpoint, but only slightly
- the main unresolved issue on the oracle line is not basic efficacy anymore; it is how to keep the gain while reducing near-saturated patch magnitude
- do not discard the oracle artifacts or their evaluation chain when cleaning old experiment outputs

## Current Best Artifacts
Primary baseline artifacts:
- baseline policy: `logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600.pt`
- 46D PPO init: `logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600_46dim_init.pt`
- residual gap model: `genesis_simulation/residual_dataset/train_out_delta_26600_posw16/best_residual_dynamics.pt`

Primary offline patch artifacts:
- oracle-supervised `p512` checkpoint: `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt`
- oracle-supervised `p1024` checkpoint: `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p1024_eta05_top50/best_delta_action_patch.pt`
- `p512` paired delta: `genesis_simulation/residual_dataset/paired_delta_patched_26600_oraclev1_p512_eta05_top50.npz`
- `p1024` paired delta: `genesis_simulation/residual_dataset/paired_delta_patched_26600_oraclev1_p1024_eta05_top50.npz`

Primary frozen-patch PPO artifacts:
- best current run: `logs/Frozen_Patch_PPO/20260417_154954-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof`
- best current checkpoint: `logs/Frozen_Patch_PPO/20260417_154954-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof/model_32500.pt`
- same-run evaluation checkpoints of interest:
  - `model_26650.pt`
  - `model_26900.pt`
  - `model_27100.pt`
  - `model_27300.pt`
  - `model_27550.pt`
  - `model_27750.pt`
  - `model_27950.pt`
  - `model_28200.pt`
  - `model_29200.pt`
  - `model_32500.pt`

Primary fixed-state evaluation artifacts:
- baseline paired delta: `genesis_simulation/residual_dataset/paired_delta_26600_on_26600states.npz`
- baseline Isaac states: `genesis_simulation/residual_dataset/isaac_26600_on_26600states.npz`

## PPO Integration Status As Of April 17, 2026
This turn finished the first working integration of a frozen offline delta-action patch into the PPO closed-loop environment.

Important algorithmic clarification:
- the project goal from `AGENTS.md` is not "train another PPO patch on top of a frozen patch"
- the intended end state is:
  - freeze the delta-action patch
  - insert it into the environment as a black-box action correction module
  - train the main PPO policy in that patched environment

What changed:
- `humanoidverse/envs/delta_a/delta_a_closed_loop.py`
  - `delta_ckpt_path` now supports offline `23`-dim delta-action patch checkpoints in addition to old `46`-dim attention-delta style checkpoints
  - frozen offline patch output is combined with the trainable PPO patch output
  - reset-time state access was fixed so frozen patch inference works before `root_states`, `dof_pos`, and `dof_vel` env caches are fully initialized
- `run_attention_delta.sh`
  - now exports `PATH=/root/miniconda3/envs/rl/bin:$PATH` so IsaacGym can find `ninja`
  - now defaults `FROZEN_PATCH_CKPT` to the oracle-supervised `p512` checkpoint

Smoke-test result on April 17, 2026:
- command used:
  - `NUM_ENVS=64 NUM_ITERS=1 SAVE_INTERVAL=1000 NUM_STEPS_PER_ENV=4 NUM_MINI_BATCHES=1 /bin/bash /root/autodl-tmp/ASAP/run_attention_delta.sh`
- result:
  - PPO env instantiated successfully
  - frozen `p512` patch loaded successfully
  - gap model loaded successfully
  - base `23D` policy checkpoint loaded successfully
  - `46D` init checkpoint loaded successfully
  - one training iteration completed and checkpoint saved successfully
- smoke-test log dir:
  - `logs/Delta_Patch_Training/20260417_105508-Train_46dim_Patch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof`
- saved checkpoint from the smoke test:
  - `logs/Delta_Patch_Training/20260417_105508-Train_46dim_Patch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof/model_26601.pt`

Important current limitation:
- the launch script defaults to frozen `p512` because that was the integration target for this turn
- if you want the slightly better offline artifact instead, override:
  - `FROZEN_PATCH_CKPT=/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p1024_eta05_top50/best_delta_action_patch.pt`

## April 18, 2026 Update: Frozen-Patch PPO Is Now the Leading Validated Path
This update supersedes the earlier April 17 "integration works but not yet successful" top-line conclusion.

Clean benchmark used in the latest README/user-reported evaluation:
- baseline `26600`: `2.020122528`

Best validated run:
- run dir: `logs/Frozen_Patch_PPO/20260417_154954-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof`
- best checkpoint: `model_32500.pt`
- best result: `1.813558698`

Same-run progression:
- `26650`: `2.017875671`
- `26900`: `1.989935994`
- `27100`: `1.940354466`
- `27300`: `1.960826874`
- `27550`: `1.953179717`
- `27750`: `1.962601423`
- `27950`: `1.912846565`
- `28200`: `1.865794301`
- `29200`: `1.866206646`
- `32500`: `1.813558698`

Safe interpretation:
- after removing confounders and fixing the PPO sigma bug, the clean frozen-patch PPO line does improve the fair fixed-state metric materially
- the gain is not a tiny fluctuation; the best checkpoint is about `10.2%` better than the cleaned baseline
- combined with the offline patch, the total improvement over the cleaned baseline is about `13.8%`
- the main question has shifted from "does this line work at all?" to "why does one-step improvement not yet translate into equally strong long-horizon motion quality?"

Operational priority:
- treat `run_frozen_patch_ppo.sh` as the main policy-training path
- treat `run_attention_delta.sh` as an ablation/debug launcher for patch-branch experiments
- preserve the old April 17 failure/debug notes below as historical diagnosis, not as the current headline result

## April 19, 2026 Update: Re-extract Constrained Knee/Ankle Patch Instead Of Masking A Full-Body Patch
Recent video checks showed that naively using a full-body oracle patch with only lower-body or lower-waist dimensions enabled can reduce one-step metrics while degrading closed-loop motion quality. The current interpretation is that a full-body oracle solution is a coupled action correction; hard-masking it after training can break the original compensation structure.

New active direction:
- re-solve the oracle action correction with the action search space constrained to knee/ankle joints only
- keep full-state inputs intact so the patch can still condition on root, torso, opposite leg, phase/contact state, and whole-body momentum
- train the patch with masked loss/regularization only on the active knee/ankle output dimensions

G1 knee/ankle action indices:
- `3`: `left_knee_joint`
- `4`: `left_ankle_pitch_joint`
- `5`: `left_ankle_roll_joint`
- `9`: `right_knee_joint`
- `10`: `right_ankle_pitch_joint`
- `11`: `right_ankle_roll_joint`

Code changes already made:
- `genesis_simulation/residual_dataset/solve_action_star_local_linear.py`
  - added `--action-dof-indices`
  - the oracle solve now computes finite-difference Jacobian columns only for those action dimensions
  - output remains 23D, but non-active dimensions are forced to zero
  - output stores `action_dof_indices` metadata
- `genesis_simulation/residual_dataset/build_oracle_patch_dataset.py`
  - now preserves `action_dof_indices` into the supervised dataset
  - does not zero or alter full-state features
- `genesis_simulation/residual_dataset/train_delta_action_patch_from_oracle.py`
  - now reads dataset `action_dof_indices` by default
  - `plain_mse`, `jacobian_mse`, `patch_l2`, and `patch_l1` are computed only on active patch joints
  - optional override: `--loss-dof-indices`

Important anti-footguns:
- do not train a knee/ankle patch by averaging loss over all 23 action dimensions; zero non-patch dimensions will make the loss look artificially good
- do not regularize all 23 dimensions when only 6 dimensions are active; report and constrain the active dimensions
- do not zero full-body state features; knee/ankle residuals may depend on root, torso, opposite leg, and whole-body momentum

Recommended next commands from repo root:
- environment:
  - `cd /root/autodl-tmp/ASAP`
  - `conda activate /root/autodl-tmp/env_genesis`
- solve constrained oracle labels:
  - `/root/miniconda3/envs/rl/bin/python genesis_simulation/residual_dataset/solve_action_star_local_linear.py --isaac-npz genesis_simulation/residual_dataset/isaac_26600_anchor.npz --out-npz genesis_simulation/residual_dataset/action_star_local_linear_pilot512_knee_ankle_lam10_delta006.npz --max-samples 512 --ridge-lambda 10.0 --max-delta 0.06 --root-weight 0.0 --dof-pos-weight 1.0 --dof-vel-weight 0.1 --action-dof-indices 3,4,5,9,10,11`
- build filtered supervised dataset:
  - `/root/miniconda3/envs/rl/bin/python genesis_simulation/residual_dataset/build_oracle_patch_dataset.py --oracle-npz genesis_simulation/residual_dataset/action_star_local_linear_pilot512_knee_ankle_lam10_delta006.npz --out-npz genesis_simulation/residual_dataset/oracle_patch_dataset_pilot512_knee_ankle_eta05_top50.npz --min-eta 0.05 --topk-frac 0.50`
- train the supervised knee/ankle patch:
  - `/root/miniconda3/envs/rl/bin/python genesis_simulation/residual_dataset/train_delta_action_patch_from_oracle.py --dataset-npz genesis_simulation/residual_dataset/oracle_patch_dataset_pilot512_knee_ankle_eta05_top50.npz --out-dir genesis_simulation/residual_dataset/train_out_delta_action_patch_knee_ankle_p512_eta05_top50 --epochs 300 --batch-size 256 --max-delta-scale 0.06 --delta-mse-weight 1.0 --jacobian-weight 1.0 --patch-l2-weight 0.1 --patch-l1-weight 0.01`

After training:
- apply the new patch to fixed-state data and rebuild paired deltas before judging it
- if one-step metrics improve without large active-joint saturation, then test it in `run_frozen_patch_ppo.sh` by overriding `FROZEN_PATCH_CKPT` to the new checkpoint
- when mounting this new patch, use an env mask consistent with the patch: either add a dedicated `knee_ankle` env mask or use the checkpoint output itself if non-active dimensions are exactly zero

## April 17 Follow-up Notes
This follow-up clarified the fair comparison target for the new frozen-patch PPO line and added a cleaner control path that removes the 46D trainable head as a variable.

### 1. Comparison target must remain the trusted baseline, not the run-local `model_26600`
Do not compare the new PPO line only against `logs/Delta_Patch_Training/.../model_26600.pt` from the same run and treat that as the main conclusion.

Reason:
- in the April 17 run `logs/Delta_Patch_Training/20260417_112304-Train_46dim_Patch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof`, the local `model_26600.pt` is not the original baseline artifact
- the trainer first loads the original baseline `model_26600.pt`, then loads `model_26600_46dim_init.pt`, and then immediately saves a new run-local `model_26600.pt`
- that run-local checkpoint is effectively the initialization point of the patched training line, not the trusted baseline used in prior fair evaluations

Operational rule:
- use the trusted baseline `2.103550434` as the primary target when judging whether the PPO-with-frozen-patch line is actually successful
- use same-run `26600 -> 27000 -> 27600` comparisons only as secondary information about local optimization progress inside that run

### 2. April 17 fair-evaluation finding for the new run
Same-run April 17 fixed-state paired results on `3748` valid samples:
- run-local `model_26600`: `2.831180334`
- `model_27000`: `2.829273462`
- `model_27600`: `2.826856613`

Safe interpretation:
- the April 17 run shows a tiny same-run improvement from `26600` to `27000/27600`
- but this does not establish success, because all of these are still much worse than the trusted baseline `2.103550434`
- therefore the fair top-line conclusion is still: the PPO-with-frozen-patch line has not yet beaten the trusted baseline

### 3. The launch path was still training a 46D head by default
Before this follow-up, `run_attention_delta.sh` defaulted to:
- `robot.actions_dim=46`
- `env.config.add_extra_action=True`
- `checkpoint=model_26600_46dim_init.pt`

That means the default trainable PPO branch was still the 46D attention-style head, even though a frozen 23D offline patch was already mounted in the environment.

### 4. 23D control path was added to isolate variables
This follow-up updated the code so the trainable PPO branch can now run in either:
- `23D` direct-delta mode
- `46D` attention-delta mode

Key implementation facts:
- `humanoidverse/envs/delta_a/delta_a_closed_loop.py` now supports both `actions_dim=num_dof` and `actions_dim=2*num_dof`
- when running with a frozen patch, a `23D` trainable actor now still produces a trainable direct delta patch; it is no longer silently bypassed
- `run_attention_delta.sh` now accepts `TRAIN_ACTION_MODE=23|46`
- the script now defaults to `TRAIN_ACTION_MODE=23` so future runs can exclude the 46D attention head as a confounder

Recommended command pattern:
- default 23D run:
  - `NUM_ENVS=4096 /bin/bash run_attention_delta.sh`
- explicit 46D run:
  - `TRAIN_ACTION_MODE=46 NUM_ENVS=4096 /bin/bash run_attention_delta.sh`

### 5. The correct "frozen patch trains main policy" path now has its own launcher
To match the project-level goal from `AGENTS.md`, a separate launcher was added:
- `run_frozen_patch_ppo.sh`

This path is different from `run_attention_delta.sh`:
- `run_attention_delta.sh`
  - still trains a patch branch (`PPODeltaA`)
  - useful only for patch-training ablations
- `run_frozen_patch_ppo.sh`
  - uses standard `PPO`
  - loads the baseline `23D` policy checkpoint as the trainable main policy
  - mounts the frozen offline patch in `DeltaA_ClosedLoop`
  - does not provide `actions_closed_loop` from a frozen base policy
  - therefore the environment treats PPO's own `actions` as the base action and executes:
    - `a_exec = a_base + delta_frozen(s_t, a_base)`

Operational note:
- `DeltaA_ClosedLoop.step()` now supports both modes:
  - if `actions_closed_loop` is supplied, it behaves like the old `PPODeltaA` path
  - if `actions_closed_loop` is absent, it falls back to using PPO's own `actions` as the base action

Recommended main-policy command:
- `NUM_ENVS=4096 /bin/bash run_frozen_patch_ppo.sh`

Important stabilization note:
- when fine-tuning from the baseline `model_26600.pt`, the checkpoint carries an actor `std` around `1.088`
- this is too large for patched-environment fine-tuning and can cause immediate jitter
- `run_frozen_patch_ppo.sh` now defaults to:
  - `LOAD_OPTIMIZER=False`
  - `RESET_POLICY_STD_ON_LOAD=True`
  - `POLICY_STD_ON_LOAD=0.15`
- `humanoidverse/agents/ppo/ppo.py` now supports:
  - `reset_policy_std_on_load`
  - `policy_std_on_load`
  so checkpoint weight loading can keep the policy mean while forcibly resetting exploration noise

### 6. April 17 actor-output comparison: checkpoint corruption is not the leading hypothesis
This follow-up added a one-shot actor-side debug print in `humanoidverse/agents/ppo/ppo.py` gated by:
- `PPO_ACTION_DEBUG_ONCE=1`

Purpose:
- compare the first real rollout action from the same `model_26600.pt` in:
  - the original `motion_tracking` environment
  - the frozen-patch PPO environment from `run_frozen_patch_ppo.sh`

Commands used:
- original env:
  - `export PATH=/root/miniconda3/envs/rl/bin:$PATH && PPO_ACTION_DEBUG_ONCE=1 HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking env=motion_tracking +device=cuda:0 +robot=g1/g1_29dof_anneal_23dof +terrain=terrain_locomotion_plane +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history +domain_rand=NO_domain_rand_finetune_with_deltaA +rewards=motion_tracking/reward_motion_tracking_dm_2real_gapppo robot.actions_dim=23 checkpoint=/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600.pt ++algo.config.learn_sigma=False ++algo.config.init_noise_std=0.15 ++algo.config.actor_learning_rate=1e-5 ++algo.config.critic_learning_rate=1e-5 ++algo.config.load_optimizer=False ++algo.config.reset_policy_std_on_load=True ++algo.config.policy_std_on_load=0.15 robot.motion.motion_file=/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl num_envs=64 project_name=Baseline_Debug experiment_name=MotionTracking_Baseline_26600_Debug algo.config.save_interval=1000 algo.config.num_learning_iterations=1 algo.config.num_steps_per_env=4 algo.config.num_mini_batches=1`
- frozen-patch env:
  - `PPO_ACTION_DEBUG_ONCE=1 DELTA_A_DEBUG_ONCE=0 NUM_ENVS=64 NUM_ITERS=1 SAVE_INTERVAL=1000 NUM_STEPS_PER_ENV=4 NUM_MINI_BATCHES=1 /bin/bash /root/autodl-tmp/ASAP/run_frozen_patch_ppo.sh`

Observed first-rollout actor outputs:
- original env:
  - `actions sample0`: `mean_abs=2.472978592`, `min=-10.678888321`, `max=5.596428394`
  - `action_mean sample0`: `mean_abs=2.466190100`, `min=-10.612179756`, `max=5.474626064`
  - `action_sigma sample0`: constant `0.15`
- frozen-patch env:
  - `actions sample0`: `mean_abs=2.301159620`, `min=-5.962260723`, `max=6.260736465`
  - `action_mean sample0`: `mean_abs=2.309742689`, `min=-6.169045448`, `max=6.238621235`
  - `action_sigma sample0`: constant `0.15`

Safe interpretation:
- the same loaded checkpoint already produces large-magnitude actions in the original environment
- the frozen-patch environment does **not** introduce a dramatic actor-side blow-up before the first env step
- therefore "the checkpoint was silently modified" is not the leading explanation for the bad fixed-state result
- the more likely failure source is downstream of actor sampling:
  - patched action execution `a_exec = a_base + delta_frozen(...)`
  - altered env dynamics / termination / reward path
  - mismatch between the baseline policy's action distribution and the patched environment

Operational consequence:
- do not keep chasing checkpoint corruption as the primary hypothesis
- if this line is debugged further, prioritize comparisons of:
  - original env rollout vs patched env rollout after env stepping
  - executed action / target position / termination behavior
  - fixed-state evaluation after zero-step or one-step rollout interventions

### 7. April 17 fair evaluation for the first frozen-patch PPO checkpoint remains poor
Trusted fixed-state result reported by the user for:
- `logs/Frozen_Patch_PPO/20260417_145442-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof/model_26601.pt`

Result:
- `mean_abs_delta_valid=2.960936069`

Interpretation:
- this is worse than the trusted baseline `2.103550434` by `0.857385635`
- relative degradation is about `40.8%`
- as of April 17, 2026, the frozen-patch PPO line is still not competitive with the baseline on the fair fixed-state metric

### 8. April 17 execution-layer comparison: the biggest confirmed difference is env-side execution/termination, not actor output
This follow-up added a one-shot execution debug print in `humanoidverse/envs/legged_base_task/legged_robot_base.py` gated by:
- `EXECUTION_DEBUG_ONCE=1`

Important behavior:
- the debug now skips reset/bootstrap samples with zero `actions_after_delay`
- for `DeltaA_ClosedLoop`, this matters because `reset_all()` still performs a zero-action bootstrap step that can already execute a frozen patch

Commands used:
- original env:
  - `export PATH=/root/miniconda3/envs/rl/bin:$PATH && EXECUTION_DEBUG_ONCE=1 PPO_ACTION_DEBUG_ONCE=0 HYDRA_FULL_ERROR=1 /root/miniconda3/envs/rl/bin/python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking env=motion_tracking +device=cuda:0 +robot=g1/g1_29dof_anneal_23dof +terrain=terrain_locomotion_plane +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history +domain_rand=NO_domain_rand_finetune_with_deltaA +rewards=motion_tracking/reward_motion_tracking_dm_2real_gapppo robot.actions_dim=23 checkpoint=/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260415_144213-MotionTracking_CR7_Siuuu_Resume_V2_Boost-motion_tracking-g1_29dof_anneal_23dof/model_26600.pt ++algo.config.learn_sigma=False ++algo.config.init_noise_std=0.15 ++algo.config.actor_learning_rate=1e-5 ++algo.config.critic_learning_rate=1e-5 ++algo.config.load_optimizer=False ++algo.config.reset_policy_std_on_load=True ++algo.config.policy_std_on_load=0.15 robot.motion.motion_file=/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass_scale092.pkl num_envs=64 project_name=Baseline_Debug experiment_name=MotionTracking_Execution_26600_Debug algo.config.save_interval=1000 algo.config.num_learning_iterations=1 algo.config.num_steps_per_env=4 algo.config.num_mini_batches=1`
- frozen-patch env:
  - `EXECUTION_DEBUG_ONCE=1 PPO_ACTION_DEBUG_ONCE=0 DELTA_A_DEBUG_ONCE=0 NUM_ENVS=64 NUM_ITERS=1 SAVE_INTERVAL=1000 NUM_STEPS_PER_ENV=4 NUM_MINI_BATCHES=1 /bin/bash /root/autodl-tmp/ASAP/run_frozen_patch_ppo.sh`

Observed first real execution sample:
- original `motion_tracking` env:
  - `episode_len=2`
  - `terminate_when_motion_far=1`
  - `actions_after_delay_norm sample0`: `mean_abs=1.338137388`, `min=-3.355384827`, `max=3.011318684`
  - `executed_action_offset_rad sample0`: `mean_abs=0.334534347`, `min=-0.838846207`, `max=0.752829671`
  - `torque_limit_ratio_sample0_max=0.936786115`
  - `dof_vel_limit_ratio_sample0_max=0.463841230`
  - `motion_far_sample0_max=0.102326311`, threshold `0.300000000`
  - `joint_pos_diff_norm=1.782528996`
- patched `DeltaA_ClosedLoop` env:
  - `episode_len=2`
  - `terminate_when_motion_far=0`
  - `actions_after_delay_norm sample0`: `mean_abs=2.525721788`, `min=-7.665585518`, `max=4.613040447`
  - `executed_action_offset_rad sample0`: `mean_abs=0.650400460`, `min=-1.820993900`, `max=1.095464826`
  - `torque_limit_ratio_sample0_max=1.000000000`
  - `dof_vel_limit_ratio_sample0_max=1.001825809`
  - `motion_far_sample0_max=0.118542291`, threshold `0.500000000`
  - `joint_pos_diff_norm=1.671318769`

Safe interpretation:
- the patched env does not mainly differ at the actor-output layer; the stronger divergence shows up at executed action / torque / velocity level
- compared with the original env, the patched env's first real executed action is much larger:
  - executed action mean abs roughly `0.6504` vs `0.3345`
  - executed action min/max roughly `[-1.8210, 1.0955]` vs `[-0.8388, 0.7528]`
- the patched env hits much harsher actuator-state regimes immediately:
  - torque ratio reaches `1.0`
  - dof velocity ratio slightly exceeds `1.0`
- the patched env also changes a major task-level rule:
  - `terminate_when_motion_far` is disabled
  - threshold is effectively relaxed from `0.3` to `0.5`

Important conclusion:
- the strongest confirmed issue so far is not checkpoint corruption and not missing patch clamp
- the strongest confirmed issue is env-side execution mismatch:
  - larger executed action after adding the frozen patch
  - more saturated torques / joint velocities
  - different termination regime than the original motion-tracking baseline

Operational consequence:
- future debugging should prioritize:
  - restoring a fairer termination regime in `delta_a_closed_loop` when comparing against baseline
  - testing a hard clamp on final executed action / target position, not just the patch itself
  - comparing fixed-state evaluation before and after such execution-level safety constraints

### 9. April 17 env cleanup: patched env now defaults closer to "baseline env + frozen patch only"
To reduce confounders, this follow-up changed two defaults.

Config changes:
- `humanoidverse/config/env/delta_a_closed_loop.yaml`
  - restored motion-far termination to match the baseline `motion_tracking` env:
    - `terminate_when_motion_far: True`
    - `termination_min_base_height: 0.2`
    - `termination_motion_far_threshold: 0.3`
- `run_frozen_patch_ppo.sh`
  - added `USE_GAP_REWARD`, default:
    - `False`
  - launcher still passes `gap_model_path`, `gap_reward_scale`, and `gap_reward_sign`, but reward-side correction is now opt-in instead of always-on

Reason:
- before this cleanup, the frozen-patch PPO path was not "just baseline env + frozen patch"
- it also changed task-level termination and defaulted gap reward on
- those were major confounders when comparing against the baseline

Smoke test after cleanup:
- command:
  - `EXECUTION_DEBUG_ONCE=1 PPO_ACTION_DEBUG_ONCE=0 DELTA_A_DEBUG_ONCE=0 NUM_ENVS=64 NUM_ITERS=1 SAVE_INTERVAL=1000 NUM_STEPS_PER_ENV=4 NUM_MINI_BATCHES=1 /bin/bash /root/autodl-tmp/ASAP/run_frozen_patch_ppo.sh`
- verified from logs:
  - launcher printed `use gap reward  : False`
  - env printed `terminate_when_motion_far=1`
  - env printed `motion_far_threshold=0.300000000`
  - frozen patch still loaded successfully
  - one training iteration still completed successfully

Observed first real execution sample after cleanup:
- patched `DeltaA_ClosedLoop` env:
  - `terminate_when_motion_far=1`
  - `executed_action_offset_rad`:
    - `mean_abs=0.561845362`
    - `min=-1.591322780`
    - `max=1.706130147`
  - `torque_limit_ratio_sample0_max=1.000000000`
  - `dof_vel_limit_ratio_sample0_max=0.677233934`
  - `motion_far_sample0_max=0.089530833`, threshold `0.300000000`
  - `joint_pos_diff_norm=1.644383192`

Interpretation:
- the patched env is now materially closer to the intended algorithmic meaning:
  - baseline env semantics
  - plus frozen patch execution
  - without reward-side gap correction by default
- however, even after removing those confounders, the patched env still produces larger executed actions than the baseline env and can still hit torque saturation
- therefore the next likely bottleneck remains execution-level safety / final-action magnitude, not actor checkpoint corruption

### 10. April 17 run-local initialization checkpoint behavior was restored for frozen-patch PPO
To make fair comparisons easier, training now writes a run-local checkpoint immediately after loading the source checkpoint and before any learning updates.

Code change:
- `humanoidverse/train_agent.py`
  - after `algo.load(config.checkpoint)`, it now saves:
    - `model_<loaded_iter>.pt`
  - for the `26600` baseline checkpoint, this means each new run now gets its own run-local:
    - `model_26600.pt`

Launcher change:
- `run_frozen_patch_ppo.sh`
  - default `SAVE_INTERVAL` changed from `100` to `50`

Why this matters:
- it gives a clean per-run initialization checkpoint in the exact patched-env config used by that run
- that makes it possible to compare:
  - run-local `model_26600.pt`
  - `model_26650.pt`
  - `model_26700.pt`
  - etc.
  without confusing those with the original baseline artifact from the old run directory

Smoke test on April 17, 2026:
- command:
  - `NUM_ENVS=64 NUM_ITERS=1 NUM_STEPS_PER_ENV=4 NUM_MINI_BATCHES=1 /bin/bash /root/autodl-tmp/ASAP/run_frozen_patch_ppo.sh`
- observed log sequence:
  - load original baseline `model_26600.pt`
  - immediately save run-local `logs/Frozen_Patch_PPO/20260417_154221-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof/model_26600.pt`
  - after one iteration, save `model_26601.pt`

Operational rule:
- for future frozen-patch PPO runs, use the run-local `model_26600.pt` as the "patched-env initialization" checkpoint
- keep using the original trusted baseline `2.103550434` only for the cross-run top-line fairness comparison against the historical baseline

### 11. April 17 cleaned frozen-patch PPO run achieved clear gains beyond the frozen-patch initialization point
Trusted fixed-state results reported by the user for:
- run dir:
  - `logs/Frozen_Patch_PPO/20260417_154221-MainPolicy_23dim_FrozenPatch_26600_Gap26600-delta_a-g1_29dof_anneal_23dof`

Key checkpoints:
- run-local initialization `model_26600.pt`: `2.020122528`
- `model_26650.pt`: `2.017875671`
- `model_26900.pt`: `1.989935994`
- `model_27100.pt`: `1.940354466`
- `model_27300.pt`: `1.960826874`
- `model_27550.pt`: `1.953179717`

Interpretation:
- this is the first strong evidence that the cleaned frozen-patch PPO line is doing useful optimization rather than only inheriting the frozen patch gain
- the run improves not only over the historical baseline `2.103550434`, but also over the cleaned patched-env initialization `2.020122528`
- current best known checkpoint from this run is:
  - `model_27100.pt`: `1.940354466`

Useful deltas:
- best-vs-historical-baseline improvement:
  - `2.103550434 - 1.940354466 = 0.163195968`
  - about `7.76%` lower
- best-vs-cleaned-initialization improvement:
  - `2.020122528 - 1.940354466 = 0.079768062`
  - about `3.95%` lower

Shape of the curve:
- training improved steadily through `27100`
- later checkpoints `27300` and `27550` regressed somewhat versus `27100`, though they remained clearly better than initialization
- operationally, keep `27100` as the current best checkpoint unless later evaluation beats it

## Recommended Next Steps
Highest-value next work:
1. Use `run_frozen_patch_ppo.sh` as the primary path for future work, because it matches the intended algorithmic separation from `AGENTS.md`.
2. Run a real main-policy PPO finetune with frozen patch inserted as an environment parameter.
3. Compare `frozen p512 + main-policy PPO` against:
   - baseline `26600`
   - frozen patch only
   - old PPO patch runs without frozen oracle patch
   - patch-training ablations from `run_attention_delta.sh`
4. Evaluate all new checkpoints with the fixed-state workflow, not self-anchor metrics.
5. Try the same integration with frozen `p1024`, since it is slightly better offline than `p512`.

If PPO still fails under fair fixed-state evaluation, the likely diagnosis remains:
- objective mismatch
- rollout distribution shift
- PPO exploiting proxy reward structure rather than shrinking real one-step sim gap

## April 17 Follow-up: Frozen-Patch `26600` Start Gap Diagnosis
This follow-up investigated why the frozen-patch training start checkpoint reported a much worse fixed-state metric:
- reported run-local `model_26600`: `mean_abs_delta_valid=2.845075130`
- trusted baseline: `2.103550434`

Key verified facts:
- original trusted baseline checkpoint still reproduces the trusted baseline artifact exactly under the current fixed-state collection path
  - current 512-sample rerun of original baseline produced identical `a` and `s_next` to `genesis_simulation/residual_dataset/isaac_26600_on_26600states.npz` on the first 512 valid samples
  - first-512 true metric for baseline remains `2.122014284`
- run-local frozen-patch `model_26600.pt` is not the original baseline artifact
  - it is saved after loading baseline and resetting actor std to about `0.15`
  - however, actor mean policy is not the main source of the gap; original baseline ONNX and run-local ONNX differ only about `0.0006` mean target-position units on matched actor observations in one diagnostic
- the bad frozen-patch start is caused by the `DeltaA_ClosedLoop` frozen-patch execution path, not by the original baseline checkpoint

Important action-level evidence:
- offline p512 patch expected action change on first 512 valid samples:
  - `mean_abs(a_p512 - a_base)=0.058632642`
  - max is bounded by the p512 checkpoint cap around `0.10`
- `DeltaA_ClosedLoop` frozen-patch path after the first unit fix still produced:
  - `mean_abs(a_env_frozen - a_base)=0.154873565`
  - `p95=0.416418850`
  - `max=0.969303131`
- this exceeds the p512 patch cap by far, so the environment is not simply executing `baseline_target + p512_delta`.

Metric evidence on first 512 valid samples:
- baseline: `2.122014284`
- offline p512 fixed-actions result: `2.032983542`
- frozen env after unit/cache fixes: `2.968364478`
- the frozen env improves root/dof_pos but badly worsens dof_vel, same pattern as the earlier full-run `2.845` anomaly.

Code fixes already applied in this follow-up:
- `humanoidverse/envs/delta_a/delta_a_closed_loop.py`
  - offline `delta_action_patch` checkpoints are now fed target joint positions, not normalized actions
  - frozen offline patch deltas are treated as radian target-position deltas, not normalized action deltas
  - frozen patch output is cached per env step so it is not recomputed on every physics substep

Important caveat:
- These fixes are necessary but not sufficient. The frozen env path still produces target-position deviations far beyond the p512 cap.
- The next highest-value debug is to instrument one `DeltaA_ClosedLoop.step()` call and print:
  - incoming `actions`
  - `base_actions`
  - `frozen_patch`
  - `train_patch`
  - `motion_action`
  - `actions_total`
  - `executed_actions_total`
- Expected invariant for p512-only main-policy path:
  - `abs(executed_actions_total + default_dof_pos - baseline_target_pos)` should be at most about `0.10` per joint.
  - Current observed max is about `0.97`, so this invariant is violated.

Do not trust the frozen-patch PPO `26600` start as a fair baseline until that invariant is restored.

## Safe Claims Going Forward
Supported:
- residual gap modeling is useful as a one-step predictor
- current PPO patch checkpoints are worse than baseline under fair fixed-state evaluation
- oracle-supervised offline patching can beat baseline on the corrected fixed-state metric
- `p1024` is slightly better than `p512`, but the gain is small
- the new frozen-patch PPO integration path now runs end-to-end at least through a smoke test
- April 17 same-run `26600 -> 27000 -> 27600` in the new frozen-patch line improved only slightly and remained far worse than the trusted baseline
- same-run `model_26600` from the April 17 Delta_Patch_Training directory is not the original trusted baseline artifact
- the codebase now supports a `23D` trainable PPO patch path with frozen offline patch mounted, so the 46D head can be ablated cleanly
- the intended mainline from `AGENTS.md` is "freeze patch, train main PPO policy", and `run_frozen_patch_ppo.sh` is now the launcher for that path

Not supported:
- any claim that the old PPO patch pipeline is already successful
- any claim that self-anchor improvement demonstrates causal sim-gap reduction
- any claim that oracle-supervised gains are already robust to stronger action-magnitude constraints

## April 19, 2026 Update: Tabula Rasa Frozen-Patch PPO + Alpha Curriculum
This update adds the clean Tabula Rasa control-vs-patch experiment path.

New launcher:
- `run_frozen_patch_ppo_tabula_rasa.sh`

Purpose:
- train the main `23D` PPO policy from random initialization
- mount the frozen offline delta-action patch as part of `DeltaA_ClosedLoop`
- compare against `run_jump.sh`, which is the no-patch from-scratch control

Current control alignment:
- `run_jump.sh` and `run_frozen_patch_ppo_tabula_rasa.sh` now both use:
  - `+domain_rand=domain_rand_base`
  - `+rewards=motion_tracking/reward_motion_tracking_dm_2real`
  - `+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history`
  - `checkpoint=null`
  - `auto_load_latest=False`
  - fixed PPO sigma: `learn_sigma=False`, `init_noise_std=0.8`
- This makes the main experimental variable:
  - no patch (`run_jump.sh`)
  - frozen patch environment (`run_frozen_patch_ppo_tabula_rasa.sh`)

Important PPO sigma note:
- `learn_sigma=False` now actually works after the PPOActor fix.
- In this mode `actor.std` is a buffer, not an optimizer parameter.
- This only affects newly started training processes.

Frozen patch alpha curriculum:
- `DeltaA_ClosedLoop` now supports a global scalar frozen-patch injection schedule:
  - `frozen_patch_alpha_start`
  - `frozen_patch_alpha_end`
  - `frozen_patch_alpha_warmup_steps`
  - `frozen_patch_alpha_delay_steps`
  - `frozen_patch_alpha_schedule` (`linear` or `smoothstep`)
- Actual action composition for the Tabula Rasa path is:
  - `a_actual = a_base + alpha(t) * delta_patch`
- The script sets:
  - `use_policy_action_as_base=True`
  - `frozen_patch_alpha_start=0.0`
  - `frozen_patch_alpha_end=0.2`
  - `frozen_patch_alpha_warmup_steps=2000`
  - `frozen_patch_alpha_delay_steps=0`
  - `frozen_patch_alpha_schedule=smoothstep`

Interpretation of the default alpha:
- the current default frozen patch is `p512`:
  - `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt`
- that checkpoint stores `max_delta_scale=0.10`
- with `alpha_end=0.2`, the final effective patch cap is roughly `0.02 rad`
- this was chosen to avoid the earlier `0.10 rad` full-strength patch causing high-Kp torque spikes and poor early Tabula Rasa learning

Step-unit caution:
- `frozen_patch_alpha_warmup_steps` is counted in environment steps via `common_step_counter`, not PPO iterations.
- With `num_steps_per_env=16`, `warmup_steps=2000` is only about `125` PPO iterations.
- If the intended warmup is about `2000` PPO iterations, set:
  - `FROZEN_PATCH_ALPHA_WARMUP_STEPS=32000`

Current working hypothesis:
- The earlier full-strength frozen patch likely disrupted early PPO exploration even if it improved one-step fixed-state metrics.
- The alpha curriculum tests whether the patch is useful when introduced as a small, delayed/smoothed environment correction rather than as a full-strength modifier from step zero.

Next intended implementation:
- add a frozen-patch joint mask to `DeltaA_ClosedLoop` and expose it in `run_frozen_patch_ppo_tabula_rasa.sh`
- motivation:
  - the current frozen patch is a full `23D` joint patch
  - upper-body residuals may inject unwanted angular momentum / COM disturbance during run-up and jump
  - a masked patch may preserve useful lower-body contact compensation while removing upper-body interference
- proposed mask options:
  - `all`: no mask, current behavior
  - `lower`: joints `0..11` only
  - `lower_waist`: joints `0..14` only, recommended first trial for CR7 because waist may matter for turning/jump
- intended formula:
  - `a_actual = a_base + alpha(t) * (M * delta_patch)`
- recommended first trial after implementation:
  - `FROZEN_PATCH_MASK=lower_waist`
  - keep `FROZEN_PATCH_ALPHA_END=0.2`
  - keep long warmup (`FROZEN_PATCH_ALPHA_WARMUP_STEPS=62000` or similar)

## April 20, 2026 Update: Alpha-Zero Tabula Rasa Patch Debug
This update records the current diagnosis for why `run_frozen_patch_ppo_tabula_rasa.sh` with patch alpha set to zero can still diverge from pure baseline training.

Runs / sessions checked:
- `0patch` tmux session: frozen-patch Tabula Rasa run
- `puredebug` tmux session: pure `run_jump.sh` baseline run
- active `0patch` no-delay debug run started with:
  - `./run_frozen_patch_ppo_tabula_rasa.sh ++domain_rand.randomize_ctrl_delay=False 2>&1 | tee /tmp/0patch_nodelay_debug.log`
- previous delayed debug log:
  - `/tmp/0patch_debug.log`
- no-delay debug log:
  - `/tmp/0patch_nodelay_debug.log`

Confirmed for the alpha-zero `0patch` runs:
- `FROZEN_PATCH_ALPHA_START=0`
- `FROZEN_PATCH_ALPHA_END=0`
- `use_gap_reward=False`
- `reward_config=motion_tracking/reward_motion_tracking_dm_2real`
- `domain_rand=domain_rand_base` unless explicitly overridden for the no-delay diagnostic
- `obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history`
- `checkpoint=null`
- `auto_load_latest=False`
- `learn_sigma=False`
- `init_noise_std=0.8`
- `env=delta_a_closed_loop`

Important debug result with normal `domain_rand_base`:
- `/tmp/0patch_debug.log` showed PPO sampled a nonzero action, but `sample0` reached `_compute_torques()` as all zeros.
- This was not caused by the patch. It was caused by `domain_rand_base` control delay:
  - `randomize_ctrl_delay=True`
  - `ctrl_delay_step_range=[0, 3]`
- The first debug sample can therefore be polluted by the initial delay queue.

Important debug result with control delay disabled:
- `/tmp/0patch_nodelay_debug.log` is the clean action-chain check.
- Key values from that log:
  - `PPO_ACTION_DEBUG actions sample0 mean_abs=0.631277978`
  - `DELTA_A_DEBUG frozen_patch_alpha=0.000000000`
  - `DELTA_A_DEBUG frozen_patch_rad sample0 mean_abs=0.000000000`
  - `DELTA_A_DEBUG train_patch_norm sample0 mean_abs=0.000000000`
  - `DELTA_A_DEBUG actions_total_norm sample0 mean_abs=0.631277978`
  - `EXECUTION_DEBUG actions_after_delay_norm sample0 mean_abs=0.631277978`
  - `EXECUTION_DEBUG executed_action_offset_rad sample0 mean_abs=0.157819495`
  - `action_scale=0.25`, so `0.631277978 * 0.25 = 0.1578194945`
- Safe conclusion:
  - with `alpha=0` and `randomize_ctrl_delay=False`, `DeltaA_ClosedLoop` executes exactly `PPO_action * action_scale`
  - frozen patch contribution is zero
  - train patch contribution is zero
  - action clipping is not active (`action_clip=100`, `action_clip_frac=0`)
- Therefore the main action post-processing chain is not the explanation for alpha-zero divergence.

Current suspect list after the no-delay check:
- highest priority: reset / bootstrap / history differences
  - `DeltaA_ClosedLoop.reset_all()` performs a zero-action `step()` through the wrapper path
  - baseline `LeggedRobotMotionTracking` uses the base path
  - this may alter initial `history_actor`, `actions` history, `last_actions`, or first rollout observations
- second priority: RNG stream differences
  - the patch environment loads a checkpoint and constructs `delta_model`
  - even with alpha zero, extra torch operations can shift random number streams before motion sampling, domain randomization, or PPO sampling
  - unless all seeds and RNG consumption are controlled, two otherwise equivalent RL runs can diverge quickly
- lower priority after the no-delay debug:
  - action scaling / clipping / patch addition
  - patch gradients or patch loss affecting shared parameters

Recommended next diagnostic:
- Add or run a one-iteration, no-training equivalence probe that compares observations rather than only actions.
- The useful check is:
  - instantiate pure `motion_tracking`
  - instantiate `delta_a_closed_loop` with `frozen_patch_alpha_start=0`, `frozen_patch_alpha_end=0`
  - disable `randomize_ctrl_delay`
  - use the same seed, same motion file, same env count
  - reset both
  - compare `actor_obs`, `critic_obs`, `hist_obs_dict`, `last_actions`, `motion_start_times`, and first-step `obs_buf_dict`
- If obs differs before the first real PPO update, the alpha-zero divergence is explained by reset/history/RNG, not by patch action composition.

What puredebug should do next:
- It does not need to keep running just to prove action equivalence; `0patch_nodelay_debug.log` already proves the patch wrapper action chain is equivalent at alpha zero with delay disabled.
- If a fair curve comparison is still needed, keep `puredebug` running as the baseline curve for the same wall-clock/iteration range.
- For the next targeted diagnostic, puredebug should be restarted or supplemented with a short `tee` run so its first-step debug is saved:
  - `PPO_ACTION_DEBUG_ONCE=1 EXECUTION_DEBUG_ONCE=1 ./run_jump.sh 2>&1 | tee /tmp/puredebug_once.log`
- Better than another long pure run:
  - implement the no-training observation equivalence probe above, because current evidence points away from action post-processing and toward initial observations/RNG.

## Files Most Relevant To The Next Person
- `run_attention_delta.sh`
- `run_frozen_patch_ppo.sh`
- `run_frozen_patch_ppo_tabula_rasa.sh`
- `run_jump.sh`
- `humanoidverse/envs/delta_a/delta_a_closed_loop.py`
- `genesis_simulation/residual_dataset/collect_isaac_from_fixed_states.py`
- `genesis_simulation/residual_dataset/collect_isaac_from_fixed_actions.py`
- `genesis_simulation/residual_dataset/build_paired_delta_from_isaac_anchors.py`
- `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p512_eta05_top50/best_delta_action_patch.pt`
- `genesis_simulation/residual_dataset/train_out_delta_action_patch_from_oracle_v1_p1024_eta05_top50/best_delta_action_patch.pt`

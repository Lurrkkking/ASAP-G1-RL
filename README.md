<p align="right">
  <b>English</b> | <a href="README_CN.md">中文</a>
</p>

# ASAP-G1-Learning: G1 Reinforcement Learning and Sim2Sim Experiments

This project is built on top of [ASAP](https://github.com/LeCAR-Lab/ASAP). It records my experiments on **Unitree G1** motion imitation, reinforcement learning, cross-physics-engine validation, ASAP delta action reproduction, and football kickup / juggling task design.

Rather than simply adding more features, this project focuses on several core questions:

- Motion tracking for complex human motions on G1
- Sim2Sim validation from Isaac Gym to Genesis / MuJoCo
- Locomotion, motion imitation, and contact-stability tuning
- ASAP delta action data collection, open-loop training, and evaluation
- RL environment construction and task redesign for football kickup / juggling

> The original ASAP documentation is kept as `README_ORIGIN.md`. Please refer to it for basic installation and usage instructions.

---

## Results

### 1. CR7 Siuuu Motion Imitation

With a 0.85 motor torque soft limit and domain randomization, G1 can complete a relatively full motion sequence: **run-up → jump and turn → arm swing in the air → landing**.

| Isaac Gym (training) | Genesis (validation) |
| :---: | :---: |
| <img src="media/CR7_Issacgym.gif" width="400"> | <img src="media/CR7_Genesis.gif" width="400"> |

### 2. Rough-terrain Locomotion

After training a basic walking policy in Isaac Gym, the policy was tested zero-shot on rough terrain in Genesis. The early policy often tripped over obstacles; later reward adjustments improved leg lifting and terrain-crossing stability.

| Before optimization | After optimization |
| :---: | :---: |
| <img src="media/Walk_fall.gif" width="400"> | <img src="media/Walk_Genesis.gif" width="400"> |

### 3. G1 Football Kickup / Juggling Task (Ongoing)

This is the current main task line. The goal is not merely to make the robot touch the ball, but to gradually train a more sustainable ball-control behavior:

> When the ball leaves the controllable region in front of the robot, the robot should use a corrective touch to send it back into that region.

Current progress:

- Built the minimal `robot + ball` simulation pipeline
- Added ball state, contact detection, and debug logs based on Isaac Gym tensors
- Implemented the initial single-hit / kickup environment and rewards
- Trained early policies that can kick the ball upward
- After removing overly hard ground-contact termination, the policy can control the ball closer to the target height
- Some rollouts already show second swing attempts, which suggests early signs of continuous correction behavior

| Ball reaches a reasonable height, but the posture is still unnatural | A second swing attempt appears |
| :---: | :---: |
| <img src="media/ball_kickup_stable.gif" width="400"> | <img src="media/second_swing_attempt.gif" width="400"> |

The task is still exploratory and has not achieved stable continuous juggling. The current intermediate goal is to obtain a robust **kickup / recovery primitive**: when the ball becomes too low, the robot can kick it up, avoid large horizontal drift, and return to a ready state for the next touch. Continuability, height control, direction control, and motion-style constraints will be added progressively.

See detailed notes: [Technical Summary](docs/football_juggling_summary.md)

---

## Main Technical Work

### Motion Tracking and Reward Tuning

For high-dynamic motion imitation on G1, I mainly worked on the following aspects:

- Introduced `soft_torque_curriculum`: the torque limit is relaxed early to help the policy explore jumping, then gradually reduced back to the 0.85 soft torque limit.
- Addressed the “single-foot sticking to the ground” local optimum by adjusting fall penalties and foot-tracking rewards, encouraging real double-foot takeoff.
- Increased `penalty_action_rate` to suppress high-frequency arm and joint oscillations during takeoff and landing.
- Added `penalty_feet_ori` to encourage flatter and more stable foot orientation during landing.
- Used high-noise checkpoint continuation to increase exploration again from an existing policy, improving motion stiffness and insufficient jump height.

### Locomotion and Rough-terrain Testing

To evaluate basic walking ability and cross-engine generalization, I also trained a G1 locomotion policy and tested it on rough terrain in Genesis:

- Trained a basic locomotion policy in Isaac Gym.
- Exported the policy to ONNX and tested it zero-shot in Genesis.
- Observed early failures such as insufficient foot clearance, foot-obstacle collision, and unstable landing.
- Improved terrain-crossing stability by adjusting foot-height, contact-stability, and landing-related rewards.

### Sim2Sim Validation Pipeline

To check whether the policy was merely overfitting Isaac Gym, I completed several cross-engine testing components:

- Added `humanoidverse/export_pt_to_onnx.py` to export trained policies to ONNX.
- Built `genesis_simulation/` to load and test ONNX policies in Genesis.
- Compared jump height, stability, and landing behavior between Isaac Gym and Genesis, revealing that contact, damping, and integration differences strongly affect high-dynamic motions.
- Re-enabled domain randomization, which significantly improved stability in Genesis.
- Completed the MuJoCo ONNX inference pipeline and aligned key parameters such as control frequency, action filtering, and target-rate limits.

### ASAP Delta Action Reproduction

This part focuses on the delta action stage in ASAP. Unlike the older “analytic residual / oracle patch” notes, the current implementation follows the recent experimental route: rollout-with-action data collection, open-loop deltaA training, and deterministic evaluation.

Current progress:

- Implemented an Isaac Gym rollout logger that generates `motion_with_action.pkl` files containing `root / dof / body state` and 23-DoF `action`.
- Completed Gym-to-Gym sanity checks for action-space semantics, action clipping, motion phase alignment, `motion_lib` loading, and deterministic actor mean.
- Exported the CR7 motion tracking policy to ONNX and built a local MuJoCo rollout collector to generate MuJoCo target rollout data.
- Trained an open-loop MuJoCo-to-Isaac-Gym deltaA model using the MuJoCo rollout pkl.
- Built zero-delta vs deterministic-deltaA evaluation scripts to verify whether the learned deltaA actually improves target-state matching.

Current observations:

- Open-loop deltaA can significantly improve root/body pose tracking; the best checkpoint observed around 12% improvement in total diff norm.
- Full-body 23-DoF deltaA may negatively affect DoF velocity and closed-loop stability.
- Closed-loop fine-tuning is sensitive to observation configuration, reward design, reference motion, and frozen deltaA integration, and is still under debugging.

Therefore, this part is not presented as a complete reproduction of the final ASAP residual results. It is kept as an **ASAP delta action data-pipeline and open-loop evaluation reproduction study**.

### Football Task Environment and Task Redesign

The football task initially used a single-hit design: the robot prepares, swings its leg, touches the ball, and receives a score based on the post-contact ball height. This version was useful for building the basic pipeline, but several problems became clear:

- The task was easily reduced to a one-shot kick scoring problem
- The reward became fragmented as more patches were added
- After a successful first kick, the body posture often failed to transition naturally into a second kick
- Purely optimizing height can reduce recoverability for the next touch

The task was therefore reframed as **maintaining the ball inside a controllable region in front of the robot**:

- When the ball is inside the target state region, the policy is rewarded for maintaining it there.
- When the ball leaves the region and starts falling, a corrective touch is needed.
- The purpose of contact is not simply to kick the ball higher, but to send it back into a sustainable control region.
- The focus shifts from “whether the robot touches the ball” to contact quality, outgoing ball trajectory, post-kick recovery, and continuability for the next touch.

The future training interface will be structured into several layers: the functional objective defines whether the ball returns to the controllable region, contact geometry defines whether the touch is physically reasonable, motion prior controls naturalness, and hard constraints remove unstable behaviors.

---

## Current Focus

The next stage mainly focuses on the football control task and deltaA reproduction wrap-up:

- [ ] Keep the kickup direction and stabilize the single recovery primitive first.
- [ ] Replace immediate ground-contact termination with softer penalty / delayed termination to avoid over-kicking.
- [ ] Add recoverability metrics after the first kick: torso stability, support stability, swing-leg recovery, and next-touch ready pose.
- [ ] Measure the minimum ball-foot distance during second swing attempts to identify whether the problem is timing, trajectory, or observation.
- [ ] Add local ball-relative-to-foot observations to reduce blind leg swinging.
- [ ] Continue reframing the task from “single-hit scoring” into “maintaining the ball inside a front controllable region”, with height, direction, and next-ready-state constraints.
- [ ] Run ankle-only or lower-body-only deltaA ablations to reduce the side effects of full-body residual actions on joint velocity and closed-loop stability.
- [ ] Continue improving Genesis / MuJoCo Sim2Sim testing to evaluate cross-engine generalization.

---

## Acknowledgements

Thanks to the [ASAP](https://github.com/LeCAR-Lab/ASAP) team for open-sourcing a strong embodied intelligence training framework.  
This project is mainly for my personal learning and experiments in reinforcement learning, robot control, and cross-physics-engine validation.

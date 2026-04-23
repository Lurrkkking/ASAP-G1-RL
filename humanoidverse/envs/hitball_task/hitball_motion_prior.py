from pathlib import Path

import torch

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import quat_rotate
from isaac_utils.rotations import calc_heading_quat_inv, quat_mul


class HitBallMotionPriorMixin:
    MOTION_STAGE_PREP = 0
    MOTION_STAGE_SWING = 1
    MOTION_STAGE_POST = 2

    def _init_hitball_motion_prior(self):
        motion_cfg = getattr(self.config.robot, "motion", None)
        motion_file = getattr(motion_cfg, "motion_file", None) if motion_cfg is not None else None
        default_anchor = float(getattr(self.config, "motion_prior_start_time_s", 0.0))
        self.motion_time_scale = float(getattr(self.config, "motion_prior_time_scale", 1.0))
        self.motion_time_anchor_prep = float(
            getattr(self.config, "motion_time_anchor_prep", default_anchor)
        )
        self.motion_time_anchor_swing = float(
            getattr(self.config, "motion_time_anchor_swing", self.motion_time_anchor_prep)
        )
        self.motion_time_anchor_post = float(
            getattr(self.config, "motion_time_anchor_post", self.motion_time_anchor_swing)
        )
        self.hitball_motion_prior_enabled = (
            motion_cfg is not None
            and motion_file is not None
            and self._motion_prior_file_is_available(motion_file)
        )

        if not self.hitball_motion_prior_enabled:
            self._init_empty_hitball_motion_prior_buffers()
            return

        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        random_sample = bool(getattr(self.config, "motion_prior_random_sample", False))
        self._motion_lib.load_motions(random_sample=random_sample)

        self.motion_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_len = self._motion_lib.get_motion_length(self.motion_ids)

        self._init_hitball_tracking_config()
        self._init_empty_hitball_motion_prior_buffers()
        self._reset_hitball_motion_phase_clock(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )
        self._update_hitball_motion_prior_reference()

    def _motion_prior_file_is_available(self, motion_file):
        motion_path = Path(str(motion_file))
        if motion_path.is_file():
            return True
        if motion_path.is_dir():
            return any(motion_path.glob("*.pkl"))
        return False

    def _init_empty_hitball_motion_prior_buffers(self):
        self.ref_body_pos = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device
        )
        self.ref_body_rot = torch.zeros(
            self.num_envs, self.num_bodies, 4, dtype=torch.float, device=self.device
        )
        self.ref_body_rot[..., 3] = 1.0
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device
        )
        self.ref_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ref_root_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ref_root_rot[:, 3] = 1.0
        self.ref_motion_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ref_motion_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ref_motion_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.hitball_tracking_phase_gate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.motion_phase_start_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_phase_local_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.motion_phase_anchor_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.motion_phase_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _init_hitball_tracking_config(self):
        rewards_cfg = self.config.rewards

        default_right_leg_dofs = [
            name
            for name in self.dof_names
            if name.startswith("right_hip")
            or name.startswith("right_knee")
            or name.startswith("right_ankle")
        ]
        if len(default_right_leg_dofs) == 0:
            default_right_leg_dofs = [
                name
                for name in self.dof_names
                if "right" in name and ("hip" in name or "knee" in name or "ankle" in name)
            ]
        right_leg_dof_names = list(
            getattr(rewards_cfg, "tracking_lower_body_dof_names", default_right_leg_dofs)
        )
        self.tracking_lower_body_dof_names = [
            name for name in right_leg_dof_names if name in self.dof_names
        ]
        if len(self.tracking_lower_body_dof_names) == 0:
            self.tracking_lower_body_dof_names = default_right_leg_dofs
        self.tracking_lower_body_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.tracking_lower_body_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        default_upper_body_dofs = list(getattr(self.config.robot, "upper_dof_names", []))
        upper_body_dof_names = list(
            getattr(rewards_cfg, "tracking_upper_body_dof_names", default_upper_body_dofs)
        )
        self.tracking_upper_body_dof_names = [
            name for name in upper_body_dof_names if name in self.dof_names
        ]
        if len(self.tracking_upper_body_dof_names) == 0:
            self.tracking_upper_body_dof_names = default_upper_body_dofs
        self.tracking_upper_body_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.tracking_upper_body_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        default_feet_body_names = [self.right_foot_body_name]
        feet_body_names = list(
            getattr(rewards_cfg, "tracking_feet_body_names", default_feet_body_names)
        )
        self.tracking_feet_body_names = [name for name in feet_body_names if name in self.body_names]
        if len(self.tracking_feet_body_names) == 0:
            self.tracking_feet_body_names = default_feet_body_names
        self.tracking_feet_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.tracking_feet_body_names],
            dtype=torch.long,
            device=self.device,
        )

        default_upper_body_names = list(getattr(self.config.robot.motion, "upper_body_link", []))
        upper_body_names = list(
            getattr(rewards_cfg, "tracking_upper_body_names", default_upper_body_names)
        )
        self.tracking_upper_body_names = [name for name in upper_body_names if name in self.body_names]
        if len(self.tracking_upper_body_names) == 0:
            self.tracking_upper_body_names = default_upper_body_names
        self.tracking_upper_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.tracking_upper_body_names],
            dtype=torch.long,
            device=self.device,
        )

        default_trunk_body_names = ["pelvis"]
        torso_name = getattr(self.config.robot, "torso_name", None)
        if torso_name is not None:
            default_trunk_body_names.append(torso_name)
        trunk_body_names = list(
            getattr(rewards_cfg, "tracking_trunk_body_names", default_trunk_body_names)
        )
        self.tracking_trunk_body_names = [name for name in trunk_body_names if name in self.body_names]
        if len(self.tracking_trunk_body_names) == 0:
            self.tracking_trunk_body_names = default_trunk_body_names
        self.tracking_trunk_body_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in self.tracking_trunk_body_names],
            dtype=torch.long,
            device=self.device,
        )

        self.tracking_joint_pos_sigma = float(getattr(rewards_cfg, "tracking_joint_pos_sigma", 0.35))
        ankle_dof_names = list(
            getattr(
                rewards_cfg,
                "tracking_right_ankle_dof_names",
                ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
            )
        )
        self.tracking_right_ankle_dof_names = [
            name for name in ankle_dof_names if name in self.dof_names
        ]
        self.tracking_right_ankle_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.tracking_right_ankle_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.tracking_right_ankle_pos_sigma = float(
            getattr(rewards_cfg, "tracking_right_ankle_pos_sigma", 0.12)
        )
        self.tracking_upper_body_joint_pos_sigma = float(
            getattr(rewards_cfg, "tracking_upper_body_joint_pos_sigma", self.tracking_joint_pos_sigma)
        )
        self.tracking_feet_pos_sigma = float(getattr(rewards_cfg, "tracking_feet_pos_sigma", 0.03))
        self.tracking_upper_body_pos_sigma = float(
            getattr(rewards_cfg, "tracking_upper_body_pos_sigma", self.tracking_feet_pos_sigma)
        )
        self.tracking_trunk_rot_sigma = float(getattr(rewards_cfg, "tracking_trunk_rot_sigma", 0.80))

        phase_gates = getattr(rewards_cfg, "tracking_phase_gates", {})
        self.tracking_phase_gate_values = {
            self.PRE_CONTACT_PREP: float(getattr(phase_gates, "pre_contact_prep", 0.25)),
            self.PRE_CONTACT_SWING: float(getattr(phase_gates, "pre_contact_swing", 1.0)),
            self.CONTACTED: float(getattr(phase_gates, "contacted", 0.45)),
            self.POST_CONTACT_EVAL: float(getattr(phase_gates, "post_contact_eval", 0.20)),
            self.DONE_PHASE: float(getattr(phase_gates, "done", 0.0)),
        }

    def _update_hitball_motion_prior_reference(self):
        if not self.hitball_motion_prior_enabled:
            self.hitball_tracking_phase_gate[:] = 0.0
            return

        self._sync_hitball_motion_phase_clock()
        phase_elapsed_steps = (
            self.episode_length_buf - self.motion_phase_start_step
        ).clamp(min=0)
        self.motion_phase_local_time[:] = (
            phase_elapsed_steps.float() * self.dt * self.motion_time_scale
        )
        motion_times = self.motion_phase_anchor_time + self.motion_phase_local_time
        motion_res = self._motion_lib.get_motion_state(
            self.motion_ids,
            motion_times,
            offset=self.env_origins,
        )

        self.ref_dof_pos[:] = motion_res["dof_pos"][:, : self.num_dof]
        self.ref_body_pos[:] = motion_res["rg_pos_t"][:, : self.num_bodies]
        self.ref_body_rot[:] = motion_res["rg_rot_t"][:, : self.num_bodies]
        self.ref_root_pos[:] = motion_res["root_pos"]
        self.ref_root_rot[:] = motion_res["root_rot"]
        self.ref_motion_time[:] = motion_times
        self.ref_motion_phase[:] = torch.clamp(motion_times / self.motion_len.clamp(min=1e-6), 0.0, 1.0)
        self.ref_motion_frame[:] = (
            self.ref_motion_phase * (self._motion_lib._motion_num_frames[self.motion_ids] - 1)
        ).long()
        self._update_hitball_tracking_phase_gate()

    def _get_hitball_motion_stage(self, task_phase=None):
        if task_phase is None:
            task_phase = self.task_phase

        motion_stage = torch.full_like(task_phase, self.MOTION_STAGE_POST)
        motion_stage[task_phase == self.PRE_CONTACT_PREP] = self.MOTION_STAGE_PREP
        motion_stage[task_phase == self.PRE_CONTACT_SWING] = self.MOTION_STAGE_SWING
        return motion_stage

    def _get_hitball_motion_anchor_time(self, motion_stage):
        anchor_time = torch.full(
            motion_stage.shape,
            self.motion_time_anchor_post,
            dtype=torch.float,
            device=self.device,
        )
        anchor_time[motion_stage == self.MOTION_STAGE_PREP] = self.motion_time_anchor_prep
        anchor_time[motion_stage == self.MOTION_STAGE_SWING] = self.motion_time_anchor_swing
        return anchor_time

    def _reset_hitball_motion_phase_clock(self, env_ids, motion_stage=None):
        if (not self.hitball_motion_prior_enabled) or len(env_ids) == 0:
            return

        if motion_stage is None:
            motion_stage = self._get_hitball_motion_stage(self.task_phase[env_ids])

        anchor_time = self._get_hitball_motion_anchor_time(motion_stage)
        self.motion_phase_stage[env_ids] = motion_stage
        self.motion_phase_start_step[env_ids] = self.episode_length_buf[env_ids]
        self.motion_phase_local_time[env_ids] = 0.0
        self.motion_phase_anchor_time[env_ids] = anchor_time
        self.ref_motion_time[env_ids] = anchor_time
        self.ref_motion_phase[env_ids] = torch.clamp(
            anchor_time / self.motion_len[env_ids].clamp(min=1e-6),
            0.0,
            1.0,
        )
        self.ref_motion_frame[env_ids] = (
            self.ref_motion_phase[env_ids]
            * (self._motion_lib._motion_num_frames[self.motion_ids[env_ids]] - 1)
        ).long()

    def _sync_hitball_motion_phase_clock(self):
        motion_stage = self._get_hitball_motion_stage()
        changed = motion_stage != self.motion_phase_stage
        changed_env_ids = changed.nonzero(as_tuple=False).flatten()
        if len(changed_env_ids) > 0:
            self._reset_hitball_motion_phase_clock(
                changed_env_ids,
                motion_stage=motion_stage[changed_env_ids],
            )

    def _update_hitball_tracking_phase_gate(self):
        self.hitball_tracking_phase_gate[:] = 0.0
        for phase, gate_value in self.tracking_phase_gate_values.items():
            self.hitball_tracking_phase_gate[self.task_phase == phase] = gate_value

    def _heading_local_body_pos(self, body_pos, root_pos, root_rot):
        body_rel = body_pos - root_pos.unsqueeze(1)
        heading_inv = calc_heading_quat_inv(root_rot.clone(), w_last=True)
        heading_inv = heading_inv.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        return quat_rotate(heading_inv.reshape(-1, 4), body_rel.reshape(-1, 3)).view_as(body_rel)

    def _heading_local_body_rot(self, body_rot, root_rot):
        heading_inv = calc_heading_quat_inv(root_rot.clone(), w_last=True)
        heading_inv = heading_inv.unsqueeze(1).expand(-1, body_rot.shape[1], -1)
        return quat_mul(heading_inv.reshape(-1, 4), body_rot.reshape(-1, 4), w_last=True).view_as(body_rot)

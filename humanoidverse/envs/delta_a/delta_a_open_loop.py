import torch
import numpy as np
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from termcolor import colored

from scipy.spatial.transform import Rotation as sRot
import joblib

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking


class DeltaA_OpenLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        # import ipdb; ipdb.set_trace()
        super().__init__(config, device)
        self.delta_action_dof_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        self.detla_action_percentage_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        self.delta_action_cnt = 1
        self.delta_action_mask_mode = self._get_delta_action_mask_mode()
        self.ankle_delta_indices = self._get_ankle_delta_indices()
        self.delta_action_mask = self._build_delta_action_mask()

    def _get_delta_action_mask_mode(self):
        algo_cfg = getattr(self.config, "algo", None)
        if algo_cfg is None:
            return "full"
        algo_inner_cfg = getattr(algo_cfg, "config", None)
        if algo_inner_cfg is None:
            return "full"
        return str(getattr(algo_inner_cfg, "delta_action_mask_mode", "full"))

    def _get_ankle_delta_indices(self):
        joint_names = list(self.config.robot.dof_names)
        ankle_joint_names = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        indices = []
        for joint_name in ankle_joint_names:
            if joint_name not in joint_names:
                raise ValueError(
                    f"Could not find ankle joint '{joint_name}' in robot.dof_names={joint_names}"
                )
            indices.append(joint_names.index(joint_name))
        return indices

    def _build_delta_action_mask(self):
        mode = self.delta_action_mask_mode
        if mode not in {"full", "ankle_only"}:
            raise ValueError(
                f"Unsupported delta_action_mask_mode={mode}. Expected one of ['full', 'ankle_only']."
            )
        mask = torch.ones(self.num_dofs, dtype=torch.float32, device=self.device)
        if mode == "ankle_only":
            mask.zero_()
            mask[self.ankle_delta_indices] = 1.0
        return mask

    def _mask_delta_action(self, delta_action):
        return delta_action * self.delta_action_mask

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if hasattr(self.config, 'zero_delta_a'):
            if self.config.zero_delta_a:                
                actions *= 0.
                print("actions", actions)
                pass
        raw_delta_action = actions
        masked_delta_action = self._mask_delta_action(raw_delta_action)
        actions_scaled = masked_delta_action * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if self.config['add_extra_action']:
            motion_action_raw = self.get_open_loop_action_at_current_timestep()
            # motion_action *= 0
            # print("motion_action", motion_action)
            # print("self.get_open_loop_action_at_current_timestep()", self.get_open_loop_action_at_current_timestep())
            motion_action_scaled = motion_action_raw * self.config.robot.control.action_scale
        else:
            motion_action_raw = torch.zeros_like(actions)
            motion_action_scaled = torch.zeros_like(actions_scaled)

        final_action_raw = motion_action_raw + masked_delta_action
        
        # add action_to_delta_a_heatmap
        
        self.delta_action_dof_heatmaps = self.delta_action_dof_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) * torch.abs(actions_scaled)
        self.detla_action_percentage_heatmaps = self.detla_action_percentage_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) *  torch.abs(actions_scaled / motion_action_scaled)
        self.delta_action_cnt += 1

        ankle_delta_raw = raw_delta_action[:, self.ankle_delta_indices]
        ankle_delta_masked = masked_delta_action[:, self.ankle_delta_indices]
        non_ankle_mask = (1.0 - self.delta_action_mask).bool()
        non_ankle_delta_after_mask = masked_delta_action[:, non_ankle_mask]
        if non_ankle_delta_after_mask.numel() > 0:
            non_ankle_delta_after_mask_mean = non_ankle_delta_after_mask.detach().abs().mean()
        else:
            non_ankle_delta_after_mask_mean = torch.zeros((), dtype=masked_delta_action.dtype, device=masked_delta_action.device)
        self.log_dict["delta_a/delta_action_raw_mean_abs"] = raw_delta_action.detach().abs().mean()
        self.log_dict["delta_a/delta_action_raw_max_abs"] = raw_delta_action.detach().abs().max()
        self.log_dict["delta_a/delta_action_raw_l2_mean"] = torch.norm(raw_delta_action.detach(), dim=-1).mean()
        self.log_dict["delta_a/delta_action_scaled_mean_abs"] = actions_scaled.detach().abs().mean()
        self.log_dict["delta_a/delta_action_scaled_max_abs"] = actions_scaled.detach().abs().max()
        self.log_dict["delta_a/motion_action_raw_mean_abs"] = motion_action_raw.detach().abs().mean()
        self.log_dict["delta_a/motion_action_raw_max_abs"] = motion_action_raw.detach().abs().max()
        self.log_dict["delta_a/motion_action_scaled_mean_abs"] = motion_action_scaled.detach().abs().mean()
        self.log_dict["delta_a/motion_action_scaled_max_abs"] = motion_action_scaled.detach().abs().max()
        self.log_dict["delta_a/delta_over_motion_scaled_mean"] = (
            actions_scaled.detach().abs() / (motion_action_scaled.detach().abs() + 1e-6)
        ).mean()
        self.log_dict["delta_a/ankle_delta_raw_mean_abs"] = ankle_delta_raw.detach().abs().mean()
        self.log_dict["delta_a/ankle_delta_raw_max_abs"] = ankle_delta_raw.detach().abs().max()
        self.log_dict["delta_a/raw_delta_action_mean_abs"] = raw_delta_action.detach().abs().mean()
        self.log_dict["delta_a/masked_delta_action_mean_abs"] = masked_delta_action.detach().abs().mean()
        self.log_dict["delta_a/ankle_delta_mean_abs"] = ankle_delta_masked.detach().abs().mean()
        self.log_dict["delta_a/non_ankle_delta_after_mask_mean_abs"] = non_ankle_delta_after_mask_mean
        self.log_dict["delta_a/delta_over_motion_mean"] = (
            masked_delta_action.detach().abs() / (motion_action_raw.detach().abs() + 1e-6)
        ).mean()
        self.log_dict["delta_a/final_action_mean_abs"] = final_action_raw.detach().abs().mean()
        self.log_dict["delta_a/final_action_max_abs"] = final_action_raw.detach().abs().max()
        # import ipdb; ipdb.set_trace()
        # print("motion_action", motion_action)
        # print("motion_action", motion_action)
        # print("motion_action", motion_action.shape)
        # print('actions', actions)
        # print('motion_action', motion_action)
        # import ipdb; ipdb.set_trace(

        # handcrafted perfect delta_a for 65Kp
        # perfect_delta_a_for_65Kp = (motion_action + self.default_dof_pos - self.simulator.dof_pos) * -0.35
        # actions_scaled = perfect_delta_a_for_65Kp

        if control_type=="P":
            # torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            torques = self._kp_scale * self.p_gains*(actions_scaled + motion_action_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques
        
    def _reward_minimal_action_norm(self):
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self._mask_delta_action(self.actions), dim=-1))


    def _reward_normalized_penalty_minimal_action_norm(self):
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self._mask_delta_action(self.actions), dim=-1)) - 1

    def _reward_penalty_minimal_action_norm(self):
        # print("self.actions", self.actions)
        # print("torch.norm(self.actions, dim=-1)", torch.norm(self.actions, dim=-1))
        # # import ipdb; ipdb.set_trace()
        # # print max abs values of rewards
        # print("max abs values of rewards", torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)))
        # # throw out the reward if the action norm is too large, and tell me which action caused it
        # if torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)) > 100000:
        #     print("action causing the reward to be thrown out", self.actions[torch.argmax(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1))])
        #     import ipdb; ipdb.set_trace()
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        #     # return torch.tensor(0.0)
        # else:
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        
        # clip the reward from -1000 to 1000
        return torch.exp(torch.norm(self._mask_delta_action(self.actions), dim=-1))-1
        # return torch.clip(torch.exp(torch.norm(self.actions, dim=-1))-1, -1000, 1000)
        
    def get_open_loop_action_at_current_timestep(self):
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_action = self._motion_lib.get_motion_actions(self.motion_ids, motion_times)

        return motion_action
    
    # NOTE: this is the perfect delta_a for 0.65Kp scenario
    def _get_perfect_delta_a(self):
        return (self.get_open_loop_action_at_current_timestep() * self.config.robot.control.action_scale + self.default_dof_pos - self.simulator.dof_pos) * -0.35

    def _get_obs_actions_open_loop(self):
        return self.get_open_loop_action_at_current_timestep()

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        # import ipdb; ipdb.set_trace()
        return self.simulator.dof_pos[:, self.ankle_delta_indices]
    
    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, self.ankle_delta_indices]

    def _get_obs_actions_ankle_pitch_roll(self):
        return self.actions[:, self.ankle_delta_indices]
    
    def _get_obs_actions_open_loop_ankle_pitch_roll(self):
        return self.get_open_loop_action_at_current_timestep()[:, self.ankle_delta_indices]
    

    

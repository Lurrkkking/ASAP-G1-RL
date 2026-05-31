"""Stand smoke env — reset to prepared pose, not yaml default."""
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
import torch


class LeggedRobotStandSmoke(LeggedRobotLocomotion):
    """Override reset to use prepared pose (prepare_sim equilibrium)."""

    def _init_buffers(self):
        super()._init_buffers()
        # Snapshot prepared pose right after init (post-prepare_sim equilibrium)
        self._prepared_dof_pos = self.simulator.dof_pos[0].clone()  # [22] single env
        # ALSO update default_dof_pos to prepared pose!
        # This fixes: PD target, observation dof_pos reference, zero-action torque
        self.default_dof_pos[:] = self._prepared_dof_pos.unsqueeze(0)

    def _reset_dofs(self, env_ids, target_state=None):
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            # Use prepared pose — NOT yaml default_joint_angles
            self.simulator.dof_pos[env_ids] = self._prepared_dof_pos.unsqueeze(0).expand(len(env_ids), -1)
            self.simulator.dof_vel[env_ids] = 0.

    def _reset_root_states(self, env_ids, target_root_states=None):
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.simulator.robot_root_states[env_ids, 7:13] = 0.

from isaacgym import gymapi, gymtorch
import torch

from humanoidverse.simulator.isaacgym.isaacgym import IsaacGym as IsaacGymBase


class IsaacGym(IsaacGymBase):
    """Minimal IsaacGym simulator for one robot actor plus one ball actor per env."""

    def load_assets(self):
        super().load_assets()
        self._load_ball_asset()

    def _load_ball_asset(self):
        ball_cfg = self.config.ball if "ball" in self.config else {}
        self.ball_radius = float(ball_cfg.get("radius", 0.10))
        self.ball_mass = float(ball_cfg.get("mass", 0.43))
        self.ball_restitution = float(ball_cfg.get("restitution", 0.45))
        self.ball_friction = float(ball_cfg.get("friction", 0.8))
        ball_init_pos = self.config.get("ball_init_pos", [0.0, 0.0, 1.0])
        ball_init_rot = self.config.get("ball_init_rot", [0.0, 0.0, 0.0, 1.0])

        ball_options = gymapi.AssetOptions()
        ball_options.density = float(ball_cfg.get("density", 1000.0))
        ball_options.angular_damping = float(ball_cfg.get("angular_damping", 0.01))
        ball_options.linear_damping = float(ball_cfg.get("linear_damping", 0.01))
        ball_options.max_angular_velocity = float(ball_cfg.get("max_angular_velocity", 100.0))
        ball_options.max_linear_velocity = float(ball_cfg.get("max_linear_velocity", 100.0))

        self.ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)
        ball_shape_props = self.gym.get_asset_rigid_shape_properties(self.ball_asset)
        for prop in ball_shape_props:
            prop.friction = self.ball_friction
            prop.restitution = self.ball_restitution
        self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_shape_props)

        self.ball_start_pose = gymapi.Transform()
        self.ball_start_pose.p = gymapi.Vec3(*ball_init_pos)
        self.ball_start_pose.r = gymapi.Quat(*ball_init_rot)
        self.ball_color = gymapi.Vec3(0.95, 0.55, 0.08)

    def _build_each_env(self, env_id, env_ptr):
        start_pose = gymapi.Transform()
        pos = self.env_origins[env_id].clone()
        start_pose.p = gymapi.Vec3(*pos)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_handle = self.gym.create_actor(
            env_ptr,
            self.robot_asset,
            start_pose,
            self.env_config.robot.asset.robot_type,
            env_id,
            self.env_config.robot.asset.self_collisions,
            0,
        )
        self._body_list = self.gym.get_actor_rigid_body_names(env_ptr, robot_handle)
        dof_props = self._process_dof_props(dof_props_asset, env_id)
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props, recomputeInertia=True)

        ball_handle = self.gym.create_actor(
            env_ptr,
            self.ball_asset,
            self.ball_start_pose,
            "ball",
            env_id,
            0,
            0,
        )
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, self.ball_color)
        ball_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
        ball_body_props[0].mass = self.ball_mass
        self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, ball_body_props, recomputeInertia=True)

        if env_id == 0:
            self.ball_handle = ball_handle

        self.envs.append(env_ptr)
        self.robot_handles.append(robot_handle)

    def prepare_sim(self):
        self.gym.prepare_sim(self.sim)
        self.refresh_sim_tensors()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        robot_name = self.env_config.robot.asset.robot_type
        self.jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, robot_name))
        self.massmatrix = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, robot_name))

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        self.refresh_sim_tensors()

        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self._get_num_actors_per_env()
        root_states = self.all_root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])
        self.robot_root_states = root_states[:, 0, :]
        self.ball_state = root_states[:, self.ball_handle, :]
        self.cubeA_state = self.ball_state
        self._cubeA_id = self.ball_handle
        self.base_quat = self.robot_root_states[..., 3:7]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        if self.offscreen_record:
            self._setup_offscreen_recording()

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        num_actors = self._get_num_actors_per_env()
        actor_ids = set_env_ids.unsqueeze(-1) * num_actors + torch.arange(
            num_actors, device=set_env_ids.device
        ).unsqueeze(0)
        actor_ids = actor_ids.reshape(-1).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        num_actors = self._get_num_actors_per_env()
        robot_actor_ids = (set_env_ids * num_actors).to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_states),
            gymtorch.unwrap_tensor(robot_actor_ids),
            len(robot_actor_ids),
        )

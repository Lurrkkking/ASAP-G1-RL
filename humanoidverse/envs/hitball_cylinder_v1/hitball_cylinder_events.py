import torch
from loguru import logger

from humanoidverse.envs.hitball_cylinder_v1.hitball_cylinder_obs import compute_cylinder_obs
from humanoidverse.utils.torch_utils import quat_rotate


def _in_box(x, ranges):
    return (
        (x[:, 0] >= ranges[0, 0])
        & (x[:, 0] <= ranges[0, 1])
        & (x[:, 1] >= ranges[1, 0])
        & (x[:, 1] <= ranges[1, 1])
        & (x[:, 2] >= ranges[2, 0])
        & (x[:, 2] <= ranges[2, 1])
    )


def _get_robot_stable_mask(env):
    gravity_xy = torch.norm(env.projected_gravity[:, :2], dim=-1)
    root_height = env.simulator.robot_root_states[:, 2]
    base_vel_xy = torch.norm(env.base_lin_vel[:, :2], dim=-1)
    return (
        (gravity_xy <= env.phase_gravity_xy_max)
        & (root_height >= env.phase_root_height_min)
        & (base_vel_xy <= env.phase_base_vel_xy_max)
    )


def _get_support_stable_mask(env):
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def _target_contact_now(env):
    if env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ball_contact_force = torch.norm(env.simulator.contact_forces[:, env.ball_body_env_idx], dim=-1)
    ball_has_contact = (
        (ball_contact_force > env.ball_contact_force_threshold)
        & (env.ball_pos[:, 2] > env.ball_contact_min_height)
    )
    candidate_pos = env.simulator._rigid_body_pos[:, env.target_foot_candidate_body_indices]
    candidate_dist = torch.norm(candidate_pos - env.ball_pos.unsqueeze(1), dim=-1)
    candidate_near_ball = torch.min(candidate_dist, dim=1).values < env.contact_dist_threshold
    return ball_has_contact & candidate_near_ball


def _non_target_contact_now(env):
    if env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot_body_forces = torch.norm(env.simulator.contact_forces[:, : env.num_bodies], dim=-1)
    force_mask = robot_body_forces > env.contact_force_threshold
    force_mask[:, env.target_foot_candidate_body_indices] = False
    return torch.any(force_mask, dim=1)


def _upper_body_ball_contact_now(env):
    if env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ball_contact_force = torch.norm(env.simulator.contact_forces[:, env.ball_body_env_idx], dim=-1)
    ball_has_contact = (
        (ball_contact_force > env.ball_contact_force_threshold)
        & (env.ball_pos[:, 2] > env.ball_contact_min_height)
    )
    near_candidates = []
    if env.wrong_contact_terminate_body_indices.numel() > 0:
        upper_body_pos = env.simulator._rigid_body_pos[:, env.wrong_contact_terminate_body_indices]
        upper_body_dist = torch.norm(upper_body_pos - env.ball_pos.unsqueeze(1), dim=-1)
        near_candidates.append(upper_body_dist < env.debug_non_target_ball_contact_dist_thresh)
    if env.elbow_proxy_body_indices.numel() > 0:
        proxy_dist_to_ball, _ = _compute_elbow_proxy_ball_dist_and_force(env)
        near_candidates.append(proxy_dist_to_ball < env.debug_non_target_ball_contact_dist_thresh)
    if not near_candidates:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    near_ball = torch.cat(near_candidates, dim=1)
    return ball_has_contact & torch.any(near_ball, dim=1)


def _compute_elbow_proxy_ball_dist_and_force(env):
    if env.elbow_proxy_body_indices.numel() == 0:
        return None, None

    robot_body_forces = torch.norm(env.simulator.contact_forces[:, : env.num_bodies], dim=-1)
    proxy_body_pos = env.simulator._rigid_body_pos[:, env.elbow_proxy_body_indices]
    proxy_body_rot = env.simulator._rigid_body_rot[:, env.elbow_proxy_body_indices]
    proxy_offsets_world = torch.zeros_like(proxy_body_pos)
    for i in range(env.elbow_proxy_body_indices.shape[0]):
        local_offset = env.elbow_proxy_local_offsets[i].unsqueeze(0).expand(env.num_envs, -1)
        proxy_offsets_world[:, i] = quat_rotate(proxy_body_rot[:, i], local_offset)
    proxy_pos = proxy_body_pos + proxy_offsets_world
    proxy_dist_to_ball = torch.norm(proxy_pos - env.ball_pos.unsqueeze(1), dim=-1)
    proxy_force = robot_body_forces[:, env.elbow_proxy_body_indices]
    return proxy_dist_to_ball, proxy_force


def _compute_ball_contact_attribution(env):
    if env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        env.ball_has_contact[:] = False
        env.ball_contact_nearest_body_index[:] = -1
        env.ball_contact_nearest_body_distance[:] = float("inf")
        env.ball_contact_nearest_body_is_target[:] = False
        env.ball_contact_topk_body_indices[:] = -1
        env.ball_contact_topk_body_distances[:] = float("inf")
        env.shank_contact_like[:] = False
        env.shank_contact_like_dist[:] = float("inf")
        return

    ball_contact_force = torch.norm(env.simulator.contact_forces[:, env.ball_body_env_idx], dim=-1)
    env.ball_has_contact[:] = (
        (ball_contact_force > env.ball_contact_force_threshold)
        & (env.ball_pos[:, 2] > env.ball_contact_min_height)
    )

    body_dist_to_ball = torch.norm(
        env.simulator._rigid_body_pos[:, : env.num_bodies] - env.ball_pos.unsqueeze(1), dim=-1
    )
    topk = min(max(1, int(getattr(env, "debug_ball_contact_topk", 5))), env.num_bodies)
    nearest_dist, nearest_idx = torch.min(body_dist_to_ball, dim=1)
    topk_dist, topk_idx = torch.topk(body_dist_to_ball, k=topk, largest=False, dim=1)

    env.ball_contact_nearest_body_index[:] = nearest_idx
    env.ball_contact_nearest_body_distance[:] = nearest_dist
    env.ball_contact_nearest_body_is_target[:] = env.target_foot_candidate_body_mask[nearest_idx]
    env.ball_contact_topk_body_indices[:] = -1
    env.ball_contact_topk_body_distances[:] = float("inf")
    env.ball_contact_topk_body_indices[:, :topk] = topk_idx
    env.ball_contact_topk_body_distances[:, :topk] = topk_dist

    if (
        getattr(env, "debug_shank_contact_like", False)
        and env.debug_shank_knee_body_idx is not None
        and env.debug_shank_ankle_body_idx is not None
    ):
        knee_pos = env.simulator._rigid_body_pos[:, env.debug_shank_knee_body_idx]
        ankle_pos = env.simulator._rigid_body_pos[:, env.debug_shank_ankle_body_idx]
        shank_vec = ankle_pos - knee_pos
        shank_len_sq = torch.sum(shank_vec * shank_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        ball_rel = env.ball_pos - knee_pos
        proj = torch.sum(ball_rel * shank_vec, dim=-1, keepdim=True) / shank_len_sq
        proj = torch.clamp(proj, 0.0, 1.0)
        closest = knee_pos + proj * shank_vec
        shank_dist = torch.norm(env.ball_pos - closest, dim=-1)
        env.shank_contact_like_dist[:] = shank_dist
        env.shank_contact_like[:] = env.ball_has_contact & (shank_dist < env.debug_shank_near_thresh)
    else:
        env.shank_contact_like[:] = False
        env.shank_contact_like_dist[:] = float("inf")


def _debug_print_non_target_ball_contact(env):
    if not getattr(env, "debug_print_non_target_ball_contact", False):
        return

    if env.debug_non_target_ball_contact_body_indices.numel() == 0 or env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        return

    ball_contact_force = torch.norm(env.simulator.contact_forces[:, env.ball_body_env_idx], dim=-1)
    ball_has_contact = (
        (ball_contact_force > env.ball_contact_force_threshold)
        & (env.ball_pos[:, 2] > env.ball_contact_min_height)
    )
    robot_body_forces = torch.norm(env.simulator.contact_forces[:, : env.num_bodies], dim=-1)
    body_dist_to_ball = torch.norm(
        env.simulator._rigid_body_pos[:, : env.num_bodies] - env.ball_pos.unsqueeze(1), dim=-1
    )
    monitor_forces = robot_body_forces[:, env.debug_non_target_ball_contact_body_indices]
    monitor_dists = body_dist_to_ball[:, env.debug_non_target_ball_contact_body_indices]
    monitor_contact_parts = [monitor_dists < env.debug_non_target_ball_contact_dist_thresh]
    proxy_dist_to_ball, proxy_force = _compute_elbow_proxy_ball_dist_and_force(env)
    if proxy_dist_to_ball is not None:
        monitor_contact_parts.append(proxy_dist_to_ball < env.debug_non_target_ball_contact_dist_thresh)
    env.debug_non_target_ball_contact_now[:] = ball_has_contact & torch.any(torch.cat(monitor_contact_parts, dim=1), dim=1)
    started = env.debug_non_target_ball_contact_now & (~env.prev_debug_non_target_ball_contact_now)
    if not torch.any(started):
        return

    topk = max(1, int(getattr(env, "debug_print_non_target_ball_contact_topk", 3)))
    started_ids = torch.nonzero(started, as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in started_ids:
        topk_entries = []
        for local_idx, world_body_idx in enumerate(env.debug_non_target_ball_contact_body_indices.detach().cpu().tolist()):
            dist_val = float(monitor_dists[env_id, local_idx].detach().cpu().item())
            if dist_val >= env.debug_non_target_ball_contact_dist_thresh:
                continue
            force_val = float(monitor_forces[env_id, local_idx].detach().cpu().item())
            topk_entries.append(
                (
                    env.body_names[int(world_body_idx)],
                    round(force_val, 4),
                    round(dist_val, 4),
                )
            )
        if proxy_dist_to_ball is not None:
            for proxy_name, dist_val, force_val in zip(
                env.elbow_proxy_body_names,
                proxy_dist_to_ball[env_id].detach().cpu().tolist(),
                proxy_force[env_id].detach().cpu().tolist(),
            ):
                if dist_val >= env.debug_non_target_ball_contact_dist_thresh:
                    continue
                topk_entries.append((proxy_name, round(float(force_val), 4), round(float(dist_val), 4)))
        topk_entries.sort(key=lambda x: x[2])
        topk_entries = topk_entries[:topk]
        best_body_name, best_force, best_dist = topk_entries[0]

        logger.warning(
            "HitBallCylinder upper-body ball contact: env={} step={} best_body={} force={:.4f} dist={:.4f} "
            "ball_force={:.4f} ball_pos={} ball_vel={} topk={}",
            env_id,
            int(env.episode_length_buf[env_id].detach().cpu().item()),
            best_body_name,
            best_force,
            best_dist,
            float(ball_contact_force[env_id].detach().cpu().item()),
            [round(float(x), 4) for x in env.ball_pos[env_id].detach().cpu().tolist()],
            [round(float(x), 4) for x in env.ball_lin_vel[env_id].detach().cpu().tolist()],
            topk_entries,
        )


def _debug_print_ball_contact_nearest_bodies(env):
    if env.ball_body_env_idx >= env.simulator.contact_forces.shape[1]:
        env.prev_ball_has_contact[:] = env.ball_has_contact
        return

    started = env.ball_has_contact & (~env.prev_ball_has_contact)
    env.prev_ball_has_contact[:] = env.ball_has_contact
    if not getattr(env, "debug_print_ball_contact_nearest_bodies", False):
        return
    if not torch.any(started):
        return

    ball_contact_force = torch.norm(env.simulator.contact_forces[:, env.ball_body_env_idx], dim=-1)
    robot_body_forces = torch.norm(env.simulator.contact_forces[:, : env.num_bodies], dim=-1)
    proxy_dist_to_ball, proxy_force = _compute_elbow_proxy_ball_dist_and_force(env)
    body_dist_to_ball = torch.norm(
        env.simulator._rigid_body_pos[:, : env.num_bodies] - env.ball_pos.unsqueeze(1), dim=-1
    )
    topk = min(max(1, int(getattr(env, "debug_ball_contact_topk", 5))), env.num_bodies)
    started_ids = torch.nonzero(started, as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in started_ids:
        dist_row = body_dist_to_ball[env_id]
        force_row = robot_body_forces[env_id]
        nearest_entries = []
        for body_idx, body_dist in zip(
            env.ball_contact_topk_body_indices[env_id].detach().cpu().tolist(),
            env.ball_contact_topk_body_distances[env_id].detach().cpu().tolist(),
        ):
            if body_idx < 0:
                continue
            nearest_entries.append(
                (
                    env.body_names[int(body_idx)],
                    round(float(body_dist), 4),
                    round(float(force_row[int(body_idx)].detach().cpu().item()), 4),
                    bool(env.target_foot_candidate_body_mask[int(body_idx)].detach().cpu().item()),
                )
            )
        if proxy_dist_to_ball is not None:
            for proxy_name, dist_val, force_val in zip(
                env.elbow_proxy_body_names,
                proxy_dist_to_ball[env_id].detach().cpu().tolist(),
                proxy_force[env_id].detach().cpu().tolist(),
            ):
                nearest_entries.append(
                    (
                        proxy_name,
                        round(float(dist_val), 4),
                        round(float(force_val), 4),
                        False,
                    )
                )
            nearest_entries.sort(key=lambda x: x[1])
            nearest_entries = nearest_entries[: max(topk, len(env.elbow_proxy_body_names))]

        upper_entries = []
        if env.wrong_contact_terminate_body_indices.numel() > 0:
            upper_dist = dist_row[env.wrong_contact_terminate_body_indices]
            upper_force = force_row[env.wrong_contact_terminate_body_indices]
            upper_topk_dist, upper_topk_idx = torch.topk(-upper_dist, k=min(topk, upper_dist.shape[0]))
            for _, local_idx in zip(upper_topk_dist.detach().cpu().tolist(), upper_topk_idx.detach().cpu().tolist()):
                world_idx = int(env.wrong_contact_terminate_body_indices[int(local_idx)].detach().cpu().item())
                upper_entries.append(
                    (
                        env.body_names[world_idx],
                        round(float(upper_dist[int(local_idx)].detach().cpu().item()), 4),
                        round(float(upper_force[int(local_idx)].detach().cpu().item()), 4),
                    )
                )
        if proxy_dist_to_ball is not None:
            for proxy_name, dist_val, force_val in zip(
                env.elbow_proxy_body_names,
                proxy_dist_to_ball[env_id].detach().cpu().tolist(),
                proxy_force[env_id].detach().cpu().tolist(),
            ):
                upper_entries.append(
                    (
                        proxy_name,
                        round(float(dist_val), 4),
                        round(float(force_val), 4),
                    )
                )
            upper_entries.sort(key=lambda x: x[1])
            upper_entries = upper_entries[: max(topk, len(env.elbow_proxy_body_names))]

        foot_entries = []
        foot_dist = dist_row[env.target_foot_candidate_body_indices]
        foot_force = force_row[env.target_foot_candidate_body_indices]
        foot_topk_dist, foot_topk_idx = torch.topk(-foot_dist, k=min(topk, foot_dist.shape[0]))
        for _, local_idx in zip(foot_topk_dist.detach().cpu().tolist(), foot_topk_idx.detach().cpu().tolist()):
            world_idx = int(env.target_foot_candidate_body_indices[int(local_idx)].detach().cpu().item())
            foot_entries.append(
                (
                    env.body_names[world_idx],
                    round(float(foot_dist[int(local_idx)].detach().cpu().item()), 4),
                    round(float(foot_force[int(local_idx)].detach().cpu().item()), 4),
                )
            )

        logger.warning(
            "HitBallCylinder ball contact attribution: env={} step={} ball_force={:.4f} ball_pos={} ball_vel={} "
            "nearest_all={} nearest_upper={} nearest_target_foot={} shank_contact_like={} shank_dist={:.4f}",
            env_id,
            int(env.episode_length_buf[env_id].detach().cpu().item()),
            float(ball_contact_force[env_id].detach().cpu().item()),
            [round(float(x), 4) for x in env.ball_pos[env_id].detach().cpu().tolist()],
            [round(float(x), 4) for x in env.ball_lin_vel[env_id].detach().cpu().tolist()],
            nearest_entries,
            upper_entries,
            foot_entries,
            bool(env.shank_contact_like[env_id].detach().cpu().item()),
            float(env.shank_contact_like_dist[env_id].detach().cpu().item()),
        )


def update_cylinder_phase(env):
    robot_stable = _get_robot_stable_mask(env)
    support_stable = _get_support_stable_mask(env)
    in_impact_window = _in_box(env.ball_pos_foot, env.impact_window)

    prev_phase = env.phase.clone()
    next_phase = prev_phase.clone()

    wait_to_prepare = (
        (prev_phase == env.PHASE_WAIT)
        & env.strike_gate
        & robot_stable
        & support_stable
        & (~env.carry_flag)
    )
    next_phase[wait_to_prepare] = env.PHASE_PREPARE

    prepare_to_strike = (
        (prev_phase == env.PHASE_PREPARE)
        & (
            (env.phase_step >= env.prepare_min_steps)
            | in_impact_window
            | (env.phase_step >= env.prepare_max_steps)
        )
    )
    next_phase[prepare_to_strike] = env.PHASE_STRIKE

    strike_to_recover = (
        (prev_phase == env.PHASE_STRIKE)
        & (
            env.contact_end
            | env.valid_kick
            | (env.phase_step >= env.strike_max_steps)
            | env.carry_flag
        )
    )
    next_phase[strike_to_recover] = env.PHASE_RECOVER

    recover_to_wait = (
        (prev_phase == env.PHASE_RECOVER)
        & (env.phase_step >= env.recover_min_steps)
        & robot_stable
        & (env.time_since_last_contact > env.strike_cooldown_steps)
    )
    next_phase[recover_to_wait] = env.PHASE_WAIT

    phase_changed = next_phase != prev_phase
    env.phase[:] = next_phase
    env.phase_step[phase_changed] = 0
    env.phase_step[~phase_changed] += 1


def update_cylinder_events(env):
    compute_cylinder_obs(env)

    env.prev_contact_now[:] = env.contact_now
    env.prev_non_target_contact_now[:] = env.non_target_contact_now
    env.prev_debug_non_target_ball_contact_now[:] = env.debug_non_target_ball_contact_now
    _compute_ball_contact_attribution(env)
    env.contact_now[:] = _target_contact_now(env)
    env.non_target_contact_now[:] = _non_target_contact_now(env)
    env.upper_body_ball_contact_now[:] = _upper_body_ball_contact_now(env)
    env.debug_contact_end_phase_strike[:] = False
    env.debug_contact_end_saved_strike_gate[:] = False
    env.debug_contact_end_short_contact[:] = False
    env.debug_contact_end_impact_geometry_good[:] = False
    env.debug_contact_end_release_good[:] = False
    env.debug_contact_end_post_release_trajectory_good[:] = False
    env.debug_contact_end_not_carry[:] = False
    _debug_print_ball_contact_nearest_bodies(env)
    _debug_print_non_target_ball_contact(env)
    env.contact_start[:] = env.contact_now & (~env.prev_contact_now)
    env.contact_end[:] = (~env.contact_now) & env.prev_contact_now
    env.valid_kick[:] = False

    idle = (~env.contact_now) & (~env.contact_end)
    env.contact_duration[idle] = 0
    env.contact_duration[env.contact_start] = 0
    env.contact_duration[env.contact_now] += 1
    env.time_since_last_contact += 1
    env.time_since_last_contact[env.contact_start] = 0
    env.contact_had_non_target_force[env.contact_now] |= env.non_target_contact_now[env.contact_now]
    env.contact_had_non_target_ball_nearest[env.contact_now] |= (
        (~env.ball_contact_nearest_body_is_target[env.contact_now]) & env.ball_has_contact[env.contact_now]
    )
    env.contact_had_non_target_force[idle] = False
    env.contact_had_non_target_ball_nearest[idle] = False

    ball_foot_dist = torch.norm(env.ball_pos_foot, dim=-1)
    ball_foot_rel_speed = torch.norm(env.ball_vel_foot, dim=-1)
    target_foot_vel_z = env.target_foot_vel[:, 2]
    env.ball_near_foot[:] = ball_foot_dist < env.near_foot_thresh
    env.ball_near_foot_duration[env.ball_near_foot] += 1
    env.ball_near_foot_duration[~env.ball_near_foot] = 0

    carry = (
        env.ball_near_foot
        & (ball_foot_rel_speed < env.carry_rel_speed_thresh)
        & (target_foot_vel_z > env.carry_foot_up_thresh)
        & (env.ball_lin_vel[:, 2] > env.carry_ball_up_thresh)
        & (env.ball_near_foot_duration > env.max_near_foot_steps)
    )
    env.carry_flag[:] = carry

    pred_pos_foot = env.ball_pred_pos_foot_tau.view(env.num_envs, len(env.prediction_horizons), 3)
    pred_vz = env.ball_pred_vz_tau
    strike_window = _in_box(pred_pos_foot[:, env.strike_gate_horizon_index], env.strike_window)
    descending = pred_vz[:, env.strike_gate_horizon_index] < env.strike_gate_vz_max
    cooldown_ok = env.time_since_last_contact > env.strike_cooldown_steps
    not_near_foot = ~env.ball_near_foot
    env.strike_gate[:] = strike_window & descending & not_near_foot & cooldown_ok

    if torch.any(env.contact_start):
        start = env.contact_start
        env.phase_at_contact_start[start] = env.phase[start]
        env.saved_strike_gate_at_contact_start[start] = env.strike_gate[start]
        env.saved_assisted_kick_at_contact_start[start] = env.assisted_kick_applied_on_contact[start]
        env.saved_ball_pos_foot_at_contact_start[start] = env.ball_pos_foot[start]
        env.saved_foot_vel_at_contact_start[start] = env.target_foot_vel[start]
        env.saved_contact_start_nearest_body_index[start] = env.ball_contact_nearest_body_index[start]
        env.saved_contact_start_nearest_body_distance[start] = env.ball_contact_nearest_body_distance[start]
        env.saved_contact_start_nearest_body_is_target[start] = env.ball_contact_nearest_body_is_target[start]

    if torch.any(env.contact_end):
        ended = env.contact_end
        start_pos = env.saved_ball_pos_foot_at_contact_start
        short_contact = env.contact_duration <= env.max_contact_steps
        impact_geometry_good = _in_box(start_pos, env.impact_window)
        release_good = (
            (ball_foot_dist > env.release_dist_thresh)
            | (ball_foot_rel_speed > env.release_rel_speed_thresh)
        )

        ball_vel_base = env.ball_vel_base
        ball_pos_base = env.ball_pos_base
        zone_delta = env.control_zone_center.unsqueeze(0) - ball_pos_base
        zone_delta_xy = zone_delta[:, :2]
        zone_dist_xy = torch.norm(zone_delta_xy, dim=-1).clamp(min=1e-6)
        vel_to_zone = torch.sum(ball_vel_base[:, :2] * zone_delta_xy, dim=-1) / zone_dist_xy
        horizontal_speed = torch.norm(ball_vel_base[:, :2], dim=-1)
        post_release_trajectory_good = (
            (env.ball_lin_vel[:, 2] > env.release_vz_min)
            & (env.ball_lin_vel[:, 2] < env.release_vz_max)
            & (horizontal_speed < env.release_horizontal_speed_max)
            & (vel_to_zone > env.release_toward_zone_speed_min)
        )
        env.debug_contact_end_phase_strike[:] = env.phase_at_contact_start == env.PHASE_STRIKE
        env.debug_contact_end_saved_strike_gate[:] = env.saved_strike_gate_at_contact_start
        env.debug_contact_end_short_contact[:] = short_contact
        env.debug_contact_end_impact_geometry_good[:] = impact_geometry_good
        env.debug_contact_end_release_good[:] = release_good
        env.debug_contact_end_post_release_trajectory_good[:] = post_release_trajectory_good
        env.debug_contact_end_not_carry[:] = ~env.carry_flag

        valid = (
            (env.phase_at_contact_start == env.PHASE_STRIKE)
            & env.saved_strike_gate_at_contact_start
            & short_contact
            & impact_geometry_good
            & release_good
            & post_release_trajectory_good
            & (~env.carry_flag)
        )
        env.valid_kick[ended] = valid[ended]
        env.last_valid_kick_step[env.valid_kick] = env.episode_length_buf[env.valid_kick]
        env.last_post_release_horizontal_speed[ended] = horizontal_speed[ended]
        if getattr(env, "debug_print_contact_event_details", False):
            ended_ids = torch.nonzero(ended, as_tuple=False).flatten().detach().cpu().tolist()
            for env_id in ended_ids:
                nearest_idx = int(env.saved_contact_start_nearest_body_index[env_id].detach().cpu().item())
                nearest_name = env.body_names[nearest_idx] if nearest_idx >= 0 else "NONE"
                logger.info(
                    "HitBallCylinder contact_end debug: env={} step={} valid_kick={} nearest_start_body=({}, {}) "
                    "nearest_start_is_target={} nearest_start_dist={:.4f} non_target_force_during_contact={} "
                    "non_target_nearest_during_contact={} shank_contact_like={} shank_dist={:.4f} "
                    "phase_strike={} strike_gate={} short_contact={} impact_geometry_good={} "
                    "release_good={} post_release_trajectory_good={} not_carry={}",
                    env_id,
                    int(env.episode_length_buf[env_id].detach().cpu().item()),
                    bool(env.valid_kick[env_id].detach().cpu().item()),
                    nearest_idx,
                    nearest_name,
                    bool(env.saved_contact_start_nearest_body_is_target[env_id].detach().cpu().item()),
                    float(env.saved_contact_start_nearest_body_distance[env_id].detach().cpu().item()),
                    bool(env.contact_had_non_target_force[env_id].detach().cpu().item()),
                    bool(env.contact_had_non_target_ball_nearest[env_id].detach().cpu().item()),
                    bool(env.shank_contact_like[env_id].detach().cpu().item()),
                    float(env.shank_contact_like_dist[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_phase_strike[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_saved_strike_gate[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_short_contact[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_impact_geometry_good[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_release_good[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_post_release_trajectory_good[env_id].detach().cpu().item()),
                    bool(env.debug_contact_end_not_carry[env_id].detach().cpu().item()),
                )

    update_cylinder_phase(env)
    publish_cylinder_event_logs(env)


def publish_cylinder_event_logs(env):
    zone = env._get_control_zone_mask()
    target_contact_ratio = env.contact_now.float().mean()
    non_target_contact_ratio = env.non_target_contact_now.float().mean()
    ball_contact_ratio = env.ball_has_contact.float().mean()
    nearest_target_ratio = (
        (env.ball_has_contact & env.ball_contact_nearest_body_is_target).float().sum()
        / env.ball_has_contact.float().sum().clamp(min=1.0)
    )
    nearest_non_target_ratio = (
        (env.ball_has_contact & (~env.ball_contact_nearest_body_is_target)).float().sum()
        / env.ball_has_contact.float().sum().clamp(min=1.0)
    )
    contact_start_nearest_target_ratio = (
        env.saved_contact_start_nearest_body_is_target[env.contact_start].float().mean()
        if torch.any(env.contact_start)
        else torch.zeros((), dtype=torch.float, device=env.device)
    )
    valid_kick_nearest_target_ratio = (
        env.saved_contact_start_nearest_body_is_target[env.valid_kick].float().mean()
        if torch.any(env.valid_kick)
        else torch.zeros((), dtype=torch.float, device=env.device)
    )
    target_whitelist_contact_ratio = (
        (env.ball_has_contact & env.ball_contact_nearest_body_is_target).float().mean()
    )
    non_target_body_contact_ratio = (
        (env.ball_has_contact & (~env.ball_contact_nearest_body_is_target)).float().mean()
    )
    action_rate = torch.norm(env.actions - env.last_actions, dim=-1)
    torque_ratio = torch.abs(env.torques) / env.torque_limits.clamp(min=1e-6)
    pre_impact_reward, pre_impact_active, pre_impact_dist = env._get_pre_impact_foot_to_ball_stats()
    pre_impact_active_float = pre_impact_active.float()
    pre_impact_active_count = pre_impact_active_float.sum()
    pre_impact_active_dist_mean = (
        pre_impact_dist * pre_impact_active_float
    ).sum() / pre_impact_active_count.clamp(min=1.0)
    pred_pos_foot = env.ball_pred_pos_foot_tau.view(env.num_envs, len(env.prediction_horizons), 3)
    pred_vz = env.ball_pred_vz_tau
    strike_window = _in_box(pred_pos_foot[:, env.strike_gate_horizon_index], env.strike_window)
    descending = pred_vz[:, env.strike_gate_horizon_index] < env.strike_gate_vz_max
    cooldown_ok = env.time_since_last_contact > env.strike_cooldown_steps
    not_near_foot = ~env.ball_near_foot

    env.log_dict["hitball_cylinder/strike_gate_rate"] = env.strike_gate.float().mean()
    env.log_dict["hitball_cylinder/strike_window_rate"] = strike_window.float().mean()
    env.log_dict["hitball_cylinder/descending_rate"] = descending.float().mean()
    env.log_dict["hitball_cylinder/cooldown_ok_rate"] = cooldown_ok.float().mean()
    env.log_dict["hitball_cylinder/not_near_foot_rate"] = not_near_foot.float().mean()
    env.log_dict["hitball_cylinder/contact_rate"] = env.contact_now.float().mean()
    env.log_dict["hitball_cylinder/contact_start_rate"] = env.contact_start.float().mean()
    env.log_dict["hitball_cylinder/contact_end_rate"] = env.contact_end.float().mean()
    env.log_dict["hitball_cylinder/contact_duration_mean"] = env.contact_duration.float().mean()
    env.log_dict["hitball_cylinder/valid_kick_rate"] = env.valid_kick.float().mean()
    env.log_dict["hitball_cylinder/carry_ratio"] = env.carry_flag.float().mean()
    env.log_dict["hitball_cylinder/phase_wait_rate"] = (env.phase == env.PHASE_WAIT).float().mean()
    env.log_dict["hitball_cylinder/phase_prepare_rate"] = (env.phase == env.PHASE_PREPARE).float().mean()
    env.log_dict["hitball_cylinder/phase_strike_rate"] = (env.phase == env.PHASE_STRIKE).float().mean()
    env.log_dict["hitball_cylinder/phase_recover_rate"] = (env.phase == env.PHASE_RECOVER).float().mean()
    env.log_dict["hitball_cylinder/phase_step_mean"] = env.phase_step.float().mean()
    env.log_dict["hitball_cylinder/zone_occupancy_ratio"] = zone.float().mean()
    env.log_dict["hitball_cylinder/post_release_horizontal_speed"] = (
        env.last_post_release_horizontal_speed.mean()
    )
    env.log_dict["hitball_cylinder/base_drift"] = torch.norm(
        env.simulator.robot_root_states[:, 0:2] - env.episode_start_base_pos[:, 0:2], dim=-1
    ).mean()
    env.log_dict["hitball_cylinder/target_foot_contact_ratio"] = target_contact_ratio
    env.log_dict["hitball_cylinder/non_target_contact_ratio"] = non_target_contact_ratio
    env.log_dict["hitball_cylinder/ball_has_contact_ratio"] = ball_contact_ratio
    env.log_dict["hitball_cylinder/target_whitelist_body_contact_ratio"] = target_whitelist_contact_ratio
    env.log_dict["hitball_cylinder/non_target_body_contact_ratio"] = non_target_body_contact_ratio
    env.log_dict["hitball_cylinder/nearest_body_target_ratio_given_ball_contact"] = nearest_target_ratio
    env.log_dict["hitball_cylinder/nearest_body_non_target_ratio_given_ball_contact"] = nearest_non_target_ratio
    env.log_dict["hitball_cylinder/contact_start_nearest_target_ratio"] = contact_start_nearest_target_ratio
    env.log_dict["hitball_cylinder/valid_kick_nearest_target_ratio"] = valid_kick_nearest_target_ratio
    env.log_dict["hitball_cylinder/contact_had_non_target_force_ratio"] = (
        env.contact_had_non_target_force.float().mean()
    )
    env.log_dict["hitball_cylinder/contact_had_non_target_nearest_ratio"] = (
        env.contact_had_non_target_ball_nearest.float().mean()
    )
    env.log_dict["hitball_cylinder/shank_contact_like_rate"] = env.shank_contact_like.float().mean()
    env.log_dict["hitball_cylinder/valid_kick_shank_contact_like_rate"] = (
        env.shank_contact_like[env.valid_kick].float().mean()
        if torch.any(env.valid_kick)
        else torch.zeros((), dtype=torch.float, device=env.device)
    )
    env.log_dict["hitball_cylinder/nearest_body_distance_mean"] = torch.where(
        env.ball_has_contact,
        env.ball_contact_nearest_body_distance,
        torch.zeros_like(env.ball_contact_nearest_body_distance),
    ).sum() / env.ball_has_contact.float().sum().clamp(min=1.0)
    env.log_dict["hitball_cylinder/shank_contact_like_dist_mean"] = torch.where(
        env.ball_has_contact,
        env.shank_contact_like_dist,
        torch.zeros_like(env.shank_contact_like_dist),
    ).sum() / env.ball_has_contact.float().sum().clamp(min=1.0)
    for body_idx, body_name in enumerate(env.body_names):
        nearest_mask = env.ball_has_contact & (env.ball_contact_nearest_body_index == body_idx)
        env.log_dict[f"hitball_cylinder/nearest_body_hist_idx_{body_idx}"] = nearest_mask.float().mean()
    env.log_dict["hitball_cylinder/non_target_contact_start_rate"] = (
        env.non_target_contact_now & (~env.prev_non_target_contact_now)
    ).float().mean()
    env.log_dict["hitball_cylinder/upper_body_ball_contact_rate"] = (
        env.upper_body_ball_contact_now.float().mean()
    )
    env.log_dict["hitball_cylinder/action_rate"] = action_rate.mean()
    env.log_dict["hitball_cylinder/torque_clip_ratio"] = (torque_ratio > 0.98).float().mean()
    env.log_dict["hitball_cylinder/reset_ball_pos_foot_mean_x"] = env.reset_ball_pos_foot_mean[0]
    env.log_dict["hitball_cylinder/reset_ball_pos_foot_mean_y"] = env.reset_ball_pos_foot_mean[1]
    env.log_dict["hitball_cylinder/reset_ball_pos_foot_mean_z"] = env.reset_ball_pos_foot_mean[2]
    env.log_dict["hitball_cylinder/pre_impact_reward_mean"] = (
        pre_impact_reward * pre_impact_active_float
    ).mean()
    env.log_dict["hitball_cylinder/pre_impact_active_rate"] = pre_impact_active_float.mean()
    env.log_dict["hitball_cylinder/pre_impact_foot_ball_dist"] = torch.where(
        pre_impact_active_count > 0,
        pre_impact_active_dist_mean,
        pre_impact_dist.mean(),
    )
    assisted_valid_kick = env.valid_kick & env.saved_assisted_kick_at_contact_start
    unassisted_valid_kick = env.valid_kick & (~env.saved_assisted_kick_at_contact_start)
    env.log_dict["assisted_kick/alpha"] = env.assisted_kick_alpha_buf.mean()
    env.log_dict["assisted_kick/active_rate"] = env.assisted_kick_active.float().mean()
    env.log_dict["assisted_kick/assisted_valid_kick_rate"] = assisted_valid_kick.float().mean()
    env.log_dict["assisted_kick/unassisted_valid_kick_rate"] = unassisted_valid_kick.float().mean()

    env.extras["hitball_cylinder"] = {
        "strike_gate": env.strike_gate.clone(),
        "strike_window": strike_window.clone(),
        "descending": descending.clone(),
        "cooldown_ok": cooldown_ok.clone(),
        "not_near_foot": not_near_foot.clone(),
        "ball_has_contact": env.ball_has_contact.clone(),
        "contact_now": env.contact_now.clone(),
        "contact_start": env.contact_start.clone(),
        "contact_end": env.contact_end.clone(),
        "contact_duration": env.contact_duration.clone(),
        "non_target_contact_now": env.non_target_contact_now.clone(),
        "upper_body_ball_contact_now": env.upper_body_ball_contact_now.clone(),
        "ball_contact_nearest_body_index": env.ball_contact_nearest_body_index.clone(),
        "ball_contact_nearest_body_distance": env.ball_contact_nearest_body_distance.clone(),
        "ball_contact_nearest_body_is_target": env.ball_contact_nearest_body_is_target.clone(),
        "ball_contact_topk_body_indices": env.ball_contact_topk_body_indices.clone(),
        "ball_contact_topk_body_distances": env.ball_contact_topk_body_distances.clone(),
        "contact_start_nearest_body_index": env.saved_contact_start_nearest_body_index.clone(),
        "contact_start_nearest_body_distance": env.saved_contact_start_nearest_body_distance.clone(),
        "contact_start_nearest_body_is_target": env.saved_contact_start_nearest_body_is_target.clone(),
        "contact_had_non_target_force": env.contact_had_non_target_force.clone(),
        "contact_had_non_target_ball_nearest": env.contact_had_non_target_ball_nearest.clone(),
        "shank_contact_like": env.shank_contact_like.clone(),
        "shank_contact_like_dist": env.shank_contact_like_dist.clone(),
        "valid_kick_phase_strike": env.debug_contact_end_phase_strike.clone(),
        "valid_kick_saved_strike_gate": env.debug_contact_end_saved_strike_gate.clone(),
        "valid_kick_short_contact": env.debug_contact_end_short_contact.clone(),
        "valid_kick_impact_geometry_good": env.debug_contact_end_impact_geometry_good.clone(),
        "valid_kick_release_good": env.debug_contact_end_release_good.clone(),
        "valid_kick_post_release_trajectory_good": env.debug_contact_end_post_release_trajectory_good.clone(),
        "valid_kick_not_carry": env.debug_contact_end_not_carry.clone(),
        "phase": env.phase.clone(),
        "phase_step": env.phase_step.clone(),
        "phase_at_contact_start": env.phase_at_contact_start.clone(),
        "valid_kick": env.valid_kick.clone(),
        "assisted_kick_active": env.assisted_kick_active.clone(),
        "assisted_valid_kick": assisted_valid_kick.clone(),
        "carry_flag": env.carry_flag.clone(),
        "zone_occupancy": zone.clone(),
        "ball_pos_base": env.ball_pos_base.clone(),
        "ball_pos_foot": env.ball_pos_foot.clone(),
        "ball_pred_pos_base_tau": env.ball_pred_pos_base_tau.clone(),
        "ball_pred_pos_foot_tau": env.ball_pred_pos_foot_tau.clone(),
    }

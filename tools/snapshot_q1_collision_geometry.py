#!/usr/bin/env python3
"""
Standalone diagnostic script: Q1 goalkeeper collision geometry snapshot.

Reads the current Q1 goalkeeper env, exports collision shape data,
computes foot overlap metrics, and generates PNG visualizations.

Does NOT:
  - Modify any existing code, URDF, config, or reward files
  - Start PPO training
  - Load or save any checkpoint

Only writes to: outputs/{robot_type}_collision_snapshot/

Usage (run from /root/autodl-tmp/ASAP):

  # Q1 goalkeeper
  python tools/snapshot_q1_collision_geometry.py \
      +exp=q1_goalkeeper_smoke num_envs=1 headless=True

  # G1 motion tracking (needs extra overrides)
  python tools/snapshot_q1_collision_geometry.py \
      +exp=motion_tracking +robot=g1/g1_29dof_anneal_23dof \
      +simulator=isaacgym +terrain=terrain_locomotion_plane \
      +domain_rand=NO_domain_rand \
      +rewards=motion_tracking/reward_motion_tracking_dm_2real \
      +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
      "robot.motion.motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/custom/kick_primitive.pkl" \
      num_envs=1 headless=True
"""

import os
import sys
import json
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import hydra
from omegaconf import OmegaConf, open_dict

# Register OmegaConf resolvers used by project configs (same as config_utils.py)
try:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception:
    pass  # resolvers already registered

# IsaacGym must be imported before torch (project convention)
import isaacgym  # noqa: F401
from isaacgym import gymapi

import torch
from loguru import logger

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ═══════════════════════════════════════════════
# Math utilities
# ═══════════════════════════════════════════════
def euler_to_quat(roll, pitch, yaw):
    """Euler (roll,pitch,yaw) → quaternion [x,y,z,w] (intrinsic XYZ)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ])


def quat_multiply(q1, q2):
    """Multiply two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q [x,y,z,w]."""
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    vq = np.array([v[0], v[1], v[2], 0.0])
    return quat_multiply(quat_multiply(q, vq), qc)[:3]


# ═══════════════════════════════════════════════
# URDF collision geometry parser
# ═══════════════════════════════════════════════
def parse_urdf_collision_shapes(urdf_path):
    """Parse URDF collision elements → list of dicts per shape."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    shapes = []
    for link in root.findall("link"):
        link_name = link.get("name")
        for collision in link.findall("collision"):
            origin_el = collision.find("origin")
            if origin_el is not None:
                xyz = [float(v) for v in origin_el.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin_el.get("rpy", "0 0 0").split()]
            else:
                xyz, rpy = [0., 0., 0.], [0., 0., 0.]

            geom = collision.find("geometry")
            if geom is None:
                continue

            shape_info = {
                "link_name": link_name,
                "shape_type": None,
                "local_pos": np.array(xyz),
                "local_rpy": np.array(rpy),
                "dimensions": None,
            }

            for tag, stype, dim_fn in [
                ("box", "box", lambda s: [float(v) for v in s.get("size").split()]),
                ("sphere", "sphere", lambda s: [float(s.get("radius"))]),
                ("capsule", "capsule", lambda s: [float(s.get("radius")), float(s.get("length"))]),
                ("cylinder", "cylinder", lambda s: [float(s.get("radius")), float(s.get("length"))]),
            ]:
                el = geom.find(tag)
                if el is not None:
                    shape_info["shape_type"] = stype
                    shape_info["dimensions"] = dim_fn(el)
                    shapes.append(shape_info)
                    break

    return shapes


# ═══════════════════════════════════════════════
# World-space AABB computation
# ═══════════════════════════════════════════════
def compute_shape_world_aabb(body_world_pos, body_world_quat,
                              shape_local_pos, shape_local_rpy,
                              shape_type, shape_dims,
                              contact_offset=0.0):
    """Compute world-space AABB for a single collision shape."""
    shape_local_quat = euler_to_quat(*shape_local_rpy)
    world_center = body_world_pos + quat_rotate(body_world_quat, shape_local_pos)

    if shape_type == "box":
        hx, hy, hz = [d / 2.0 for d in shape_dims[:3]]
    elif shape_type == "sphere":
        r = shape_dims[0] + contact_offset
        hx = hy = hz = r
    elif shape_type in ("capsule", "cylinder"):
        r = shape_dims[0] + contact_offset
        half_len = shape_dims[1] / 2.0
        hx = hy = r
        hz = half_len + r
    else:
        hx = hy = hz = 0.05

    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [-hx, hy,  hz], [hx, hy,  hz],
    ])

    world_quat = quat_multiply(body_world_quat, shape_local_quat)
    corners_world = np.array([world_center + quat_rotate(world_quat, c)
                               for c in corners_local])

    return world_center, corners_world.min(axis=0), corners_world.max(axis=0)


def bbox_overlap(min1, max1, min2, max2):
    return np.all(min1 <= max2) and np.all(min2 <= max1)


def bbox_min_distance(min1, max1, min2, max2):
    dist2 = 0.0
    for i in range(3):
        if max1[i] < min2[i]:
            dist2 += (min2[i] - max1[i]) ** 2
        elif max2[i] < min1[i]:
            dist2 += (min1[i] - max2[i]) ** 2
    return math.sqrt(dist2)


# ═══════════════════════════════════════════════
# Matplotlib 3D helpers
# ═══════════════════════════════════════════════
def draw_bbox_3d(ax, bmin, bmax, color, alpha=0.3, linewidth=1.0, label=None):
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]], [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]], [bmin[0], bmax[1], bmax[2]],
    ])
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]
    for face in faces:
        poly = Poly3DCollection([face], alpha=alpha, facecolor=color,
                                edgecolor=color, linewidth=linewidth)
        ax.add_collection3d(poly)
    if label:
        ctr = (bmin + bmax) / 2
        ax.text(ctr[0], ctr[1], ctr[2], label, fontsize=6, ha='center',
                color='black', bbox=dict(boxstyle='round,pad=0.1',
                                         facecolor='white', alpha=0.7))


def draw_ground_plane(ax, z=0.0, x_range=(-1, 1), y_range=(-1, 1), alpha=0.12):
    xx, yy = np.meshgrid(np.linspace(*x_range, 20), np.linspace(*y_range, 20))
    ax.plot_surface(xx, yy, np.full_like(xx, z), color='tan', alpha=alpha,
                    zorder=0)


def setup_3d_axes(ax, title, xlim=None, ylim=None, zlim=None,
                  elev=None, azim=None):
    ax.set_xlabel("X (forward)"); ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (up)"); ax.set_title(title, fontsize=10)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)
    ax.set_aspect('equal', adjustable='box')


def get_body_color(name):
    if "left" in name: return "steelblue"
    if "right" in name: return "coral"
    if name in ("pelvis", "waist_roll_link", "torso_link"): return "seagreen"
    return "gray"


# ═══════════════════════════════════════════════
# Hydra entry point
# ═══════════════════════════════════════════════
@hydra.main(config_path="../humanoidverse/config", config_name="base",
            version_base="1.1")
def main(config: OmegaConf):
    # ── Output directory ──
    orig_cwd = hydra.utils.get_original_cwd()
    robot_type = config.robot.asset.robot_type
    out_dir = Path(orig_cwd) / f"outputs/{robot_type}_collision_snapshot"
    # Allow override via: snapshot.out_dir=some/path
    snap = config.get("snapshot", None)
    if snap is not None and hasattr(snap, "out_dir"):
        out_dir = Path(orig_cwd) / snap.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-process config ──
    from humanoidverse.utils.helpers import pre_process_config
    pre_process_config(config)

    with open_dict(config.env.config):
        config.env.config.robot = config.robot
        config.env.config.obs = config.obs
        if hasattr(config, "algo") and hasattr(config.algo, "config"):
            config.env.config.algo = config.algo

    config.env.config.save_rendering_dir = str(out_dir / "renderings")

    from humanoidverse.envs.base_task.base_task import BaseTask
    from hydra.utils import instantiate

    # ── Print header ──
    print("\n" + "=" * 65)
    print("  Q1 COLLISION GEOMETRY SNAPSHOT")
    print("=" * 65)

    exp_name = config.get("experiment_name", "unknown")
    actual_urdf = config.robot.asset.urdf_file
    asset_root = config.robot.asset.asset_root
    root_z = config.robot.init_state.pos[2]

    print(f"  Experiment:        {exp_name}")
    print(f"  URDF (config):     {actual_urdf}")
    print(f"  Asset root:        {asset_root}")
    print(f"  root_z config:     {root_z}")
    print(f"  action_dim:        {config.robot.actions_dim}")
    print(f"  num_envs:          {config.num_envs}")

    # ── [1/8] Create env ──
    print("\n[1/8] Creating headless env (num_envs=1)...")
    device = config.get("device", None)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env: BaseTask = instantiate(config=config.env, device=device)
    print("  ✓ Env created")

    sim = env.simulator
    gym = sim.gym
    env_ptr = sim.envs[0]
    robot_handle = sim.robot_handles[0]

    # ── [2/8] Asset info ──
    print("\n[2/8] Asset info...")
    asset_path = os.path.join(asset_root, actual_urdf)
    print(f"  Resolved URDF:     {asset_path}")
    print(f"  File exists:       {os.path.exists(asset_path)}")

    expected_urdf = "q1/q1_22dof_rl_collision.urdf"
    urdf_warning = ""
    if actual_urdf != expected_urdf:
        urdf_warning = f"URDF MISMATCH: expected {expected_urdf}, got {actual_urdf}"
        print(f"  ⚠️  {urdf_warning}")

    num_dof, num_bodies = sim.num_dof, sim.num_bodies
    dof_names = list(sim.dof_names)
    body_names = list(sim.body_names)
    print(f"  DOFs={num_dof}  Bodies={num_bodies}")
    print(f"  Bodies: {body_names}")

    asset_info = {
        "experiment_name": exp_name,
        "urdf_file_config": actual_urdf,
        "urdf_path_resolved": asset_path,
        "urdf_exists": os.path.exists(asset_path),
        "urdf_warning": urdf_warning,
        "asset_root": asset_root,
        "root_z_init": root_z,
        "action_dim": config.robot.actions_dim,
        "num_bodies": num_bodies, "num_dofs": num_dof,
        "body_names": body_names, "dof_names": dof_names,
        "num_envs": config.num_envs,
        "robot_type": config.robot.asset.robot_type,
        "self_collisions": config.robot.asset.self_collisions,
    }
    with open(out_dir / "asset_info.json", "w") as f:
        json.dump(asset_info, f, indent=2, default=str)
    print("  ✓ asset_info.json")

    # ── [3/8] Set robot to standing pose ──
    print("\n[3/8] Setting standing pose + settle...")
    env_target = str(config.env._target_)
    is_motion_env = 'motion_tracking' in env_target
    is_goalkeeper_env = 'goalkeeper' in env_target

    if is_goalkeeper_env:
        # Use normal reset (env has prepared equilibrium pose)
        print("  Goalkeeper env — using reset_envs_idx (prepared pose)")
        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        env.reset_envs_idx(env_ids)
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
        sim.refresh_sim_tensors()
    elif is_motion_env:
        # Skip motion reset; manually set config default standing pose
        print("  Motion-tracking env — using config defaults (skip motion pose)")
        default_angles = config.robot.init_state.default_joint_angles
        init_pos = list(config.robot.init_state.pos)
        init_rot = list(config.robot.init_state.rot)
        for i, dof_name in enumerate(sim.dof_names):
            if dof_name in default_angles:
                sim.dof_pos[0, i] = default_angles[dof_name]
        sim.dof_vel[0, :] = 0.0
        sim.robot_root_states[0, :3] = torch.tensor(init_pos, device=env.device)
        sim.robot_root_states[0, 3:7] = torch.tensor(init_rot, device=env.device)
        sim.robot_root_states[0, 7:13] = 0.0
        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        sim.set_dof_state_tensor(env_ids,
            torch.cat([sim.dof_pos[env_ids].unsqueeze(-1),
                       sim.dof_vel[env_ids].unsqueeze(-1)], dim=-1))
        sim.set_actor_root_state_tensor(env_ids, sim.robot_root_states[env_ids])
        for _ in range(10):
            sim.simulate_at_each_physics_step()
        sim.refresh_sim_tensors()
    else:
        # Generic: use config defaults
        print("  Generic env — using config defaults")
        default_angles = config.robot.init_state.default_joint_angles
        init_pos = list(config.robot.init_state.pos)
        init_rot = list(config.robot.init_state.rot)
        for i, dof_name in enumerate(sim.dof_names):
            if dof_name in default_angles:
                sim.dof_pos[0, i] = default_angles[dof_name]
        sim.dof_vel[0, :] = 0.0
        sim.robot_root_states[0, :3] = torch.tensor(init_pos, device=env.device)
        sim.robot_root_states[0, 3:7] = torch.tensor(init_rot, device=env.device)
        sim.robot_root_states[0, 7:13] = 0.0
        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        sim.set_dof_state_tensor(env_ids,
            torch.cat([sim.dof_pos[env_ids].unsqueeze(-1),
                       sim.dof_vel[env_ids].unsqueeze(-1)], dim=-1))
        sim.set_actor_root_state_tensor(env_ids, sim.robot_root_states[env_ids])
        for _ in range(10):
            sim.simulate_at_each_physics_step()
        sim.refresh_sim_tensors()
    body_positions = sim._rigid_body_pos[0].cpu().numpy()  # [B,3]
    body_quats = sim._rigid_body_rot[0].cpu().numpy()      # [B,4] xyzw
    pelvis_z = body_positions[body_names.index("pelvis"), 2]
    print(f"  Pelvis world Z =   {pelvis_z:.4f}")

    # ── [4/8] Enumerate rigid shapes ──
    print("\n[4/8] Enumerating rigid shapes...")

    # Actor-level shape properties (runtime friction/restitution/offsets)
    actor_shape_props = gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
    asset_shape_count = gym.get_asset_rigid_shape_count(sim.robot_asset)
    print(f"  actor shapes = {len(actor_shape_props)}  asset shapes = {asset_shape_count}")

    # Parse URDF for geometry type and dimensions
    urdf_shapes = parse_urdf_collision_shapes(asset_path)
    urdf_by_link = defaultdict(list)
    for s in urdf_shapes:
        urdf_by_link[s["link_name"]].append(s)
    print(f"  URDF collision elements = {len(urdf_shapes)}")

    # For each body, query asset rigid body shape indices
    # API: get_asset_rigid_body_shape_indices(asset) → List[IndexRange]
    # Each IndexRange has .start (first shape index) and .count (num shapes)
    rigid_shapes = []
    shape_ranges = gym.get_asset_rigid_body_shape_indices(sim.robot_asset)
    print(f"  Bodies from shape ranges: {len(shape_ranges)}")

    for body_idx in range(num_bodies):
        body_name = body_names[body_idx]
        ir = shape_ranges[body_idx]  # IndexRange with .start, .count
        urdf_for_body = urdf_by_link.get(body_name, [])

        for k in range(ir.count):
            shape_idx = ir.start + k

            # URDF geometry (shape type + dims + local pose)
            if k < len(urdf_for_body):
                ud = urdf_for_body[k]
                shape_type = ud["shape_type"]
                dims = ud["dimensions"]
                local_pos = ud["local_pos"]
                local_rpy = ud["local_rpy"]
            else:
                shape_type = "unknown"
                dims = [0.05, 0.05, 0.05]
                local_pos = np.zeros(3)
                local_rpy = np.zeros(3)

            # Actor-level properties (friction, restitution, offsets)
            if shape_idx < len(actor_shape_props):
                p = actor_shape_props[shape_idx]
                friction, restitution = p.friction, p.restitution
                contact_offset, rest_offset = p.contact_offset, p.rest_offset
            else:
                friction = restitution = contact_offset = rest_offset = 0.0

            entry = {
                "body_index": body_idx,
                "body_name": body_name,
                "shape_index": int(shape_idx),
                "shape_type": shape_type,
                "local_pos_x": float(local_pos[0]),
                "local_pos_y": float(local_pos[1]),
                "local_pos_z": float(local_pos[2]),
                "local_rpy_r": float(local_rpy[0]),
                "local_rpy_p": float(local_rpy[1]),
                "local_rpy_y": float(local_rpy[2]),
                "dim_x": float(dims[0]) if len(dims) > 0 else None,
                "dim_y": float(dims[1]) if len(dims) > 1 else None,
                "dim_z": float(dims[2]) if len(dims) > 2 else None,
                "radius": float(dims[0]) if shape_type in ("sphere", "capsule", "cylinder") else None,
                "length": float(dims[1]) if shape_type in ("capsule", "cylinder") and len(dims) > 1 else None,
                "contact_offset": float(contact_offset),
                "rest_offset": float(rest_offset),
                "friction": float(friction),
                "restitution": float(restitution),
            }
            rigid_shapes.append(entry)

    print(f"  Enumerated {len(rigid_shapes)} shapes across {num_bodies} bodies")

    # Save CSV/JSON
    csv_fields = [
        "body_index", "body_name", "shape_index", "shape_type",
        "local_pos_x", "local_pos_y", "local_pos_z",
        "local_rpy_r", "local_rpy_p", "local_rpy_y",
        "dim_x", "dim_y", "dim_z", "radius", "length",
        "contact_offset", "rest_offset", "friction", "restitution",
    ]
    with open(out_dir / "rigid_shapes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rigid_shapes)
    with open(out_dir / "rigid_shapes.json", "w") as f:
        json.dump(rigid_shapes, f, indent=2, default=float)
    print("  ✓ rigid_shapes.csv + .json")

    # ── [5/8] World-space AABBs ──
    print("\n[5/8] Computing world-space AABBs...")
    world_bboxes = []
    bodies_with_shapes = set()
    for s in rigid_shapes:
        bodies_with_shapes.add(s["body_name"])
        body_idx = s["body_index"]
        bwp = body_positions[body_idx]
        bwq = body_quats[body_idx]
        lp = np.array([s["local_pos_x"], s["local_pos_y"], s["local_pos_z"]])
        lr = np.array([s["local_rpy_r"], s["local_rpy_p"], s["local_rpy_y"]])

        if s["shape_type"] == "box":
            dims = [s["dim_x"], s["dim_y"], s["dim_z"]]
        elif s["shape_type"] == "sphere":
            dims = [s["radius"]]
        elif s["shape_type"] in ("capsule", "cylinder"):
            dims = [s["radius"], s["length"]]
        else:
            dims = [0.05, 0.05, 0.05]

        ctr, bmin, bmax = compute_shape_world_aabb(
            bwp, bwq, lp, lr, s["shape_type"], dims, s["contact_offset"])

        world_bboxes.append({
            "body_name": s["body_name"], "shape_index": s["shape_index"],
            "shape_type": s["shape_type"],
            "world_center_x": float(ctr[0]),
            "world_center_y": float(ctr[1]),
            "world_center_z": float(ctr[2]),
            "bbox_min_x": float(bmin[0]), "bbox_min_y": float(bmin[1]),
            "bbox_min_z": float(bmin[2]),
            "bbox_max_x": float(bmax[0]), "bbox_max_y": float(bmax[1]),
            "bbox_max_z": float(bmax[2]),
            "bbox_size_x": float(bmax[0]-bmin[0]),
            "bbox_size_y": float(bmax[1]-bmin[1]),
            "bbox_size_z": float(bmax[2]-bmin[2]),
        })

    bodies_without_shapes = [bn for bn in body_names
                             if bn not in bodies_with_shapes]

    with open(out_dir / "collision_world_bbox.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(world_bboxes[0].keys()))
        w.writeheader(); w.writerows(world_bboxes)
    print(f"  ✓ collision_world_bbox.csv ({len(world_bboxes)} bboxes)")

    # Per-body merged AABB
    body_merged = {}
    for bb in world_bboxes:
        bn = bb["body_name"]
        bmin = np.array([bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]])
        bmax = np.array([bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]])
        if bn not in body_merged:
            body_merged[bn] = {"min": bmin, "max": bmax}
        else:
            body_merged[bn]["min"] = np.minimum(body_merged[bn]["min"], bmin)
            body_merged[bn]["max"] = np.maximum(body_merged[bn]["max"], bmax)

    # ── [6/8] Foot collision metrics ──
    print("\n[6/8] Foot collision analysis...")
    foot_keys = [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_knee_link", "right_knee_link",
    ]

    left_foot = body_merged.get("left_ankle_roll_link")
    right_foot = body_merged.get("right_ankle_roll_link")
    left_ap = body_merged.get("left_ankle_pitch_link")
    right_ap = body_merged.get("right_ankle_pitch_link")

    ground_z = 0.0
    metrics = {}

    if left_foot is not None:
        metrics["left_foot_bottom_z"] = float(left_foot["min"][2])
        metrics["left_foot_clearance"] = float(left_foot["min"][2] - ground_z)
        print(f"  left_foot_bottom_z   = {metrics['left_foot_bottom_z']:.4f}")
        print(f"  left_foot_clearance  = {metrics['left_foot_clearance']:.4f}")

    if right_foot is not None:
        metrics["right_foot_bottom_z"] = float(right_foot["min"][2])
        metrics["right_foot_clearance"] = float(right_foot["min"][2] - ground_z)
        print(f"  right_foot_bottom_z  = {metrics['right_foot_bottom_z']:.4f}")
        print(f"  right_foot_clearance = {metrics['right_foot_clearance']:.4f}")

    if left_foot is not None and right_foot is not None:
        overlap = bbox_overlap(left_foot["min"], left_foot["max"],
                               right_foot["min"], right_foot["max"])
        min_d = bbox_min_distance(left_foot["min"], left_foot["max"],
                                  right_foot["min"], right_foot["max"])
        ly = (left_foot["min"][1] + left_foot["max"][1]) / 2
        ry = (right_foot["min"][1] + right_foot["max"][1]) / 2
        metrics["left_right_foot_bbox_overlap"] = bool(overlap)
        metrics["left_right_foot_bbox_min_distance"] = float(min_d)
        metrics["foot_y_separation"] = float(abs(ly - ry))
        print(f"  foot_bbox_overlap    = {overlap}")
        print(f"  foot_bbox_min_dist   = {min_d:.4f}")
        print(f"  foot_y_separation    = {abs(ly-ry):.4f}")

    if left_ap is not None and left_foot is not None:
        metrics["left_ankle_pitch_roll_overlap"] = bool(
            bbox_overlap(left_ap["min"], left_ap["max"],
                         left_foot["min"], left_foot["max"]))
        print(f"  left_ap_roll_overlap = {metrics['left_ankle_pitch_roll_overlap']}")
    if right_ap is not None and right_foot is not None:
        metrics["right_ankle_pitch_roll_overlap"] = bool(
            bbox_overlap(right_ap["min"], right_ap["max"],
                         right_foot["min"], right_foot["max"]))
        print(f"  right_ap_roll_overlap= {metrics['right_ankle_pitch_roll_overlap']}")

    metrics["bodies_without_shapes"] = bodies_without_shapes

    suspicious_large = []
    for bb in world_bboxes:
        sz = max(bb["bbox_size_x"], bb["bbox_size_y"], bb["bbox_size_z"])
        if sz > 0.3:
            suspicious_large.append(f"{bb['body_name']}[s{bb['shape_index']}]: {sz:.3f}m")
    metrics["suspicious_large_shapes"] = suspicious_large

    suspicious_low = []
    for bb in world_bboxes:
        if bb["bbox_min_z"] < -0.02:
            suspicious_low.append(
                f"{bb['body_name']}[s{bb['shape_index']}]: min_z={bb['bbox_min_z']:.4f}")
    metrics["suspicious_low_shapes_below_ground"] = suspicious_low

    with open(out_dir / "feet_collision_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("  ✓ feet_collision_metrics.json")

    # ── [7/8] Visualizations ──
    print("\n[7/8] Generating PNG visualizations...")

    all_bmins = np.array([[bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]]
                           for bb in world_bboxes])
    all_bmaxs = np.array([[bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]]
                           for bb in world_bboxes])
    go_min = all_bmins.min(axis=0) - 0.1
    go_max = all_bmaxs.max(axis=0) + 0.1

    LP = [mpatches.Patch(color='steelblue', alpha=0.5, label='Left'),
          mpatches.Patch(color='coral', alpha=0.5, label='Right'),
          mpatches.Patch(color='seagreen', alpha=0.5, label='Torso/Pelvis'),
          mpatches.Patch(color='gray', alpha=0.5, label='Other')]

    def _draw_all(ax, bboxes, show_labels=False):
        for bb in bboxes:
            c = get_body_color(bb["body_name"])
            draw_bbox_3d(ax,
                         np.array([bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]]),
                         np.array([bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]]),
                         color=c, alpha=0.35)
        for bi, bn in enumerate(body_names):
            if bn in body_merged:
                pos = body_positions[bi]
                ax.scatter(pos[0], pos[1], pos[2], c=get_body_color(bn),
                           marker='o', s=15, zorder=10)
                if show_labels:
                    ax.text(pos[0], pos[1], pos[2]+0.02, bn, fontsize=5,
                            ha='center', va='bottom')

    # 1. Front view
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    _draw_all(ax, world_bboxes)
    draw_ground_plane(ax, 0, (go_min[0], go_max[0]), (go_min[1], go_max[1]))
    setup_3d_axes(ax, "Q1 Collision Geometry — Front View", elev=10, azim=0,
                  xlim=(go_min[0], go_max[0]),
                  ylim=(go_min[1], go_max[1]),
                  zlim=(min(go_min[2], -0.1), go_max[2]))
    ax.legend(handles=LP, loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"collision_front.png", dpi=150)
    plt.close(); print("  ✓ collision_front.png")

    # 2. Side view
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    _draw_all(ax, world_bboxes)
    draw_ground_plane(ax, 0, (go_min[0], go_max[0]), (go_min[1], go_max[1]))
    setup_3d_axes(ax, "Q1 Collision Geometry — Side View", elev=10, azim=-90,
                  xlim=(go_min[0], go_max[0]),
                  ylim=(go_min[1], go_max[1]),
                  zlim=(min(go_min[2], -0.1), go_max[2]))
    ax.legend(handles=LP, loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"collision_side.png", dpi=150)
    plt.close(); print("  ✓ collision_side.png")

    # 3. Top view
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    _draw_all(ax, world_bboxes, show_labels=True)
    setup_3d_axes(ax, "Q1 Collision Geometry — Top View", elev=90, azim=0,
                  xlim=(go_min[0], go_max[0]),
                  ylim=(go_min[1], go_max[1]),
                  zlim=(go_min[2], go_max[2]))
    ax.legend(handles=LP, loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"collision_top.png", dpi=150)
    plt.close(); print("  ✓ collision_top.png")

    # 4. Feet zoom top
    feet_bboxes = [bb for bb in world_bboxes if bb["body_name"] in foot_keys]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for bb in feet_bboxes:
        c = "blue" if "left" in bb["body_name"] else "red"
        bmin = np.array([bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]])
        bmax = np.array([bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]])
        draw_bbox_3d(ax, bmin, bmax, c, alpha=0.45, linewidth=1.5)
        ctr = (bmin+bmax)/2
        ax.text(ctr[0], ctr[1], ctr[2]+0.03, bb["body_name"], fontsize=7,
                ha='center', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    for bi, bn in enumerate(body_names):
        if bn in foot_keys:
            pos = body_positions[bi]
            ax.scatter(pos[0], pos[1], pos[2], c="blue" if "left" in bn else "red",
                       marker='o', s=30, zorder=10)

    if feet_bboxes:
        fbmin = np.array([[bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]]
                           for bb in feet_bboxes])
        fbmax = np.array([[bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]]
                           for bb in feet_bboxes])
        fmn, fmx = fbmin.min(axis=0)-0.08, fbmax.max(axis=0)+0.08
    else:
        fmn, fmx = np.array([-0.5,-0.5,-0.5]), np.array([0.5,0.5,0.5])

    if left_foot is not None and right_foot is not None:
        lc = (left_foot["min"]+left_foot["max"])/2
        rc = (right_foot["min"]+right_foot["max"])/2
        status = "OVERLAP!" if metrics.get("left_right_foot_bbox_overlap") else "OK"
        sc = "red" if metrics.get("left_right_foot_bbox_overlap") else "green"
        ax.text((lc[0]+rc[0])/2, (lc[1]+rc[1])/2, fmx[2]+0.02,
                f"MinDist={metrics.get('left_right_foot_bbox_min_distance','?'):.3f}m | {status}",
                fontsize=10, ha='center', color=sc, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))

    setup_3d_axes(ax, "Q1 Feet Collision — Zoom Top", elev=90, azim=0,
                  xlim=(fmn[0], fmx[0]), ylim=(fmn[1], fmx[1]),
                  zlim=(fmn[2], fmx[2]))
    ax.legend(handles=[mpatches.Patch(color='blue', alpha=0.5, label='Left'),
                        mpatches.Patch(color='red', alpha=0.5, label='Right')],
              loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"feet_collision_zoom_top.png", dpi=150)
    plt.close(); print("  ✓ feet_collision_zoom_top.png")

    # 5. Feet zoom front
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for bb in feet_bboxes:
        c = "blue" if "left" in bb["body_name"] else "red"
        bmin = np.array([bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]])
        bmax = np.array([bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]])
        draw_bbox_3d(ax, bmin, bmax, c, alpha=0.45, linewidth=1.5)
    draw_ground_plane(ax, 0, (fmn[0], fmx[0]), (fmn[1], fmx[1]), alpha=0.2)
    if left_foot is not None:
        lfz = left_foot["min"][2]; lfc = (left_foot["min"]+left_foot["max"])/2
        ax.text(lfc[0], lfc[1], lfz-0.02,
                f"L bot={lfz:.3f} clr={metrics.get('left_foot_clearance','?'):.3f}",
                fontsize=7, ha='center', color='blue',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    if right_foot is not None:
        rfz = right_foot["min"][2]; rfc = (right_foot["min"]+right_foot["max"])/2
        ax.text(rfc[0], rfc[1], rfz-0.02,
                f"R bot={rfz:.3f} clr={metrics.get('right_foot_clearance','?'):.3f}",
                fontsize=7, ha='center', color='red',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    setup_3d_axes(ax, "Q1 Feet Collision — Zoom Front", elev=10, azim=0,
                  xlim=(fmn[0], fmx[0]), ylim=(fmn[1], fmx[1]),
                  zlim=(min(fmn[2], -0.1), fmx[2]))
    ax.legend(handles=[mpatches.Patch(color='blue', alpha=0.5, label='Left'),
                        mpatches.Patch(color='red', alpha=0.5, label='Right')],
              loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"feet_collision_zoom_front.png", dpi=150)
    plt.close(); print("  ✓ feet_collision_zoom_front.png")

    # 6. Arm/torso front view
    upper_bodies = [
        "pelvis", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
    ]
    arm_bboxes = [bb for bb in world_bboxes if bb["body_name"] in upper_bodies]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for bb in arm_bboxes:
        c = get_body_color(bb["body_name"])
        draw_bbox_3d(ax,
                     np.array([bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]]),
                     np.array([bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]]),
                     color=c, alpha=0.4, linewidth=1.5)
        ctr = np.array([bb["world_center_x"], bb["world_center_y"],
                         bb["world_center_z"]])
        ax.text(ctr[0], ctr[1], ctr[2], bb["body_name"], fontsize=6,
                ha='center', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    for bi, bn in enumerate(body_names):
        if bn in upper_bodies:
            ax.scatter(body_positions[bi, 0], body_positions[bi, 1],
                       body_positions[bi, 2], c=get_body_color(bn),
                       marker='o', s=25, zorder=10)

    if arm_bboxes:
        abmin = np.array([[bb["bbox_min_x"], bb["bbox_min_y"], bb["bbox_min_z"]]
                           for bb in arm_bboxes])
        abmax = np.array([[bb["bbox_max_x"], bb["bbox_max_y"], bb["bbox_max_z"]]
                           for bb in arm_bboxes])
        amn, amx = abmin.min(axis=0)-0.1, abmax.max(axis=0)+0.1
    else:
        amn, amx = np.array([-0.5,-0.5,0.]), np.array([0.5,0.5,1.5])

    setup_3d_axes(ax, "Q1 Arm & Torso Collision — Front View", elev=10, azim=0,
                  xlim=(amn[0], amx[0]), ylim=(amn[1], amx[1]),
                  zlim=(amn[2], amx[2]))
    ax.legend(handles=[mpatches.Patch(color='steelblue', alpha=0.5, label='Left arm'),
                        mpatches.Patch(color='coral', alpha=0.5, label='Right arm'),
                        mpatches.Patch(color='seagreen', alpha=0.5, label='Torso')],
              loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/"arm_collision_front.png", dpi=150)
    plt.close(); print("  ✓ arm_collision_front.png")

    # ── [8/8] Final report ──
    print("\n" + "=" * 65)
    print("[8/8] FINAL REPORT")
    print("=" * 65)

    def _fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)

    rows = [
        ("Actual URDF", actual_urdf, urdf_warning),
        ("Total bodies", num_bodies, ""),
        ("Total shapes", len(rigid_shapes), ""),
        ("Bodies w/o shapes", str(bodies_without_shapes),
         "⚠️ " if bodies_without_shapes else ""),
        ("L/R foot overlap",
         str(metrics.get("left_right_foot_bbox_overlap", "N/A")),
         "⚠️ OVERLAP!" if metrics.get("left_right_foot_bbox_overlap") else ""),
        ("L/R foot min dist",
         _fmt(metrics.get("left_right_foot_bbox_min_distance", "N/A")),
         "⚠️ <5cm" if isinstance(metrics.get("left_right_foot_bbox_min_distance"), float)
                      and metrics["left_right_foot_bbox_min_distance"] < 0.05 else ""),
        ("Left foot clearance",
         _fmt(metrics.get("left_foot_clearance", "N/A")),
         "⚠️ BELOW GROUND" if isinstance(metrics.get("left_foot_clearance"), float)
                             and metrics["left_foot_clearance"] < -0.01 else ""),
        ("Right foot clearance",
         _fmt(metrics.get("right_foot_clearance", "N/A")),
         "⚠️ BELOW GROUND" if isinstance(metrics.get("right_foot_clearance"), float)
                             and metrics["right_foot_clearance"] < -0.01 else ""),
        ("L ankle pitch/roll overlap",
         str(metrics.get("left_ankle_pitch_roll_overlap", "N/A")),
         "⚠️ DUPLICATE" if metrics.get("left_ankle_pitch_roll_overlap") else ""),
        ("R ankle pitch/roll overlap",
         str(metrics.get("right_ankle_pitch_roll_overlap", "N/A")),
         "⚠️ DUPLICATE" if metrics.get("right_ankle_pitch_roll_overlap") else ""),
        ("Suspicious large shapes", len(suspicious_large),
         "⚠️ " + ", ".join(suspicious_large) if suspicious_large else ""),
        ("Shapes below ground", len(suspicious_low),
         "⚠️ " + ", ".join(suspicious_low) if suspicious_low else ""),
        ("Foot Y separation",
         _fmt(metrics.get("foot_y_separation", "N/A")), ""),
    ]

    print(f"  {'Item':<38s} | {'Value':<22s} | Warning")
    print(f"  {'-'*38}-+-{'-'*22}-+-{'-'*35}")
    for item, val, warn in rows:
        print(f"  {item:<38s} | {str(val):<22s} | {warn}")

    imgs = ["collision_front.png", "collision_side.png",
            "collision_top.png", "feet_collision_zoom_top.png",
            "feet_collision_zoom_front.png", "arm_collision_front.png"]
    data_files = ["asset_info.json", "rigid_shapes.csv", "rigid_shapes.json",
                  "collision_world_bbox.csv", "feet_collision_metrics.json"]

    print(f"\n  Images ({len(imgs)}):")
    for img in imgs:
        print(f"    {out_dir/img}")
    print(f"\n  Data ({len(data_files)}):")
    for f in data_files:
        print(f"    {out_dir/f}")

    print("\n" + "=" * 65)
    print("  DIAGNOSTIC COMPLETE — no files modified.")
    print("=" * 65 + "\n")

    # Clean shutdown
    gym.destroy_sim(sim.sim)


if __name__ == "__main__":
    main()

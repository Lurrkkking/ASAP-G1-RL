import os
import sys

import genesis as gs
import numpy as np
import onnxruntime as ort
import torch

sys.path.append("/root/autodl-tmp/ASAP")
from humanoidverse.utils.torch_utils import quat_rotate_inverse


URDF_PATH = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf"
ONNX_PATH = os.environ.get(
    "ONNX_PATH",
    "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260407_155156-MotionTracking_CR7_V2_Fixed-motion_tracking-g1_29dof_anneal_23dof/exported/model_2000.onnx",
)
OUTPUT_VIDEO = os.environ.get("OUT_VIDEO", "g1_siuuu_genesis.mp4")

SCALES = {
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "actions": 1.0,
    "ref_motion_phase": 1.0,
}

OBS_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "dof_pos": 23,
    "dof_vel": 23,
    "actions": 23,
    "ref_motion_phase": 1,
}

# Match config.obs.obs_auxiliary.history_actor for motion_tracking.
HISTORY_CFG = {
    "base_ang_vel": 4,
    "projected_gravity": 4,
    "dof_pos": 4,
    "dof_vel": 4,
    "actions": 4,
    "ref_motion_phase": 4,
}

# Training concatenates obs keys using sorted(obs_config).
ACTOR_OBS_ORDER = sorted([
    "base_ang_vel",
    "projected_gravity",
    "dof_pos",
    "dof_vel",
    "actions",
    "ref_motion_phase",
    "history_actor",
])
HISTORY_KEY_ORDER = sorted(HISTORY_CFG.keys())

ACTION_SCALE = float(os.environ.get("ACTION_SCALE", "0.25"))
ACTION_CLIP_VALUE = float(os.environ.get("ACTION_CLIP_VALUE", "100.0"))
CONTROL_DECIMATION = 4
SIM_FPS = 200
SIM_DT = 1.0 / SIM_FPS
SIM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# CR7 motion file used in this run has 134 frames at 30 FPS -> 4.4667s.
MOTION_DURATION = float(os.environ.get("MOTION_DURATION", "4.4666666667"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "600"))
KD_SCALE = float(os.environ.get("KD_SCALE", "1.5"))

RL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# Keep only training-side body set, same as locomotion Genesis runner.
BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]

DEFAULT_DOF_POS = np.array(
    [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

KP = np.array(
    [
        100,
        100,
        100,
        200,
        20,
        20,
        100,
        100,
        100,
        200,
        20,
        20,
        400,
        400,
        400,
        90,
        60,
        20,
        60,
        90,
        60,
        20,
        60,
    ],
    dtype=np.float32,
)

KD = np.array(
    [
        2.5,
        2.5,
        2.5,
        5.0,
        0.2,
        0.1,
        2.5,
        2.5,
        2.5,
        5.0,
        0.2,
        0.1,
        5.0,
        5.0,
        5.0,
        2.0,
        1.0,
        0.4,
        1.0,
        2.0,
        1.0,
        0.4,
        1.0,
    ],
    dtype=np.float32,
)
KD = KD * KD_SCALE

TORQUE_LIMITS = np.array(
    [
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
    ],
    dtype=np.float32,
)


def as_tensor_2d(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=SIM_DEVICE)
    else:
        x = x.to(device=SIM_DEVICE, dtype=torch.float32)
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def to_numpy_1d(x):
    x = as_tensor_2d(x)
    return x[0].detach().cpu().numpy().astype(np.float32)


def make_history_buffers():
    return {
        key: np.zeros((HISTORY_CFG[key], OBS_DIMS[key]), dtype=np.float32)
        for key in HISTORY_CFG
    }


def query_history_actor(history_buffers):
    chunks = []
    for key in HISTORY_KEY_ORDER:
        chunks.append(history_buffers[key].reshape(-1))
    history_actor = np.concatenate(chunks, axis=0).astype(np.float32)
    return history_actor


def update_history_buffers(history_buffers, curr_features):
    # Match HistoryHandler.add: shift old -> index+1, write newest at index 0.
    for key in HISTORY_CFG:
        history_buffers[key][1:] = history_buffers[key][:-1]
        history_buffers[key][0] = curr_features[key]


def build_curr_features(dof_pos, dof_vel, base_ang_vel_body, projected_gravity, last_actions, phase):
    return {
        "actions": (last_actions * SCALES["actions"]).astype(np.float32),
        "base_ang_vel": (base_ang_vel_body * SCALES["base_ang_vel"]).astype(np.float32),
        "dof_pos": ((dof_pos - DEFAULT_DOF_POS) * SCALES["dof_pos"]).astype(np.float32),
        "dof_vel": (dof_vel * SCALES["dof_vel"]).astype(np.float32),
        "projected_gravity": (projected_gravity * SCALES["projected_gravity"]).astype(np.float32),
        "ref_motion_phase": np.array([phase * SCALES["ref_motion_phase"]], dtype=np.float32),
    }


def build_actor_obs(curr_features, history_actor):
    parts = {
        "actions": curr_features["actions"],
        "base_ang_vel": curr_features["base_ang_vel"],
        "dof_pos": curr_features["dof_pos"],
        "dof_vel": curr_features["dof_vel"],
        "history_actor": history_actor,
        "projected_gravity": curr_features["projected_gravity"],
        "ref_motion_phase": curr_features["ref_motion_phase"],
    }
    obs = np.concatenate([parts[k] for k in ACTOR_OBS_ORDER], axis=0).astype(np.float32)
    if obs.shape[0] != 380:
        raise RuntimeError(f"actor_obs dim mismatch: got {obs.shape[0]}, expected 380")
    return obs


def main():
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not os.path.isfile(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=10),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=URDF_PATH,
            merge_fixed_links=True,
            links_to_keep=BODY_NAMES,
            pos=(0.0, 0.0, 0.8),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(2.0, 2.0, 1.0),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=False,
    )
    scene.build()

    motor_dofs = [robot.get_joint(name).dof_idx_local for name in RL_JOINT_NAMES]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX input={session.get_inputs()[0].shape}, output={session.get_outputs()[0].shape}")
    print(f"[INFO] MOTION_DURATION={MOTION_DURATION:.4f}, KD_SCALE={KD_SCALE:.3f}")

    robot.set_dofs_position(torch.tensor(DEFAULT_DOF_POS, dtype=torch.float32, device=SIM_DEVICE), motor_dofs)
    robot.set_dofs_velocity(torch.zeros(23, dtype=torch.float32, device=SIM_DEVICE), motor_dofs)

    gravity_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=SIM_DEVICE)
    last_actions = np.zeros(23, dtype=np.float32)
    history_buffers = make_history_buffers()

    cam.start_recording()
    print("[INFO] Start motion-tracking ONNX rollout")

    for step in range(NUM_STEPS):
        dof_pos = to_numpy_1d(robot.get_dofs_position(motor_dofs))
        dof_vel = to_numpy_1d(robot.get_dofs_velocity(motor_dofs))

        quat_wxyz = as_tensor_2d(robot.get_quat())
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        base_ang_vel_world = as_tensor_2d(robot.get_ang())

        base_ang_vel_body = to_numpy_1d(quat_rotate_inverse(quat_xyzw, base_ang_vel_world))
        projected_gravity = to_numpy_1d(quat_rotate_inverse(quat_xyzw, gravity_world))

        # Training computes phase from motion time; +1 step offset approximates episode_length_buf+1.
        t = (step + 1) * (SIM_DT * CONTROL_DECIMATION)
        phase = np.float32((t % MOTION_DURATION) / MOTION_DURATION)

        curr_features = build_curr_features(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            base_ang_vel_body=base_ang_vel_body,
            projected_gravity=projected_gravity,
            last_actions=last_actions,
            phase=phase,
        )

        # Match training timing: actor_obs uses history from previous steps (not including current step).
        history_actor = query_history_actor(history_buffers)
        actor_obs = build_actor_obs(curr_features, history_actor)

        action = session.run([output_name], {input_name: actor_obs[None, :]})[0][0].astype(np.float32)
        action = np.clip(action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)

        # Update history after observation/inference.
        update_history_buffers(history_buffers, curr_features)
        last_actions = action.copy()

        target_pos = action * ACTION_SCALE + DEFAULT_DOF_POS

        for _ in range(CONTROL_DECIMATION):
            dof_pos_sub = to_numpy_1d(robot.get_dofs_position(motor_dofs))
            dof_vel_sub = to_numpy_1d(robot.get_dofs_velocity(motor_dofs))
            torques = KP * (target_pos - dof_pos_sub) - KD * dof_vel_sub
            torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
            robot.control_dofs_force(torch.tensor(torques, dtype=torch.float32, device=SIM_DEVICE), motor_dofs)
            scene.step()

        root_pos = to_numpy_1d(robot.get_pos())
        cam.set_pose(
            pos=(root_pos[0] + 2.0, root_pos[1] + 2.0, 1.0),
            lookat=(root_pos[0], root_pos[1], 0.5),
            up=(0.0, 0.0, 1.0),
        )
        cam.render()

        if step % 50 == 0:
            print(
                f"[step {step}] phase={phase:.3f}, action_norm={np.linalg.norm(action):.3f}, "
                f"root_z={root_pos[2]:.3f}"
            )

    cam.stop_recording(save_to_filename=OUTPUT_VIDEO, fps=25)
    print(f"[DONE] saved video: {os.path.abspath(OUTPUT_VIDEO)}")


if __name__ == "__main__":
    main()

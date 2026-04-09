import os

import genesis as gs
import numpy as np
import onnxruntime as ort
import torch

import sys
sys.path.append("/root/autodl-tmp/ASAP") # 强行把项目根目录塞给 Python
from humanoidverse.utils.torch_utils import quat_rotate_inverse
from collections import deque

URDF_PATH = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf"
ONNX_PATH = "/root/autodl-tmp/ASAP/logs/TEST/20260406_220219-TEST_Locomotion_resume_from_2300/exported/model_3700.onnx"

# From humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml
ACTION_SCALE = 0.25
ACTION_CLIP_VALUE = 100.0
CONTROL_DECIMATION = 41
SIM_FPS = 200
SIM_DT = 1.0 / SIM_FPS
SIM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# actor_obs fields from obs config; NOTE: env code sorts keys before concatenation
ACTOR_OBS_FIELDS = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "command_lin_vel",
    "command_ang_vel",
    "dof_pos",
    "dof_vel",
    "actions",
]
OBS_ORDER = sorted(ACTOR_OBS_FIELDS)

OBS_SCALES = {
    "base_lin_vel": 2.0,
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "command_lin_vel": 1.0,
    "command_ang_vel": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "actions": 1.0,
}

# Same order as robot.dof_names in config
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

# Same as robot.body_names in config, used by training-side Genesis loader
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
        8.0,
        0.6,
        0.1,
        2.5,
        2.5,
        2.5,
        8.0,
        0.6,
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
KD = KD * 1.5

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


def as_tensor_2d(x) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=SIM_DEVICE)
    else:
        x = x.to(device=SIM_DEVICE, dtype=torch.float32)
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def to_numpy_1d(x) -> np.ndarray:
    x = as_tensor_2d(x)
    return x[0].detach().cpu().numpy().astype(np.float32)


def build_actor_obs(
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    base_lin_vel_body: np.ndarray,
    base_ang_vel_body: np.ndarray,
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    last_actions: np.ndarray,
) -> np.ndarray:
    parts = {
        "base_lin_vel": base_lin_vel_body,
        "base_ang_vel": base_ang_vel_body,
        "projected_gravity": projected_gravity,
        "command_lin_vel": commands[:2],
        "command_ang_vel": commands[2:3],
        "dof_pos": dof_pos - DEFAULT_DOF_POS,
        "dof_vel": dof_vel,
        "actions": last_actions,
    }

    obs_chunks = []
    for key in OBS_ORDER:
        obs_chunks.append(parts[key] * OBS_SCALES[key])

    obs = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    if obs.shape[0] != 81:
        raise RuntimeError(f"actor_obs dim mismatch: got {obs.shape[0]}, expected 81")
    return obs


def main() -> None:
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not os.path.isfile(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    print("[INFO] init Genesis")
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=10),
        rigid_options=gs.options.RigidOptions(enable_self_collision=True),
        show_viewer=False,
        show_FPS=False,
    )

    # ========================================================
    # 🏔️ 复杂地形：适配 Genesis 0.4.5 API
    # 1. 生成 201x201 的随机噪声图
    #h_field = 0.02 * np.random.randn(201, 201).astype(np.float32)

    #plane = scene.add_entity(
    #    gs.morphs.Terrain(
    #        height_field=h_field,
    #        horizontal_scale=0.1,  # 采样间隔 0.1米，201个点正好覆盖 20m x 20m
    #        vertical_scale=1.0,    # 垂直比例设为 1.0，表示 h_field 里的数值单位直接就是“米”
    #        pos=(-10, -10, 0),         # 地形中心点位置
    #   ),
    #    material=gs.materials.Rigid(friction=0.7) 
    #)

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.7) 
    )
    # ========================================================

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

    # Resolve dof ids dynamically from joint names to avoid hard-coded index drift.
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in RL_JOINT_NAMES]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX input={session.get_inputs()[0].shape}, output={session.get_outputs()[0].shape}")

    # Initialize robot close to training init pose.
    robot.set_dofs_position(
        position=torch.tensor(DEFAULT_DOF_POS, dtype=torch.float32, device=SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )
    robot.set_dofs_velocity(
        velocity=torch.zeros(23, dtype=torch.float32, device=SIM_DEVICE),
        dofs_idx_local=motor_dofs,
    )

    cmd_x = float(os.environ.get("CMD_X", "0.5"))
    cmd_y = float(os.environ.get("CMD_Y", "0.0"))
    cmd_yaw = float(os.environ.get("CMD_YAW", "0.0"))
    commands = np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)

    last_actions = np.zeros(23, dtype=np.float32)
    gravity_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=SIM_DEVICE)

    # ========================================================
    # 🕵️‍♂️ 状态估计器初始化 (State Estimator)
    # ========================================================
    # 创建一个容量为 5 的历史队列 (相当于延迟/平滑过去 100ms 的数据)
    # 容量越大，速度越平滑，但反应越迟钝！(这就是真机调试中最经典的 Trade-off)
    vel_history = deque(maxlen=15) 
    last_root_pos = to_numpy_1d(robot.get_pos())
    # 每次网络推理的真实时间跨度 dt
    control_dt = SIM_DT * CONTROL_DECIMATION 
    # ========================================================

    cam.start_recording()
    print("[INFO] start control loop")

    for step in range(500):
 

        # Current state in policy joint order.
        dof_pos = to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
        dof_vel = to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
        base_quat_wxyz = as_tensor_2d(robot.get_quat())
        base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]

        # ========================================================
        # 🧠 真实生存模式：用微分与滤波估算速度
        # ========================================================
        current_root_pos = to_numpy_1d(robot.get_pos())

        # 1. 微分算速度 (模拟真实 IMU 积分与里程计)
        inst_vel = (current_root_pos - last_root_pos) / control_dt

        # 2. 塞入历史记忆库
        vel_history.append(inst_vel)

        # 3. 滑动平均滤波 (过滤掉每一步踏地带来的震动高频噪声)
        estimated_vel_world = np.mean(vel_history, axis=0)

        # 机器人绝对跑不到 5m/s，超过这个数值一定是位移微分产生的数值爆炸
        estimated_vel_world = np.clip(estimated_vel_world, -5.0, 5.0)

        # 4. 把估算出来的速度，伪装成张量喂给神经网络！(彻底删除了 robot.get_vel())
        base_lin_vel_world = as_tensor_2d(estimated_vel_world)

        # 更新记忆
        last_root_pos = current_root_pos.copy()
        # ========================================================

        # 真机上的陀螺仪测角速度通常极其精准，暂时无需剥离
        base_ang_vel_world = as_tensor_2d(robot.get_ang())

        base_lin_vel_body = to_numpy_1d(quat_rotate_inverse(base_quat_xyzw, base_lin_vel_world))
        base_ang_vel_body = to_numpy_1d(quat_rotate_inverse(base_quat_xyzw, base_ang_vel_world))
        projected_gravity = to_numpy_1d(quat_rotate_inverse(base_quat_xyzw, gravity_world))

        # ========================================================
        # 🦿 恶劣环境测试：飞来横祸（传感器欺骗）
        # ========================================================
        # 在第 200 帧的时候，突然给它的倾角传感器（投影重力）加上极大的噪声
        #if step > 200 and step < 205:
            # 告诉网络：你的身体突然向右严重倾斜了！
            #projected_gravity[1] += 0.8  
            # 告诉网络：你被一股神秘力量猛推了一下，产生了极大的横向速度！
            #base_lin_vel_body[1] += 4.7
           # print(f"⚠️ [系统警告] 第 {step} 帧，机器人遭到猛烈侧向撞击！")

        obs = build_actor_obs(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            base_lin_vel_body=base_lin_vel_body,
            base_ang_vel_body=base_ang_vel_body,
            projected_gravity=projected_gravity,
            commands=commands,
            last_actions=last_actions,
        )

        action = session.run([output_name], {input_name: obs[None, :]})[0][0].astype(np.float32)
        action = np.clip(action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE)
        last_actions = action.copy()

        target_dof_pos = action * ACTION_SCALE + DEFAULT_DOF_POS

        # Match training-style PD torque control + torque clipping.
        for _ in range(CONTROL_DECIMATION):
            dof_pos_sub = to_numpy_1d(robot.get_dofs_position(dofs_idx_local=motor_dofs))
            dof_vel_sub = to_numpy_1d(robot.get_dofs_velocity(dofs_idx_local=motor_dofs))
            torques = KP * (target_dof_pos - dof_pos_sub) - KD * dof_vel_sub
            torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
            robot.control_dofs_force(
                torch.tensor(torques, dtype=torch.float32, device=SIM_DEVICE),
                dofs_idx_local=motor_dofs,
            )
            scene.step()
        # ========================================================
        # 🎬 动态运镜：第三人称跟随视角
        # ========================================================
        # 1. 实时获取机器人躯干的世界坐标 (x, y, z)
        root_pos = to_numpy_1d(robot.get_pos())
        
         # 2. 机位：X和Y跟着走，但高度 (Z) 永远锁死在 1.0 米！
        cam_pos = (root_pos[0], root_pos[1] + 2.0, 1.0)
        
        # 3. 焦点：看向上半身，焦点高度也锁死在 0.8 米！
        cam_lookat = (root_pos[0], root_pos[1], 0.8)
        
        # 4. 更新摄像机姿态并渲染
        cam.set_pose(pos=cam_pos, lookat=cam_lookat,up=(0.0, 0.0, 1.0))
        cam.render()
        # ========================================================

        if step % 50 == 0:
            print(
                f"[step {step}] action_norm={np.linalg.norm(action):.3f}, "
                f"base_lin={base_lin_vel_body}, proj_g={projected_gravity}"
            )

    out = "g1_walking_onnx.mp4"
    cam.stop_recording(save_to_filename=out, fps=25)
    print(f"[DONE] saved video: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

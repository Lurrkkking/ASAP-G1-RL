# 技术复盘（0408）

> 说明：本复盘仅覆盖你要求的“之前做过的工作”，不包含最后一轮 reference-init（按 `TIME_OFFSET` 从 motion state 初始化）的新增实现。

## 1. 背景与症状

- 目标模型：
  - `logs/TEST_CR7_Siuuu/20260408_094756-MotionTracking_CR7_FullSystem_V1-motion_tracking-g1_29dof_anneal_23dof/model_5000.pt`
- 现象：
  - 在 IsaacGym 评估视频中动作接近正确。
  - 在 Genesis ONNX 回放中出现明显异常（扭曲、动作不完整、疑似足端黏地等）。

## 2. 代码与配置排查链路

### 2.1 Reset 与 RSI 机制核查

- 重点文件：
  - `humanoidverse/envs/legged_base_task/legged_robot_base.py`
  - `humanoidverse/envs/motion_tracking/motion_tracking.py`
- 结论：
  - `legged_base` 默认 reset 不是 motionlib 状态采样，主要是默认姿态 + 随机扰动。
  - `motion_tracking` reset 路径确实使用 motionlib：
    - 重采样 motion start time：`_motion_lib.sample_time(...)`
    - 重置 `dof_pos/dof_vel` 与 `root states` 使用 `_motion_lib.get_motion_state(...)`
  - 因此 motion tracking 任务具备 RSI 风格的状态切入能力。

### 2.2 Termination 阈值是否“死数”核查

- 文件：
  - `humanoidverse/envs/motion_tracking/motion_tracking.py`
  - `humanoidverse/config/env/motion_tracking.yaml`
- 结论：
  - 代码中对应变量不是 `tracking_error_threshold`，而是 `terminate_when_motion_far_threshold`。
  - 该阈值可静态也可课程化动态更新：
    - 若 `terminate_when_motion_far_curriculum=false`：等效静态阈值。
    - 若开启 curriculum：依据 `average_episode_length` 做增减与 clip（不是按 `current_epoch`）。

### 2.3 PT->ONNX 与 IsaacGym 视频链路确认

- 导出入口：`humanoidverse/export_pt_to_onnx.py`
- 核心参数：`+checkpoint=/abs/path/model_5000.pt`
- 输出路径默认：`<checkpoint_dir>/exported/model_5000.onnx`
- IsaacGym 视频导出入口：`humanoidverse/eval_agent.py`

### 2.4 Hydra 报错根因定位

- 报错：`LexerNoViableAltException` 出现在 `checkpoint=...` 覆盖参数处。
- 根因：长路径在 shell/粘贴时发生换行，Hydra 解析失败。
- 处理：
  - 提供单行命令写法。
  - 提供软链接短路径规避方案（例如 `/tmp/model_5000.pt`）。

## 3. Genesis 与 IsaacGym 差异专项排查

### 3.1 23维 DOF 映射与顺序核查

- 新增诊断脚本：
  - `genesis_simulation/debug_dof_pd_alignment.py`
- 做了什么：
  - 打印训练 config 中 23 维关节顺序、PD 参数。
  - 在 IsaacGym 与 Genesis 侧分别按关节名解析索引并打印映射。
- 发现：
  - IsaacGym 侧在该配置下 `gym_idx` 与 RL 顺序对齐（0..22）。
  - Genesis 侧是“非连续本地索引”（例如 6/9/12/...），但按关节名解析后顺序可对齐到 RL joint list。
  - 结论：更像控制实现差异，不是简单的 23 维名称顺序错位。

### 3.2 PD / KD 侧不一致核查

- 训练配置中：
  - `domain_rand.randomize_pd_gain = false`
  - 即训练/评估环境下 `_kp_scale=1.0, _kd_scale=1.0`
- Genesis ONNX 脚本中：
  - `KD_SCALE` 默认曾为 `1.5`，会将 `KD` 统一放大 50%。
- 结论：
  - 这是一个强可疑项，可能导致动作发僵、过阻尼、形态异常。

### 3.3 Genesis 调用约定风险核查

- 现象：Genesis 日志反复提示参数形式弃用（建议使用 `dofs_idx_local`）。
- 处理：
  - 已将 ONNX runner 中所有带 DOF 索引的 `set/get/control` 调用统一改为关键字参数 `dofs_idx_local=...`。
- 目的：
  - 避免位置参数在不同版本 API 中被解释错误，降低“扭麻花”类风险。

### 3.4 TIME_OFFSET 生效性核查

- 发现：
  - 原 `run_onnx_motiontracking.py` 中 `TIME_OFFSET` 未实际参与 phase 计算（传了但不生效）。
- 处理：
  - 接入 `TIME_OFFSET` 到时间相位计算，保证 phase 可控。

### 3.5 接触参数可调性

- 处理：
  - 将地面摩擦暴露为环境变量（`FLOOR_FRICTION`），便于快速 A/B 排查“足端粘滞/打滑”问题。

## 4. 产出物清单（不含最后一轮 reference-init）

- 新增：
  - `genesis_simulation/debug_dof_pd_alignment.py`
- 修改：
  - `genesis_simulation/run_onnx_motiontracking.py`
  - `genesis_simulation/run_onnx_locomotion.py`
- 核心改动类型：
  - DOF API 调用统一关键字化 (`dofs_idx_local=`)
  - `TIME_OFFSET` 真正接入 phase
  - `FLOOR_FRICTION` 参数化

## 5. 截止该阶段的结论

1. PT 与 ONNX 导出链路正常，IsaacGym 评估链路基本可信。  
2. Genesis 异常主要嫌疑不在“简单 DOF 名称顺序错位”，而在“运行时控制/物理实现偏差”。  
3. 已定位并修正多处高风险不一致项：
   - `KD_SCALE` 过大风险
   - `TIME_OFFSET` 未生效
   - DOF 位置参数调用可能引发的 API 歧义

## 6. 下一步建议（当时阶段）

1. 固定 `KD_SCALE=1.0` 做基线。  
2. 使用关键字 `dofs_idx_local` 后重新验证视频。  
3. 对 `FLOOR_FRICTION` 做小范围网格测试（例如 `0.5/0.7/1.0`）。  
4. 若仍异常，再继续做“训练 reset 等价初始化”与“观测构造逐项对齐”排查。


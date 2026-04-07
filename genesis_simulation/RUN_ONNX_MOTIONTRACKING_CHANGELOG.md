# run_onnx_motiontracking 详细复盘（排查过程 + 修改思路 + 证据）

## 0. 先回答“是不是很简单就改好了？”
不是。

这次不是单点 bug，而是 4 层问题叠加：
1. 脚本用错任务（loco 81 维输入）
2. 改成 380 维后，观测“维度对但语义错”（排序/历史时序错位）
3. 语义修正后，进入 sim2sim 域差问题（IsaacGym -> Genesis）
4. 动作爆发段（起跳）对动力学偏差敏感，导致“手有动作、腿起跳失败”

也就是说：脚本层关键错误已经修到位，但“稳定起跳”还受策略与动力学域差限制。

---

## 1. 需求与现象

### 1.1 目标
让 `genesis_simulation/run_onnx_motiontracking.py` 用 motion-tracking ONNX 在 Genesis 正常回放动作，并尽量接近 IsaacGym 评估表现。

### 1.2 初始现象
- 报错：`Got invalid dimensions ... actor_obs index 1 Got:81 Expected:380`
- 修成 380 后：能动、手部动作明显，但很快摔倒。

用户给出的典型日志（旧）：
- `step 50: phase=0.204, root_z=0.773`
- `step 100: phase=0.404, root_z=0.156`
- `step 150: phase=0.604, root_z=0.073`

含义：机器人在进入中段相位后迅速失稳贴地。

---

## 2. 排查时间线（按问题层次）

## 2.1 第一层：模型输入维度是否正确

### 检查点
- `run_onnx.py` 是 locomotion（81 维）
- motion-tracking ONNX 需要 380 维

### 结论
必须单独维护 `run_onnx_motiontracking.py`，不能共用 loco 的观测构造。

---

## 2.2 第二层：380 维是否“语义一致”

这一步是关键转折。不是只要凑够 380 就行。

### 训练侧真实规则（代码证据）
1. `actor_obs` 拼接顺序按 `sorted(obs_config)`，不是 yaml 中写的顺序。
2. `history_actor` 是“按 key 分开存历史”，不是整帧堆叠。
3. history 写入时序是：
   - 先用旧 history 计算本步 obs
   - 本步结束后再把当前特征写进 history
4. history 的索引 0 是“最新帧”。

### 原脚本问题
- 按“单帧固定顺序 + 整帧堆叠”拼 380
- 先 append 当前帧再推理

这会导致：
- 虽然 shape 对，但 feature 语义错位（网络读到错误字段、错误时序）

---

## 2.3 第三层：物理初始化与结构偏差

对齐项：
- 初始高度改回训练配置附近（`z=0.8`）
- 引入 `merge_fixed_links=True`
- 引入 `links_to_keep=BODY_NAMES`（与已有 Genesis loco 稳定脚本一致思路）

目的：减小 URDF 构型和接触链条差异。

---

## 2.4 第四层：相位周期是否对应真实动作

检查 motion 文件：
- 文件：`...CR7_level2_filter_amass.pkl`
- 统计：`frames=134, fps=30, duration=4.4667s`
- `root_z` 振幅约 `0.35m`，峰值在 `2.63s`

因此将 `MOTION_DURATION` 默认改为 `4.4666666667`，而非经验值 5.0。

---

## 3. 具体改动（脚本层）

目标文件：
- `genesis_simulation/run_onnx_motiontracking.py`

## 3.1 观测构造
- 新增 `ACTOR_OBS_ORDER = sorted([...])`
- 按排序后的 key 拼接 actor_obs
- 增加 380 维断言，防止 silent mismatch

## 3.2 history 机制
- 新增 `history_buffers`（按 key 维护）
- 新增 `query_history_actor()`：按训练 key 顺序拼接
- 新增 `update_history_buffers()`：最新写 index 0
- 推理流程改为：
  1. 用旧 history 构造 obs
  2. ONNX 推理
  3. 更新 history

## 3.3 动力学/初始化
- `z=0.8`
- `merge_fixed_links=True`
- `links_to_keep=BODY_NAMES`
- PD 与 torque limits 对齐训练配置

## 3.4 可调参数（不改代码可扫参）
支持环境变量：
- `ONNX_PATH`
- `OUT_VIDEO`
- `NUM_STEPS`
- `MOTION_DURATION`
- `KD_SCALE`
- `ACTION_SCALE`
- `ACTION_CLIP_VALUE`

---

## 4. 实验矩阵与结果（关键）

> 评估口径：观察 `root_z` 在 step 50/100/150 的变化，判断是否在中段相位崩掉。

| 方案 | ONNX | 关键参数 | step50 root_z | step100 root_z | step150 root_z | 结论 |
|---|---|---|---:|---:|---:|---|
| 用户原始现象 | model_2000 | 未对齐版本 | 0.773 | 0.156 | 0.073 | 很早崩 |
| 修语义后短测 | model_2000 | `MOTION_DURATION=4.4667, KD=1.5` | 0.774 | 0.175 | 0.079 | 稍好但仍早崩 |
| 更高 checkpoint | model_2900 | `KD=1.5` | 0.780 | 0.649 | 0.175 | 明显改善 |
| 增阻尼 | model_2900 | `KD=2.5` | 0.783 | 0.742 | 0.132 | 中段更稳 |
| 增阻尼+降幅 | model_2900 | `KD=2.5, ACTION_SCALE=0.22` | 0.780 | 0.420 | 0.089 | 爆发不足，后段仍崩 |

结论：
- checkpoint 提升（2000 -> 2900）确实有帮助。
- 增阻尼可延后崩溃，但仍无法保证完整起跳/落地稳定。
- 降 `ACTION_SCALE` 不一定更好，可能抑制起跳动量。

---

## 5. “手有动作但没跳起来”的原因分解

这不是单一原因，而是组合效应：

1. 策略优先学会上肢模仿
- 训练中上肢/VR 跟踪权重较高（如 `teleop_vr_3point=3.6`）

2. 起跳需要下肢爆发 + 接触稳定
- 这部分对仿真器接触模型、惯量、阻尼、摩擦非常敏感

3. 训练中存在强平滑/限制项
- `penalty_action_rate`, `limits_torque`, `soft_torque_limit` 等会抑制爆发边界动作

4. sim2sim 域差在“爆发段”被放大
- 同策略在 IsaacGym 可行，不代表在 Genesis 的同相位同样稳定

所以看到的现象是：
- 上肢动作先对齐（视觉上“像了”）
- 下肢起跳链条失稳（`root_z` 进中段后掉落）

---

## 6. 已解决 vs 未解决

## 6.1 已解决（脚本工程问题）
- 输入维度错误（81 vs 380）
- 观测拼接顺序错误
- history 时序和布局错误
- phase 默认时长与 reference 不一致
- 部分动力学结构差异（links_to_keep）

## 6.2 未彻底解决（策略层/域差层）
- 仍可能在起跳相位后段失稳
- 落地与躯干恢复能力不足

这部分不是再改一行推理脚本就能根治，核心在策略鲁棒性与跨引擎动力学对齐。

---

## 7. 产物与命令

## 7.1 关键产物
- 修改后的脚本：`genesis_simulation/run_onnx_motiontracking.py`
- 新导出的 ONNX：
  - `.../exported/model_2900.onnx`

## 7.2 推荐运行命令（当前较稳）
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/env_genesis
cd /root/autodl-tmp/ASAP

ONNX_PATH=/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260407_155156-MotionTracking_CR7_V2_Fixed-motion_tracking-g1_29dof_anneal_23dof/exported/model_2900.onnx \
MOTION_DURATION=4.4666666667 \
KD_SCALE=2.5 \
ACTION_SCALE=0.25 \
NUM_STEPS=600 \
OUT_VIDEO=/root/autodl-tmp/ASAP/genesis_simulation/g1_siuuu_genesis_2900.mp4 \
python genesis_simulation/run_onnx_motiontracking.py
```

---

## 8. 下一步建议（先不改训练代码版本）

若只做评估侧动作：
1. 固定 `model_2900.onnx`，小范围网格扫参（`KD_SCALE ∈ [2.0, 3.0]`, `ACTION_SCALE ∈ [0.23, 0.27]`）
2. 统计每组“首次 root_z < 0.2 的步数”作为稳定性指标
3. 选最佳组再看视频细节（是否有“假稳定”拖地行为）

若要真正“跳起来并稳落地”：
- 需要回到训练侧调整 reward/课程与 sim2sim 对齐，不是仅靠推理脚本就能保证。


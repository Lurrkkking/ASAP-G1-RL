# ONNX 在 Genesis 生成仿真视频

## 目标
使用训练得到的 `.pt` 模型先导出 `.onnx`，再通过 `genesis_simulation/run_onnx.py` 在 Genesis 中运行策略并录制视频。

## 前置条件
- 已有可用 checkpoint（例如 `model_3700.pt`）
- 已安装并可用：`genesis`、`onnxruntime`、`torch`
- 建议在项目根目录执行命令：`/root/autodl-tmp/ASAP`

## 1. 从 `.pt` 导出 `.onnx`（IsaacGym）
> 该项目通过 `eval_agent.py` 自动导出 ONNX。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl
cd /root/autodl-tmp/ASAP

python humanoidverse/eval_agent.py \
'+simulator=isaacgym' \
'+exp=locomotion' \
'+robot=g1/g1_29dof_anneal_23dof' \
'+terrain=terrain_locomotion_plane' \
'+domain_rand=NO_domain_rand' \
'+rewards=loco/reward_g1_locomotion' \
'+obs=loco/leggedloco_obs_singlestep_withlinvel' \
'headless=True' \
'checkpoint=/root/autodl-tmp/ASAP/logs/TEST/20260406_220219-TEST_Locomotion_resume_from_2300/model_3700.pt'
```

导出结果通常在：

```text
/root/autodl-tmp/ASAP/logs/TEST/20260406_220219-TEST_Locomotion_resume_from_2300/exported/model_3700.onnx
```

## 2. 检查/修改 Genesis 脚本
文件：`/root/autodl-tmp/ASAP/genesis_simulation/run_onnx.py`

需要确认这两个常量：
- `URDF_PATH`
- `ONNX_PATH`

可选控制命令（脚本支持环境变量）：
- `CMD_X`：前进速度命令（默认 `0.5`）
- `CMD_Y`：侧向速度命令（默认 `0.0`）
- `CMD_YAW`：偏航角速度命令（默认 `0.0`）

## 3. 在 Genesis 运行并录视频
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genesis
cd /root/autodl-tmp/ASAP

# 例：先低速验证稳定性
CMD_X=0.2 CMD_Y=0.0 CMD_YAW=0.0 python genesis_simulation/run_onnx.py
```

## 4. 视频输出位置
脚本当前输出文件名为：

```text
g1_walking_onnx.mp4
```

输出到你执行命令时的当前目录（建议在 `/root/autodl-tmp/ASAP` 下运行，便于管理）。

## 5. 常见问题
### 5.1 `ONNX not found`
- 检查 `ONNX_PATH` 是否与导出路径一致。

### 5.2 `ModuleNotFoundError: humanoidverse`
- 确保在项目根目录运行：`cd /root/autodl-tmp/ASAP`。

### 5.3 机器人扭曲/站不稳
- 先把命令降到 `CMD_X=0.0` 验证站立，再逐步提高到 `0.2/0.3/0.5`。
- Genesis 与 IsaacGym 接触模型不同，动作表现有差异是正常现象。

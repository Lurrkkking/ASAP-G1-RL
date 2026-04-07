# Eval/训练问题排查总结（2026-04-05）

## 1. 背景
本次主要排查了三个方向：
- `eval_agent` 导出视频黑屏/原地踏步问题
- 从 checkpoint 续训时迭代计数未继承问题
- 评估命令与相机/分辨率配置问题

---

## 2. 关键发现

### 2.1 黑屏不是持续性问题，历史文件里有“好/坏混合”
- 同一实验目录下存在黑屏视频（例如早期 12s 文件）和正常视频（后续 12s 文件）。
- 通过逐帧亮度统计确认：黑屏视频均值接近 `0`，正常视频均值约 `63`。

### 2.2 `offscreen_record` 需要图形上下文
- 直接 headless 跑时，存在不产视频或行为不稳定的情况。
- 使用 `xvfb-run` 后，12s/600 帧视频可稳定写出。

### 2.3 Eval 时默认 command 会被清零，导致“原地踏步/轻微转圈”
- `PPO.evaluate_policy()` 默认调用 `self.env.set_is_evaluating()`。
- locomotion 环境的 `set_is_evaluating()` 会把命令置零（未显式传 command）。
- 结果：常见表现是原地踏步；再叠加 heading/yaw 逻辑会出现轻微转圈。

### 2.4 `eval_agent` 会优先加载 checkpoint 同目录 `config.yaml`
- 这意味着你改了工作区 YAML，不一定会影响某个旧 checkpoint 的 eval。
- 如果该 checkpoint 目录里的 `config.yaml` 是旧配置，eval 表现会跟旧配置走。

### 2.5 Hydra 严格模式：新增 key 必须用 `+`
- 例如：
  - `+env.config.eval_command=[1.0,0.0,0.0]`
  - `+env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]`
- 不加 `+` 会报 `Key ... is not in struct`。

### 2.6 续训计数不继承的根因
- 老版本中间 checkpoint 的 `iter` 被写成 `0`（即便文件名是 `model_9500.pt`）。
- 所以 resume 时显示从 0 开始。

### 2.7 分辨率变低原因
- 是命令里显式传了：
  - `offscreen_record_width=640`
  - `offscreen_record_height=360`
- 删掉这两个参数后，会恢复默认分辨率配置。

### 2.8 相机角度当前为硬编码
- IsaacGym offscreen 相机位置在代码中固定设置，不是现成 YAML 参数。
- 左转 45°需要改 `cam_pos/cam_target` 的代码值。

---

## 3. 本次已完成的修复

### 3.1 修复续训迭代号继承（`ppo.py`）
- `save()` 支持显式 `iter_num`，中间 checkpoint 按当前迭代写入。
- `load()` 对老 checkpoint 做兼容：当 `iter=0` 时，尝试从文件名 `model_xxx.pt` 恢复迭代号。

### 3.2 增强 eval：支持固定评估命令（`eval_command`）
- 在 `PPO._pre_evaluate_policy()` 增加读取 `env.config.eval_command` 的逻辑。
- 这样可直接导出固定直走/绕弯视频（例如 `[1.0, 0.0, 0.4]`）。

---

## 4. 推荐命令模板

### 4.1 固定直走导出（无分辨率参数）
```bash
LOGURU_LEVEL=INFO xvfb-run -a -s "-screen 0 1280x720x24" \
python humanoidverse/eval_agent.py \
+checkpoint="$CKPT" \
+simulator=isaacgym \
+exp=locomotion \
auto_record=true \
auto_record_num_frames=600 \
offscreen_record=true \
offscreen_record_fps=30 \
disable_keyboard_listener=true \
'+env.config.eval_command=[1.0,0.0,0.0]' \
'+env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
'+env.config.locomotion_command_ranges.heading=[0.0,0.0]'
```

### 4.2 固定左转前进导出
把 `eval_command` 改为：
```bash
'+env.config.eval_command=[1.0,0.0,0.4]'
```
右转则用 `-0.4`。

---

## 5. 相机左转 45°做法
在 `humanoidverse/simulator/isaacgym/isaacgym.py` 的 offscreen 相机设置处，把：
```python
cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, 3.0)
```
改为：
```python
cam_pos = gymapi.Vec3(0.0, 7.07, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, 3.0)
```
即可得到相对原视角左转约 45° 的画面。

---

## 6. 后续建议
- 若要“稳定直走+可控绕弯”，训练配置建议保留一定的 `ang_vel_yaw` 分布，不要全程固定为 0。
- 若长期需要多机位，建议把 offscreen 相机参数做成 Hydra 可覆盖字段（`offscreen_cam_pos/offscreen_cam_target`），避免每次改代码。

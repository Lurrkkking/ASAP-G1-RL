# Eval 视频导出命令（基于 `model_2100.pt`）

## 1) 直走视频（已删除分辨率参数）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl
cd /root/autodl-tmp/ASAP

CKPT="/root/autodl-tmp/ASAP/logs/TEST/20260405_161507-TEST_Locomotion_Fresh_Start-locomotion-g1_29dof_anneal_23dof/model_2100.pt"

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
'env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
'env.config.locomotion_command_ranges.heading=[0.0,0.0]'
```

## 2) 左转弯前进视频

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl
cd /root/autodl-tmp/ASAP

CKPT="/root/autodl-tmp/ASAP/logs/TEST/20260405_161507-TEST_Locomotion_Fresh_Start-locomotion-g1_29dof_anneal_23dof/model_2100.pt"

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
'+env.config.eval_command=[1.0,0.0,0.4]' \
'env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
'env.config.locomotion_command_ranges.heading=[0.0,0.0]'
```

## 3) 右转弯前进视频

把上面命令中的 `+env.config.eval_command=[1.0,0.0,0.4]` 改成：

```bash
'+env.config.eval_command=[1.0,0.0,-0.4]'
```

## 4) 摄像头角度能不能更换？

可以更换，但你当前 offscreen 录制相机是代码里写死的，不是 YAML 参数。

位置在：

- `humanoidverse/simulator/isaacgym/isaacgym.py` 的 `_setup_offscreen_recording()`
- 当前值：
  - `cam_pos = (5.0, 5.0, 3.0)`
  - `cam_target = (0.0, 0.0, 3.0)`

如果你想改镜头，可以直接改这两行后再跑 eval。

## 5) 视频输出目录

会输出到：

```text
/root/autodl-tmp/ASAP/logs/TEST/20260405_161507-TEST_Locomotion_Fresh_Start-locomotion-g1_29dof_anneal_23dof/renderings/ckpt_2100/
```

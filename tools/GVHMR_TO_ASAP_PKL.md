# 用法：

  cd /root/autodl-tmp/GVHMR
  ./run_video_to_asap_pkl.sh /root/autodl-tmp/GVHMR/myvideo/own_kickball_eg.mp4 50

  如果不手动传 FPS，它会自己从视频里读：

  ./run_video_to_asap_pkl.sh /root/autodl-tmp/GVHMR/myvideo/own_kickball_eg.mp4


# GVHMR 视频转 ASAP 可训练 PKL 备忘

这份备忘记录的是这次实际跑通的路径：

`视频 -> GVHMR hmr4d_results.pt -> GMR robot pkl -> ASAP motion pkl`

适用文件：

- 输入视频：`/root/autodl-tmp/GVHMR/myvideo/own_kickball_eg.mp4`
- GVHMR 输出：`/root/autodl-tmp/GVHMR/outputs/demo/own_kickball_eg/hmr4d_results.pt`
- GMR 输出：`/root/autodl-tmp/GMR/unitree_g1_gmr/own_kickball_eg.pkl`
- ASAP 训练动作：`/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl`

## 1. 跑 GVHMR

在 `GVHMR` 环境里执行：

```bash
cd /root/autodl-tmp/GVHMR
conda run -n GVHMR python tools/demo/demo.py --video /root/autodl-tmp/GVHMR/myvideo/own_kickball_eg.mp4 -s
```

说明：

- `-s` 表示静态相机，跳过 VO。
- 这一步会产出 `outputs/demo/<video_stem>/hmr4d_results.pt`。

## 2. 重要坑：保留原视频 FPS

这套流程里最容易忘的是：

- 原视频 `own_kickball_eg.mp4` 是 `50 FPS`
- GVHMR 原版脚本默认会把输入和渲染视频硬编码成 `30 FPS`
- 如果不修，`0_input_video.mp4`、`1_incam.mp4`、后续 GMR / ASAP motion 的时间轴都会变慢

这次已经修过的文件：

- `GVHMR/tools/demo/demo.py`
- `GVHMR/hmr4d/utils/video_io_utils.py`
- `GVHMR/hmr4d/configs/demo.yaml`
- `GMR/general_motion_retargeting/utils/smpl.py`
- `GMR/scripts/gvhmr_to_robot.py`

现在逻辑是：

- GVHMR 会保留源视频 FPS
- GMR 可以显式传 `--src_fps`

## 3. 用 GMR 把 GVHMR 结果转成 robot pkl

```bash
cd /root/autodl-tmp/GMR
PYTHONPATH=/root/autodl-tmp/GMR:/root/autodl-tmp/GMR/third_party \
xvfb-run -a conda run -n gmr python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file /root/autodl-tmp/GVHMR/outputs/demo/own_kickball_eg/hmr4d_results.pt \
  --src_fps 50 \
  --robot unitree_g1 \
  --save_as_pkl True
```

说明：

- 这里 `--src_fps 50` 很关键，必须和原视频一致。
- 输出会自动写到：
  ` /root/autodl-tmp/GMR/unitree_g1_gmr/own_kickball_eg.pkl`

## 4. 把 GMR pkl 转成 ASAP 训练用 motion file

```bash
cd /root/autodl-tmp/ASAP
/root/miniconda3/envs/rl/bin/python tools/convert_gmr_pkl_to_asap_motion.py \
  --input /root/autodl-tmp/GMR/unitree_g1_gmr/own_kickball_eg.pkl \
  --output /root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl \
  --motion-name 0-own_kickball_eg_gvhmr
```

成功后应看到类似输出：

```text
frames: 66 fps: 50
pose_aa shape: (66, 27, 3)
dof shape: (66, 23)
```

## 5. 用工具脚本播放检查

```bash
cd /root/autodl-tmp/ASAP
/root/miniconda3/envs/rl/bin/python tools/play_motion_pkl_isaacgym.py \
  --motion-file /root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl \
  --record-fps 50 \
  --out /root/autodl-tmp/ASAP/logs_eval/0-own_kickball_eg_gvhmr_playback.mp4 \
  --device cuda:0
```

播放输出：

- `/root/autodl-tmp/ASAP/logs_eval/0-own_kickball_eg_gvhmr_playback.mp4`

## 6. 用 run_kick_primitive.sh 训练

当前脚本：

- `/root/autodl-tmp/ASAP/run_kick_primitive.sh`

已经指向：

- `/root/autodl-tmp/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-own_kickball_eg_gvhmr.pkl`

直接运行：

```bash
cd /root/autodl-tmp/ASAP
bash run_kick_primitive.sh
```

## 7. 最小检查清单

下次如果结果不对，优先检查这 4 件事：

1. 原视频 FPS 是多少。
2. `outputs/demo/.../0_input_video.mp4` 的 FPS 是否和原视频一致。
3. `gvhmr_to_robot.py` 是否传了正确的 `--src_fps`。
4. 最终 ASAP motion file 打印出来的 `fps` 是否正确。

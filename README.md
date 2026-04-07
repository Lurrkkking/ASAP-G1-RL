# ASAP-G1-Learning: G1 机器人高难度动作模仿学习记录

本项目是基于 [ASAP](https://github.com/LeCAR-Lab/ASAP) 框架的学习与复现项目。主要目标是让 **Unitree G1** 机器人学会模仿复杂的动作轨迹（如 C罗 Siuuu 起跳），并尝试打通从训练到部署的中间环节。

> **注**：原项目文档已重命名为 `README_ORIGIN.md`，方便查阅环境配置与官方指南。

## 📝 学习与改进进展 (Current Progress)

为了解决 G1 在模仿爆发性动作时出现的平衡与姿态问题，我进行了以下尝试：

### 1. 奖励函数微调 (Reward Tuning)
- **解决“大风车”甩臂**：通过增加 `penalty_action_rate`（动作速率惩罚）至 -1.5，强迫模型输出更平滑的指令，改善了起跳瞬间手臂乱挥的代偿行为。
- **上半身保真度优化**：拉高了 `teleop_vr_3point` 的权重，尝试在保持平衡的前提下更精准地还原参考轨迹。
- **落地稳定性**：针对 G1 落地容易崴脚的问题，引入了 `penalty_feet_ori` 约束，迫使脚掌与地面平行。

### 2. Sim2Sim 链路尝试 (ONNX & Genesis)
为了验证策略的通用性，我额外添加了以下脚本：
- `humanoidverse/export_pt_to_onnx.py`: 尝试将训练好的 `.pt` 权重导出为 **ONNX** 格式。
- `genesis_simulation/`: 尝试在 **Genesis** 物理引擎中加载导出的 ONNX 策略。目前已基本跑通推理流程，这比单纯在 Isaac Gym 里验证更接近真实物理表现。运行 `run_onnx.py`可直接导出`Genesis`仿真环境下的视频，补充了原仓库没有`sim2sim`的空缺，便于没有实机的人实验。

## 🙂 目前成果
1、在Isaacgym训练行走模型后，sim2sim到Genesis进行崎岖地面仿真，频繁摔倒，增加`feet_max_height_for_this_air`抬腿惩罚，解决了被崎岖地面绊倒的问题。


## ⚠️ 待解决的问题
- [ ] 起跳高度目前受限于电机力矩软限位，后续考虑微调课程学习（Curriculum）阈值。
- [ ] 落地后的震荡依然存在，需要进一步优化 PD 参数或动作平滑奖励。

---
## 致谢
感谢 [ASAP](https://github.com/LeCAR-Lab/ASAP) 团队开源了如此优秀的具身智能框架，本项目仅作为个人学习与技术探索使用。
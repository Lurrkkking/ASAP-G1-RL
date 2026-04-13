# ASAP-G1-Learning: G1 机器人动作模仿学习记录

本项目是基于 [ASAP](https://github.com/LeCAR-Lab/ASAP) 框架的进阶实践项目，用于个人学习和探索 Isaac Gym、Genesis 等具身智能仿真平台。主要目标是让 **Unitree G1** 机器人学会模仿复杂的爆发性动作轨迹（如 C罗 Siuuu 起跳），并重点打通从强化学习训练到多引擎物理验证（Sim-to-Sim）的部署闭环。

> **注**：原项目官方文档已重命名为 `README_ORIGIN.md`，方便查阅环境配置与基础指南。

---

## 🚀 核心成果展示 (Showcase)

### 1. C罗 Siuuu 动作模仿 (Dynamic Motion Tracking)
在 0.85 电机力矩软限位及域随机化（Domain Randomization）的严苛条件下，成功实现了完整动作链：**助跑 → 起跳转身 → 空中挥臂 → 落地**。

<details>
<summary><b>👉 点击展开：Isaac Gym vs Genesis 多视角对比 (CR7 Siuuu)</b></summary>

| Isaac Gym (训练环境) | Genesis (验证环境) |
| :---: | :---: |
| ![CR7 in Isaac Gym](media/CR7_Issacgym.gif) | ![CR7 in Genesis](media/CR7_Genesis.gif) |

</details>

### 2. 崎岖地形自适应 (Rough Terrain Robustness)
在 Isaac Gym 训练出基础行走模型后，零样本（Zero-shot）迁移至 Genesis 的崎岖地面进行测试。针对初期频繁被地形绊倒的问题，通过修改奖励函数进行了针对性优化。

<details>
<summary><b>👉 点击展开：崎岖地面优化前后对比 (Walk)</b></summary>

| 优化前（频繁摔倒） | 优化后（稳定跨越） |
| :---: | :---: |
| ![Walk fall before tuning](media/Walk_fall.gif) | ![Walk stable after tuning](media/Walk_Genesis.gif) |
*注：右侧为增加 `feet_max_height_for_this_air` 抬腿惩罚后的表现，大幅提升了地形适应力。*
</details>

---

## 🛠️ 学习与改进进展 (Technical Details)

为了解决 G1 在模仿爆发性动作时出现的平衡崩溃与姿态畸变，我进行了以下技术探索：

### 1. 奖励函数工程 (Reward Tuning)
- **抑制“大风车”代偿行为**：大幅增加 `penalty_action_rate`（动作速率惩罚）至 `-1.5`，强迫模型输出更平滑的关节指令，有效改善了起跳瞬间为了维持平衡而导致的手臂乱挥现象。
- **上半身姿态保真度**：拉高了 `teleop_vr_3point` 的权重，在保证下盘稳定的前提下，更精准地还原了参考动作的躯干轨迹。
- **落地稳定性约束**：针对 G1 落地容易崴脚失控的痛点，引入 `penalty_feet_ori`（脚部朝向惩罚），迫使机器人脚掌在触地瞬间与地面保持平行。

### 2. Sim2Sim 链路打通 (ONNX -> Genesis)
为了验证 RL 策略的泛化能力，补充了原仓库缺失的 Sim2Sim 验证链路，方便没有实机硬件的开发者进行物理学交叉验证：
- **模型导出**：新增 `humanoidverse/export_pt_to_onnx.py`，支持将训练好的 `.pt` 权重直接导出为跨平台的 **ONNX** 格式。
- **物理验证引擎**：新增 `genesis_simulation/` 模块，成功在 **Genesis** 物理引擎中加载 ONNX 策略进行推理。
- **意义**：运行 `run_onnx.py` 即可在 Genesis 中直观观察机器人表现。由于 Genesis 的接触动力学通常比 Isaac Gym 更严谨，这种交叉验证能有效暴露模型在真实物理世界中可能出现的“动能损耗”与“穿模依赖”。

---

## ⚠️ 待解决的问题 (To-Do List)
- [ ] **高度突破**：起跳高度目前受限于电机力矩软限位，后续计划微调课程学习（Curriculum）阈值，进一步释放爆发力。
- [ ] **高频震荡**：落地后的躯干震荡依然存在，需要进一步优化 PD 刚度参数或引入更平滑的动作惩罚项。

---

## 🙏 致谢
感谢 [ASAP](https://github.com/LeCAR-Lab/ASAP) 团队开源了如此优秀的具身智能框架。本项目仅作为个人强化学习与机器人控制技术的探索使用。
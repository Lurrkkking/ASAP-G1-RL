<p align="right">
  <a href="README.md">English</a> | <b>中文</b>
</p>

# ASAP-G1-Learning: G1 机器人强化学习与 Sim2Sim 实验记录

本项目基于 [ASAP](https://github.com/LeCAR-Lab/ASAP) 框架，记录我在 **Unitree G1** 上进行动作模仿、强化学习、跨物理引擎验证、ASAP delta action 复现以及足球控球任务探索的过程。

相比于简单堆功能，本项目更关注几件事：

- 复杂人体动作在 G1 上的 motion tracking 训练
- Isaac Gym → Genesis / MuJoCo 的 Sim2Sim 验证
- locomotion、动作模仿与接触稳定性调试
- ASAP delta action 阶段的数据采集、open-loop 训练与评估
- 面向足球 kickup / juggling 的 RL 环境构建与任务重构

> 原 ASAP 官方说明已保留为 `README_ORIGIN.md`，环境配置和基础使用方法可参考原文件。

---

## 结果展示

### 1. C罗 Siuuu 动作模仿

在 0.85 电机力矩软限位和域随机化条件下，G1 可以完成较完整的动作链：**助跑 → 起跳转身 → 空中挥臂 → 落地**。

| Isaac Gym（训练环境） | Genesis（验证环境） |
| :---: | :---: |
| <img src="media/CR7_Issacgym.gif" width="400"> | <img src="media/CR7_Genesis.gif" width="400"> |

### 2. 崎岖地形 locomotion

在 Isaac Gym 中训练基础行走策略后，将策略零样本迁移到 Genesis 崎岖地面测试。早期模型容易被障碍绊倒，后续通过奖励调整提升了抬腿积极性和过障稳定性。

| 优化前（频繁摔倒） | 优化后（稳定跨越） |
| :---: | :---: |
| <img src="media/Walk_fall.gif" width="400"> | <img src="media/Walk_Genesis.gif" width="400"> |

### 3. G1 足球 kickup / juggling 任务（进行中）

这是目前最主要的任务线。目标不是简单让机器人“踢到球”，而是逐步训练出一种更可持续的控球能力：

> 当球偏离人体前方可控区域时，机器人通过一次触球修正，把球重新送回可控区域。

当前已经完成：

- `robot + ball` 最小仿真链路
- 基于 Isaac Gym tensor 的球状态、接触检测和调试日志
- single-hit / kickup 版本环境与奖励
- 能将球踢起的初步策略
- 去掉过硬落地终止后，策略能更稳定地把球控制到目标高度附近
- 部分 rollout 中已出现二次摆腿尝试，说明策略开始出现连续修正的早期迹象

| 控制到目标高度附近，但姿态仍不自然 | 已出现二次摆腿尝试 |
| :---: | :---: |
| <img src="media/ball_kickup_stable.gif" width="400"> | <img src="media/second_swing_attempt.gif" width="400"> |

当前任务仍处于探索阶段，尚未实现稳定连续颠球。现阶段更可行的中间目标是先得到一个稳定的 **kickup / recovery primitive**：球低了能踢起来，球不要飞太偏，人踢完后仍能回到下一次触球的准备状态。后续再逐步加入可续性、控高、控方向和动作风格约束。

详细总结见：[G1 足球颠球任务技术总结](docs/G1_football_juggling_technical_summary.md)

---

## 主要技术工作

### Motion tracking 与奖励调整

围绕 G1 的高动态动作模仿，主要做了以下实验：

- 引入 `soft_torque_curriculum`，早期放宽力矩帮助探索起跳动作，后期逐步收回到 0.85 力矩软限位。
- 针对“单脚赖地”的局部最优，调整跌倒惩罚和脚部跟踪奖励，使模型更愿意真正双脚离地。
- 增大 `penalty_action_rate`，抑制起跳和落地时的手臂、关节高频乱抖。
- 加入 `penalty_feet_ori`，约束落地脚掌姿态，降低落地失稳概率。
- 使用高噪声断点续训，在已有策略基础上重新提高探索强度，改善动作发僵和起跳不足的问题。

### Locomotion 与粗糙地形测试

为了验证基础行走能力和跨引擎泛化性，本项目还进行了 G1 locomotion 训练与 Genesis 粗糙地形测试：

- 在 Isaac Gym 中训练基础 locomotion policy。
- 将策略导出为 ONNX，并在 Genesis 中进行零样本测试。
- 观察到早期策略在崎岖地形上存在抬腿不足、足端碰撞和落地不稳问题。
- 通过调整脚部高度、接触稳定性、落地相关奖励，提高了跨越障碍时的稳定性。

### Sim2Sim 验证链路

为了检查策略是否只是在 Isaac Gym 中“过拟合仿真器”，补齐了跨引擎测试流程：

- 新增 `humanoidverse/export_pt_to_onnx.py`，支持将策略导出为 ONNX。
- 搭建 `genesis_simulation/`，支持在 Genesis 中加载 ONNX 策略测试。
- 对比 Isaac Gym 与 Genesis 中的起跳高度、稳定性和落地行为，发现接触、阻尼和积分差异会显著影响高动态动作。
- 重新启用域随机化后，Genesis 中的动作稳定性明显改善。
- 补齐 MuJoCo ONNX 推理流程，并对齐控制频率、动作滤波和目标限速等参数。

### ASAP delta action 复现探索

这部分围绕 ASAP 论文中的 delta action 阶段展开。与旧版 README 中的“解析 residual / oracle patch”不同，当前实现按最近的实验路线，直接围绕 ASAP 的 rollout-with-action 数据链路和 open-loop deltaA 训练进行复现。

当前完成的工作包括：

- 实现 Isaac Gym rollout logger，生成包含 `root / dof / body state` 与 23-DoF `action` 的 `motion_with_action.pkl`。
- 完成 Gym → Gym sanity check，验证 `action` 空间、clip 语义、motion phase、`motion_lib` 读取和 deterministic actor mean 的一致性。
- 将 CR7 motion tracking policy 导出为 ONNX，并在 MuJoCo 中构建本地 rollout 采集脚本，生成 MuJoCo target rollout pkl。
- 使用 MuJoCo rollout pkl 训练 MuJoCo → Isaac Gym 的 open-loop deltaA。
- 实现 zero-delta 与 deterministic-deltaA 对比评估，用于判断 deltaA 是否真实改善目标状态匹配。

当前观察到的现象：

- open-loop deltaA 能明显改善 root/body pose tracking；最佳 checkpoint 下 total diff norm 曾取得约 12% 的改善。
- full-body 23-DoF deltaA 容易对 DoF velocity 和 closed-loop stability 产生副作用。
- closed-loop fine-tuning 对 obs、reward、reference motion 和 frozen deltaA 接入方式非常敏感，目前仍处于调试阶段。

因此，当前不把该部分表述为“完整复现 ASAP residual 结果”，而是作为 **ASAP delta action 数据链路与 open-loop 评估复现探索** 保留。

### 足球任务环境与任务重构

足球任务最初采用 single-hit 设计，即机器人准备、摆腿、触球，然后根据触球后球的高度打分。这个版本能跑通基本链路，但逐渐暴露出问题：

- 容易把任务写成“一次性击球得分”
- reward 越补越碎，语义不够统一
- 第一脚成功后，身体姿态往往无法自然衔接第二脚
- 单纯追求高度会削弱后续连续控球的可恢复性

因此后续将任务重新抽象为“前方控球区维持”：

- 球处于人体前方目标状态区时，主要奖励维持稳定。
- 球离开目标状态区并进入下落趋势时，触发修正需求。
- 触球的目标不是单纯把球踢高，而是把球送回一个可持续控制的区域。
- 后续重点从“是否踢到球”转向“触球质量、出球轨迹、踢后恢复姿态和下一拍可续性”。

后续训练接口会按层次拆分：功能目标负责球是否回到可控区，接触几何负责是否以合理方式触球，风格先验负责动作自然性，硬约束负责剔除失稳行为。

---

## 当前重点

接下来主要集中在足球控球任务和 deltaA 复现收尾：

- [ ] 保留能起球的 kickup 方向，先稳定单次 recovery primitive。
- [ ] 将球落地立即终止改为更温和的 soft penalty / delayed termination，避免策略过度踢高。
- [ ] 加入第一脚后的 recoverability 指标：躯干稳定、支撑稳定、摆腿回收、下一拍 ready pose。
- [ ] 统计二次摆腿时的球-脚最近距离，判断是时机问题、轨迹问题还是观测问题。
- [ ] 增加球相对脚的局部几何观测，减少盲目摆腿。
- [ ] 将“单次击球”继续重构为“前方可控区维持”任务，逐步加入高度、方向和下一拍准备状态约束。
- [ ] 对 deltaA 做 ankle-only 或 lower-body-only 对照，减少 full-body residual 对关节速度和闭环稳定性的副作用。
- [ ] 继续完善 Genesis / MuJoCo 的 Sim2Sim 测试，评估策略是否具有跨引擎泛化能力。

---

## 致谢

感谢 [ASAP](https://github.com/LeCAR-Lab/ASAP) 团队开源的具身智能训练框架。  
本项目主要用于个人在强化学习、机器人控制和跨物理引擎验证方面的学习与实验。

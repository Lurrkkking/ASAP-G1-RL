<p align="right">
  <a href="README.md">English</a> | <b>中文</b>
</p>

# ASAP-G1-Learning: G1 机器人强化学习与 Sim2Sim 实验记录

本项目基于 [ASAP](https://github.com/LeCAR-Lab/ASAP) 框架，记录我在 **Unitree G1** 上进行动作模仿、强化学习、跨物理引擎验证、ASAP delta action 复现以及足球控球任务探索的过程。

相比于简单堆功能，本项目更关注几件事：

- 复杂人体动作在 G1 上的 motion tracking 训练
- Isaac Gym → Genesis / MuJoCo 的 Sim2Sim 验证
- locomotion、动作模仿与接触稳定性调试
- ASAP delta action 阶段的数据采集、open-loop 训练与 closed-loop fine-tuning
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

### 3. ASAP Delta Action Residual：MuJoCo-to-IsaacGym Sim2Sim

该部分尝试复现 ASAP 中的 residual / delta action 思路，但目标并非真机复现，而是先在 **MuJoCo → Isaac Gym** 的 Sim2Sim 场景中验证残差补偿链路。

核心目标是：

> 用 MuJoCo 作为目标动力学环境，采集带 action 的 rollout；在 Isaac Gym 中训练 frozen deltaA；再将 frozen deltaA 注入 Isaac Gym，构造 corrected simulator，并对原始 motion tracking 主策略进行 closed-loop fine-tuning。

#### Open-loop deltaA 评估

open-loop 阶段使用 MuJoCo rollout-with-action 数据训练 deltaA。确定性评估时，在 Isaac Gym 中分别执行：

- **Zero delta**：只执行 MuJoCo rollout 中记录的 action；
- **Deterministic deltaA**：执行 MuJoCo action + learned residual action。

在 paper-style open-loop 指标下，ankle-only deltaA 能显著降低 MuJoCo-to-IsaacGym replay error：

| 指标 | Zero delta | Deterministic deltaA | 改善 |
| :--- | ---: | ---: | ---: |
| Eg-mpjpe (mm) | 1038.984 | 82.724 | 92.04% |
| Empjpe (mm) | 316.271 | 40.539 | 87.18% |
| Eacc (mm/frame²) | 7.569 | 4.559 | 39.77% |
| Evel (mm/frame) | 24.483 | 6.484 | 73.51% |
| **Paper mean improvement** | - | - | **73.13%** |
| **Paper normalized improvement** | - | - | **73.13%** |

该结果说明，deltaA 不是简单输出随机扰动，而是能够在 open-loop replay 中显著降低目标状态匹配误差。

#### Closed-loop fine-tuning

closed-loop 是 ASAP residual 流程的最后一环。训练时将 frozen deltaA 注入 Isaac Gym，作为 simulator dynamics correction；然后 fine-tune 原始 motion tracking 主策略。部署到 MuJoCo 时，只使用 fine-tuned main policy，不再叠加 deltaA。

复现过程中发现一个关键问题：旧 closed-loop 配置在 `delta_action_scale=0` 时也不能退化为普通 motion tracking continue-train。也就是说，即便 frozen deltaA 没有实际参与动作，主策略仍然会因为 env、reset、termination、reward 和 PPO 超参不等价而发生 policy drift，尤其集中在 ankle / hip action 上。

因此重新构建了 **baseline-equivalent closed-loop** 配置：

- 保留 baseline motion tracking 的 reward 和 PPO 设置；
- 主策略 actor observation 保持不变；
- frozen deltaA 只通过 corrected action path 注入；
- 当 `delta_action_scale=0` 时，closed-loop 退化为普通 baseline continue-train；
- 通过 scale sweep 保守地测试非零 residual correction。

目前最稳定的 closed-loop candidate 使用：

- `delta_action_mask_mode = ankle_only`
- `delta_action_scale = 0.05`
- frozen open-loop deltaA checkpoint：`model_6000.pt`
- fine-tuned main policy checkpoint：`model_13100.pt`

MuJoCo closed-loop 数值评估如下：

| 指标 | Closed-loop scale=0.05 |
| :--- | ---: |
| Eg-mpjpe | 317.775 |
| Empjpe | 207.820 |
| Eacc | 27.935 |
| Evel | 31.004 |
| **Paper mean improvement vs baseline** | **22.79%** |

该策略在 MuJoCo 中保持稳定，而更大的 scale（如 `0.15`）表现明显变差。这说明 closed-loop residual correction 的 scale 需要保持保守，否则容易导致 ankle / hip action drift，并在目标动力学环境中被放大为抖动或摔倒。

| Baseline MuJoCo rollout | Closed-loop scale=0.05 MuJoCo rollout |
| :---: | :---: |
| <img src="imgs/baseline.gif" width="400"> | <img src="imgs/005closed.gif" width="400"> |

该部分不表述为完整真机 ASAP residual 复现，而是定位为 **ASAP residual 的 Sim2Sim 复现探索**：包括 MuJoCo rollout 采集、ankle-only open-loop deltaA 训练、paper-style open-loop evaluation、baseline-equivalent closed-loop fine-tuning 和 MuJoCo deployment evaluation。

### 4. G1 足球 kickup / juggling 任务（进行中）

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
- 补齐 MuJoCo ONNX 推理流程，并对齐控制频率、动作滤波、目标限速和 policy action 语义。

### ASAP Delta Action 复现探索

这部分围绕 ASAP 论文中的 delta action 阶段展开。与旧版笔记中的“解析 residual / oracle patch”不同，当前实现按更接近 ASAP 的数据驱动路线，围绕 rollout-with-action 数据采集、open-loop deltaA 训练、deterministic evaluation 和 closed-loop fine-tuning 进行复现。

#### 为什么需要 delta action

在 Isaac Gym 中训练出的 motion tracking policy，放到 MuJoCo、Genesis 或真实机器人中，不一定产生相同的状态转移。原因包括接触模型、阻尼、积分器、马达模型和刚体动力学差异。

delta action 的目标不是替代原始动作，而是在原始动作上增加一个小修正：

`final_action = base_action + delta_action`

在 open-loop 阶段，`base_action` 来自记录好的 target rollout；在 closed-loop 阶段，`base_action` 来自正在训练的 main policy。

#### Rollout-with-action 数据采集

原始 motion pkl 只包含参考动作状态，不包含 policy 在目标环境中执行过的 action。因此，deltaA 训练需要额外采集 `motion_with_action.pkl`，其中包括：

- root state
- DoF position / velocity
- body position / velocity
- body rotation
- 23-DoF policy action

这一步是 deltaA 训练的基础，因为 deltaA 需要知道目标环境中某一步执行了什么 action，以及该 action 导致了什么状态转移。

#### Open-loop deltaA 训练

在当前 Sim2Sim residual 实验中，先将 CR7 motion tracking policy 导出为 ONNX，并在 MuJoCo 中 rollout 得到 target trajectory。然后回到 Isaac Gym 中训练 deltaA，使 Isaac Gym 在执行相同 action 加 residual 后，更接近 MuJoCo 的状态转移。

open-loop evaluation 比较两种情况：

- **Zero delta**：在 Isaac Gym 中直接执行 MuJoCo rollout 里记录的 action；
- **Deterministic deltaA**：执行 MuJoCo action + deterministic residual action。

一个关键实现细节是：确定性评估中使用 actor mean，而不是 PPO sampled action。这样可以避免把 exploration noise 误认为 learned residual。

当前最佳 ankle-only open-loop checkpoint 的 paper-style deterministic replay 结果如下：

| 指标 | Zero delta | Deterministic deltaA | 改善 |
| :--- | ---: | ---: | ---: |
| Eg-mpjpe (mm) | 1038.984 | 82.724 | 92.04% |
| Empjpe (mm) | 316.271 | 40.539 | 87.18% |
| Eacc (mm/frame²) | 7.569 | 4.559 | 39.77% |
| Evel (mm/frame) | 24.483 | 6.484 | 73.51% |
| **Paper mean improvement** | - | - | **73.13%** |

这说明 ankle-only deltaA 能显著降低 MuJoCo-to-IsaacGym 的 open-loop replay error。

#### 为什么选择 ankle-only residual

Full-body 23-DoF residual 虽然可以改善 root / body pose tracking，但容易对 DoF velocity 和 closed-loop stability 产生副作用。相比之下，ankle-only residual 更局部、更保守，也更接近 ASAP 真实部署中对 G1 的使用方式。

当前 ankle-only mask 只保留四个 residual 维度：

- left ankle pitch
- left ankle roll
- right ankle pitch
- right ankle roll

注意：主 motion action 仍然是 23 维。被限制的是 residual correction，而不是原始动作。

#### Closed-loop fine-tuning

初始 closed-loop 尝试不稳定。排查后发现，旧 closed-loop 配置在 `delta_action_scale=0` 时也不能退化为普通 baseline motion tracking continuation。它使用了不同的 environment、reset 逻辑、termination 设置、reward 权重和 PPO 超参数，导致即使 frozen deltaA 实际不生效，主策略也会发生 policy drift。

因此重新构建了 **baseline-equivalent closed-loop** 配置。核心 sanity check 是：

`delta_action_scale = 0` 时，closed-loop training 应该等价于普通 baseline motion tracking continue-train。

通过该等价性检查后，再保守地测试非零 residual scale。目前最稳定的结果为：

- `delta_action_scale = 0.05`
- `delta_action_mask_mode = ankle_only`
- fine-tuned main policy: `model_13100.pt / model_13100.onnx`

MuJoCo closed-loop deployment 结果如下：

| 指标 | 数值 |
| :--- | ---: |
| Eg-mpjpe | 317.775 |
| Empjpe | 207.820 |
| Eacc | 27.935 |
| Evel | 31.004 |
| Paper mean improvement vs baseline | 22.79% |
| Ankle action drift vs baseline | 1.470 |
| Hip action drift vs baseline | 0.964 |
| Action rate | 3.431 |

该 closed-loop policy 在 MuJoCo 中能够稳定运行；但 `scale=0.15` 已经明显变差，说明 residual correction 的 scale 不能过大。当前结果更适合表述为一个稳定的 Sim2Sim closed-loop candidate，而不是完整真机 ASAP 复现。

#### 当前 residual 结论

当前阶段性结论：

- open-loop MuJoCo-to-IsaacGym deltaA 在 paper-style replay 指标上有效；
- ankle-only residual 比 full-body residual 更稳定；
- closed-loop fine-tuning 对 baseline 等价性和 residual scale 非常敏感；
- baseline-equivalent closed-loop 修复了 `scale=0` 也会漂移的问题；
- `delta_action_scale=0.05` 得到了一个稳定的 MuJoCo candidate，并有正向 paper-style improvement；
- 更大的 scale 容易导致 ankle / hip action drift，并在 MuJoCo 中引发抖动或摔倒。

因此，该部分定位为 **ASAP delta action 的 Sim2Sim 复现探索**，而不是完整 real-world residual 复现。

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
- [ ] 继续进行 baseline-equivalent closed-loop residual 的 scale sweep。
- [ ] 通过更小 residual scale、early stopping 或 action regularization 降低 closed-loop 中的 ankle / hip action drift。
- [ ] 继续完善 Genesis / MuJoCo 的 Sim2Sim 测试，评估策略是否具有跨引擎泛化能力。

---

## 致谢

感谢 [ASAP](https://github.com/LeCAR-Lab/ASAP) 团队开源的具身智能训练框架。  
本项目主要用于个人在强化学习、机器人控制和跨物理引擎验证方面的学习与实验。
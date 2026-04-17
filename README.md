# ASAP-G1-Learning: G1 机器人动作模仿学习记录

本项目是基于 [ASAP](https://github.com/LeCAR-Lab/ASAP) 框架的进阶实践项目，用于个人学习和探索 Isaac Gym、Genesis 等具身智能仿真平台。主要目标是让 **Unitree G1** 机器人学会模仿复杂的爆发性动作轨迹（如 C罗 Siuuu 起跳），并重点打通从强化学习训练到多引擎物理验证（Sim-to-Sim）的部署闭环。

> **注**：原项目官方文档已重命名为 `README_ORIGIN.md`，方便查阅环境配置与基础指南。

---

## 🚀 目前成果展示 (Showcase)

### 1. C罗 Siuuu 动作模仿 (Dynamic Motion Tracking)
在 0.85 电机力矩软限位及域随机化（Domain Randomization）的严苛条件下，成功实现了完整动作链：**助跑 → 起跳转身 → 空中挥臂 → 落地**。



| Isaac Gym (训练环境) | Genesis (验证环境) |
| :---: | :---: |
| <img src="media/CR7_Issacgym.gif" width="400"> | <img src="media/CR7_Genesis.gif" width="400"> |



### 2. 崎岖地形自适应 (Rough Terrain Robustness)
在 Isaac Gym 训练出基础行走模型后，零样本（Zero-shot）迁移至 Genesis 的崎岖地面进行测试。针对初期频繁被地形绊倒的问题，通过修改奖励函数进行了针对性优化。



| 优化前（频繁摔倒） | 优化后（稳定跨越） |
| :---: | :---: |
| <img src="media/Walk_fall.gif" width="400"> | <img src="media/Walk_Genesis.gif" width="400"> |

*注：右侧为增加 `feet_max_height_for_this_air` 抬腿惩罚后的表现，大幅提升了地形适应力。*


---

## 🛠️ 学习与改进进展 (Technical Details)

为了解决 G1 在模仿爆发性动作（如 CR7 Siuuu 起跳）时出现的平衡崩溃、姿态畸变以及“不思进取”等问题，我进行了以下深度的技术探索与排障：

### 1. 奖励函数与局部最优博弈 (Reward Engineering & Hacking)
- **开启课程学习**：开启 `soft_torque_curriculum` ，最大值为 `1.2`，初期给模型更大的力矩以探索起跳的发力技巧，后期逐渐缩减至`0.85`，解决了机器人不起跳的现象。
- **破解“踮脚”陷阱 (Reward Hacking)**：在训练 Siuuu 腾空动作时，模型因极度恐惧跌倒惩罚（`termination = -200`），进化出了“单脚死死踩地、另一只脚踮起”的狡猾代偿行为。通过动态降低跌倒惩罚，并翻倍脚部位置跟踪奖励（`teleop_body_position_feet`），成功逼迫模型打破舒适区，实现双脚离地。
- **抑制“大风车”代偿**：大幅增加 `penalty_action_rate` 至 `-1.5`，迫使模型输出更平滑的关节指令，解决了起跳瞬间依靠手臂乱挥来强行维持平衡的现象。
- **落地姿态约束**：引入 `penalty_feet_ori`（脚部朝向惩罚），迫使机器人脚掌在触地瞬间与地面保持平行，降低了落地崴脚失控的概率。

### 2. 物理与控制参数破局 (Dynamics & Control)
- **高噪断点续训 (Noise-injected Resume)**：针对 25k 轮后模型陷入“动作发僵”的局部最优解，采用修改版脚本，在加载权重时强制重置探索噪声（`init_noise_std=0.85`），给模型重新注入试错勇气，成功拉升了动作的高度与舒展度。

### 3. Sim2Sim 链路打通与动力学诊断 (ONNX -> Genesis)
为验证 RL 策略的真实物理泛化能力，补充了原仓库缺失的跨引擎验证链路：
- **模型导出与推理**：新增 `humanoidverse/export_pt_to_onnx.py` 导出策略，并构建 `genesis_simulation/` 环境，运行 `run_onnx.py` 即可在 Genesis 中进行物理检验。
- **发现“动能小偷”**：在比对测试中发现，相同的 0.85 极限力矩在 Isaac Gym 中足以腾空，但在 Genesis 中高度受损。这暴露了高精度物理引擎（如 Genesis）由于更严谨的接触刚度（Stiffness/Restitution）和隐式积分，吸收并损耗了瞬态冲量。这为后续迈向真实物理部署（Sim2Real）敲响了警钟，指明了域随机化（DR）需要进一步覆盖摩擦与阻尼参数的方向。
- **开启域随机化训练**：Isaac Gym 中动作理想且完整，但是在 Genesis 仿真测试中经常抽搐摔倒，后面推测是模型训练过拟合，Genesis 高保真的环境让模型稳定性下降。后续重新开启了域随机化训练，模型在 Genesis 中的鲁棒性和动作完整度显著提升。
- **MuJoCo 复现实验进展**：目前已补齐 MuJoCo ONNX 推理链，并对齐了与 Genesis 的关键时序参数（控制频率、动作滤波、目标限速等）。现阶段 MuJoCo 已从“直接抽搐摔倒”改善到“可正常起步但在起跳前失衡”，说明问题已从推理链硬错误收敛到更细粒度的相位、接触和模型动力学差异。

### 4. 基于 Oracle 和强化学习的残差补丁优化 (Residual Patch)
- **从纯 RL 转向 Oracle 残差建模**：早期直接用 46 维 Attention-Delta 网络配合 PPO 盲探索 `Isaac -> Genesis` gap，但很快发现纯黑盒 RL 在接触非连续、高刚度 PD 和瞬态冲击下样本效率极差，还会滑向“为了不摔倒而扭曲姿态”的局部最优。后续改为先在固定状态上构造局部动力学雅可比 `J`，再用带阻尼的加权岭回归求解 Oracle 残差 $\delta a^* = (J^T W J + \lambda I)^{-1} J^T W r$，把“先找局部最优动作，再教给网络”的物理先验引入到残差补丁学习中。
- **离线 `a*` 蒸馏与 OOD 约束**：围绕 $\pi^\Delta(s, a_{base})$ 的 Asymmetric Distillation 做了多轮 fixed-state 实验。128 样本时误差从 `2.10` 反升到 `2.73`，并出现平均约 `0.065 rad` 的幻觉动作，说明网络会过拟合数值噪声并在 OOD 状态下放大不稳定性。后续加入 `Lower-Body Masking`、`Deadzone Filtering` 和补丁幅值约束后，扩到 512/1024 样本时误差终于稳定优于 baseline，改善约 **4%~5%**，但也暴露出“单帧状态无法表达接触/相位历史”的 state aliasing 上限。
- **闭环执行病理定位**：将冻结残差补丁接回 `DeltaA_ClosedLoop` 后，日志里出现了最高约 `1.7 rad` 的最终执行偏移和电机扭矩饱和。进一步核实后确认 `max_delta_scale = ±0.10` 的代码硬钳位本身是生效的，因此问题不是约束失效，而是高 `K_p` 下肢关节上的 `Bang-Bang` 高频震荡和 PD 扭矩尖峰。这个结论把补丁设计从“大胆纠偏”收敛成“微调修正”，后续幅值需要进一步压到 `0.02 rad` 量级，并考虑低通滤波。
- **Confounder 清洗与 PPO 底层 bug 修复**：为做纯净 A/B，对“打补丁环境”额外清洗了被放宽的 termination threshold 和诱导性 gap reward，避免实验变量被环境私货污染。同时向下追踪动作噪声曲线漂移问题，最终定位到 `PPOActor.std` 被底层无条件注册为 `nn.Parameter`，导致 `learn_sigma=False` 实际无效。后续已用最小向后兼容方式修复：`learn_sigma=False` 时将 `std` 改为同名 `register_buffer`，并保留旧 checkpoint 的无损加载能力。
- **纯净验证台上的残差收益确认**：在清洗 confounder 后，用“旧主策略 `26600` + 冻结黄金补丁”做固定状态 one-step gap 验证，指标从 `2.02` 稳定下降到 `1.86`，在离线残差补丁的基础上再次物理改善约 **7.9%**，比初始策略改善约**11.4%**。这说明残差补丁在局部动力学层面确实有效，收益不是日志假象，也不是奖励塑形带来的伪提升。
- **策略塑性丧失的诊断**：虽然 one-step gap 数据明显变好，但渲染中旧 PPO 策略的动作风格仍然拘谨保守，说明问题已经从“补丁有没有用”上升到“旧策略是否还有可塑性”。当前判断是：经历约 `26600` 轮动力学训练后，主策略已经固化了代偿性步态记忆，仅靠冻结补丁无法完成运动先验重塑。
---

## ⚠️ 待解决的问题 (To-Do List)
- [ ] **Root 级误差补偿**：当前残差补丁已经明显降低了 `dof_pos / dof_vel` 级别的 Sim2Sim Gap，但对 `root` 级轨迹改善有限。下一步需要针对质心/躯干高度与整体姿态设计更直接的奖励或补偿机制。
- [ ] **MuJoCo 相位与接触对齐**：MuJoCo 链路目前仍落后于 Genesis，后续需要继续排查 `cycle_time / time_offset`、地面接触参数、XML 动力学与执行器配置，缩小 MuJoCo 与 Genesis/IsaacGym 之间的差异（有点奇怪哈哈，之前能用的。。😭）
- [ ] **长时 Rollout 量化**：目前已经完成同锚点 one-step gap 量化，下一步计划补充多步 rollout 指标（存活步数、累计 tracking error、root/body/joint 分项误差），更完整地评估残差补丁在真实 Sim2Sim 场景下的收益。
- [ ] **历史信息建模 / 解决 state aliasing**：当前 `\pi^\Delta(s, a_{base})` 主要依赖单帧状态，已经暴露出无法完整表征接触力矩和相位历史的问题。下一步需要系统评估 frame stacking、历史动作/速度上下文或显式接触特征，验证是否能继续突破 512/1024 样本后的收益天花板。
- [ ] **补丁执行平滑化**：闭环测试已经表明，哪怕代码层面满足 `±0.10` 约束，也可能因为高 `K_p` 关节上的 PD 扭矩尖峰导致系统炸毁。后续需要继续压缩 `max_delta_scale` 到更保守区间，并尝试低通滤波、动作速率约束或 torque-aware regularization。
- [ ] **Tabula Rasa 主策略重训**：当前最关键的开放问题不是补丁能否降低局部 gap，而是旧 PPO 主策略是否已经发生 `Loss of Plasticity`。下一步应在挂载冻结补丁的干净环境中，从随机初始化重新训练 PPO 主脑，验证是否能摆脱旧策略的代偿性“猥琐步态”记忆。
- [ ] **动作自主标注**：准备下一步自己着手标注更具挑战性的动作，如羽毛球杀球动作（Smash），并完成其 Sim2Sim 验证。
---

## 🙏 致谢
感谢 [ASAP](https://github.com/LeCAR-Lab/ASAP) 团队开源了如此优秀的具身智能框架。本项目仅作为个人强化学习与机器人控制技术的探索使用。

# Attention-Delta 门控改造说明（中文）


虽然原仓库里有完整的残差网络代码骨架，但原仓库的官方实现是“全局粗放式”的残差，没有注意力机制。原版 ASAP 的残差（23 维）：只输出 $\Delta a_t$。代码里的开关（如 with_delta_a_or_not）是通过人为配置的静态权重或布尔值来强行缩放残差。我们刚才改造的版本（46 维）：在他们原有的骨架上，我们加入了 $\alpha_t$ (Attention 门控权重)。这是原版代码里没有的创新点。总结来说： 你看到的 delta_a 文件夹下的代码，就是 ASAP 原文里用来弥合 Sim2Real / Sim2Sim 物理鸿沟的“残差网络”。你现在的代码库，是基于官方“基础版残差”升级改造后的“自适应注意力残差”版本。

本文档说明本次在仓库中完成的改造文件、核心逻辑和使用方式。
目标是将原先随机全局残差逻辑升级为可学习的 Attention-Delta 门控机制：

\[
a_{patch}=a_t+\alpha_t\odot\Delta a_t
\]

其中：
- \(a_t\)：闭环基础策略输出（旧策略）
- \(\Delta a_t\)：当前可训练策略输出的残差动作分量
- \(\alpha_t\)：当前可训练策略输出的关节级注意力权重

---

## 1. 文件变更总览

本次修改的文件：

1. `humanoidverse/envs/delta_a/delta_a_closed_loop.py`
2. `humanoidverse/config/exp/train_delta_a_closed_loop.yaml`
3. `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_simfinetuning.yaml`
4. `humanoidverse/config/rewards/motion_tracking/delta_a/reward_motion_tracking_use_deltaA_to_train_2real.yaml`
5. `humanoidverse/agents/delta_a/train_delta_a.py`

---

## 2. 各文件详细说明

## 2.1 `humanoidverse/config/exp/train_delta_a_closed_loop.yaml`

### 改动内容
- 将该实验的 `robot.actions_dim` 从 23 提升到 46。
- 新增 `env.config.max_delta_scale` 配置项（默认取 `robot.control.action_scale`）。

### 目的
- 让 Actor 输出 46 维：
  - 前 23 维：`raw_delta_a`
  - 后 23 维：`raw_alpha`

### 说明
- 该改动只作用于 `train_delta_a_closed_loop` 这个实验配置，不会全局污染其他实验。

---

## 2.2 `humanoidverse/envs/delta_a/delta_a_closed_loop.py`

这是本次核心改造文件。

### A) 新增/调整的关键状态变量
- `self.alpha_t`：保存当前步的注意力权重（用于 reward）
- `self.raw_delta_a`：保存当前步原始残差动作（用于 obs 输出）
- `self.max_delta_scale`：残差动作幅值上限
- `self.closed_loop_actions`：闭环策略动作固定按 `num_dof(23)` 管理

### B) 动作剥离与门控融合（核心）
新增方法：`_compute_attention_delta_action(self, actions)`

执行流程：
1. 切片：
   - `raw_delta_a = actions[:, :num_dof]`
   - `raw_alpha   = actions[:, num_dof:2*num_dof]`
2. 激活：
   - `delta_a = tanh(raw_delta_a) * max_delta_scale`
   - `alpha_t = sigmoid(raw_alpha)`
3. 门控：
   - `actions_scaled = delta_a * alpha_t`

### C) 替换旧随机逻辑
在 `_compute_torques()` 中，删除了旧的 `with_delta_a_or_not / delta_a_scale` 随机缩放路径，改为使用上面的 `actions_scaled`。

保留并延续原有物理叠加骨架：
- `torque ~ (actions_scaled + motion_action + default_dof_pos - dof_pos)`

其中 `motion_action` 是闭环基础策略动作（即 \(a_t\)）。

### D) 观测维度兼容处理
因为动作空间改为 46，而现有 obs 配置里 `actions` 仍按 23 维定义，做了覆盖：
- `_get_obs_actions()` 返回 `raw_delta_a`（23 维）
- `_get_obs_actions_sim2real_policy()` 返回 `raw_delta_a`（23 维）

这样不会破坏既有 obs 维度定义。

### E) 新增稀疏性奖励函数
新增：`_reward_attention_sparsity(self)`

```python
return torch.sum(self.alpha_t, dim=-1)
```

用于惩罚注意力权重总量，避免所有关节始终满开。

---

## 2.3 `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_simfinetuning.yaml`

### 改动内容
- 在 `reward_scales` 中新增：
  - `attention_sparsity: -0.01`
- 在 `reward_penalty_reward_names` 中加入：
  - `"attention_sparsity"`

### 目的
- 将注意力稀疏惩罚接入训练总 reward，并纳入现有 penalty curriculum 管理。

---

## 2.4 `humanoidverse/config/rewards/motion_tracking/delta_a/reward_motion_tracking_use_deltaA_to_train_2real.yaml`

### 改动内容
- 同步新增：
  - `attention_sparsity: -0.01`
- 并加入 `reward_penalty_reward_names`。

### 目的
- 保持另一套 delta_a 训练奖励配置的一致性，避免切配置后行为不一致。

---

## 2.5 `humanoidverse/agents/delta_a/train_delta_a.py`

### 改动内容
在 `PPODeltaA` 初始化加载闭环策略 checkpoint 时，增加动作维度兼容逻辑：

1. 暂存当前训练动作维度（现在是 46）。
2. 临时将 `env.config.robot.actions_dim` 切到闭环 checkpoint 对应维度（通常 23）。
3. 实例化并加载闭环策略。
4. 加载完成后恢复训练动作维度为 46。

### 目的
- 防止“训练动作维度 46 与旧闭环策略权重 23 维”导致的加载维度不匹配错误。

---

## 3. 训练时的重要约定

1. 当前训练策略（delta 分支）输出 46 维。
2. 闭环基础策略输出仍是 23 维（历史 checkpoint）。
3. 环境中最终参与物理叠加的是：
   - `motion_action(23)` + `delta_a(23) * alpha_t(23)`
4. `alpha_t` 不作为附加监督目标，仅通过任务 reward + 稀疏惩罚共同塑形。

---

## 4. 参数调节建议（起点）

1. `attention_sparsity` 权重起点：`-0.01`
2. 若发现 `alpha_t` 过大（几乎全 1），可调到 `-0.02 ~ -0.05`
3. 若发现残差几乎被抑制（`alpha_t` 过小），可回调到 `-0.005`
4. `max_delta_scale` 默认跟随 `action_scale`，若动作过激可单独减小。

---

## 5. 已完成的基础检查

1. 相关 Python 文件已通过语法检查（`py_compile`）。
2. 修改过的 YAML 已通过 `yaml.safe_load` 解析。

说明：
- 本次未执行完整长时训练，只完成了代码级与配置级静态检查。

# Q1 Init Fix Log

## 问题

Q1 机器人加载后第一帧正常，第二帧被整体弹飞/翻转，lv 达到 3-12 m/s。

## 排查过的方向（逐一排除）

| 假设 | 结论 |
|---|---|
| root_z 过高导致悬空坠落 | 部分相关，但不是主因 |
| STL collision mesh 触碰地面 | 改为 primitive box/sphere 后问题依然存在 |
| contact_offset=0.02 过大 | 改为 0.001 后问题依然存在 |
| gym.add_ground 无限平面有问题 | 换 static box ground 后问题依然存在 |
| 手写 PD torque 导致 | noPD 下问题依然存在 |
| noPD 没写 zero actuation force tensor | 显式写 zero torque 后问题依然存在 |
| collision filter 导致不碰撞 | 不是 |

## 根因

**`prepare_sim` 后不应该用 `set_dof_state_tensor` 覆盖 DOF positions。**

流程是：

1. `set_actor_dof_states` 设置 DOF positions 为目标值（例如 knee=0.30）
2. `prepare_sim` 将 DOF positions 调整到关节链自洽的平衡值（例如 knee=0.37）
3. 如果此时用 `set_dof_state_tensor` 把 knee 强行写回 0.30，相当于把关节"瞬移"回非平衡位置
4. 第一次 `simulate` 时，PhysX 约束求解器产生 ~9000N 的 phantom 冲量，把 robot 弹飞

**证据 — 四种 fix 策略对比：**

| 策略 | f1 lv (m/s) | f1 av (rad/s) |
|---|---|---|
| fix_all（覆盖 root pos + dof pos，清零速度） | 12.01 | 14.14 |
| fix_root_only（覆盖 root pos，清零速度） | 10.89 | 21.80 |
| fix_nothing（不做任何修正） | 3.27 | 1.86 |
| **fix_root_vel_only（只清零速度，不动 pos）** | **0.025** | **0.079** |

## 正确的初始化流程

```python
# 1. 创建 actor
ah = gym.create_actor(env, asset, pose, "q1", -1, 0, 0)

# 2. 设置 DOF positions（BEFORE prepare_sim）
dof_st = gym.get_actor_dof_states(env, ah, gymapi.STATE_ALL)
for i in range(ndof):
    dof_st[i]["pos"] = target_pos[i]
    dof_st[i]["vel"] = 0.0
gym.set_actor_dof_states(env, ah, dof_st, gymapi.STATE_ALL)

# 3. 设置 shape props（contact_offset=0.001 等）
props = gym.get_actor_rigid_shape_properties(env, ah)
for i in range(len(props)):
    props[i].contact_offset = 0.001
    props[i].rest_offset = 0.0
    props[i].restitution = 0.0
    props[i].friction = 0.8
gym.set_actor_rigid_shape_properties(env, ah, props)

# 4. prepare_sim
gym.prepare_sim(sim)

# 5. 获取 tensors
rt = gym.acquire_actor_root_state_tensor(sim)
r_v = gymtorch.wrap_tensor(rt).view(-1, 13)
dt = gym.acquire_dof_state_tensor(sim)
d_v = gymtorch.wrap_tensor(dt).view(-1, 2)

# 6. 只清零速度，不动 position！
r_v[:, 7:13] = 0.0   # 清零 root lin_vel + ang_vel
d_v[:, 1] = 0.0       # 清零 DOF velocities

# 注意：不要覆盖 r_v[:, 0:3]（root pos）
#       不要覆盖 r_v[:, 3:7]（root quat）
#       不要覆盖 d_v[:, 0]  （dof pos）

gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(r_all))
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(d_all))

# 7. 每帧显式写 zero torque（noPD 模式）
torque_tensor.zero_()
gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_tensor))
```

## 新的 STATE CLEAN 定义

只检查速度，不检查位置：
- root_lin_vel = 0
- root_ang_vel = 0  
- dof_vel = 0
- quat finite
- 不要求 dof_pos 等于 default_joint_angles

## 验证结果

修复后四组对照（add_ground 原生地面）：

| Case | ground | grav | PD | f1_rz | f1_lv | f1_av | 状态 |
|---|---|---|---|---|---|---|---|
| A | add_ground | on | off | 0.398 | 0.05 | 0.08 | 正常 |
| B | add_ground | off | off | 0.399 | 0.00 | 0.00 | 完全静止 |
| C | none | on | off | 0.388 | 0.20 | 0.00 | 自由落体 |
| D | add_ground | on | on | 0.400 | 0.60 | 2.65 | PD 正常 |

## 关键脚本

`q1_standing_final.py` — 最终验收脚本，使用 add_ground + 正确 init 流程

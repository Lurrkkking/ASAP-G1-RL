import torch
from pathlib import Path

# 你的旧 23 维权重路径
old_ckpt_path = "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260411_104945-MotionTracking_CR7_FullSystem_V2_Fresh_8192-motion_tracking-g1_29dof_anneal_23dof/model_13000.pt"
# 新的 46 维权重输出路径
new_ckpt_path = "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260411_104945-MotionTracking_CR7_FullSystem_V2_Fresh_8192-motion_tracking-g1_29dof_anneal_23dof/model_13000_46dim_init.pt"

print(f"正在读取旧权重: {old_ckpt_path}")
ckpt = torch.load(old_ckpt_path, map_location="cpu")

# 明确锁定 Actor（动作输出网络）
state_dict = ckpt['actor_model_state_dict']
upgraded_keys = []

# === 第一部分：权重手术 (在循环内完成) ===
for key, tensor in list(state_dict.items()):
    # 1. 自动追踪 std
    if ("std" in key.lower() or "log_std" in key.lower()) and tensor.shape == torch.Size([23]):
        new_tensor = torch.zeros(46)
        new_tensor[:23] = tensor
        new_tensor[23:] = tensor.mean()
        state_dict[key] = new_tensor
        upgraded_keys.append(key)
        print(f"✔️ 自动扩容 std 层: {key}")

    # 2. 自动追踪输出层 weight
    elif len(tensor.shape) == 2 and tensor.shape[0] == 23:
        hidden_dim = tensor.shape[1]
        new_tensor = torch.zeros((46, hidden_dim))
        new_tensor[:23, :] = tensor
        state_dict[key] = new_tensor
        upgraded_keys.append(key)
        print(f"✔️ 自动扩容 weight 层: {key} (前置维度: {hidden_dim})")

    # 3. 自动追踪输出层 bias
    elif len(tensor.shape) == 1 and tensor.shape[0] == 23 and "std" not in key.lower():
        new_tensor = torch.zeros(46)
        new_tensor[:23] = tensor
        new_tensor[23:] = -10.0  # 🔴 数学静默
        state_dict[key] = new_tensor
        upgraded_keys.append(key)
        print(f"✔️ 自动扩容 bias 层: {key}")

# === 第二部分：优化器清理与保存 (🔴 必须在循环结束后执行) ===

# 我们要的是大脑的肌肉记忆（权重），而不是之前的训练惯性（优化器状态）
if 'actor_optimizer_state_dict' in ckpt:
    del ckpt['actor_optimizer_state_dict']
    print("🧹 已清除 Actor 旧维度优化器状态")
        
if 'critic_optimizer_state_dict' in ckpt:
    del ckpt['critic_optimizer_state_dict']
    print("🧹 已清除 Critic 旧维度优化器状态")

# 将改好的 Actor 放回原来的大字典中
ckpt['actor_model_state_dict'] = state_dict

# 保存
torch.save(ckpt, new_ckpt_path)
print(f"\n🎉 完美！手术大获成功。")
print(f"提示：优化器状态已重置，系统将在启动时自动初始化 46 维 Adam 缓冲区。")
print(f"新的基座文件已保存至: {new_ckpt_path}")
import torch

# 指向你的 46 维基座文件
ckpt_path = "/root/autodl-tmp/ASAP/logs/TEST_CR7_Siuuu/20260411_104945-MotionTracking_CR7_FullSystem_V2_Fresh_8192-motion_tracking-g1_29dof_anneal_23dof/model_13000_46dim_init.pt"

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt['actor_model_state_dict']

for key, tensor in state_dict.items():
    if "std" in key.lower() or "log_std" in key.lower():
        print(f"🔍 找到噪声张量: {key}, 当前均值: {tensor.mean().item():.4f}")
        
        # 判断是 std 还是 log_std 并进行强制修改
        if tensor.mean() > 0:
            # 如果是正数，说明是 raw std，直接设为 0.02
            state_dict[key] = torch.ones_like(tensor) * 0.15
            print("   -> 确认为 std，已强制修改为 0.02")
        else:
            # 如果是负数，说明是 log_std，设为 log(0.02) ≈ -3.912
            state_dict[key] = torch.ones_like(tensor) * (-3.912)
            print("   -> 确认为 log_std，已强制修改为 -3.912 (等效于 std=0.02)")

# 保存修改后的文件
torch.save(ckpt, ckpt_path)
print("\n🎉 噪声物理切除成功！可以重新启动训练了。")
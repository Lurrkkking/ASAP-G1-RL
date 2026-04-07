import genesis as gs
import numpy as np
import os

def main():
    print("🚀 正在初始化 Genesis (GPU Headless 模式)...")
    gs.init(backend=gs.gpu)

    # 1. 创建场景 (适配 Genesis 0.4.5+ 最新版本，删除过时的 viewer 参数)
    scene = gs.Scene(
        show_viewer=False,   # 强制无头模式
    )

    # 2. 添加地板
    plane = scene.add_entity(gs.morphs.Plane())

    # 3. 添加 G1 机器人
    urdf_path = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf" 
    print(f"📦 正在加载机器人模型: {urdf_path}")
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0, 0, 0.8),    # 在半空中
            euler=(0, 0, 0),
        ),
    )

    # 4. 架设虚拟摄像机
    cam = scene.add_camera(
        res=(640, 480),
        pos=(2.0, 2.0, 1.0),
        lookat=(0, 0, 0.5),
        fov=60,
        GUI=False
    )

    # 5. 编译场景
    print("🔨 正在编译物理场景...")
    scene.build()
    print("✅ 场景编译完成！开始录制自由落体视频...")

    # 6. 录制循环：跑 150 帧
    cam.start_recording()
    for i in range(150):
        scene.step()
        cam.render()
        
    # 保存视频
    video_path = "g1_falling_test.mp4"
    cam.stop_recording(save_to_filename=video_path, fps=50)
    
    print(f"🎉 测试完成！视频已保存至: {os.path.abspath(video_path)}")

if __name__ == "__main__":
    main()
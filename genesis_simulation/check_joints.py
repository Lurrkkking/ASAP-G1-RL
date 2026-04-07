import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
urdf_path = "/root/autodl-tmp/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf" 
robot = scene.add_entity(gs.morphs.URDF(file=urdf_path))
scene.build()

print("========== Genesis 识别到的 29 个关节顺序 ==========")

# 针对 Genesis 最新版本的不同 API 进行探测
names = []
if hasattr(robot, 'dof_names'):
    names = robot.dof_names
elif hasattr(robot, 'dofs'):
    names = [dof.name for dof in robot.dofs]
elif hasattr(robot, 'joints'):
    # 有些关节可能不是自由度，过滤掉固定关节
    names = [joint.name for joint in robot.joints if joint.n_dofs > 0]
else:
    print("❌ 找不到关节属性！请打印 robot 内部结构：", dir(robot))

for i, name in enumerate(names):
    print(f"索引 {i}: {name}")
print("==================================================")
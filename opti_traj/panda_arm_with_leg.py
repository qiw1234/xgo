import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import matplotlib.pyplot as plt
import json
# 规划臂足协同动作，扭身子，同时手臂末端在世界系下固定位置不动
# 输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 质心保持不动，机身x方向转动在yz平面投影是一个圆
# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)


num_row = 200
num_col = 68
fps = 50

ref = np.ones((num_row - 1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row - 1, 3))
root_ang_vel = np.zeros((num_row - 1, 3))
root_rot_dot = np.zeros((num_row - 1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row - 1, 12))
arm_pos = np.zeros((num_row, 3))
arm_rot = np.zeros((num_row, 4))
arm_dof_pos = np.zeros((num_row, 6))
arm_dof_vel = np.zeros((num_row - 1, 6))

# 质心轨迹
root_pos[:, 2] = 0.55
# 质心线速度 默认为0

# 姿态
q0 = [0, 0, 0, 1]  # [x,y,z,w]
q1 = [0, 0, np.sin(np.pi / 36), np.cos(np.pi / 36)]  # 绕z轴转4°

root_rot[:] = q0
interval = 20
start = 5
end = start + interval

for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q0, q1, frac)

start = end
end = start + interval
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q0, frac)

# 计算关节角度
# 获取足端初始位置，初始位置在世界系下保持不变，需要计算质心系的足端位置
toe_pos[:] = panda7.toe_pos_init
def z_1(t):
    # z方向轨迹，0.4s
    x = 0.2  # 对称轴0.2s
    h = 0.1
    return h - 5 / 2 * (t - x) ** 2

pos1 = [0, 0, 0]
pos2 = [0.1, -0.1, 0]
temp_pos_1= np.linspace(pos1, pos2, 20) #伸腿
temp_pos_2= np.linspace(pos2, pos1, 20) #收腿
for i in range(20):
    t = i/fps
    temp_pos_1[i, 2] = z_1(t)
    temp_pos_2[i, 2] = z_1(t)

toe_pos[5:25, :3] += temp_pos_1
toe_pos[5:25, 9:11] -= temp_pos_1[:, :2]
toe_pos[5:25, 11] += temp_pos_1[:, 2]
toe_pos[25:45, :3] += temp_pos_2
toe_pos[25:45, 9:11] -= temp_pos_2[:, :2]
toe_pos[25:45, 11] += temp_pos_2[:, 2]

# 重复四次
for i in range(3):
    root_rot[50*(i+1):50*(i+2), :] = root_rot[:50,:]
    toe_pos[50*(i+1):50*(i+2), :] = toe_pos[:50,:]

# 四元数的导数
for i in range(num_row - 1):
    root_rot_dot[i, :] = (root_rot[i + 1, :] - root_rot[i, :]) * fps
# 质心角速度
for i in range(num_row - 1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i, :]) @ root_rot_dot[i, :]


# 计算质心系下的足端轨迹
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, :3]
    toe_pos[i, 3:6] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 3:6]
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 6:9]
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 9:12]

# go2的关节上下限
q = SX.sym('q', 3, 1)
for j in range(4):
    for i in range(num_row):
        # toe_pos是质心系的足端位置，所以rpy以及质心位置都是[0 0 0]
        # 这里不把上面的部分合起来放到一个式子里是因为下面使用casadi，上面使用了numpy，混合运算会出问题
        pos = (casadi.SX(casadi.DM(utils.quaternion2rotm(root_rot[i, :])).full()) @
               (panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe)[:3])
        cost = 500 * dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=panda7.lb[3 * j:3 * j + 3], ubx=panda7.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps


#  --------第一个关节来回运动---------
# 定义各阶段的时间步数

phase2_end = 25
phase4_end = 50

# 设置角度范围（单位：弧度）
initial_angle1 = 0
final_angle1 = -np.pi / 6  # 60度
# end_angle2 = -np.pi / 22.5  # -60度
# end_angle3 = np.pi / 3  # 60度
initial_angle2 = 0  # 初始角度 0度
final_angle2 = -np.pi / 12  # 目标角度 -30度
initial_angle3 = 0  # 初始角度 0度
final_angle3 = np.pi / 6  # 目标角度 -30度
initial_angle4 = 0  # 初始角度 0度
final_angle4 = -np.pi / 12  # 目标角度 -30度
initial_angle5 = 0  # 初始角度 0度
final_angle5 = np.pi / 6  # 目标角度 -30度
initial_angle6 = 0  # 初始角度 0度
final_angle6 = np.pi / 24  # 目标角度 -30度
# 计算每步的角度变化量
angle2_step1 = (final_angle2 - initial_angle2) / phase2_end
# 计算每步的角度变化量
angle3_step1 = (final_angle3 - initial_angle3) / phase2_end
angle4_step1 = (final_angle4 - initial_angle4) / phase2_end
angle5_step1 = (final_angle5 - initial_angle5) / phase2_end
angle6_step1 = (final_angle6 - initial_angle6) / phase2_end
# 计算每步的角度变化量
angle2_step2 = (initial_angle2 - final_angle2) / phase2_end
# 计算每步的角度变化量
angle3_step2 = (initial_angle3 - final_angle3) / phase2_end
angle4_step2 = (initial_angle4 - final_angle4) / phase2_end
angle5_step2 = (initial_angle5 - final_angle5) / phase2_end
angle6_step2 = (initial_angle6 - final_angle6) / phase2_end
# 阶段1:
for i in range(phase2_end):
    frac = i / phase2_end
    arm_dof_pos[i, 1] = initial_angle2 + (final_angle2 - initial_angle2) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 2] = initial_angle3 + (final_angle3 - initial_angle3) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 3] = initial_angle4 + (final_angle4 - initial_angle4) * (0.5 - 0.5 * np.cos(np.pi * frac))
    # arm_dof_pos[i, 4] = initial_angle5 + angle5_step1 * (i + 1)
    arm_dof_pos[i, 5] = initial_angle6 + (final_angle6 - initial_angle6) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 0] = initial_angle1 + (final_angle1 - initial_angle1) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 4] = initial_angle5 + (final_angle5 - initial_angle5) * (0.5 - 0.5 * np.cos(np.pi * frac))

# 阶段3:
for i in range(phase2_end, phase4_end):
    frac = (i - phase2_end) / (phase4_end - phase2_end)
    arm_dof_pos[i, 0] = final_angle1 + (initial_angle1 - final_angle1) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 1] = final_angle2 + (initial_angle2 - final_angle2) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 2] = final_angle3 + (initial_angle3 - final_angle3) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 3] = final_angle4 + (initial_angle4 - final_angle4) * (0.5 - 0.5 * np.cos(np.pi * frac))
    # arm_dof_pos[i, 4] = final_angle5 + angle5_step2 * (i - phase3_end + 1)
    arm_dof_pos[i, 5] = final_angle6 + (initial_angle6 - final_angle6) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 4] = final_angle5 + (initial_angle5 - final_angle5) * (0.5 - 0.5 * np.cos(np.pi * frac))

# 反方向再来一遍
# 设置角度范围（单位：弧度）
initial_angle1 = 0
final_angle1 = np.pi / 6  # 30度
initial_angle2 = 0  # 初始角度 0度
final_angle2 = -np.pi / 12  # 目标角度 -15度
initial_angle3 = 0  # 初始角度 0度
final_angle3 = np.pi / 6  # 目标角度 -30度
initial_angle4 = 0  # 初始角度 0度
final_angle4 = -np.pi / 12  # 目标角度 -30度
initial_angle5 = 0  # 初始角度 0度
final_angle5 = -np.pi / 6  # 目标角度 -30度
initial_angle6 = 0  # 初始角度 0度
final_angle6 = -np.pi / 24  # 目标角度 -30度

start = 50
end = 75
for i in range(start, end):
    frac = (i - start) / (end - start)
    arm_dof_pos[i, 1] = initial_angle2 + (final_angle2 - initial_angle2) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 2] = initial_angle3 + (final_angle3 - initial_angle3) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 3] = initial_angle4 + (final_angle4 - initial_angle4) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 5] = initial_angle6 + (final_angle6 - initial_angle6) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 0] = initial_angle1 + (final_angle1 - initial_angle1) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 4] = initial_angle5 + (final_angle5 - initial_angle5) * (0.5 - 0.5 * np.cos(np.pi * frac))
start = 75
end = 100
for i in range(start, end):
    frac = (i - start) / (end - start)
    arm_dof_pos[i, 0] = final_angle1 + (initial_angle1 - final_angle1) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 1] = final_angle2 + (initial_angle2 - final_angle2) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 2] = final_angle3 + (initial_angle3 - final_angle3) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 3] = final_angle4 + (initial_angle4 - final_angle4) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 5] = final_angle6 + (initial_angle6 - final_angle6) * (0.5 - 0.5 * np.cos(np.pi * frac))
    arm_dof_pos[i, 4] = final_angle5 + (initial_angle5 - final_angle5) * (0.5 - 0.5 * np.cos(np.pi * frac))

# 重复一遍
arm_dof_pos[100:200,:] = arm_dof_pos[:100,:]

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps
print("dof_vel is :", dof_vel)

# arm fk
robot_arm_rot, robot_arm_pos = utils.arm_fk([0, 0, 0, 0, 0, 0])
# 机械臂末端在机身坐标系下的位置
arm_pos[:] = robot_arm_pos
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i,:]) @ robot_arm_rot)




# 画图
colors = ['b', 'g', 'r', 'c', 'm', 'y']
plt.plot(range(num_row), arm_dof_pos[:, 0], label='Joint 1 (Radians)', color=colors[0])
plt.plot(range(num_row), arm_dof_pos[:, 1], label='Joint 2 (Radians)', color=colors[1], linestyle='--', marker='x', markersize=4, alpha=0.7)
plt.plot(range(num_row), arm_dof_pos[:, 2], label='Joint 3 (Radians)', color=colors[2], linestyle='-', marker='o', markersize=4, alpha=0.7)
plt.plot(range(num_row), arm_dof_pos[:, 3], label='Joint 4 (Radians)', color=colors[3], linestyle='-.', marker='s', markersize=4, alpha=0.7)
plt.plot(range(num_row), arm_dof_pos[:, 4], label='Joint 5 (Radians)', color=colors[4], linestyle=':', marker='^', markersize=4, alpha=0.7)
plt.plot(range(num_row), arm_dof_pos[:, 5], label='Joint 6 (Radians)', color=colors[5])
# plt.plot(range(num_row), root_rot, label='root_rot', color=colors[5])
# 设置图形标签
plt.xlabel('Time Step (Frames)')
plt.ylabel('Joint Angle (Radians)')

# 设置图形标题
plt.title('Joint Space Trajectories')

# 显示图例
plt.legend()

# 显示图形
# plt.show()

# 创建一个3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每条腿的足端位置
# 假设每条腿的足端位置分别存储在 toe_pos 的不同部分
# 例如：左前腿：toe_pos[:, 0:3], 右前腿：toe_pos[:, 3:6], 左后腿：toe_pos[:, 6:9], 右后腿：toe_pos[:, 9:12]

ax.plot(toe_pos[:, 0], toe_pos[:, 1], toe_pos[:, 2], label="Right Front Leg", color="r")
ax.plot(toe_pos[:, 3], toe_pos[:, 4], toe_pos[:, 5], label="Left Front Leg", color="g")
ax.plot(toe_pos[:, 6], toe_pos[:, 7], toe_pos[:, 8], label="Right Back Leg", color="b")
ax.plot(toe_pos[:, 9], toe_pos[:, 10], toe_pos[:, 11], label="Left Back Leg", color="y")

# 设置标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Toe Positions of the Four Legs')

# 显示图例
ax.legend()
# 展示图形
# plt.show()
num_steps = toe_pos.shape[0]

# 创建一个2D图
plt.figure(figsize=(10, 6))

# 绘制每条腿在z方向上的变化
plt.plot(range(num_steps), toe_pos[:, 2], label="Right Front Leg (Z)", color="r", linestyle='-', marker='o', markersize=4, alpha=0.7)
plt.plot(range(num_steps), toe_pos[:, 5], label="Left Front Leg (Z)", color="g", linestyle='--', marker='x', markersize=4, alpha=0.7)
plt.plot(range(num_steps), toe_pos[:, 8], label="Right Back Leg (Z)", color="b", linestyle='-.', marker='s', markersize=4, alpha=0.7)
plt.plot(range(num_steps), toe_pos[:, 11], label="Left Back Leg (Z)", color="y", linestyle=':', marker='^', markersize=4, alpha=0.7)
# 显示时间信息（在时间点处添加标注）
for i in range(0, num_steps, int(num_steps / 10)):  # 在10个时间步上添加标注
    plt.text(i, toe_pos[i, 2], f'{i}', fontsize=8, ha='right', va='bottom')
# 设置标签和标题
plt.xlabel('Time Step')
plt.ylabel('Z Position')
plt.title('Toe Positions in Z Direction Over Time')

# 显示图例
plt.legend()

# 展示图形
plt.show()






# 组合轨迹
# 最终输出的末端位置是在世界系中，末端相对质心的位置
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
ref[:, 49:52] = arm_pos[:num_row - 1, :]
ref[:, 52:56] = arm_rot[:num_row - 1, :]
ref[:, 56:62] = arm_dof_pos[:num_row - 1, :]
ref[:, 62:68] = arm_dof_vel

# ref[100:200,:] = ref[:99,:]

# # 导出完整轨迹
# outfile = 'output_panda_fixed_gripper/panda_leg_with_arm_1.txt' # 初始角度不为0
# outfile = 'output_panda_fixed_gripper/panda_leg_with_arm_2.txt' # 初始角度为0
# outfile = 'output_panda_fixed_gripper/panda_leg_with_arm_3.txt' # 两段迈步
# outfile = 'output_panda_fixed_gripper/panda_leg_with_arm_4.txt' # 一段 末端不动
# np.savetxt(outfile, ref, delimiter=',')
# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_leg_with_arm.txt'
np.savetxt(outfile, ref, delimiter=',')
np.savetxt('output_panda_fixed_gripper/arm_dof_pos.txt', arm_dof_pos, delimiter=',')
# 保存json
files = 'output_panda_fixed_gripper'
file = "panda_leg_with_arm.txt"

name = file.split('.')[0]

motion = np.loadtxt(os.path.join(files, file), delimiter=',')
json_data = {
    'frame_duration': 1 / fps,
    'frames': motion.tolist()
}
with open(files + '_json/' + name + '.json', 'w') as f:
    json.dump(json_data, f, indent=4)

# outfile = 'output_panda_fixed_gripper/arm_dof_pos.txt'
# np.savetxt(outfile, arm_dof_pos, delimiter=',')
# # 保存json
# files = 'output_panda_fixed_gripper'
# file = "arm_dof_pos.txt"

# name = file.split('.')[0]
#
# motion = np.loadtxt(os.path.join(files, file), delimiter=',')
# json_data = {
#     'frame_duration': 1 / fps,
#     'frames': motion.tolist()
# }
# with open(files + '_json/' + name + '.json', 'w') as f:
#     json.dump(json_data, f, indent=4)
import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import matplotlib.pyplot as plt

# panda的打拍子动作，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7的顺序不一致，需要重新写motion_loader函数
# 这里都用这个顺序
# 关节角度和关节速度
# 额外添加末端位姿7维，关节角度8维角速度， 关节角速度8维

# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 150
num_col = 72
fps = 50

beat_ref = np.ones((num_row - 1, num_col))
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
arm_dof_pos = np.zeros((num_row, 8))
arm_dof_vel = np.zeros((num_row-1, 8))


# beat traj
def get_pos_beat(num_frames, tend):
    h = 0.12
    a1 = 0.01
    a2 = 0.01
    T = 0.6  # 周期0.6s
    traj = np.zeros((num_frames, 3))
    for item, t in enumerate(np.linspace(0, tend, num_frames)):
        n = np.floor((t - 0) / T)
        z = -4 * h * (t - (2 * n + 1) * T / 2) ** 2 + h
        if np.sin(2 * np.pi * t / T) > 0:
            x = -a1 * np.sin(2 * pi * t / T)
        else:
            x = -a2 * np.sin(2 * pi * t / T)
        traj[item, 0] = x
        traj[item, 2] = z

    return traj


# 质心轨迹
root_pos[:, 2] = 0.55
# 质心线速度 默认为0

# 姿态 默认为[0, 0, 0, 1]
root_rot[:] = np.array([0, 0, 0, 1])
# 质心角速度 默认为0

# 世界系下足端位置
toe_pos_init = panda7.toe_pos_init  # 默认足端位置
toe_pos[:] = toe_pos_init

toe_pos[:, 0:3] += get_pos_beat(num_row, num_row/fps)
plt.figure()
plt.plot(toe_pos[:, 0], toe_pos[:, 2])
# plt.show()
# 计算关节角度
# panda7的关节上下限
q = SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是世界系下的足端轨迹(也是质心系)，但质心位置和方向都保持在原点，正前方不变，所以欧拉角和质心都是[0, 0, 0]
        pos = panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe
        cost = 500 * dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=panda7.lb[3 * j:3 * j + 3], ubx=panda7.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps

# arm fk
robot_arm_rot, robot_arm_pos=utils.arm_fk([0, 0, 0, 0, 0, 0])
# 机械臂末端在机身坐标系下的位置
arm_pos[:] = robot_arm_pos
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i,:]) @ robot_arm_rot)



# 组合轨迹
beat_ref[:, :3] = root_pos[:num_row - 1, :]
beat_ref[:, 3:7] = root_rot[:num_row - 1, :]
beat_ref[:, 7:10] = root_lin_vel
beat_ref[:, 10:13] = root_ang_vel
beat_ref[:, 13:25] = toe_pos[:num_row - 1, :]
beat_ref[:, 25:37] = dof_pos[:num_row - 1, :]
beat_ref[:, 37:49] = dof_vel
beat_ref[:, 49:52] = arm_pos[:num_row - 1, :]
beat_ref[:, 52:56] = arm_rot[:num_row - 1, :]
beat_ref[:, 56:64] = arm_dof_pos[:num_row - 1, :]
beat_ref[:, 64:72] = arm_dof_vel

# 导出完整轨迹
outfile = 'output_panda/panda_beat.txt'
np.savetxt(outfile, beat_ref, delimiter=',')

# # 导出fixed arm轨迹
outfile = 'output_panda_fixed_arm/panda_beat.txt'
np.savetxt(outfile, beat_ref[:, :49], delimiter=',')

# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_beat.txt'
out = np.hstack((beat_ref[:, :56], beat_ref[:, 56:62], beat_ref[:, 64:70]))
np.savetxt(outfile, out, delimiter=',')

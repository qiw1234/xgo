import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import json

# 此轨迹仅用于机器人状态初始化，且仅能用于训练站立，其他动作不能使用此轨迹，需要删除
# 输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]



go2 = utils.QuadrupedRobot()
num_row = 80
num_col = 49
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


# 质心轨迹
root_pos[:, 2] = 0.075
# 质心线速度 默认为0

# 姿态
root_rot[:] = np.array([0, sin(-0.05), 0, cos(-0.05)]) # [0, -0.1, 0] zyx euler
# 四元数的导数

# 质心角速度

# 关节角度
dof_pos[:] =  [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,-0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
# 足端位置
# 计算足端位置在质心坐标系的坐标
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, :3], 0, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, 3:6] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, 3:6], 1, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, 6:9] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, :3], 2, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, 9:12] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, :3], 3, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps




# 组合轨迹
# 最终输出的末端位置是在世界系中，末端相对质心的位置
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel


# # 导出完整轨迹
outfile = 'output/down.txt'
np.savetxt(outfile, ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': ref.tolist()
}
with open('go2ST/down.json', 'w') as f:
    json.dump(json_data, f, indent=4)

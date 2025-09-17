import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import json

# 规划挥手的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]

# 挥手动作：右前腿抬起来，挥舞右前腿，轨迹轮廓在y-z平面来看类似一片银杏叶
# 包含两段轨迹，一是抬腿到指定位置，不在一个平面内，需要通过两个平面图形进行组合，二是在yz平面内挥舞
# 由于机器狗的手臂关节的结构，横着挥手感觉不太好看，所以我想着做一个招财猫的动作
# 具体来说就是身体仰起来30°，然后右手做招财猫的动作


go2 = utils.QuadrupedRobot()
num_row = 210
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
root_pos[:, 2] = 0.3
# 质心线速度 默认为0

# 姿态
q0 = [0, 0, 0, 1]
q1 = [0, np.sin(-np.pi / 24), 0, np.cos(-np.pi / 24)]
root_rot[:] = q1
end = 20 #0.4s
for i in range(end):
    frac = (i+1) / (end - 1)
    root_rot[i, :] = quaternion_slerp(q0, q1, frac)
start = end
end = 190

# 四元数的导数
for i in range(num_row-1):
    root_rot_dot[i,:] = (root_rot[i+1,:] - root_rot[i,:]) * fps
# 质心角速度
for i in range(num_row-1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i,:])@ root_rot_dot[i,:]

# 足端轨迹
# 除右前腿，其他的都是世界系的位置，都保持不动
# 这是默认的足端位置，坐标系是固定在root处的世界系
# 右前腿的动作通过关节角度规划
toe_pos[:] = go2.toe_pos_init
q_FR_0 = [-0.0905, 0.844, -1.107]  # 初始位置
q_FR_1 = [-0.1, -0.8, -1.5]  # 抬手中间位置
q_FR_2 = [-0.1, -0.8, -1]  # 抬手下方
q_FR_3 = [-0.1, -0.8, -2.5]  # 抬手上方
q_FL_0 = [0.1, 0.8, -1.5]
q_FL_1 = [-0.4, 0.8, -1.5]
q_FL_2 = [0.1, 0.8, -1.9]
# 右前腿关节角度
dof_pos[:20, :3] = q_FR_0
dof_pos[20:70, :3] = np.linspace(q_FR_0, q_FR_1, 50)
dof_pos[70:80, :3] = np.linspace(q_FR_1, q_FR_2, 10)
dof_pos[80:100, :3] = np.linspace(q_FR_2, q_FR_3, 20)
dof_pos[100:120, :3] = np.linspace(q_FR_3, q_FR_2, 20)
dof_pos[120:140, :3] = np.linspace(q_FR_2, q_FR_3, 20)
dof_pos[140:160, :3] = np.linspace(q_FR_3, q_FR_2, 20)
dof_pos[160:180, :3] = np.linspace(q_FR_2, q_FR_3, 20)
dof_pos[180:200, :3] = np.linspace(q_FR_3, q_FR_2, 20)
dof_pos[200:210, :3] = np.linspace(q_FR_2, q_FR_1, 10)
# 左前腿关节角度
dof_pos[:, 3:6] = q_FL_0
dof_pos[:8, 3:6] = np.linspace(q_FL_0, q_FL_2, 8)
dof_pos[8:20, 3:6] = np.linspace(q_FL_2, q_FL_1, 12)
dof_pos[20:, 3:6] = q_FL_1
# dof_pos[190:202, 3:6] = np.linspace(q_FL_1, q_FL_2, 12)
# dof_pos[202:210, 3:6] = np.linspace(q_FL_2, q_FL_0, 8)

# 计算足端位置在质心坐标系的坐标
for i in range(toe_pos.shape[0]):
    if i < 20:
        toe_pos[i, :3] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ go2.toe_pos_init[:3]
    else:
        toe_pos[i, :3] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, :3], 0, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, 3:6] = np.transpose(casadi.DM(go2.transrpy(dof_pos[i, 3:6], 1, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ toe_pos[i, 6:9]
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ toe_pos[i, 9:12]

# go2的关节上下限
q = SX.sym('q', 3, 1)
last_q = [0,0,0]
for j in range(4):
    for i in range(num_row):
        # 这里的toe_pos是世界系足端轨迹，需要考虑质心姿态，因此左乘一个质心姿态
        pos = (go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe)[:3]
        cost = 500 * casadi.dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))\
                + casadi.dot((q - last_q), (q - last_q))
        nlp = {'x': q, 'f': cost}
        S = casadi.nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=go2.lb[3 * j:3 * j + 3], ubx=go2.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        last_q = q_opt
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

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



# 导出完整轨迹
outfile = 'output/wave.txt'
np.savetxt(outfile, ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': ref.tolist()
}
with open('output_json/wave.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('go2ST/wave.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('/home/pcpc/webots_project/GO2/controllers/dog_supervisor/wave.json', 'w') as f:
    json.dump(json_data, f, indent=4)

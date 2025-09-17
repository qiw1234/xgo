import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import json

# 规划摇摆的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 机身右转30°，左转60°，右转60°，左转30°
go2 = utils.QuadrupedRobot()
num_row = 160
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
toe_pos_world = np.zeros((num_row, 12))


def h_1(t):
    x = 0.2  # 对称轴
    h = g / 2 * x ** 2
    return h - g / 2 * (t - x) ** 2


# 质心轨迹
body_height = 0.3
for i in range(20):
    root_pos[i, 2] = h_1(i / 50) + body_height
for i in range(20,40):
    root_pos[i, 2] = -0.5*h_1((i-20)/50) + body_height
for i in range(3):
    root_pos[40 * (i + 1):40 * (i + 2), :] = root_pos[:40, :]
# 质心线速度
for i in range(num_row - 1):
    root_lin_vel[i, :] = (root_pos[i + 1, :] - root_pos[i, :]) * fps
# 姿态
# 计算姿态有点小问题，但是影响不大
q1 = [0, 0, 0, 1]  # [x,y,z,w]
q2 = [0, 0, np.sin(-np.pi / 12), np.cos(-np.pi / 12)]
q3 = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
interval = 20
interval2 = 20
start = 0
end = start + interval

for i in range(end):
    frac = (i + 1 - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q2, frac)

start = end
end = start + interval2
root_rot[start:end, :] = q2

start = end
end = start + interval
for i in range(start, end):
    frac = (i + 1 - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q3, frac)

start = end
end = start + interval2
root_rot[start:end, :] = q3

start = end
end = start + interval
for i in range(start, end):
    frac = (i + 1 - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q3, q2, frac)

start = end
end = start + interval2
root_rot[start:end, :] = q2

start = end
end = start + interval
for i in range(start, end):
    frac = (i + 1 - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q1, frac)

start = end
end = start + interval2
root_rot[start:end, :] = q1

# 四元数的导数
for i in range(num_row - 1):
    root_rot_dot[i, :] = (root_rot[i + 1, :] - root_rot[i, :]) * fps
# 质心角速度
for i in range(num_row - 1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i, :]) @ root_rot_dot[i, :]

# 获取足端初始位置
toe_pos_world[:] = go2.toe_pos_init


# 足端轨迹,跳跃伸腿时间为0.04s
# for i in range(5):
#     toe_pos[i, 2] -= h_1(i / 50)
#     toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
# for i in range(5,15):
#     toe_pos[i,2] = toe_pos[2, 2]
#     toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
# for i in range(15,20):
#     toe_pos[i, 2] -= h_1(i / 50)
#     toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
# for i in range(3):
#     toe_pos[20 * (i + 1):20 * (i + 2), :] = toe_pos[:20, :]

# 世界系下足端轨迹 一点一点规划
def get_toepos(angle):
    toepos_rf = casadi.DM(go2.transrpy([-0.1, 0.8, -1.5], 0, [0, 0, angle], [0, 0, body_height]) @ go2.toe).full().T
    toepos_lf = casadi.DM(go2.transrpy([0.1, 0.8, -1.5], 1, [0, 0, angle], [0, 0, body_height]) @ go2.toe).full().T
    toepos_rh = casadi.DM(go2.transrpy([-0.1, 0.8, -1.5], 2, [0, 0, angle], [0, 0, body_height]) @ go2.toe).full().T
    toepos_lh = casadi.DM(go2.transrpy([0.1, 0.8, -1.5], 3, [0, 0, angle], [0, 0, body_height]) @ go2.toe).full().T
    return np.hstack(
        (toepos_rf.flatten()[:3], toepos_lf.flatten()[:3], toepos_rh.flatten()[:3], toepos_lh.flatten()[:3]))


# 朝向正前方足端x_y,只考虑xy，
toepos_0 = get_toepos(0)
# 右转30°的足端位置
toepos_1 = get_toepos(-np.pi / 6)
# 左转30°的足端位置
toepos_2 = get_toepos(np.pi / 6)

# 世界系下足端轨迹，这里说的世界系是固定在初始root处的世界坐标系
delta_h = h_1(4 / 50)
for i in range(5):
    toe_pos_world[i, :] = toepos_0

angle = np.linspace(0, -np.pi / 6, 10)
for i in range(5, 15):
    toe_pos_world[i, :] = get_toepos(angle[i - 5])
    toe_pos_world[i, 2] = toe_pos_world[i, 5] = toe_pos_world[i, 8] = toe_pos_world[i, 11] = root_pos[i, 2] - delta_h + go2.toe_pos_init[2]

for i in range(15, 45):
    toe_pos_world[i, :] = toepos_1

angle = np.linspace(-np.pi / 6, np.pi / 6, 10)
for i in range(45, 55):
    toe_pos_world[i, :] = get_toepos(angle[i - 45])
    toe_pos_world[i, 2] = toe_pos_world[i, 5] = toe_pos_world[i, 8] = toe_pos_world[i, 11] = root_pos[i, 2] - delta_h + go2.toe_pos_init[2]

for i in range(55, 85):
    toe_pos_world[i, :] = toepos_2

angle = np.linspace(np.pi / 6, -np.pi / 6, 10)
for i in range(85, 95):
    toe_pos_world[i, :] = get_toepos(angle[i - 85])
    toe_pos_world[i, 2] = toe_pos_world[i, 5] = toe_pos_world[i, 8] = toe_pos_world[i, 11] = root_pos[i, 2] - delta_h + go2.toe_pos_init[2]

for i in range(95, 125):
    toe_pos_world[i, :] = toepos_1

angle = np.linspace(-np.pi / 6, 0, 10)
for i in range(125, 135):
    toe_pos_world[i, :] = get_toepos(angle[i - 125])
    toe_pos_world[i, 2] = toe_pos_world[i, 5] = toe_pos_world[i, 8] = toe_pos_world[i, 11] = root_pos[i, 2] - delta_h + go2.toe_pos_init[2]

for i in range(135, 160):
    toe_pos_world[i, :] = toepos_0

# 质心系的足端位置
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ (toe_pos_world[i, :3] - root_pos[i, :])
    toe_pos[i, 3:6] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ (toe_pos_world[i, 3:6] - root_pos[i, :])
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ (toe_pos_world[i, 6:9] - root_pos[i, :])
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ (toe_pos_world[i, 9:12] - root_pos[i, :])

toe_pos_body = toe_pos[:]

# go2的关节上下限
q = SX.sym('q', 3, 1)
for j in range(4):
    for i in range(num_row):
        # toe_pos是质心系的足端位置，所以rpy以及质心位置都是[0 0 0]
        # 这里不把上面的部分合起来放到一个式子里是因为下面使用casadi，上面使用了numpy，混合运算会出问题
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500 * dot((toe_pos_body[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos_body[i, 3 * j:3 * j + 3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=go2.lb[3 * j:3 * j + 3], ubx=go2.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps


# 组合轨迹 质心系
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
a = root_pos[:num_row - 1, 2] - toe_pos[:num_row - 1, 2]
# # 导出完整轨迹
outfile = 'output/turn_and_jump.txt'
np.savetxt(outfile, ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': ref.tolist()
}
with open('output_json/turn_and_jump.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('go2ST/turn_and_jump.json', 'w') as f:
    json.dump(json_data, f, indent=4)
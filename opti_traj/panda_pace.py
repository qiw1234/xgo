import numpy as np
import utils
import casadi as ca
import CPG
import matplotlib.pyplot as plt
import os
import json

# 使用CPG模型设计左右平移的pace步态，使用hopf振荡器获得周期性的相位信号，根据这个相位信号计算足端轨迹

# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 800
num_col = 72
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
arm_dof_pos = np.zeros((num_row, 8))
arm_dof_vel = np.zeros((num_row-1, 8))
hopf_signal = np.zeros((num_row,8))

# CPG信号
initPos = np.array([-0.5, 0.5, -0.5, 0.5, 0, 0, 0, 0])
gait = 'pace' # trot spacetrot pace 改步态的时候initPos也需要改
cpg = CPG.cpgBuilder(initPos, gait=gait)

# 欧拉法获取振荡信号
t = np.linspace(0, num_row/fps, num_row)
temp = cpg.initPos.reshape(8,-1)

for i in range(num_row):
    v = cpg.hopf_osci(temp)
    temp += v/fps
    hopf_signal[i] = temp.flatten()
# 相位
phase = np.arctan2(hopf_signal[:,4:8], hopf_signal[:,0:4])
# 微操一下相位
phase[:] = 0
# 右侧腿
phase[:40, 0] = phase[:40, 2] = np.linspace(3.14, -3.14, 40)
phase[40:80, 0] = phase[40:80, 2] = np.linspace(3.14, -3.14, 40)
phase[80:100, 0] = phase[80:100, 2] = -3.14
# 左侧腿
phase[:20, 1] = phase[:20, 3] = np.linspace(0, -3.14, 20)
phase[20:60, 1] = phase[20:60, 3] = np.linspace(3.14, -3.14, 40)
phase[60:80, 1] = phase[60:80, 3] = np.linspace(3.14, 0, 20)
phase[80:100, 1] = phase[80:100, 3] = 0

for k in range(7):
    phase[100*(k+1):100*(k+2),:] = phase[:100,:]

# print(phase)
# plt.figure()
# plt.plot(t, hopf_signal[:,0], linewidth=5)
# # plt.plot(t, hopf_signal[:,4], linewidth=5)
# plt.plot(t, hopf_signal[:,1], linewidth=3)
# # plt.plot(t, hopf_signal[:,5], linewidth=2)
# plt.plot(t, hopf_signal[:,2], linewidth=3)
# plt.plot(t, hopf_signal[:,3], linewidth=3)
#
# plt.show()

if gait == 'spacetrot':
    # np.savetxt('phase.csv', phase, delimiter=',')
    phase = np.loadtxt('phase.csv', delimiter=',')

plt.figure()
plt.plot(t, phase[:,0], linewidth=6, c='g')
plt.plot(t, phase[:,1], linewidth=6, c='b')
plt.plot(t, phase[:,2], linewidth=2, c='r')
plt.plot(t, phase[:,3], linewidth=2, c='y')


# 足端位置
vx = 0

vy = np.zeros((num_row, ))
vy[:80] =  1.2
vy[100:180] = -1.2
for k in range(3):
    vy[200*(k+1):200*(k+2)] = vy[:200]


ax = vx*cpg.T*cpg.beta
ay = vy*cpg.T*cpg.beta
az = 0.12
az2 = 0.01
for i in range(4):
    for j in range(num_row):
        if phase[j,i]<0:
            p = -phase[j,i]/np.pi
            toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az, p)
        else:
            p = phase[j,i]/np.pi
            if gait == 'spacetrot':
                toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az2, p)
            else:
                toe_pos[j, 3 * i + 2] = 0
        if gait == 'spacetrot':
            toe_pos[j, 3*i] = CPG.endEffectorPos_xy_spacetrot(ax, p)
            toe_pos[j, 3*i+1] = CPG.endEffectorPos_xy_spacetrot(ay, p)
        else:
            toe_pos[j, 3*i] = CPG.endEffectorPos_xy(ax, p)
            toe_pos[j, 3*i+1] = CPG.endEffectorPos_xy(ay[j], p)

plt.figure()
plt.plot(toe_pos[:,1], toe_pos[:,2], linewidth=5)
plt.plot(t, toe_pos[:, 1], linewidth=5, marker='o')
plt.plot(t, toe_pos[:, 2], linewidth=5, marker='*')
plt.show()
# 足端相对质心的坐标
toe_pos += panda7.toe_pos_init
q = ca.SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe
        cost = 500*ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0 = [0.1, 0.8, -1.5], lbx = panda7.lb[3*j:3*j+3], ubx = panda7.ub[3*j:3*j+3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps

# 质心位置
if gait != 'spacetrot':
    for i in range(num_row-1):
        root_pos[i+1:,0] = vx/fps + root_pos[i, 0]
        root_pos[i+1:,1] = vy[i]/fps + root_pos[i, 1]
else:
    root_pos[:,0] = 0
    root_pos[:,1] = 0
root_pos[:,2] = 0.55
# 质心速度
if gait != 'spacetrot':
    root_lin_vel[:,0] = vx
    root_lin_vel[:, 1] = vy[:num_row-1]
else:
    root_lin_vel[:,0] = 0
    root_lin_vel[:, 1] = 0
# 机身方向
root_rot[:,3] = 1
# 机身角速度默认为0

# arm fk
robot_arm_rot, robot_arm_pos=utils.arm_fk([0, 0, 0, 0, 0, 0])
# 机械臂末端在机身坐标系下的位置
arm_pos[:] = robot_arm_pos
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i,:]) @ robot_arm_rot)

# 组合轨迹
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
ref[:, 49:52] = arm_pos[:num_row - 1, :]
ref[:, 52:56] = arm_rot[:num_row - 1, :]
ref[:, 56:64] = arm_dof_pos[:num_row - 1, :]
ref[:, 64:72] = arm_dof_vel

# 太空步
# gait = 'spacetrot'
# 导出完整轨迹
outfile = 'output_panda/panda_'+gait+'.txt'
np.savetxt(outfile, ref, delimiter=',')

# 导出fixed arm轨迹
outfile = 'output_panda_fixed_arm/panda_'+gait+'.txt'
np.savetxt(outfile, ref[:, :49], delimiter=',')

# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_'+gait+'.txt'
out = np.hstack((ref[:, :56], ref[:, 56:62], ref[:, 64:70]))
np.savetxt(outfile, out, delimiter=',')

# 保存json
# files = 'output_panda_fixed_gripper'
# file = "panda_spacetrot.txt"
# name = file.split('.')[0]
# motion = np.loadtxt(os.path.join(files,file), delimiter=',')
# json_data={
#     'frame_duration':1/fps,
#     'frames':motion.tolist()
# }
# with open(files+'_json/'+name+'.json', 'w') as f:
#     json.dump(json_data, f, indent=4)

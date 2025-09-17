import casadi as ca
import numpy as np
import CPG
import utils
import matplotlib.pyplot as plt

theta = ca.SX.sym('x')
rot_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -ca.sin(theta)], [0, ca.sin(theta), ca.cos(theta)]])
pos = [1, 2, 3]


# 3by3旋转矩阵变成4by4变换矩阵

# temp = ca.SX.zeros(4,4)
#
# print(temp[2, 2])
# print(rot_matrix)
# temp[:3, :3] = rot_matrix
# temp[2,2] = rot_matrix[2][2]
# temp[:3,3] = pos
# temp[3,:] = ca.SX([0, 0, 0, 1])
#
# pos1 = [2, 3, 4]
# print(ca.dot(pos1,pos1))
# a = ca.DM(pos1)
# b = ca.DM([[1, 2],[3, 4]])
# c = ca.DM(temp).full()
# print(a, b, c)

# 欧拉法获取振荡信号
num_row = 500
num_col = 49
fps = 100
go2 = utils.QuadrupedRobot()
initPos = np.array([0.5, -0.5, -0.5, 0.5, 0, 0, 0, 0])
cpg = CPG.cpgBuilder(initPos, gait='trot')
hopf_signal = np.zeros((num_row,8))

t = np.linspace(0, num_row/fps, num_row)
temp = cpg.initPos.reshape(8,-1)

for i in range(num_row):
    v = cpg.hopf_osci(temp)
    temp += v/fps
    hopf_signal[i] = temp.flatten()
# 相位
phase = np.arctan2(hopf_signal[:,4:8], hopf_signal[:,0:4])

# print(phase)
plt.figure()
plt.plot(t, hopf_signal[:,0], linewidth=5)
# plt.plot(t, hopf_signal[:,4], linewidth=5)
plt.plot(t, hopf_signal[:,1], linewidth=3)
# plt.plot(t, hopf_signal[:,5], linewidth=2)
plt.plot(t, hopf_signal[:,2], linewidth=3)
# plt.plot(t, hopf_signal[:,3], linewidth=3)

plt.show()

# plt.figure()
# plt.plot(t, phase[:,0], linewidth=5, linestyle='--')
# plt.plot(t, phase[:,1], linewidth=4)
# plt.plot(t, phase[:,2], linewidth=3)
# plt.plot(t, phase[:,3], linewidth=2)
# plt.show()





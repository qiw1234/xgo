from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import utils
import casadi as ca

go2 = utils.QuadrupedRobot()
# go2的关节上下限
lb = [-0.8378,-np.pi/2,-2.7227,-0.8378,-np.pi/2,-2.7227,-0.8378,-np.pi/6,-2.7227,-0.8378,-np.pi/6,-2.7227]
ub = [0.8378,3.4907,-0.8378,0.8378,3.4907,-0.8378,0.8378,4.5379,-0.8378,0.8378,4.5379,-0.8378]
q0 = np.linspace(lb[0], ub[0], 20)
q1 = np.linspace(lb[1], ub[1], 20)
q2 = np.linspace(lb[2], ub[2], 10)
x, y, z = [], [], []
toe_pos_r = [0.1934, -0.142, -0.3]
for i in q0:
    for j in q1:
        for k in q2:
            pos = go2.transrpy([i, j, k], 0, [0, 0, 0], [0, 0, 0]) @ go2.toe
            # a = ca.DM(pos)
            # b = a.full()
            x.append(ca.DM(pos[0]).full()[0][0] - toe_pos_r[0])# 这里是因为数据是个列向量
            y.append(ca.DM(pos[1]).full()[0][0] - toe_pos_r[1])
            z.append(ca.DM(pos[2]).full()[0][0] - toe_pos_r[2])




def scatter_3d(x, y, z):
    # 散点图
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    # s：marker标记的大小
    # c: 颜色  可为单个，可为序列
    # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
    # marker：样式
    ax.scatter(xs=x, ys=y, zs=z,
               zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker="^")
    ax.set_xlabel('x_label')
    ax.set_ylabel('y_label')
    ax.set_zlabel('z_label')
    ax.set_title('work space')

    plt.tight_layout()
    plt.show()


scatter_3d(x, y, z)






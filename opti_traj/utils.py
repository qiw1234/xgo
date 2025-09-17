import casadi as ca
import numpy as np


def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return e


def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return e


def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return e


def rot2trans(rot_matrix, pos):
    '''3by3旋转矩阵变成4by4变换矩阵
    '''
    # temp = np.zeros([4, 4])
    # temp[:3, :3] = rot_matrix[:3,:3]
    # temp[:3, 3] = pos
    # temp[3, :] = np.array([0, 0, 0, 1])
    # return temp
    temp = ca.SX.eye(4)
    temp[:3, :3] = rot_matrix
    temp[:3, 3] = pos
    # temp[3,:] = ca.SX([0, 0, 0, 1])
    return temp


def quaternion2rotm(quat):
    '''四元数变换成旋转矩阵

    Args:
        quat: x, y, z, w

    Returns:四元数对应的旋转矩阵

    '''
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    rotm = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return rotm


def rotm2quaternion(rotm):
    w = np.sqrt(1 + rotm[0, 0] + rotm[1, 1] + rotm[2, 2]) / 2.
    x = (rotm[2, 1] - rotm[1, 2]) / (4 * w)
    y = (rotm[0, 2] - rotm[2, 0]) / (4 * w)
    z = (rotm[1, 0] - rotm[0, 1]) / (4 * w)
    quat = [x, y, z, w]
    return quat


def quat2angvel_map(q):
    '''
    四元数的导数到角速度的映射矩阵
    Args:
        q: (x, y, z, w)

    Returns:
        返回映射矩阵，3*4，omega = 2*return @ q_dot(x,y,z,w)

    '''
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return np.array([[w, -z, y, -x], [z, w, -x, -y], [-y, x, w, -z]])


def arm_fk(theta):
    """
    这是panda7上的机械臂的正运动学
    根据输入的6个关节角度，计算机械臂末端的变换矩阵、旋转矩阵以及对应的四元数。
    参数:
    theta1, theta2, theta3, theta4, theta5, theta6: 6个关节角度，单位为弧度

    返回:
    ARM_end: 机械臂末端的齐次变换矩阵
    ARM_R: 机械臂末端的旋转矩阵
    quat: 机械臂末端姿态对应的四元数
    """
    # ------DH参数---------------
    ARM_DH = np.array([
        [-0.02, np.pi / 2, 0.1005, np.pi / 2 + theta[0]],
        [0.264, 0, 0, -theta[1]],
        [0.26078, 0, 0, 0.9172 * np.pi - theta[2]],
        [0.06, np.pi / 2, 0, 0.0828 * np.pi - theta[3]],
        [0, -np.pi / 2, -0.01047, -np.pi / 2 + theta[4]],
        [0, 0, 0.0285, 0]
    ])

    # ------机身质心到机械臂基座的变换矩阵---------------
    R = np.array([[0, -1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    P = np.array([[1, 0, 0, 0.332],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0.105],
                  [0, 0, 0, 1]])

    # 初始化变换矩阵列表
    ARM = [np.eye(4)]  # 初始变换矩阵为单位矩阵
    ARM[0] = np.dot(ARM[0], np.dot(P, R))  # 机身质心到机械臂基座的变换

    for i in range(len(ARM_DH)):
        a = ARM_DH[i, 0]
        alpha = ARM_DH[i, 1]
        d = ARM_DH[i, 2]
        theta = ARM_DH[i, 3]

        # 计算变换矩阵
        A = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha),
             a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha),
             a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

        # 累积变换矩阵
        ARM.append(np.dot(ARM[i], A))

    ARM_end = ARM[-1]
    ARM_R = ARM_end[0:3, 0:3]
    ARM_P = ARM_end[0:3, 3]
    return ARM_R, ARM_P


class QuadrupedRobot:
    def __init__(self, l=0.3868, w=0.093, l1=0.0955, l2=0.213, l3=0.213,
                 lb=[-0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 6, -2.7227,
                     -0.8378, -np.pi / 6, -2.7227],
                 ub=[0.8378, 3.4907, -0.8378, 0.8378, 3.4907, -0.8378, 0.8378, 4.5379, -0.8378, 0.8378,
                     4.5379, -0.8378],
                 toe_pos_init=[0.178, -0.173, -0.3, 0.178, 0.173, -0.3, -0.178, -0.173, -0.3, -0.178,
                               0.173, -0.3]
                 ):
        # go2模型参数
        self.L = l
        self.W = w
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.eye = np.eye(3)
        self.p0 = np.array([0, 0, 0])
        self.toe = np.array([self.l3, 0, 0, 1])
        self.lb = lb
        self.ub = ub
        self.toe_pos_init = toe_pos_init

    def rightfoot(self, q):
        "右侧腿末端到髋关节基坐标系的变换矩阵"
        trans01_right = rot2trans(rx(q[0]), self.p0) @ rot2trans(ry(np.pi / 2), self.p0)
        trans12_right = (rot2trans(rx(-np.pi / 2), self.p0) @
                         rot2trans(self.eye, np.array([0, 0, -self.l1])) @
                         rot2trans(rz(q[1]), self.p0))
        trans23_right = rot2trans(self.eye, np.array([self.l2, 0, 0])) @ rot2trans(rz(q[2]),
                                                                                   self.p0)
        return trans01_right @ trans12_right @ trans23_right

    def leftfoot(self, q):
        "左侧腿的末端到髋关节的基座标系的变换矩阵"
        trans01_left = rot2trans(rx(q[0]), self.p0) @ rot2trans(ry(np.pi / 2), self.p0)
        trans12_left = (rot2trans(rx(-np.pi / 2), self.p0) @
                        rot2trans(self.eye, np.array([0, 0, self.l1])) @
                        rot2trans(rz(q[1]), self.p0))
        trans23_left = rot2trans(self.eye, np.array([self.l2, 0, 0])) @ rot2trans(rz(q[2]),
                                                                                  self.p0)
        return trans01_left @ trans12_left @ trans23_left

    def trans(self, q, legnum):
        # "所有腿部的运动学变换矩阵,变换到形心处的机身坐标系下，就是还没考虑质心的位姿"
        # "legnum=0:FR  1:FL   2:HR    3:HL"
        # q表示关节角度
        if legnum == 0:
            trans_b0 = rot2trans(self.eye, np.array([self.L / 2, -self.W / 2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 1:
            trans_b0 = rot2trans(self.eye, np.array([self.L / 2, self.W / 2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        elif legnum == 2:
            trans_b0 = rot2trans(self.eye, np.array([-self.L / 2, -self.W / 2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 3:
            trans_b0 = rot2trans(self.eye, np.array([-self.L / 2, self.W / 2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        return trans

    def transrpy(self, q, legnum, rpy, p):
        '''
        考虑机身欧拉角的运动学模型
        legnum=0:FR  1:FL   2:HR    3:HL
        rpy=（r,p,y）,分别是绕x,y,z轴的转动角度
        p = (x,y,z) p表示形心的坐标3*1
        '''
        rotm = rz(rpy[2]) @ ry(rpy[1]) @ rx(rpy[0])
        transm = rot2trans(rotm, p)
        return transm @ self.trans(q, legnum)

# # 这是用来验证正运动学公式的
# go2 = QuadrupedRobot()
# toe_pos = np.array([go2.l3, 0, 0, 1])
# # dof_pos = np.array([0.8378, 2.48545, -2.18294])
#
# dof_pos = np.array([-0.1, 0.8, -1.5])
# #np中向量用一行表示
# root_rpy = np.array([0, 0, 0])
# root_pos = np.array([0, 0, 0.3])
# a = toe_pos.reshape(-1,1) # 求转置
# print(a.shape)  # (4, 1)
# print(toe_pos.shape)  # (4, )
# pos = go2.transrpy(dof_pos, 0, root_rpy, root_pos) @ toe_pos.reshape(-1, 1)
# print(pos)
# # quat = [0.2415334, 0.6404592, -0.5108982, 0.5200544 ]
# # rotm = quaternion2rotm(quat)
# # print(rotm)

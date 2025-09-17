import casadi as ca
import numpy as np


def rx(theta):
    """绕x轴旋转矩阵"""
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return e


def ry(theta):
    """绕y轴旋转矩阵"""
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return e


def rz(theta):
    """绕z轴旋转矩阵"""
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return e


def rot2trans(rot_matrix, pos):
    """3x3旋转矩阵转换为4x4变换矩阵"""
    temp = ca.SX.eye(4)
    temp[:3, :3] = rot_matrix
    temp[:3, 3] = pos
    return temp


def quaternion2rotm(quat):
    """四元数转换为旋转矩阵
    
    Args:
        quat: [x, y, z, w]
    
    Returns:
        四元数对应的旋转矩阵
    """
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    rotm = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return rotm


def rotm2quaternion(rotm):
    """旋转矩阵转换为四元数"""
    w = np.sqrt(1 + rotm[0, 0] + rotm[1, 1] + rotm[2, 2]) / 2.
    x = (rotm[2, 1] - rotm[1, 2]) / (4 * w)
    y = (rotm[0, 2] - rotm[2, 0]) / (4 * w)
    z = (rotm[1, 0] - rotm[0, 1]) / (4 * w)
    quat = [x, y, z, w]
    return quat


def quat2angvel_map(q):
    """四元数导数到角速度的映射矩阵
    
    Args:
        q: (x, y, z, w)
    
    Returns:
        映射矩阵，3x4，omega = 2 * return @ q_dot(x,y,z,w)
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return np.array([[w, -z, y, -x], [z, w, -x, -y], [-y, x, w, -z]])


class XGORobot:
    """XGO机器人运动学模型"""
    
    def __init__(self, 
                 # XGO机器人的运动学参数（从URDF文件提取）
                 l=0.14975,  # 前后腿距离
                 w=0.0446,   # 左右腿距离
                 l1=0.049813,  # hip到thigh的偏移量（y方向）
                 l2=0.0595,    # thigh长度
                 l3=0.0715,    # calf长度
                 # 关节限制（从URDF文件提取）
                 lb=[-0.8, -2.0, -2.69, -0.8, -2.0, -2.69, -0.8, -2.0, -2.69, -0.8, -2.0, -2.69],
                 ub=[0.8, 2.0, -0.9, 0.8, 2.0, -0.9, 0.8, 2.0, -0.9, 0.8, 2.0, -0.9],
                 # 初始脚端位置
                 toe_pos_init=[0.093049, -0.022303, -0.15, 0.093049, 0.022297, -0.15, 
                              -0.056701, -0.022303, -0.15, -0.056701, 0.022297, -0.15]
                 ):
        # XGO模型参数
        self.L = l  # 前后腿距离
        self.W = w  # 左右腿距离
        self.l1 = l1  # hip到thigh的偏移量
        self.l2 = l2  # thigh长度
        self.l3 = l3  # calf长度
        self.eye = np.eye(3)
        self.p0 = np.array([0, 0, 0])
        self.toe = np.array([0, 0, -self.l3, 1])  # 脚端在calf坐标系中的位置
        self.lb = lb  # 关节下限
        self.ub = ub  # 关节上限
        self.toe_pos_init = toe_pos_init  # 初始脚端位置
        
        # XGO特有的偏移参数（从URDF提取）
        self.hip_offset = np.array([-0.018825, 0, 0])  # hip到thigh的x偏移
        self.foot_offset = np.array([-0.022663, -0.018825, 0])  # calf到foot的偏移

    def rightfoot(self, q):
        """右侧腿末端到髋关节基坐标系的变换矩阵"""
        # hip关节：绕x轴旋转
        trans01_right = rot2trans(rx(q[0]), self.p0)
        
        # thigh关节：有偏移 + 绕y轴旋转
        trans12_right = (rot2trans(self.eye, self.hip_offset) @
                        rot2trans(self.eye, np.array([0, -self.l1, 0])) @
                        rot2trans(ry(q[1]), self.p0))
        
        # calf关节：thigh长度偏移 + 绕y轴旋转
        trans23_right = (rot2trans(self.eye, np.array([0, 0, -self.l2])) @
                        rot2trans(ry(q[2]), self.p0))
        
        return trans01_right @ trans12_right @ trans23_right

    def leftfoot(self, q):
        """左侧腿末端到髋关节基坐标系的变换矩阵"""
        # hip关节：绕x轴旋转
        trans01_left = rot2trans(rx(q[0]), self.p0)
        
        # thigh关节：有偏移 + 绕y轴旋转
        trans12_left = (rot2trans(self.eye, self.hip_offset) @
                       rot2trans(self.eye, np.array([0, self.l1, 0])) @
                       rot2trans(ry(q[1]), self.p0))
        
        # calf关节：thigh长度偏移 + 绕y轴旋转
        trans23_left = (rot2trans(self.eye, np.array([0, 0, -self.l2])) @
                       rot2trans(ry(q[2]), self.p0))
        
        return trans01_left @ trans12_left @ trans23_left

    def trans(self, q, legnum):
        """所有腿部的运动学变换矩阵，变换到机身坐标系
        
        Args:
            q: 关节角度 [hip, thigh, calf]
            legnum: 腿编号 0:FR, 1:FL, 2:BR, 3:BL
        
        Returns:
            变换矩阵
        """
        if legnum == 0:  # FR - 前右腿
            trans_b0 = rot2trans(self.eye, np.array([self.L/2, -self.W/2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 1:  # FL - 前左腿
            trans_b0 = rot2trans(self.eye, np.array([self.L/2, self.W/2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        elif legnum == 2:  # BR - 后右腿
            trans_b0 = rot2trans(self.eye, np.array([-self.L/2, -self.W/2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 3:  # BL - 后左腿
            trans_b0 = rot2trans(self.eye, np.array([-self.L/2, self.W/2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        
        return trans

    def transrpy(self, q, legnum, rpy, p):
        """考虑机身欧拉角的运动学模型
        
        Args:
            q: 关节角度 [hip, thigh, calf]
            legnum: 腿编号 0:FR, 1:FL, 2:BR, 3:BL
            rpy: 欧拉角 (roll, pitch, yaw)
            p: 机身位置 (x, y, z)
        
        Returns:
            考虑机身姿态的变换矩阵
        """
        rotm = rz(rpy[2]) @ ry(rpy[1]) @ rx(rpy[0])
        transm = rot2trans(rotm, p)
        return transm @ self.trans(q, legnum)

    def get_foot_position(self, q, legnum, rpy=np.array([0, 0, 0]), p=np.array([0, 0, 0])):
        """获取脚端位置
        
        Args:
            q: 关节角度 [hip, thigh, calf]
            legnum: 腿编号 0:FR, 1:FL, 2:BR, 3:BL
            rpy: 机身欧拉角
            p: 机身位置
        
        Returns:
            脚端位置 [x, y, z]
        """
        trans_matrix = self.transrpy(q, legnum, rpy, p)
        foot_pos_homo = trans_matrix @ self.toe.reshape(-1, 1)
        return foot_pos_homo[:3, 0]

    def inverse_kinematics(self, target_pos, legnum, rpy=np.array([0, 0, 0]), p=np.array([0, 0, 0])):
        """逆运动学求解（简化版本）
        
        Args:
            target_pos: 目标脚端位置 [x, y, z]
            legnum: 腿编号
            rpy: 机身欧拉角
            p: 机身位置
        
        Returns:
            关节角度 [hip, thigh, calf]
        """
        # 这里可以实现逆运动学求解算法
        # 暂时返回零角度，实际应用中需要实现具体的逆运动学算法
        return np.array([0.0, 0.8, -1.5])


# 测试代码
if __name__ == "__main__":
    # 创建XGO机器人实例
    xgo = XGORobot()
    
    # 测试正运动学
    dof_pos = np.array([-0.1, 0.8, -1.5])  # 示例关节角度
    root_rpy = np.array([0, 0, 0])
    root_pos = np.array([0, 0, 0.15])  # XGO的典型高度
    
    # 计算FR腿的脚端位置
    foot_pos = xgo.get_foot_position(dof_pos, 0, root_rpy, root_pos)
    print(f"FR腿脚端位置: {foot_pos}")
    
    # 计算所有腿的脚端位置
    for leg in range(4):
        pos = xgo.get_foot_position(dof_pos, leg, root_rpy, root_pos)
        leg_names = ['FR', 'FL', 'BR', 'BL']
        print(f"{leg_names[leg]}腿脚端位置: {pos}") 
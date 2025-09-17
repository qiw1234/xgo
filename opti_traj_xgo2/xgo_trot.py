import numpy as np
import xgo_utils
import casadi as ca
import CPG
import matplotlib.pyplot as plt
import json

# 使用CPG模型设计trot步态，使用hopf振荡器获得周期性的相位信号，根据这个相位信号计算足端轨迹

num_row = 100
num_col = 49
fps = 50

trot_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))
hopf_signal = np.zeros((num_row,8))

# 使用XGO机器人模型
xgo = xgo_utils.XGORobot()
initPos = np.array([-0.5, 0.5, 0.5, -0.5, 0, 0, 0, 0])
gait = 'trot'
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

print("生成CPG相位信号...")
plt.rcParams['toolbar'] = 'none'
plt.figure()
plt.plot(t, phase[:,0], linewidth=6, label='FR')
plt.plot(t, phase[:,1], linewidth=6, label='FL')
plt.plot(t, phase[:,2], linewidth=2, label='BR')
plt.plot(t, phase[:,3], linewidth=2, label='BL')
plt.legend()
plt.title('XGO步态相位信号')
plt.xlabel('时间 (s)')
plt.ylabel('相位 (rad)')
plt.show()

# 足端位置计算（适用于XGO的步长和高度）
vx = 0.3  # XGO的前进速度，调整为更适合小型机器人的速度
ax = vx*cpg.T*cpg.beta  # 步长
ay = 0  # 侧向步长
az = 0.05  # 抬腿高度，适合XGO的尺寸

print("计算足端轨迹...")
for i in range(4):
    for j in range(num_row):
        if phase[j,i] < 0:
            p = -phase[j,i]/np.pi
            toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az, p)
        else:
            p = phase[j,i]/np.pi
            toe_pos[j,3*i] = 0

        toe_pos[j,3*i] = CPG.endEffectorPos_xy(ax, p)
        toe_pos[j,3*i+1] = CPG.endEffectorPos_xy(ay, p)

# 显示足端轨迹
plt.figure()
plt.plot(toe_pos[:,0], toe_pos[:,2], linewidth=5, label='FR leg trajectory')
plt.xlabel('X位置 (m)')
plt.ylabel('Z位置 (m)')
plt.title('XGO前右腿足端轨迹')
plt.legend()
plt.grid(True)
plt.show()

# 足端相对质心的坐标
toe_pos += np.array(xgo.toe_pos_init).reshape(1, -1)

print("求解逆运动学...")
q = ca.SX.sym('q', 3, 1)

for j in range(4):
    print(f"处理第{j+1}条腿...")
    for i in range(num_row):
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = xgo.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ xgo.toe
        cost = 500*ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        
        nlp = {'x': q, 'f': cost}
        opts = {'ipopt.print_level': 0, 'print_time': 0}  # 减少输出信息
        S = ca.nlpsol('S', 'ipopt', nlp, opts)
        
        # 使用XGO的关节限制
        r = S(x0=[0.0, 0.8, -1.5], 
              lbx=xgo.lb[3*j:3*j+3], 
              ubx=xgo.ub[3*j:3*j+3])
        q_opt = r['x']
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
print("计算关节角速度...")
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps

# 质心位置（适合XGO的高度）
x = vx*t
root_pos[:,0] = x
root_pos[:,2] = 0.15  # XGO的典型高度，比GO2低一些

# 质心速度
root_lin_vel[:,0] = vx

# 机身方向（四元数：w, x, y, z）
root_rot[:,3] = 1  # w分量为1，表示无旋转

# 机身角速度默认为0

print("组合轨迹数据...")
# 组合轨迹
trot_ref[:, :3] = root_pos[:num_row-1,:]
trot_ref[:, 3:7] = root_rot[:num_row-1,:]
trot_ref[:, 7:10] = root_lin_vel
trot_ref[:, 10:13] = root_ang_vel
trot_ref[:, 13:25] = toe_pos[:num_row-1,:]
trot_ref[:, 25:37] = dof_pos[:num_row-1,:]
trot_ref[:, 37:49] = dof_vel

# 创建输出目录
import os
os.makedirs('output', exist_ok=True)
os.makedirs('output_json', exist_ok=True)

# 导出txt
outfile = 'output/xgo_' + gait + '.txt'
np.savetxt(outfile, trot_ref, delimiter=',')
print(f"轨迹数据已保存到: {outfile}")

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': trot_ref.tolist()
}

with open('output_json/xgo_' + gait + '.json', 'w') as f:
    json.dump(json_data, f, indent=4)
print(f"JSON数据已保存到: output_json/xgo_{gait}.json")

print("XGO步态生成完成！")

# 显示一些统计信息
print(f"\n轨迹统计信息:")
print(f"总帧数: {num_row}")
print(f"帧率: {fps} Hz")
print(f"总时间: {num_row/fps:.2f} 秒")
print(f"前进速度: {vx:.2f} m/s")
print(f"步长: {ax:.3f} m")
print(f"抬腿高度: {az:.3f} m")
print(f"机身高度: {root_pos[0,2]:.3f} m")

# 显示关节角度范围
print(f"\n关节角度范围:")
for i in range(4):
    leg_names = ['FR', 'FL', 'BR', 'BL']
    print(f"{leg_names[i]}腿:")
    for j in range(3):
        joint_names = ['Hip', 'Thigh', 'Calf']
        angles = dof_pos[:, 3*i+j]
        print(f"  {joint_names[j]}: [{np.min(angles):.3f}, {np.max(angles):.3f}] rad") 
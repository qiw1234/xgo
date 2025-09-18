#!/usr/bin/env python3
"""
XGO Sim2Sim MuJoCo Deployment Script
使用训练好的ONNX模型在MuJoCo环境中进行仿真

依赖:
- mujoco
- onnxruntime
- numpy
- torch (用于观察值预处理)
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import time
import argparse
from pathlib import Path

class XGOSim2SimController:
    def __init__(self, onnx_model_path, mujoco_xml_path):
        """
        初始化XGO Sim2Sim控制器
        
        Args:
            onnx_model_path: ONNX模型文件路径
            mujoco_xml_path: MuJoCo XML模型文件路径
        """
        self.onnx_model_path = onnx_model_path
        self.mujoco_xml_path = mujoco_xml_path
        
        # 加载ONNX模型
        print(f"Loading ONNX model from: {onnx_model_path}")
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
        # 获取模型输入输出信息
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_name = self.ort_session.get_outputs()[0].name
        
        print(f"Model input shape: {self.input_shape}")
        print(f"Model input name: {self.input_name}")
        print(f"Model output name: {self.output_name}")
        
        # 加载MuJoCo模型
        print(f"Loading MuJoCo model from: {mujoco_xml_path}")
        self.model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取关节信息
        self.num_joints = self.model.nq - 7  # 减去base的7个自由度（位置+四元数）
        self.joint_names = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and 'base' not in joint_name:  # 排除base关节
                self.joint_names.append(joint_name)
        
        print(f"Found {len(self.joint_names)} actuated joints: {self.joint_names}")
        
        # 初始化观察值缓存
        self.obs_history = []
        self.max_history = 3  # 保存最近3帧的观察值
        
        # XGO机器人参数（根据训练配置调整）
        self.obs_scales = {
            'lin_vel': 10.0,  # 从配置文件中获取
            'ang_vel': 0.5,   # 从配置文件中获取
            'dof_pos': 1.0,
            'dof_vel': 0.1,   # 从配置文件中获取
            'height_measurements': 5.0
        }
        
        # 动作缩放
        self.action_scale = 0.25
        self.default_dof_pos = np.zeros(self.num_joints)
        
    def get_observations(self):
        """
        获取机器人当前状态的观察值，格式与训练时一致
        
        Returns:
            观察值numpy数组
        """
        obs = []
        
        # 1. Base线速度 (3维)
        base_lin_vel = self.data.qvel[:3] * self.obs_scales['lin_vel']
        obs.extend(base_lin_vel)
        
        # 2. Base角速度 (3维)
        base_ang_vel = self.data.qvel[3:6] * self.obs_scales['ang_vel']
        obs.extend(base_ang_vel)
        
        # 3. 重力向量在base坐标系下的投影 (3维)
        # 通过base姿态计算重力向量
        base_quat = self.data.qpos[3:7]  # [qw, qx, qy, qz] in MuJoCo
        # 转换为旋转矩阵
        R = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R, base_quat)
        gravity_vec = R.T @ np.array([0, 0, -1])  # 重力向量在base系下
        obs.extend(gravity_vec)
        
        # 4. 关节位置 (12维)
        dof_pos = (self.data.qpos[7:7+self.num_joints] - self.default_dof_pos) * self.obs_scales['dof_pos']
        obs.extend(dof_pos)
        
        # 5. 关节速度 (12维)
        dof_vel = self.data.qvel[6:6+self.num_joints] * self.obs_scales['dof_vel']
        obs.extend(dof_vel)
        
        # 6. 上一次的动作 (12维) - 如果有历史记录
        if hasattr(self, 'last_actions'):
            obs.extend(self.last_actions)
        else:
            obs.extend(np.zeros(self.num_joints))
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        """
        执行动作并更新仿真
        
        Args:
            actions: 动作数组
        """
        # 缩放动作
        scaled_actions = actions * self.action_scale + self.default_dof_pos
        
        # 设置关节目标位置（PD控制）
        self.data.ctrl[:self.num_joints] = scaled_actions
        
        # 保存动作用于下一次观察
        self.last_actions = actions.copy()
        
        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        """重置仿真环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始姿态
        self.data.qpos[2] = 0.25  # 设置base高度
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 设置base姿态为单位四元数
        
        # 初始化关节位置（根据配置文件设置默认角度）
        default_angles = [0.1, 0.6, -1.5,  # fl_hip, fl_thigh, fl_calf
                         -0.1, 0.6, -1.5,  # fr_hip, fr_thigh, fr_calf
                          0.1, 0.6, -1.5,  # bl_hip, bl_thigh, bl_calf
                         -0.1, 0.6, -1.5]  # br_hip, br_thigh, br_calf
        self.data.qpos[7:7+self.num_joints] = default_angles
        self.default_dof_pos = np.array(default_angles)
        
        # 前向运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 清空历史记录
        self.obs_history.clear()
        if hasattr(self, 'last_actions'):
            delattr(self, 'last_actions')
    
    def run_simulation(self, duration=30.0, viewer=True):
        """
        运行仿真
        
        Args:
            duration: 仿真时长（秒）
            viewer: 是否显示可视化界面
        """
        print("Starting simulation...")
        
        # 重置环境
        self.reset()
        
        if viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step_count = 0
                
                while time.time() - start_time < duration:
                    # 获取观察值
                    obs = self.get_observations()
                    
                    # 使用ONNX模型预测动作
                    obs_input = obs.reshape(1, -1)  # 添加batch维度
                    ort_inputs = {self.input_name: obs_input}
                    actions = self.ort_session.run([self.output_name], ort_inputs)[0]
                    actions = actions.flatten()
                    
                    # 执行动作
                    self.step(actions)
                    
                    # 更新视图
                    viewer.sync()
                    
                    step_count += 1
                    if step_count % 500 == 0:
                        print(f"Simulation step: {step_count}, Time: {time.time() - start_time:.2f}s")
                    
                    # 控制仿真频率
                    time.sleep(self.model.opt.timestep)
        else:
            # 无界面模式
            start_time = time.time()
            step_count = 0
            
            while time.time() - start_time < duration:
                # 获取观察值
                obs = self.get_observations()
                
                # 使用ONNX模型预测动作
                obs_input = obs.reshape(1, -1)
                ort_inputs = {self.input_name: obs_input}
                actions = self.ort_session.run([self.output_name], ort_inputs)[0]
                actions = actions.flatten()
                
                # 执行动作
                self.step(actions)
                
                step_count += 1
                if step_count % 500 == 0:
                    print(f"Simulation step: {step_count}, Time: {time.time() - start_time:.2f}s")
        
        print("Simulation completed!")

def main():
    parser = argparse.ArgumentParser(description='XGO Sim2Sim MuJoCo Deployment')
    parser.add_argument('--onnx_model', type=str, 
                        default='logs/xgo/Sep17_11-20-53_/model_10000.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--mujoco_xml', type=str,
                        default='simple_xgo.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--no_viewer', action='store_true',
                        help='Run without viewer')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model file not found: {args.onnx_model}")
        sys.exit(1)
    
    if not os.path.exists(args.mujoco_xml):
        print(f"Error: MuJoCo XML file not found: {args.mujoco_xml}")
        sys.exit(1)
    
    # 创建控制器
    controller = XGOSim2SimController(args.onnx_model, args.mujoco_xml)
    
    # 运行仿真
    controller.run_simulation(duration=args.duration, viewer=not args.no_viewer)

if __name__ == "__main__":
    main()
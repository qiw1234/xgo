#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Gym 到 MuJoCo 策略模型转换脚本
将 Isaac Gym 训练的 XGO 策略转换为 MuJoCo 仿真可用的格式

作者: AI Assistant
日期: 2024
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append('/home/ubuntu/XGO-Simulation')
from legged_gym import LEGGED_GYM_ROOT_DIR


class ObservationMapper:
    """观测空间映射器 - 将 Isaac Gym 观测转换为 MuJoCo 观测"""
    
    def __init__(self):
        # Isaac Gym 观测空间 (48维):
        # 0:3   - base_lin_vel (被设为0)
        # 3:6   - base_ang_vel  
        # 6:9   - projected_gravity
        # 9:12  - commands[:3]
        # 12:24 - (dof_pos - default_dof_pos)
        # 24:36 - dof_vel
        # 36:48 - actions
        
        # MuJoCo 观测空间 (50维):
        # 0:2   - 时间信息 (sin, cos)
        # 2:5   - commands
        # 5:17  - (dof_pos - default_dof_pos) 
        # 17:29 - dof_vel
        # 29:41 - actions
        # 41:44 - base_ang_vel
        # 44:47 - euler_angles
        # 47:50 - projected_gravity
        
        self.isaac_obs_dim = 48
        self.mujoco_obs_dim = 50
        
    def isaac_to_mujoco(self, isaac_obs, time_step=0.0):
        """
        将 Isaac Gym 观测转换为 MuJoCo 观测
        
        Args:
            isaac_obs: Isaac Gym 观测 [batch_size, 48]
            time_step: 当前时间步 (用于生成时间信息)
        
        Returns:
            mujoco_obs: MuJoCo 观测 [batch_size, 50]
        """
        batch_size = isaac_obs.shape[0]
        mujoco_obs = torch.zeros(batch_size, self.mujoco_obs_dim, dtype=isaac_obs.dtype, device=isaac_obs.device)
        
        # 时间信息 (0:2)
        mujoco_obs[:, 0] = torch.sin(2 * torch.pi * time_step / 0.64)
        mujoco_obs[:, 1] = torch.cos(2 * torch.pi * time_step / 0.64)
        
        # 命令 (2:5) <- Isaac (9:12)
        mujoco_obs[:, 2:5] = isaac_obs[:, 9:12]
        
        # 关节位置 (5:17) <- Isaac (12:24)
        mujoco_obs[:, 5:17] = isaac_obs[:, 12:24]
        
        # 关节速度 (17:29) <- Isaac (24:36)
        mujoco_obs[:, 17:29] = isaac_obs[:, 24:36]
        
        # 动作 (29:41) <- Isaac (36:48)
        mujoco_obs[:, 29:41] = isaac_obs[:, 36:48]
        
        # 角速度 (41:44) <- Isaac (3:6)
        mujoco_obs[:, 41:44] = isaac_obs[:, 3:6]
        
        # 欧拉角 (44:47) - 从重力向量估算
        gravity = isaac_obs[:, 6:9]
        # 简化处理：从重力向量估算欧拉角
        roll = torch.atan2(gravity[:, 1], gravity[:, 2])
        pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1]**2 + gravity[:, 2]**2))
        yaw = torch.zeros_like(roll)  # 假设yaw为0
        mujoco_obs[:, 44] = roll
        mujoco_obs[:, 45] = pitch
        mujoco_obs[:, 46] = yaw
        
        # 重力向量 (47:50) <- Isaac (6:9)
        mujoco_obs[:, 47:50] = isaac_obs[:, 6:9]
        
        return mujoco_obs


class MuJoCoCompatiblePolicy(torch.nn.Module):
    """MuJoCo 兼容的策略包装器"""
    
    def __init__(self, isaac_policy, obs_mapper):
        super().__init__()
        self.isaac_policy = isaac_policy
        self.obs_mapper = obs_mapper
        self.time_step = 0
        
    def forward(self, mujoco_obs):
        """
        前向传播
        
        Args:
            mujoco_obs: MuJoCo 观测 [batch_size, 50]
        
        Returns:
            action: 动作输出 [batch_size, 12]
        """
        # 将 MuJoCo 观测转换回 Isaac Gym 格式
        batch_size = mujoco_obs.shape[0]
        isaac_obs = torch.zeros(batch_size, 48, dtype=mujoco_obs.dtype, device=mujoco_obs.device)
        
        # 线速度设为0 (0:3)
        isaac_obs[:, 0:3] = 0.0
        
        # 角速度 (3:6) <- MuJoCo (41:44)
        isaac_obs[:, 3:6] = mujoco_obs[:, 41:44]
        
        # 重力向量 (6:9) <- MuJoCo (47:50)
        isaac_obs[:, 6:9] = mujoco_obs[:, 47:50]
        
        # 命令 (9:12) <- MuJoCo (2:5)
        isaac_obs[:, 9:12] = mujoco_obs[:, 2:5]
        
        # 关节位置 (12:24) <- MuJoCo (5:17)
        isaac_obs[:, 12:24] = mujoco_obs[:, 5:17]
        
        # 关节速度 (24:36) <- MuJoCo (17:29)
        isaac_obs[:, 24:36] = mujoco_obs[:, 17:29]
        
        # 动作 (36:48) <- MuJoCo (29:41)
        isaac_obs[:, 36:48] = mujoco_obs[:, 29:41]
        
        # 使用原始 Isaac Gym 策略进行推理
        action = self.isaac_policy(isaac_obs)
        
        return action


def find_latest_model(log_dir):
    """查找最新的模型文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")
    
    # 查找所有训练目录
    train_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith(('Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'))]
    
    if not train_dirs:
        raise FileNotFoundError(f"在 {log_dir} 中未找到训练目录")
    
    # 按修改时间排序，获取最新的
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    # 查找最新的模型文件
    model_files = list(latest_dir.glob("model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"在 {latest_dir} 中未找到模型文件")
    
    # 按模型编号排序，获取最新的
    def get_model_number(path):
        try:
            return int(path.stem.split('_')[1])
        except:
            return 0
    
    latest_model = max(model_files, key=get_model_number)
    return str(latest_model)


def load_isaac_model(model_path):
    """加载 Isaac Gym 训练的模型"""
    print(f"正在加载 Isaac Gym 模型: {model_path}")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print(f"检查点信息:")
        print(f"  - 训练步数: {checkpoint.get('iter', 'N/A')}")
        print(f"  - 模型参数数量: {len(model_state_dict)}")
    else:
        raise ValueError(f"无效的模型格式: {type(checkpoint)}")
    
    return model_state_dict


def create_mujoco_policy(isaac_state_dict, obs_mapper):
    """创建 MuJoCo 兼容的策略网络"""
    
    # 创建策略网络结构 (基于 XGO 配置)
    class ActorNetwork(torch.nn.Module):
        def __init__(self, input_dim=48, output_dim=12, hidden_dims=[256, 128, 64]):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    torch.nn.Linear(prev_dim, hidden_dim),
                    torch.nn.ELU()
                ])
                prev_dim = hidden_dim
            
            layers.append(torch.nn.Linear(prev_dim, output_dim))
            self.network = torch.nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    # 创建网络实例
    actor = ActorNetwork()
    
    # 加载权重 - 正确映射 actor 参数
    actor_state_dict = actor.state_dict()
    mapped_params = {}
    
    # 映射 actor 网络参数
    actor_param_mapping = {
        'actor.0.weight': 'network.0.weight',
        'actor.0.bias': 'network.0.bias',
        'actor.2.weight': 'network.2.weight',
        'actor.2.bias': 'network.2.bias',
        'actor.4.weight': 'network.4.weight',
        'actor.4.bias': 'network.4.bias',
        'actor.6.weight': 'network.6.weight',
        'actor.6.bias': 'network.6.bias',
    }
    
    for isaac_name, target_name in actor_param_mapping.items():
        if isaac_name in isaac_state_dict and target_name in actor_state_dict:
            mapped_params[target_name] = isaac_state_dict[isaac_name]
    
    try:
        actor.load_state_dict(mapped_params, strict=False)
        print(f"✓ 成功加载权重，映射 {len(mapped_params)}/{len(actor_state_dict)} 个参数")
    except Exception as e:
        print(f"✗ 加载权重失败: {e}")
    
    # 创建 MuJoCo 兼容包装器
    mujoco_policy = MuJoCoCompatiblePolicy(actor, obs_mapper)
    
    return mujoco_policy


def convert_and_save(input_path, output_path):
    """转换并保存模型"""
    
    print("=== Isaac Gym 到 MuJoCo 策略转换 ===")
    
    # 1. 加载 Isaac Gym 模型
    isaac_state_dict = load_isaac_model(input_path)
    
    # 2. 创建观测映射器
    obs_mapper = ObservationMapper()
    
    # 3. 创建 MuJoCo 兼容策略
    mujoco_policy = create_mujoco_policy(isaac_state_dict, obs_mapper)
    
    # 4. 测试模型
    print("\n正在测试转换后的模型...")
    mujoco_policy.eval()
    
    with torch.no_grad():
        # 创建测试输入 (MuJoCo 格式)
        test_input = torch.randn(1, 50)
        test_output = mujoco_policy(test_input)
        
        print(f"✓ 测试成功:")
        print(f"  输入维度: {test_input.shape}")
        print(f"  输出维度: {test_output.shape}")
        print(f"  输出范围: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    # 5. 转换为 TorchScript 并保存
    print(f"\n正在保存模型到: {output_path}")
    
    # 创建示例输入用于 tracing
    example_input = torch.randn(1, 50)
    
    try:
        # 使用 torch.jit.trace 转换
        traced_model = torch.jit.trace(mujoco_policy, example_input)
        traced_model.save(output_path)
        print("✓ 成功保存为 TorchScript 格式")
    except Exception as e:
        print(f"TorchScript 转换失败: {e}")
        # 降级保存为普通 PyTorch 模型
        torch.save(mujoco_policy.state_dict(), output_path.replace('.pt', '_state_dict.pt'))
        torch.save(mujoco_policy, output_path.replace('.pt', '_full_model.pt'))
        print("✓ 保存为普通 PyTorch 格式")
    
    print("\n=== 转换完成 ===")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Isaac Gym 到 MuJoCo 策略转换器')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入的 Isaac Gym 模型路径 (默认自动查找最新模型)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出的 MuJoCo 模型路径 (默认: converted_policy_mujoco.pt)')
    parser.add_argument('--log_dir', type=str, default='/home/ubuntu/XGO-Simulation/logs/xgo',
                        help='训练日志目录')
    
    args = parser.parse_args()
    
    try:
        # 确定输入模型路径
        if args.input is None:
            input_path = find_latest_model(args.log_dir)
            print(f"自动找到最新模型: {input_path}")
        else:
            input_path = args.input
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入模型不存在: {input_path}")
        
        # 确定输出路径
        if args.output is None:
            output_dir = os.path.join(args.log_dir, 'mujoco_converted')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'xgo_policy_mujoco.pt')
        else:
            output_path = args.output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 执行转换
        result_path = convert_and_save(input_path, output_path)
        
        print(f"\n🎉 转换成功！")
        print(f"📁 输入: {input_path}")
        print(f"📁 输出: {result_path}")
        print(f"\n使用方法:")
        print(f"python legged_gym/scripts/sim2sim_xgo.py --load_model {result_path}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试转换后的模型
"""

import torch
import sys
import os

sys.path.append('/home/ubuntu/XGO-Simulation')

def test_converted_model():
    """测试转换后的模型"""
    
    model_path = '/home/ubuntu/XGO-Simulation/logs/xgo/mujoco_converted/xgo_policy_mujoco.pt'
    
    print("=== 转换模型测试 ===")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先运行转换脚本:")
        print("python scripts/convert_isaac_to_mujoco.py")
        return False
    
    try:
        # 加载模型
        policy = torch.jit.load(model_path, map_location='cpu')
        print("✅ 模型加载成功")
        
        # 测试单次推理
        obs = torch.randn(1, 50)
        with torch.no_grad():
            action = policy(obs)
        
        print(f"✅ 单次推理成功: {obs.shape} -> {action.shape}")
        print(f"   动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 测试批量推理
        batch_obs = torch.randn(10, 50)
        with torch.no_grad():
            batch_actions = policy(batch_obs)
        
        print(f"✅ 批量推理成功: {batch_obs.shape} -> {batch_actions.shape}")
        
        # 测试动作合理性
        action_std = action.std().item()
        if 0.1 < action_std < 5.0:
            print(f"✅ 动作标准差合理: {action_std:.3f}")
        else:
            print(f"⚠️  动作标准差异常: {action_std:.3f}")
        
        print("\n🎉 所有测试通过！模型转换成功")
        print("\n📋 使用方法:")
        print("python legged_gym/scripts/sim2sim_xgo.py")
        print("python legged_gym/scripts/sim2sim_xgo.py --terrain")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_converted_model()
    sys.exit(0 if success else 1)
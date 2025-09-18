# Isaac Gym 到 MuJoCo 模型转换工具

## 概述

这个工具可以将在 Isaac Gym 中训练的 XGO 机器狗策略模型转换为 MuJoCo 仿真环境中可用的格式。

## 主要功能

### 1. 观测空间转换
- **Isaac Gym 观测 (48维)**:
  - 0:3   - base_lin_vel (设为0，盲视觉)
  - 3:6   - base_ang_vel  
  - 6:9   - projected_gravity
  - 9:12  - commands[:3]
  - 12:24 - (dof_pos - default_dof_pos)
  - 24:36 - dof_vel
  - 36:48 - actions

- **MuJoCo 观测 (50维)**:
  - 0:2   - 时间信息 (sin, cos)
  - 2:5   - commands
  - 5:17  - (dof_pos - default_dof_pos) 
  - 17:29 - dof_vel
  - 29:41 - actions
  - 41:44 - base_ang_vel
  - 44:47 - euler_angles
  - 47:50 - projected_gravity

### 2. 网络结构适配
- 自动映射 Isaac Gym 的 actor 网络参数
- 创建 MuJoCo 兼容的策略包装器
- 支持 TorchScript 导出

## 使用方法

### 1. 自动转换最新模型
```bash
cd /home/ubuntu/XGO-Simulation
python scripts/convert_isaac_to_mujoco.py
```

### 2. 指定输入模型
```bash
python scripts/convert_isaac_to_mujoco.py --input /path/to/your/model.pt
```

### 3. 指定输出路径
```bash
python scripts/convert_isaac_to_mujoco.py --output /path/to/output/model.pt
```

### 4. 完整参数
```bash
python scripts/convert_isaac_to_mujoco.py \
    --input /home/ubuntu/XGO-Simulation/logs/xgo/Sep17_11-20-53_/model_10000.pt \
    --output /home/ubuntu/XGO-Simulation/logs/xgo/mujoco_converted/custom_model.pt \
    --log_dir /home/ubuntu/XGO-Simulation/logs/xgo
```

## 运行仿真

转换完成后，使用转换后的模型运行 MuJoCo 仿真：

```bash
# 使用默认转换模型
python legged_gym/scripts/sim2sim_xgo.py

# 或指定模型路径
python legged_gym/scripts/sim2sim_xgo.py --load_model /path/to/converted/model.pt

# 使用地形
python legged_gym/scripts/sim2sim_xgo.py --terrain
```

## 文件结构

```
XGO-Simulation/
├── scripts/
│   ├── convert_isaac_to_mujoco.py    # 转换脚本
│   └── README_conversion.md          # 本说明文件
├── legged_gym/scripts/
│   └── sim2sim_xgo.py               # MuJoCo 仿真脚本
└── logs/xgo/
    ├── Sep17_11-20-53_/             # Isaac Gym 训练日志
    │   ├── model_10000.pt           # 原始模型
    │   └── ...
    └── mujoco_converted/            # 转换后的模型
        └── xgo_policy_mujoco.pt     # MuJoCo 兼容模型
```

## 转换过程详解

1. **加载 Isaac Gym 模型**: 从 `.pt` 检查点文件中提取 actor 网络参数
2. **创建网络结构**: 根据 XGO 配置创建相同的网络架构
3. **参数映射**: 将 `actor.X.weight/bias` 映射到 `network.X.weight/bias`
4. **包装器创建**: 创建观测空间转换包装器
5. **TorchScript 转换**: 转换为 JIT 脚本以提高推理性能
6. **验证测试**: 测试输入输出维度和数值范围

## 技术细节

### 观测空间映射逻辑
- Isaac Gym 使用盲视觉策略（线速度设为0）
- MuJoCo 需要时间信息用于步态同步
- 重力向量用于估算机器人姿态
- 关节状态直接映射

### 网络架构
- 输入层: 48维 → 50维 (通过包装器转换)
- 隐藏层: [256, 128, 64] 
- 输出层: 12维 (对应12个关节)
- 激活函数: ELU

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否存在
   - 确认模型是 Isaac Gym 训练的检查点格式

2. **权重映射警告**
   - 这是正常的，脚本会自动处理参数名称差异

3. **推理结果异常**
   - 检查观测空间维度是否正确 (50维)
   - 确认动作输出范围合理 (-3 到 3)

### 调试模式
在转换脚本中添加详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

- 使用 TorchScript JIT 编译提高推理速度
- 批处理支持多环境并行仿真
- CPU 和 GPU 推理兼容

## 贡献

如需改进转换逻辑或添加新功能，请修改：
- `convert_isaac_to_mujoco.py`: 核心转换逻辑
- `sim2sim_xgo.py`: MuJoCo 仿真接口
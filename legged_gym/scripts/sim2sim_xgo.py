# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # 从 IMU 传感器获取数据
    gyro_data = data.sensor('gyro').data.astype(np.double)
    accel_data = data.sensor('accelerometer').data.astype(np.double)
    
    # 从加速度计数据估算四元数（简化处理）
    # 这里使用一个简化的方法，实际应用中可能需要更复杂的姿态估计
    gvec_local = accel_data / np.linalg.norm(accel_data)
    # 假设初始姿态，从重力向量估算roll和pitch
    roll = np.arctan2(gvec_local[1], gvec_local[2])
    pitch = np.arctan2(-gvec_local[0], np.sqrt(gvec_local[1]**2 + gvec_local[2]**2))
    yaw = 0.0  # 简化处理，假设yaw为0
    
    # 构造四元数 [x, y, z, w]
    quat = np.array([
        np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2),
        np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2),
        np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2),
        np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    ])
    
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = gyro_data
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0

    default_angle = np.zeros((cfg.env.num_actions), dtype=np.double)

    # XGO 四足机器狗关节映射 (12个关节)
    # 前左腿: fl_hip, fl_thigh, fl_calf
    # 前右腿: fr_hip, fr_thigh, fr_calf  
    # 后左腿: bl_hip, bl_thigh, bl_calf
    # 后右腿: br_hip, br_thigh, br_calf
    default_angle[0] = cfg.init_state.default_joint_angles['fl_hip_joint']
    default_angle[1] = cfg.init_state.default_joint_angles['fl_thigh_joint']
    default_angle[2] = cfg.init_state.default_joint_angles['fl_calf_joint']
    default_angle[3] = cfg.init_state.default_joint_angles['fr_hip_joint']
    default_angle[4] = cfg.init_state.default_joint_angles['fr_thigh_joint']
    default_angle[5] = cfg.init_state.default_joint_angles['fr_calf_joint']
    default_angle[6] = cfg.init_state.default_joint_angles['bl_hip_joint']
    default_angle[7] = cfg.init_state.default_joint_angles['bl_thigh_joint']
    default_angle[8] = cfg.init_state.default_joint_angles['bl_calf_joint']
    default_angle[9] = cfg.init_state.default_joint_angles['br_hip_joint']
    default_angle[10] = cfg.init_state.default_joint_angles['br_thigh_joint']
    default_angle[11] = cfg.init_state.default_joint_angles['br_calf_joint']

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz (根据decimation调整)
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # 构建观测向量 (基于XGO配置的48维观测)
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            
            # 关节位置 (12个关节)
            obs[0, 5:17] = (q - default_angle) * cfg.normalization.obs_scales.dof_pos
            # 关节速度 (12个关节)
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            # 上一次动作 (12个关节)
            obs[0, 29:41] = action
            # 角速度 (3维)
            obs[0, 41:44] = omega * cfg.normalization.obs_scales.ang_vel
            # 欧拉角 (3维)
            obs[0, 44:47] = eu_ang
            # 重力向量 (3维)
            obs[0, 47:50] = gvec

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # 在开始400步后才应用策略动作，之前保持默认姿态
            if count_lowlevel > 400:
                target_q = action * cfg.control.action_scale + default_angle
            else:
                target_q = default_angle

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='XGO Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/ubuntu/XGO-Simulation/logs/xgo/mujoco_converted/xgo_policy_mujoco.pt',
                        help='Path to load policy model from.')
    parser.add_argument('--terrain', action='store_true', default=False, help='Use terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg:
        
        class env:
            num_actions = 12  # XGO有12个关节
            num_single_obs = 50  # 单帧观测维度: 2(时间) + 3(cmd) + 12(pos) + 12(vel) + 12(action) + 3(omega) + 3(euler) + 3(gravity) = 50
            frame_stack = 1  # 历史帧数
            num_observations = num_single_obs * frame_stack

        class init_state:
            pos = [0.0, 0.0, 0.15]  # x,y,z [m]
            default_joint_angles = {  # = target angles [rad] when action = 0.0
                'fl_hip_joint': 0.1,   # [rad]
                'bl_hip_joint': 0.1,   # [rad]
                'fr_hip_joint': -0.1,  # [rad]
                'br_hip_joint': -0.1,  # [rad]
                'fl_thigh_joint': 0.6,  # [rad]
                'bl_thigh_joint': 0.6,  # [rad]
                'fr_thigh_joint': 0.6,  # [rad]
                'br_thigh_joint': 0.6,  # [rad]
                'fl_calf_joint': -1.5,  # [rad]
                'bl_calf_joint': -1.5,  # [rad]
                'fr_calf_joint': -1.5,  # [rad]
                'br_calf_joint': -1.5,  # [rad]
            }

        class control:
            action_scale = 0.25

        class normalization:
            clip_observations = 100.
            clip_actions = 100.
            
            class obs_scales:
                lin_vel = 10.0
                ang_vel = 0.5
                dof_pos = 1.0
                dof_vel = 0.1

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/xgo/mjcf/xgo.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/xgo/mjcf/xgo.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # XGO机器狗的PD控制参数 (12个关节)
            kps = np.array([2.0, 2.0, 2.0,  # fl_hip, fl_thigh, fl_calf
                           2.0, 2.0, 2.0,  # fr_hip, fr_thigh, fr_calf
                           2.0, 2.0, 2.0,  # bl_hip, bl_thigh, bl_calf
                           2.0, 2.0, 2.0], dtype=np.double)  # br_hip, br_thigh, br_calf
            kds = np.array([0.005, 0.005, 0.005,  # fl_hip, fl_thigh, fl_calf
                           0.005, 0.005, 0.005,  # fr_hip, fr_thigh, fr_calf
                           0.005, 0.005, 0.005,  # bl_hip, bl_thigh, bl_calf
                           0.005, 0.005, 0.005], dtype=np.double)  # br_hip, br_thigh, br_calf
            tau_limit = 0.45 * np.ones(12, dtype=np.double)  # 扭矩限制

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
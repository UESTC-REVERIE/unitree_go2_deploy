import numpy as np
import time
import torch
import torch.nn.functional as F
import sys
import os
from unitree_sdk2py.core.channel import ChannelPublisher,ChannelSubscriber,ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_,unitree_go_msg_dds__LowState_, unitree_go_msg_dds__HeightMap_, sensor_msgs_msg_dds__PointField_Constants_PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_ as Go2FrontVideoDataGo
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as PointCloud2Go

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as HighStateGo
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient

from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_zero_cmd, create_damping_cmd
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap

from typing import Tuple
from utils import *
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def init_cmd_go2(cmd:LowCmdGo):
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    PosStopF = 2.146e9
    VelStopF = 16000.0
    for i in range(12):
        cmd.motor_cmd[i].mode = 0x0A
        cmd.motor_cmd[i].q = PosStopF
        cmd.motor_cmd[i].dq = VelStopF  # or qd
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

def quat_to_rpy(quat):
    w, x, y, z = quat
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

VOXEL_SIZE_XY = 0.06
RANGE_X = (-0.8,  0.2 + 1e-9)
RANGE_Y = (-0.8,  0.8 + 1e-9)
RANGE_Z = ( 0.0,  5.0)
OFFSET    = 0.0 
V_FOV = (0.0, 90.0) # 垂直视场
H_FOV = (-180.0, 180.0) # 水平视场
# 17 * 27
heightmap = np.array(
            [ 5.1991e-02, -2.9083e-01,  9.6685e-02, -2.9408e-02, -6.2699e-02,
        -1.3660e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00, -7.4978e-02,
         7.6787e-02,  3.0985e-02, -1.4704e-02,  1.9062e-01, -2.6642e-02,
         1.7040e-01, -8.5774e-02,  1.1926e-01, -1.2127e-01, -1.2325e-01,
         3.7327e-02, -5.3844e-01,  8.3207e-01,  6.5266e-01, -4.7913e-01,
         1.4805e+00, -1.3350e+00,  1.2560e+00,  1.5541e+00,  4.5108e-01,
        -4.1393e-01, -1.6038e+00,  3.8533e-01, -1.4291e+00,  1.6031e-01,
         2.7835e-01,  1.2363e+00, -3.3955e-01,  4.6753e-02,  9.0873e-01,
        -1.4352e-02,  2.3764e+00, -4.9279e-01, -5.7383e-01,  2.1729e+00,
         0.0000e+00,  0.0000e+00,  5.6295e-01,  5.8430e-01,  5.8430e-01,
         5.8430e-01,  5.6232e-01,  5.7414e-01,  5.7414e-01,  5.7414e-01,
         5.4416e-01,  5.4503e-01,  5.4503e-01,  5.4503e-01,  5.4230e-01,
         5.6740e-01,  5.6740e-01,  5.6740e-01,  5.5450e-01,  5.5002e-01,
         5.5002e-01,  5.5002e-01,  5.4948e-01,  5.4948e-01,  5.4948e-01,
         5.3601e-01,  5.3601e-01,  0.0000e+00,  0.0000e+00,  5.6295e-01,
         5.8430e-01,  5.8430e-01,  5.8430e-01,  5.6232e-01,  5.7414e-01,
         5.7414e-01,  5.7414e-01,  5.4416e-01,  5.4503e-01,  5.4503e-01,
         5.4503e-01,  5.4230e-01,  5.6740e-01,  5.6740e-01,  5.6740e-01,
         5.5450e-01,  5.5002e-01,  5.5002e-01,  5.5002e-01,  5.4948e-01,
         5.4948e-01,  5.4948e-01,  5.3601e-01,  5.3601e-01,  0.0000e+00,
         0.0000e+00,  5.6295e-01,  5.6295e-01,  5.6295e-01,  5.4874e-01,
         5.6232e-01,  5.6232e-01,  5.6232e-01,  5.4059e-01,  5.4416e-01,
         5.4503e-01,  5.4503e-01,  5.4503e-01,  5.4230e-01,  5.3756e-01,
         5.3091e-01,  5.4005e-01,  5.4005e-01,  5.4005e-01,  5.2782e-01,
         5.2782e-01,  5.2782e-01,  5.2066e-01,  5.2066e-01,  5.0147e-01,
         5.0147e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.1769e-01,  5.3375e-01,  5.3375e-01,  5.3375e-01,  5.2768e-01,
         5.4059e-01,  5.4059e-01,  5.4059e-01,  5.0628e-01,  5.0628e-01,
         5.3091e-01,  5.3091e-01,  5.3091e-01,  5.2439e-01,  5.2439e-01,
         5.2439e-01,  5.0786e-01,  5.0559e-01,  5.0559e-01,  4.9308e-01,
         4.9308e-01,  5.0147e-01,  5.0147e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  5.1769e-01,  5.1769e-01,  5.1769e-01,
         5.1867e-01,  5.1867e-01,  5.1867e-01,  5.0561e-01,  5.0628e-01,
         5.0628e-01,  5.0628e-01,  5.0416e-01,  5.0045e-01,  5.0152e-01,
         5.0152e-01,  5.0152e-01,  4.9079e-01,  4.9079e-01,  4.8365e-01,
         4.8365e-01,  4.9308e-01,  4.9308e-01,  4.9308e-01,  4.6971e-01,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  5.0091e-01,
         5.0091e-01,  5.0091e-01,  4.9662e-01,  4.9662e-01,  4.9850e-01,
         4.9850e-01,  4.9850e-01,  4.7659e-01,  4.7659e-01,  4.8868e-01,
         4.8868e-01,  4.8868e-01,  4.7631e-01,  4.7631e-01,  4.7631e-01,
         4.7345e-01,  4.7345e-01,  4.6226e-01,  4.6693e-01,  4.6693e-01,
         4.6971e-01,  4.6971e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  4.7080e-01,  4.7080e-01,  4.7080e-01,
         4.6807e-01,  4.6807e-01,  4.7344e-01,  4.7659e-01,  4.7659e-01,
         4.7659e-01,  4.7451e-01,  4.6736e-01,  4.5574e-01,  4.6274e-01,
         4.6274e-01,  4.6274e-01,  4.5611e-01,  4.5611e-01,  4.4162e-01,
         4.4232e-01,  4.4232e-01,  4.4232e-01,  4.4063e-01,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         1.0864e-01,  4.4532e-01,  4.5288e-01,  4.5288e-01,  4.5288e-01,
         4.5163e-01,  4.5163e-01,  4.5163e-01,  4.5069e-01,  4.4485e-01,
         4.4052e-01,  4.4052e-01,  4.3471e-01,  4.3896e-01,  4.3896e-01,
         4.3896e-01,  4.2217e-01,  4.2185e-01,  4.1930e-01,  4.1930e-01,
         4.1410e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  9.7251e-02,  4.2817e-01,  4.2817e-01,
         4.2817e-01,  4.2669e-01,  4.2482e-01,  4.2482e-01,  4.2260e-01,
         4.2260e-01,  4.2260e-01,  4.1329e-01,  4.1754e-01,  4.1754e-01,
         4.1754e-01,  4.0670e-01,  4.0670e-01,  4.0589e-01,  4.0305e-01,
         3.9786e-01,  3.8993e-01,  3.8993e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  2.0863e-01,
         4.0454e-01,  4.0528e-01,  4.0528e-01,  4.0528e-01,  4.0213e-01,
         3.9976e-01,  3.9976e-01,  3.9886e-01,  3.9433e-01,  3.9144e-01,
         3.9342e-01,  3.9403e-01,  3.9403e-01,  3.9403e-01,  3.9304e-01,
         3.9021e-01,  3.8528e-01,  3.8993e-01,  3.8993e-01,  3.8993e-01,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  2.0863e-01,  3.8797e-01,  3.8797e-01,  3.8797e-01,
         3.8198e-01,  3.8198e-01,  3.7909e-01,  3.7655e-01,  3.7450e-01,
         3.7270e-01,  3.7228e-01,  3.7228e-01,  3.7103e-01,  3.7103e-01,
         3.7103e-01,  3.6856e-01,  3.6856e-01,  3.6856e-01,  3.6796e-01,
         3.6796e-01,  3.6796e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  2.0863e-01,  2.0863e-01,
         3.6219e-01,  3.6219e-01,  3.6219e-01,  3.6002e-01,  3.5862e-01,
         3.5607e-01,  3.5529e-01,  3.5156e-01,  3.5156e-01,  3.5156e-01,
         3.5156e-01,  3.4919e-01,  3.4919e-01,  3.4739e-01,  3.4739e-01,
         3.4801e-01,  3.4801e-01,  3.4801e-01,  3.3354e-01,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  3.4387e-01,  3.4387e-01,  3.4387e-01,
         3.4024e-01,  3.3845e-01,  3.3460e-01,  3.3336e-01,  3.3206e-01,
         3.3068e-01,  3.2795e-01,  3.2891e-01,  3.2891e-01,  3.2891e-01,
         3.2699e-01,  3.2699e-01,  3.2990e-01,  3.2990e-01,  3.2990e-01,
         3.1448e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  3.2275e-01,
         3.2275e-01,  3.2275e-01,  3.1932e-01,  3.1774e-01,  3.1411e-01,
         3.1255e-01,  3.1121e-01,  3.1049e-01,  3.0970e-01,  3.0970e-01,
         3.0970e-01,  3.0141e-01,  3.0085e-01,  3.0021e-01,  2.9946e-01,
         2.9855e-01,  2.9745e-01,  2.9607e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         2.1639e-01,  3.0116e-01,  3.0116e-01,  3.0116e-01,  2.9875e-01,
         2.9570e-01,  2.9377e-01,  2.9077e-01,  2.9077e-01,  2.8931e-01,
         2.8778e-01,  2.8628e-01,  2.8628e-01,  2.8738e-01,  2.8738e-01,
         2.8738e-01,  2.8504e-01,  2.8504e-01,  2.8221e-01,  2.7871e-01,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  2.1639e-01,  2.1639e-01,  2.7870e-01,
         2.7870e-01,  2.7870e-01,  2.7672e-01,  2.7367e-01,  2.7232e-01,
         2.7133e-01,  2.6955e-01,  2.6797e-01,  2.6797e-01,  2.6633e-01,
         2.6620e-01,  2.6620e-01,  2.6620e-01,  2.6167e-01,  2.6338e-01,
         2.6338e-01,  2.6338e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  2.1639e-01,
         2.1639e-01,  2.5793e-01,  2.5793e-01,  2.5793e-01,  2.5609e-01,
         2.5419e-01,  2.5300e-01,  2.5056e-01,  2.5009e-01,  2.4681e-01,
         2.4665e-01,  2.4665e-01,  2.4665e-01,  2.4533e-01,  2.4533e-01,
         2.4982e-01,  2.4982e-01,  2.4982e-01,  2.4176e-01,  2.7584e-02,
         5.9673e-01,  4.0457e-01, -2.5209e-02,  9.2824e-02, -1.4547e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -1.5063e-03, -1.9259e-02,
        -1.0405e-01,  4.6694e-03, -4.7944e-02,  1.1012e-01,  6.0694e-02,
         6.7851e-03,  1.9812e-02,  1.0184e-01,  1.8123e-01, -7.0271e-02,
        -1.9427e-02, -1.4465e+00, -7.1673e-01,  9.2141e-02,  1.8028e+00,
         7.7583e-01, -8.9566e-01, -2.1746e+00, -1.1164e+00,  2.4026e+00,
         3.2415e+00, -1.7799e+00, -2.4671e-01,  8.4903e-01, -7.4210e-01,
        -5.9206e-02, -3.3082e-01, -1.1871e-01,  6.4257e-02, -1.2804e-01,
         1.0765e+00,  1.9097e+00,  2.0941e+00, -4.0975e-01,  1.3657e-01,
         2.2013e-02,  1.8841e-01,  7.1563e-02,  7.1517e-02, -1.2999e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -2.3622e-02, -3.0757e-02,
        -1.0400e-01,  1.1794e-02, -2.9979e-02,  1.1294e-01,  5.6251e-02,
        -7.1537e-02,  1.5832e-02,  1.4510e-01,  2.2707e-01, -1.2661e-01,
         3.7331e-01,  3.6783e-01, -1.0950e+00,  1.2011e+00,  1.5676e+00,
         1.0322e+00, -1.0443e+00, -4.3449e+00, -1.2868e+00,  3.0372e+00,
         3.2458e+00, -3.8249e+00, -6.7998e-02,  8.1699e-01, -7.3311e-01,
        -1.4541e-01, -7.1274e-01, -2.7878e-01, -1.7539e-01, -4.9993e-01,
         1.6831e+00,  1.8747e+00,  2.0293e+00, -6.5024e-01, -8.2526e-01,
        -5.6767e-01,  2.7572e-01, -2.3073e-02,  6.9026e-02, -1.4061e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -2.0963e-02, -3.0257e-02,
        -1.0113e-01,  2.3349e-02,  1.3384e-02,  9.8051e-02,  6.2573e-02,
        -1.9186e-01,  3.4205e-02,  2.0801e-01,  2.3989e-01, -8.8417e-02,
         1.3015e+00, -1.0334e+00,  8.4990e-01,  7.0181e-01,  1.8107e+00,
        -6.2241e-01,  1.6801e+00, -4.8248e+00,  1.0760e+00,  4.4344e+00,
        -5.6995e-02,  6.2411e-02, -3.1537e-01,  5.7836e-01, -4.5238e-01,
         4.7855e-02, -9.4083e-01, -6.4445e-01,  1.7912e-02, -8.8945e-01,
         1.5557e+00,  1.6055e+00,  8.7411e-01, -3.3326e-01, -2.8182e-01,
        -1.0992e+00,  2.2977e-01, -1.0235e-01, -3.7565e-02, -1.3019e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -2.0904e-02, -2.1732e-02,
        -9.9920e-02,  3.5871e-02,  3.3620e-02,  1.0108e-01,  1.2905e-01,
        -2.7724e-01,  4.1928e-02,  2.7517e-01,  1.7736e-01,  2.4783e-02,
         4.9315e-01, -1.0018e+00,  2.9353e-01, -6.5372e-01,  1.1081e+00,
         1.6083e+00,  2.9712e+00, -2.8327e+00,  2.1090e+00,  4.2281e+00,
        -3.4188e+00,  4.8980e+00, -4.7583e-01,  3.4577e-01, -2.3530e-01,
         1.6889e-01, -7.1675e-01, -3.3562e-01,  3.4291e-01, -1.1034e+00,
         1.4163e+00,  1.1325e+00,  4.2736e-01,  4.1383e-01, -3.8024e-01,
        -6.9145e-01,  5.7365e-02,  1.6940e-02,  4.7740e-02, -1.3137e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -2.6767e-02, -1.4939e-02,
        -6.9284e-02,  2.7757e-02,  7.9288e-02,  1.2362e-01,  1.4696e-01,
        -2.5644e-01,  5.4544e-02,  1.4970e-01,  1.0277e-01,  5.8834e-02,
         7.4362e-01,  6.3484e-01,  7.9310e-01, -1.2989e+00,  3.0748e+00,
        -1.3326e+00,  6.2386e-01,  2.8639e+00,  1.8044e+00, -4.8679e+00,
        -3.5151e+00, -2.2204e+00, -6.2289e-01,  2.9632e-02, -2.8162e-02,
         5.6293e-01, -5.3785e-01,  1.8329e-03,  6.2301e-01, -9.7998e-01,
         1.5075e+00,  9.5557e-02,  1.9222e-01,  1.4211e+00,  5.2721e-01,
        -4.1380e-01, -1.2467e-01,  6.0098e-02,  4.9220e-03, -1.3304e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -4.0811e-02,  9.7575e-03,
        -5.6313e-02,  1.4171e-02,  1.0004e-01,  1.2196e-01,  1.5761e-01,
        -1.9697e-01,  8.4445e-02, -2.8801e-02,  1.8088e-02,  5.8109e-02,
         1.8208e-01,  9.1795e-01,  2.3327e+00,  3.1509e-01,  1.3119e+00,
        -1.2150e+00, -1.3701e-01,  3.3990e+00, -6.3132e-01, -6.2710e+00,
        -2.8635e+00, -1.8431e+00, -1.5460e+00,  1.2590e-01,  7.6847e-02,
         9.4931e-01, -4.1531e-01,  4.4189e-01,  6.3927e-01, -7.3044e-01,
         1.9088e+00, -4.3230e-01, -8.8361e-02,  2.1178e+00,  3.4147e-01,
        -3.9114e-01,  3.4396e-02, -9.6532e-02, -8.7809e-02, -1.3567e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -5.5162e-02,  3.7853e-02,
        -1.4672e-02,  1.2427e-02,  1.2520e-01,  5.8616e-02,  1.4125e-01,
        -1.6591e-01,  1.0227e-01, -9.8544e-02, -3.0737e-02,  3.0934e-02,
        -1.4630e-01,  2.4463e+00,  8.8033e-01,  4.3296e-01,  1.7727e+00,
        -3.1706e+00,  2.6683e-01,  1.2813e+00,  7.2614e-01, -1.4920e+00,
        -2.2541e+00, -4.6898e-01, -1.5897e+00,  8.0487e-02,  1.7708e-01,
         1.2799e+00, -2.1488e-01,  2.9210e-01,  7.1110e-01, -4.5239e-01,
         2.0937e+00, -4.9396e-01, -2.5096e-01,  2.1870e+00,  1.0973e-01,
        -5.0181e-01, -5.3906e-02,  2.6803e-02, -4.5483e-02, -1.2881e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -6.6253e-02,  4.9981e-02,
         5.6803e-03, -1.1245e-03,  1.5342e-01,  1.2043e-02,  1.4373e-01,
        -1.1982e-01,  1.0321e-01, -1.0417e-01, -7.1583e-02,  3.3973e-02,
        -1.7185e+00,  1.3765e+00,  1.3682e+00, -1.2575e+00,  2.3021e+00,
        -1.6885e+00,  1.1807e+00,  2.2243e+00, -3.3386e-01, -9.2538e-01,
         2.6820e-01,  6.6908e-01, -1.3832e+00,  2.0907e-02,  2.7372e-01,
         1.3461e+00, -2.7576e-01,  1.4778e-01,  8.6307e-01, -2.1580e-01,
         2.2988e+00, -4.7220e-01, -3.1707e-01,  2.1399e+00,  8.2767e-03,
        -1.0582e-01,  3.2409e-02,  3.4590e-02,  9.8961e-03, -1.3744e+00,
         5.0000e-01,  0.0000e+00,  0.0000e+00, -6.6920e-02,  7.7602e-02,
         2.7316e-02, -2.3777e-02,  1.8770e-01, -3.6366e-02,  1.6887e-01,
        -8.1642e-02,  1.2901e-01, -1.1454e-01, -1.2563e-01,  2.9770e-02,
        -6.0272e-01,  3.9684e-02, -5.8881e-01, -9.7389e-01,  8.9123e-01,
        -2.0679e+00, -2.1881e-01,  7.5019e-01,  1.6523e+00, -1.2578e+00,
        -1.3101e+00, -6.2466e-01, -1.4291e+00,  1.6031e-01,  2.7835e-01,
         1.2363e+00, -3.3955e-01,  4.6753e-02,  9.0873e-01, -1.4352e-02,
         2.3764e+00, -4.9279e-01, -5.7383e-01,  2.1729e+00]
        )

def cloud_to_xyz_numpy(
    cloud_msg,
    fields: Tuple[str, ...] = ("x", "y", "z"),
) -> np.ndarray:
    """
    NumPy 方式提取 PointCloud2 中指定的 float32 字段。
    """

    offset_dict = {
        f.name.decode() if isinstance(f.name, bytes) else f.name: f.offset
        for f in cloud_msg.fields
    }
    # print(offset_dict) # {'x': 0, 'y': 4, 'z': 8, 'intensity': 16, 'ring': 20, 'time': 24}
    point_step = cloud_msg.point_step

    offsets = [offset_dict[fld] for fld in fields]
    dtype = np.dtype({
        'names': fields, # xyz
        'formats': [np.float32] * len(fields), # [np.float32] * 3 
        'offsets': offsets, # 每个字段的字节偏移量列表
        'itemsize': point_step  # 设置 dtype 的总大小
    })

    raw = bytes(cloud_msg.data)
    assert len(raw) % point_step == 0, "data size mismatch"

    pts = np.frombuffer(raw, dtype=dtype)

    xyz = np.column_stack([pts[fld] for fld in fields]).astype(np.float32)

    if cloud_msg.is_bigendian:
        xyz = xyz.byteswap().newbyteorder()

    return xyz # (N,3)
def pointcloud2_to_heightmap(
        msg,
        fields: Tuple[str, ...] = ("x", "y", "z"),
) -> torch.Tensor:
    # 读取点云数据 CloudPoint2 格式到 numpy (N,3)
    pts = cloud_to_xyz_numpy(msg) 
    # 无限数置0，与 hit_vec[torch.isinf(hit_vec)] = 0.0 hit_vec[torch.isnan(hit_vec)] = 0.0 逻辑对应
    pts[~np.isfinite(pts)] = 0.0

    np.save("deploy/deploy_real/cloud_points.npy", pts)
    save_pointcloud_2_5D_view(pts, "deploy/deploy_real/cloud_points.jpg")
    
    pts[:,0] = pts[:,0]*-1
    save_pointcloud_2_5D_view(pts, "deploy/deploy_real/cloud_points-x.jpg")

    # 32 通道？
    # FOV 裁剪，
    # if V_FOV != (None, None) or H_FOV != (None, None):
    #     xy_norm  = np.linalg.norm(pts[:, :2], axis=1) + 1e-6
    #     horiz_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    #     vert_deg  = np.degrees(np.arctan2(pts[:, 2], xy_norm))

    #     v_min, v_max = V_FOV
    #     h_min, h_max = H_FOV
    #     mask = (
    #         (vert_deg >= v_min) & (vert_deg <= v_max) &
    #         (horiz_deg >= h_min) & (horiz_deg <= h_max)
    #     )
    #     pts = pts[mask]
    
    # 转成 torch，后续流程与 height_map_lidar 一致
    pts_t = torch.from_numpy(pts).to(device) # (N,3)
    # print(f"converted cloud points xyz shape: {pts_t.shape}")
    # print("------------converted cloud points------------")
    # print(pts_t)
    # print("-------------------END-----------------------")
    num_envs = 1

    x, y, z = pts_t[:, 0], pts_t[:, 1], pts_t[:, 2]

    # 体素网格
    x_bins = torch.arange(RANGE_X[0], RANGE_X[1], VOXEL_SIZE_XY, device=device)
    y_bins = torch.arange(RANGE_Y[0], RANGE_Y[1], VOXEL_SIZE_XY, device=device)

    # 有效范围过滤，valid.shape=x.shape=y.shape=z.shape
    valid = (
        (x > RANGE_X[0]) & (x <= RANGE_X[1]) &
        (y > RANGE_Y[0]) & (y <= RANGE_Y[1]) &
        (z >= RANGE_Z[0]) & (z <= RANGE_Z[1])
    )
    x, y, z = x[valid], y[valid], z[valid]

    # 每个点在网格 bins 的索引 idx 
    x_idx = torch.bucketize(x, x_bins) - 1
    y_idx = torch.bucketize(y, y_bins) - 1

    # env 索引，单环境全0
    env_idx = torch.zeros_like(valid, device=device)
    flat_env_idx = env_idx[valid]
    H, W = len(x_bins), len(y_bins)
    map_2_5D = torch.full((num_envs, H, W), float('inf'), device=device)
    # 平铺到线性后每个点应该落入的位置
    linear_idx = flat_env_idx * H * W + x_idx * W + y_idx # x_idx * W + y_idx
    # 取每个位置中的 z 最小值
    map_2_5D = map_2_5D.view(-1).scatter_reduce_(0, linear_idx, z, reduce="amin")

    map_2_5D = map_2_5D - OFFSET
    map_2_5D = torch.where(map_2_5D < 0.05,
                           torch.tensor(0.0, device=device),
                           map_2_5D)

    map_2_5D = torch.where(torch.isinf(map_2_5D),
                           torch.tensor(0.0, device=device),
                           map_2_5D)

    # 3×3 max‑pool
    map_2_5D = map_2_5D.view(num_envs, H, W)
    pooled   = F.max_pool2d(map_2_5D, kernel_size=3, stride=1, padding=1)

    # 输出与 height_map_lidar 等价 (1, H*W)
    # pooled = torch.flip(pooled, dims=[2])
    return pooled.view(num_envs, -1)

class Controller:
    def __init__(self,config:Config) -> None:
        self.config = config
        # Initialize the cotroller
        self.remote_controller = RemoteController()

        # Create a video client to capture camera data
        if self.config.save_image:
            self.video_client = VideoClient()
            self.video_client.SetTimeout(3.0)
            self.video_client.Init()
            self.save_image_path = os.path.dirname(os.path.realpath(__file__)) + "/front_camera/front_image.jpg"
            self.save_image_interval = 1.0  # Save image every 1 second
            self.last_save_time = 0.0
            self.code, self.data = self.video_client.GetImageSample()
            if self.code != 0:
                print("Failed to get image sample from the robot camera.")
            else:
                print("Successfully connected to the robot camera.")
                _ = convert_image(self.data, self.save_image_path)
        
        # load the policy model
        self.policy = torch.jit.load(config.policy_path).to(device)
        self._warm_up()
        
        self.qj = np.zeros(config.num_actions,dtype=np.float32)
        self.dqj = np.zeros(config.num_actions,dtype=np.float32)
        self.action = np.zeros(config.num_actions,dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        
        # 保存 scale 后的 priorio_obs 历史记录
        self.proprio_obs_buf = torch.zeros(1, config.history_length, config.proprio_obs_dim, dtype=torch.float32, device=device)
        
        self.obs = np.zeros(config.num_obs,dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0],dtype=np.float32)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.point_cloud = sensor_msgs_msg_dds__PointField_Constants_PointCloud2_()
        self.heightmap = torch.from_numpy(heightmap[45:45+459].copy())
        self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic,LowCmdGo)
        self.lowcmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic,LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateHandler,10)

        self.cloud_subscriber = ChannelSubscriber(config.cloud_topic,PointCloud2Go)
        self.cloud_subscriber.Init(self.PointCloudHandler,10)
        
        # try to shutdown the robot sport controller
        self.shutdown_sport_controller()

        self.wait_for_low_state()
        init_cmd_go2(self.low_cmd)
        self.counter = 0

    def shutdown_sport_controller(self):
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        self.mode_name = result['name']
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        print(f'Sport controller {self.mode_name} mode has been shutdown.') # {ai}
            
    def resume_sport_controller(self):
        self.msc.SelectMode(self.mode_name)
        # self.sc.StandUp()
        print(f'Sport controller {self.mode_name} mode has been resumed.')

    def _warm_up(self):
        _obs = torch.ones((1,config.num_obs)).to(device)
        for _ in range(10):
            _ = self.policy(_obs)
        print('Network has been warmed up.')

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def LowStateHandler(self,msg:LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
        # print("Low state Running...")
    
    def PointCloudHandler(self,msg:PointCloud2Go):
        self.point_cloud = msg
        # print(f"Points Number: {msg.width}") #4xxx
        # print(f"Received point cloud with width: {self.point_cloud.width}, height: {self.point_cloud.height}")
        # print(f"Point step: {self.point_cloud.point_step}, Row step: {self.point_cloud.row_step}")
        # print(f"length of point cloud data: {len(self.point_cloud.data)}")
        # xyz = parse_pointcloud2_np(self.point_cloud)
        # self.heightmap = pointcloud_to_heightmap(xyz)
        # print("Heightmap shape:", self.heightmap.shape)
    
    def send_cmd(self,cmd:LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print('Moving to default pos.')
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        # dof_idx = self.config.joint2motor_idx
        default_pos = self.config.default_angles


        init_dof_pos = np.zeros(12,dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.low_state.motor_state[i].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[j].dq = 0.0  # qd
                self.low_cmd.motor_cmd[j].kp = 40.0
                self.low_cmd.motor_cmd[j].kd = 0.6
                self.low_cmd.motor_cmd[j].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[i].dq = 0.0  # qd
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 0.6
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    def keep_default_pos(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.config.default_angles[i]
            self.low_cmd.motor_cmd[i].dq = 0.0  # qd
            self.low_cmd.motor_cmd[i].kp = 60.0
            self.low_cmd.motor_cmd[i].kd = 0.5
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd(self.low_cmd)
        # time.sleep(self.config.control_dt)
    def run(self):
        if self.config.save_image:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_image_interval:
                self.code, self.data = self.video_client.GetImageSample()
                _ = convert_image(self.data, self.save_image_path)
                self.last_save_time = current_time
            
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        quat = self.low_state.imu_state.quaternion
        # base_rpy = quat_to_rpy(quat)
        base_rpy = self.low_state.imu_state.rpy
        
        base_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32) * self.config.obs_scales_ang_vel
        
        qj_obs = self.qj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.obs_scales_dof_pos
        
        dqj_obs = self.dqj.copy()
        dqj_obs = dqj_obs * self.config.obs_scales_dof_vel
        
        # mapping qj and dqj to match the policy input order
        qj_obs = qj_obs[self.config.joint2motor_idx]
        dqj_obs = dqj_obs[self.config.joint2motor_idx]
        
        # Controller inputs
        self.cmd[0] = np.clip(self.remote_controller.ly,     -0.0,1.0)
        self.cmd[1] = np.clip(self.remote_controller.lx * -1, 0.0,0.0)
        self.cmd[2] = np.clip(self.remote_controller.rx * -1,-1.0,1.0)
        
        self.obs[:3] = base_ang_vel
        self.obs[3:6] = base_rpy
        # self.obs[3:6] = get_gravity_orientation(quat)
        self.obs[6:9] = self.cmd * self.config.command_scale
        self.obs[9:21] = qj_obs
        self.obs[21:33] = dqj_obs
        self.obs[33:45] = self.action
        
        # heightmap = pointcloud2_to_heightmap(self.point_cloud)
        # heightmap = heightmap.squeeze()
        # print("------------observations[:45]--------------")
        # print(self.obs)
        # print("--------------END-------------------")
        self.counter += 1
        if self.counter % 4 == 0:
            self.heightmap = pointcloud2_to_heightmap(self.point_cloud)
            self.counter = 0
        # self.heightmap = pointcloud2_to_heightmap(self.point_cloud)
        self.obs[45:45+459] = torch.clip(self.heightmap,-10.0,10.0).squeeze().cpu().numpy()
        # self.obs[45:45+459] = heightmap[45:45+459]
        # print(self.obs[45:45+459])
        
        proprio_obs = self.obs[:45].copy()
        proprio_obs_tensor = torch.from_numpy(proprio_obs).unsqueeze(0).unsqueeze(1).to(device)
        # update history buffer
        self.proprio_obs_buf = torch.cat([
            self.proprio_obs_buf[:, 1:],
            proprio_obs_tensor
        ], dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(1, -1).squeeze().cpu().numpy()
        
        self.obs[45+459:] = proprio_obs_history
        # print("------------observations[45+459:]--------------")
        # print(self.obs[45+459:])
        # print("--------------END-------------------")
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).to(device)
        # print(f"obs_tensor shape: {obs_tensor.shape}")  # input shape: (1, 909)
        # action_joint = self.policy(obs_tensor).detach()
        # action_joint = action_joint.cpu().numpy().squeeze()
        
        # mapping policy action order to model action order
        # action_motion = np.empty_like(action_joint)
        # action_motion[self.config.joint2motor_idx] = action_joint
        
        # self.target_dof_pos = action_motion * self.config.action_scale + self.config.default_angles
        
        # self.action = action_joint
        
        # for i in range(12):
        #     self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
        #     self.low_cmd.motor_cmd[i].dq = 0.0
        #     self.low_cmd.motor_cmd[i].kp = self.config.kps[i]
        #     self.low_cmd.motor_cmd[i].kd = self.config.kds[i]
        #     self.low_cmd.motor_cmd[i].tau = 0
        # self.send_cmd(self.low_cmd)
        # time.sleep(self.config.control_dt)
        # keep ?
        self.keep_default_pos()
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()

    config_path = f"{os.path.dirname(os.path.realpath(__file__))}/config/go2.yaml"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()
    
    print("Starting the control loop. Press Button 'SELECT' to exit.")
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print('Exit')
    
    controller.resume_sport_controller()
    

import copy
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
# device = "cpu"
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

VOXEL_SIZE_XY = 0.06
RANGE_X = (-0.8,  0.2 + 1e-9)
RANGE_Y = (-0.8,  0.8 + 1e-9)
RANGE_Z = ( 0.0,  5.0)
OFFSET    = 0.0
V_FOV = (0.0, 90.0)
H_FOV = (-180.0, 180.0)
X_BINS = torch.arange(RANGE_X[0], RANGE_X[1], VOXEL_SIZE_XY, device=device)
Y_BINS = torch.arange(RANGE_Y[0], RANGE_Y[1], VOXEL_SIZE_XY, device=device)
H, W   = len(X_BINS), len(Y_BINS)
# 17 * 27
def cloud_to_xyz_numpy(
    cloud_msg,
    fields: Tuple[str, ...] = ("x", "y", "z"),
) -> np.ndarray:
    """
    NumPy 方式提取 PointCloud2 中指定的 float32 字段。
    """

    offset_dict = {
        (f.name.decode() if isinstance(f.name, bytes) else f.name): f.offset
        for f in cloud_msg.fields
    }
    # print(offset_dict) # {'x': 0, 'y': 4, 'z': 8, 'intensity': 16, 'ring': 20, 'time': 24}
    point_step = cloud_msg.point_step
    # num_points  = cloud_msg.width * cloud_msg.height
    
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
    
    # 坐标系转换
    xyz[:,0] = xyz[:,0]*-1
    xyz = xyz[:, [1, 0, 2]]
    
    theta = - np.pi / 5.5  # 45 deg
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    xyz = xyz @ Rz.T
    
    return xyz # (N,3)
def pointcloud2_to_heightmap(
        msg,
        fields: Tuple[str, ...] = ("x", "y", "z"),
) -> torch.Tensor:
    """与 isaaclab_go2/source/isaaclab_tasks/isaaclab_tasks/manager_based/vision/mdp/observations.py 中的 
    height_map_lidar 逻辑对应

    Args:
        msg (_type_): 与 ROS 的 PointCloud2 数据类似的原始点云数据

    Returns:
        torch.Tensor: heightmap
    """
    # 读取点云数据 CloudPoint2 格式到 numpy (N,3)
    pts = cloud_to_xyz_numpy(msg) 
    # 无限数置0，与 hit_vec[torch.isinf(hit_vec)] = 0.0 hit_vec[torch.isnan(hit_vec)] = 0.0 逻辑对应
    pts[~np.isfinite(pts)] = 0.0
    
    # 转成 torch，后续流程与 height_map_lidar 一致
    pts_t = torch.from_numpy(pts).to(device) # (N,3)

    num_envs = 1

    x, y, z = pts_t[:, 0], pts_t[:, 1], pts_t[:, 2]

    # 有效范围过滤，valid.shape=x.shape=y.shape=z.shape
    valid = (
        (x > RANGE_X[0]) & (x <= RANGE_X[1]) &
        (y > RANGE_Y[0]) & (y <= RANGE_Y[1]) &
        (z >= RANGE_Z[0]) & (z <= RANGE_Z[1])
    )
    x, y, z = x[valid], y[valid], z[valid]

    # 每个点在网格 bins 的索引 idx 
    x_idx = torch.bucketize(x, X_BINS) - 1
    y_idx = torch.bucketize(y, Y_BINS) - 1

    # env 索引，单环境全0
    env_idx = torch.zeros_like(valid, device=device)
    flat_env_idx = env_idx[valid]
    H, W = len(X_BINS), len(Y_BINS)
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
    return pooled.view(num_envs, -1)

class Controller:
    def __init__(self,config:Config) -> None:
        self.pc_cache = []  # 新增：点云缓存列表
        self.pc_cache_pts = 0  # 当前累计点数
        
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
        self.heightmap = torch.zeros((1, 459), dtype=torch.float32, device=device)
        self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic,LowCmdGo)
        self.lowcmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic,LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateHandler)

        self.cloud_subscriber = ChannelSubscriber(config.cloud_topic,PointCloud2Go)
        self.cloud_subscriber.Init(self.PointCloudHandler)
        
        # try to shutdown the robot sport controller
        self.shutdown_sport_controller()

        self.wait_for_low_state()
        init_cmd_go2(self.low_cmd)
        self.counter = 0
        # self.pts = np.load("/home/unitree/deploy/deploy_real/data/cloud_points_21.npy") 
        self.hm = np.load("/home/unitree/deploy/deploy_real/heightmap.npy")
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
        # 累计缓存
        self.pc_cache.append(msg)
        self.pc_cache_pts += msg.width * msg.height

        # 当累计达到“伪全局帧”要求
        if self.pc_cache_pts >= self.config.full_frame_min_pts or len(self.pc_cache) >= self.config.pc_cache_len:
            self.point_cloud = self._merge_pointcloud_msgs(self.pc_cache)
            self.pc_cache.clear()
            self.pc_cache_pts = 0
    def _merge_pointcloud_msgs(self, msgs):
            base = msgs[0]
            # base = copy.deepcopy(msgs[0])
            
            merged = b''.join([bytes(m.data) for m in msgs])
            base.data = merged
            base.width = len(merged) // base.point_step
            base.row_step = base.width * base.point_step
            base.height = 1
            return base
    
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

    def run(self):
        start_time = time.time()
        if self.config.save_image:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_image_interval:
                self.code, self.data = self.video_client.GetImageSample()
                _ = convert_image(self.data, self.save_image_path)
                self.last_save_time = current_time
                
        # self.counter += 1
        # if self.counter % 4 == 0:
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        # quat = self.low_state.imu_state.quaternion
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

        self.heightmap = pointcloud2_to_heightmap(self.point_cloud)
        self.obs[45:45+459] = torch.clip(self.heightmap,-10.0,10.0).squeeze().cpu().numpy()
        
        proprio_obs = self.obs[:45].copy()
        proprio_obs_tensor = torch.from_numpy(proprio_obs).unsqueeze(0).unsqueeze(1).to(device)
        # update history buffer
        self.proprio_obs_buf = torch.cat([
            self.proprio_obs_buf[:, 1:],
            proprio_obs_tensor
        ], dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(1, -1).squeeze().cpu().numpy().copy()
        
        self.obs[45+459:] = proprio_obs_history

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).to(device)
        # print(f"obs_tensor shape: {obs_tensor.shape}")  # input shape: (1, 909)
        # policy_start_t = time.time()
        action_joint = self.policy(obs_tensor).detach()
        # print(f"model inference resume: {time.time()-policy_start_t} s")
        action_joint = action_joint.cpu().numpy().squeeze()
        
        # mapping policy action order to model action order
        action_motion = np.empty_like(action_joint)
        action_motion[self.config.joint2motor_idx] = action_joint
        
        self.target_dof_pos = action_motion * self.config.action_scale + self.config.default_angles
        
        self.action[:] = action_joint
        
        # for _ in range(4): # decimation: 4
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[i].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0
        
        self.send_cmd(self.low_cmd)
        
        time_resumed = time.time() - start_time
        # print(f"resumed time: {time_resumed} s")
        wait_time = self.config.control_dt - time_resumed
        if wait_time > 0:
            time.sleep(wait_time)
        # time.sleep(self.config.control_dt)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()

    config_path = f"{os.path.dirname(os.path.realpath(__file__))}/config/go2_navila.yaml"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()
    # last_time = time.time()
    print("Starting the control loop. Press Button 'SELECT' to exit.")
    while True:
        try:
            # last_time = time.time()
            controller.run()
            # print(f"Time resumed: {time.time()-last_time} s")
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print('Exit')
    
    controller.resume_sport_controller()
    

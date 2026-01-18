import numpy as np
import time
import torch
import os
from unitree_sdk2py.core.channel import ChannelPublisher,ChannelSubscriber,ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_,unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient

from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_zero_cmd, create_damping_cmd
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from utils import convert_image
from config import Config


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
        self.policy = torch.jit.load(config.policy_path)
        self._warm_up()
        
        self.qj = np.zeros(config.num_actions,dtype=np.float32) # pos
        self.dqj = np.zeros(config.num_actions,dtype=np.float32) # vec
        self.action = np.zeros(config.num_actions,dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs,dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0],dtype=np.float32)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic,LowCmdGo)
        self.lowcmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic,LowStateGo)
        # self.lowstate_subscriber.Init(self.LowStateHandler,10)
        self.lowstate_subscriber.Init(self.LowStateHandler)
        
        # try to shutdown the robot sport controller
        self.shutdown_sport_controller()

        self.wait_for_low_state()
        init_cmd_go2(self.low_cmd)
        
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
        #obs_nav = torch.ones((1,10))
        _obs = torch.ones((1,config.num_obs))
        for _ in range(10):
           # _ = self.policy_nav(obs_nav)
            _ = self.policy(_obs)
        print('Network has been warmed up.')

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def LowStateHandler(self,msg:LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

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
        proj_gravity = get_gravity_orientation(quat)  # imu_state quaternion: w, x, y, z
        
        base_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32) * self.config.obs_scales_ang_vel
        
        qj_obs = self.qj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.obs_scales_dof_pos
        
        dqj_obs = self.dqj.copy()
        dqj_obs = dqj_obs * self.config.obs_scales_dof_vel
        
        # mapping qj and dqj to match the policy input order
        qj_obs = qj_obs[self.config.joint2motor_idx]
        dqj_obs = dqj_obs[self.config.joint2motor_idx]
        
        # Controller inputs
        self.cmd[0] = np.clip(self.remote_controller.ly,     -1.0,1.0)
        self.cmd[1] = np.clip(self.remote_controller.lx * -1,-1.0,1.0)
        self.cmd[2] = np.clip(self.remote_controller.rx * -1,-1.0,1.0)
        
        self.obs[:3] = base_ang_vel
        self.obs[3:6] = proj_gravity
        self.obs[6:9] = self.cmd * self.config.command_scale
        self.obs[9:21] = qj_obs
        self.obs[21:33] = dqj_obs
        self.obs[33:45] = self.action

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        action_joint = self.policy(obs_tensor).detach().numpy().squeeze()

        # mapping policy action order to model action order
        action_motion = np.empty_like(action_joint)
        action_motion[self.config.joint2motor_idx] = action_joint
        
        self.target_dof_pos = action_motion * self.config.action_scale + self.config.default_angles
        
        self.action = action_joint
        
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[i].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


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
    

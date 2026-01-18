import time
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys
from typing import Tuple
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import deploy_mujoco.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOXEL_SIZE_XY = 0.06
RANGE_X = (-0.8,  0.2 + 1e-9)
RANGE_Y = (-0.8,  0.8 + 1e-9)
RANGE_Z = ( 0.0,  5.0)
OFFSET    = 0.0 
V_FOV = (0.0, 90.0) # 垂直视场
H_FOV = (-180.0, 180.0) # 水平视场
# X_BINS = torch.arange(RANGE_X[0], RANGE_X[1], VOXEL_SIZE_XY, device=device)
# Y_BINS = torch.arange(RANGE_Y[0], RANGE_Y[1], VOXEL_SIZE_XY, device=device)
# H, W   = len(X_BINS), len(Y_BINS)
# 17 * 27

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32
KEY_BINDINGS = {
    KEY_UP: (0, 1.0),    # x 方向正向
    # 's': (0, -1.0),    # x 方向反向
    # 'd': (1, +0.10),    # y 方向正向
    # 'a': (1, -0.10),    # y 方向反向
    KEY_RIGHT: (2, -1.0),    # +yaw
    KEY_LEFT: (2,  1.0),    # -yaw
    KEY_SPACE: 'reset'        # 清零
}

cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

def key_callback(keycode: int):
    """MuJoCo viewer"""
    global cmd
    if keycode not in KEY_BINDINGS:        # 非映射键直接忽略
        return

    binding = KEY_BINDINGS[keycode]
    if binding == 'reset':
        cmd[:] = 0.0                       # 立即清零速度指令
    else:
        axis, delta = binding
        cmd[axis] = np.clip(cmd[axis] + delta, -1.0, 1.0)

    print(f"[Keyboard] cmd -> {cmd}")

def pointcloud2_to_heightmap(
        pts,
        fields: Tuple[str, ...] = ("x", "y", "z"),
) -> torch.Tensor:
    # 读取点云数据 CloudPoint2 格式到 numpy (N,3)
    # 无限数置0，与 hit_vec[torch.isinf(hit_vec)] = 0.0 hit_vec[torch.isnan(hit_vec)] = 0.0 逻辑对应
    pts[~np.isfinite(pts)] = 0.0

    # np.save("deploy/deploy_real/cloud_points.npy", pts)
    # pts[:,0] = pts[:,0]*-1
    # save_pointcloud_2_5D_view(pts, "deploy/deploy_real/cloud_points-x.jpg")
    
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

# Implement the following functions refer in velocity_command.py
def update_command(data, cmd, heading_stiffness, heading_target, heading_command = True):
    """Post-processes the velocity command.

    This function sets velocity command to zero for standing environments and computes angular
    velocity from heading direction if the heading_command flag is set.
    """
    if heading_command:
        current_heading = utils.quat_to_heading_w(data.qpos[3:7])
        heading_err = utils.wrap_to_pi(heading_target - current_heading)
        cmd[2] = np.clip(heading_err*heading_stiffness,-1,1 )
    return cmd
        
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

import threading

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        policy_decimation = config["policy_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        policy2model = np.array(config["mapping_joints"], dtype=np.int32)
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        heading_stiffness = config["heading_stiffness"]
        heading_target = config["heading_target"]
        heading_command = config["heading_command"]
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    # target_dof_pos_shared  = default_angles.copy()
    action_prev_shared     = np.zeros(num_actions, np.float32)
    obs_tmp                = np.zeros(num_obs,  np.float32)
    
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    d.qpos[7:] = target_dof_pos.copy()
    d.qvel[:] = 0.0

    mujoco.mj_forward(m, d)

    for _ in range(20):
        mujoco.mj_step(m, d)
    
    m.opt.timestep = simulation_dt
    # -------- Joint order | default angle (rad)--------
    for j in range(m.njnt):
        name = mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_JOINT, j
        )
        if name is None:
            name = f"joint_{j}"

        angle = float(default_angles[j-1]) if j > 0 else float("nan")
        print(f"{j-1:2d}: {name:<15} | {angle: .4f}")
    print("-" * 50)
    # --------------------------------------------------
    # load policy
    policy = torch.jit.load(policy_path)
    print(f"Loaded policy from {policy_path}")
    deploy_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    pts = np.load(f"{deploy_root_path}/deploy_real/cloud_points_full/cloud_points_21.npy") 
    heightmap = torch.load(f"{deploy_root_path}/deploy_real/data_full/21/heightmap.pt", weights_only=True)
    
    action_policy_prev = np.zeros(num_actions, dtype=np.float32)
    proprio_obs_buf = torch.zeros(1, 9, 45, dtype=torch.float32)
    stop_event = threading.Event()
    # data_lock   = threading.Lock()
    # def control_loop():
    #     global d,proprio_obs_buf,viewer
    #     # if counter % policy_decimation == 0:
    #     while viewer.is_running():
    #         policy_start = time.time()
    #         # # Apply control signal here.
    #         # # cmd = update_command(d, cmd, heading_stiffness, heading_target, heading_command)
    #         # # create observation
    #         # with data_lock:
    #         #     qpos = d.qpos.copy()
    #         #     qvel = d.qvel.copy()
    #         # qj = qpos[7:]
    #         # dqj = qvel[6:]
    #         # quat = qpos[3:7]
    #         # lin_vel = qvel[:3]
    #         # ang_vel = qvel[3:6]

    #         # qj = (qj - default_angles) * dof_pos_scale
    #         # dqj = dqj * dof_vel_scale
            
    #         # # mapping qj and dqj to match the policy input order
    #         # qj  = qj[policy2model]
    #         # dqj = dqj[policy2model]
            
    #         # base_rpy = utils.quat_to_rpy(quat)
    #         # lin_vel = lin_vel * lin_vel_scale
    #         # ang_vel = ang_vel * ang_vel_scale

    #         # obs[:3] = ang_vel
    #         # obs[3:6] = base_rpy
    #         # obs[6:9] = cmd * cmd_scale
    #         # obs[9:21] = qj
    #         # obs[21:33] = dqj
    #         # obs[33:45] = action_policy_prev
            
    #         # # _pts = pts.copy()
    #         # # _pts[:,0] = _pts[:,0]*-1
    #         # # _pts = _pts[:, [1, 0, 2]]
            
    #         # # theta = - np.pi / 6  # 45 deg
    #         # # c, s = np.cos(theta), np.sin(theta)
    #         # # Rz = np.array([
    #         # #     [c, -s, 0],
    #         # #     [s,  c, 0],
    #         # #     [0,  0, 1]
    #         # # ], dtype=np.float32)

    #         # # _pts = _pts @ Rz.T               
    #         # # hm = torch.from_numpy(heightmap[45:45+459])
    #         # # hm=hm.view(1,17,27)
    #         # # # hm=torch.flip(hm,dims=[1,2])
    #         # # hm=torch.flip(hm,dims=[1])
    #         # # obs[45:45+459] = pointcloud2_to_heightmap(_pts).squeeze().cpu().numpy()
    #         # obs[45:45+459] = np.load("/home/penghm/workspace/isaaclab_go2/deploy/deploy_real/data_full/21/heightmap.npy")
    #         # proprio_obs = obs[:45].copy()
    #         # proprio_obs_tensor = torch.from_numpy(proprio_obs).unsqueeze(0).unsqueeze(1)
    #         # proprio_obs_buf = torch.cat([
    #         #     proprio_obs_buf[:, 1:],
    #         #     proprio_obs_tensor
    #         # ], dim=1)
    #         # proprio_obs_history = proprio_obs_buf.view(1, -1).numpy().squeeze().copy()
    #         # obs[45+459:] = proprio_obs_history
            
    #         # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    #         # # policy inference
    #         # action_policy = policy(obs_tensor).detach().numpy().squeeze()
            
    #         # # mapping policy action order to model action order
    #         # action_model = np.empty_like(action_policy)
    #         # action_model[policy2model] = action_policy

    #         # # model action order
    #         # target_dof_pos = action_model * action_scale + default_angles
    #         # # policy action order  used for next step
    #         # action_policy_prev[:] = action_policy
    #         # time.sleep(0.02)
            
    #         with data_lock:
    #             # for _ in range(10):
    #             tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
    #             d.ctrl[:] = tau
    #             mujoco.mj_step(m, d)
            
    #         resumed_time  = time.time() - policy_start
    #         # print(f"policy resume time: {resumed_time} s")
    #         wait_time = 0.005 - resumed_time
    #         if wait_time > 0:
    #             time.sleep(wait_time)
                
    # control  = threading.Thread(target=control_loop,  daemon=True)
    
    with mujoco.viewer.launch_passive(m, d,key_callback=key_callback) as viewer:
        viewer.cam.azimuth   = 0
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.5
        viewer.cam.lookat[:] = d.qpos[:3]
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # control.start()
        init_flat = False
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            if config["lock_camera"]:
                viewer.cam.lookat[:] = d.qpos[:3] # lock camera focus on the robot base
            
            
            # counter += 10
            # counter += 1
            counter += 1
            if counter % control_decimation == 0:
                policy_start = time.time()
                # Apply control signal here.
                # cmd = update_command(d, cmd, heading_stiffness, heading_target, heading_command)
                # create observation
                # with data_lock:
                # qpos = d.qpos.copy()
                # qvel = d.qvel.copy()
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                ang_vel = d.qvel[3:6]
                # print(qj)
                # print(default_angles)
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                
                # mapping qj and dqj to match the policy input order
                qj  = qj[policy2model]
                dqj = dqj[policy2model]
                # print(qj)
                
                base_rpy = utils.quat_to_rpy(quat)

                ang_vel = ang_vel * ang_vel_scale

                obs[:3] = ang_vel
                obs[3:6] = base_rpy
                obs[6:9] = cmd * cmd_scale
                obs[9:21] = qj
                obs[21:33] = dqj
                obs[33:45] = action_policy_prev

                obs[45:45+459] = heightmap.cpu().numpy()
                proprio_obs = obs[:45].copy()
                proprio_obs_tensor = torch.from_numpy(proprio_obs).unsqueeze(0).unsqueeze(1)
                if init_flat is False:
                    proprio_obs_buf = torch.cat([proprio_obs_tensor] * 9, dim=1)
                    init_flat = True
                else:
                    proprio_obs_buf = torch.cat([
                        proprio_obs_buf[:, 1:],
                        proprio_obs_tensor
                    ], dim=1)
                proprio_obs_history = proprio_obs_buf.view(1, -1).numpy().squeeze().copy()
                # print(proprio_obs_tensor)
                obs[45+459:909] = proprio_obs_history
                # print(obs)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # print(obs_tensor.size())
                # policy inference
                action_policy = policy(obs_tensor).detach().numpy().squeeze()
                # print(action_policy)
                # mapping policy action order to model action order
                action_model = np.empty_like(action_policy)
                action_model[policy2model] = action_policy

                # model action order
                target_dof_pos = action_model * action_scale + default_angles
                # policy action order  used for next step
                action_policy_prev[:] = action_policy
                # time.sleep(0.1)
                
                # with data_lock:
                for _ in range(4):
                    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)
                
                # resumed_time  = time.time() - policy_start
                # print(f"policy resume time: {resumed_time} s")
                # wait_time = 0.02 - resumed_time
                # if wait_time > 0:
                #     time.sleep(wait_time)
                    
                # if counter % control_decimation == 0: # 0.02
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                # with data_lock:
                
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    # stop_event.set()
    # control.join()
    print("Simulation finished.")

import time
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import deploy_mujoco.utils as utils

# Implement the following functions refer in pose_2d_command.py
def update_command(data, pos_command_w, heading_command_w):
    """Post-processes the velocity command.

    Re-target the position command to the current root state.
    """
    target_vec = pos_command_w - data.qpos[:3]
    pos_command_b = utils.quat_rotate_inverse(utils.yaw_quat(data.qpos[3:7]), target_vec)
    heading_command_b = utils.wrap_to_pi(heading_command_w - utils.quat_to_heading_w(data.qpos[3:7]))

    return np.concatenate([pos_command_b, np.array([heading_command_b])])


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


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        high_policy_path = config["policy_path"]["high_level"]
        low_policy_path = config["policy_path"]["low_level"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        high_level_decimation = config["control_decimation"]["high_level"]
        low_level_decimation = config["control_decimation"]["low_level"]

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
        high_num_obs = config["num_obs"]["high_level"]
        low_num_obs = config["num_obs"]["low_level"]
        policy2model = np.array(config["mapping_joints"], dtype=np.int32)
        
        pos_cmd_b = np.array(config["cmd_init"][:3], dtype=np.float32) * cmd_scale[:3]
        pos_cmd_b[2] = 0.0  # ensure no z command
        heading_cmd_b = np.array(config["cmd_init"][3], dtype=np.float32) * cmd_scale[3]
        vel_cmd = np.zeros(3, dtype=np.float32)
        
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    high_level_obs = np.zeros(high_num_obs, dtype=np.float32)
    low_level_obs = np.zeros(low_num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    # print(d.qpos[:7])  # print initial base frame position and orientation [0,0,0,445], [1,0,0,0]
    pos_cmd_w = d.qpos[:3].copy()
    heading_cmd_w = utils.quat_to_heading_w(d.qpos[3:7].copy()) 
    # accumulator for continuous input commands(TODO)
    pos_cmd_w += pos_cmd_b
    heading_cmd_w += heading_cmd_b
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
    high_policy = torch.jit.load(high_policy_path)
    low_policy = torch.jit.load(low_policy_path)
    print(f"Loaded high level policy from {high_policy_path}")
    print(f"Loaded low level policy from {low_policy_path}")
    
    action_policy_prev = np.zeros(num_actions, dtype=np.float32)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.5
        viewer.cam.lookat[:] = d.qpos[:3]
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            if config["lock_camera"]:
                # lock camera focus on the robot base
                viewer.cam.lookat[:] = d.qpos[:3]
            
            counter += 1
            # low_level_decimation times physics step per low-level policy/control step
            if counter % low_level_decimation == 0:
                # Apply control signal here.
                
                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                
                # mapping qj and dqj to match the policy input order
                qj  = qj[policy2model]
                dqj = dqj[policy2model]
                
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale
                
                # high_level_decimation times physics step per high-level policy/control step
                if counter % high_level_decimation == 0:
                    # update command 
                    cmd_b = update_command(d, pos_cmd_w, heading_cmd_w)
                    # create obs
                    high_level_obs[:3] = lin_vel
                    high_level_obs[3:6] = gravity_orientation
                    high_level_obs[6:10] = cmd_b
                    high_obs_tensor = torch.from_numpy(high_level_obs).unsqueeze(0)
                    vel_cmd = high_policy(high_obs_tensor).detach().numpy().squeeze()
                
                low_level_obs[:3] = lin_vel
                low_level_obs[3:6] = ang_vel
                low_level_obs[6:9] = gravity_orientation
                low_level_obs[9:12] = vel_cmd
                low_level_obs[12 : 12 + num_actions] = qj
                low_level_obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                low_level_obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action_policy_prev
                obs_tensor = torch.from_numpy(low_level_obs).unsqueeze(0)
                # policy inference
                
                action_policy = low_policy(obs_tensor).detach().numpy().squeeze()
                
                # mapping policy action order to model action order
                action_model = np.empty_like(action_policy)
                action_model[policy2model] = action_policy

                # model action order
                target_dof_pos = action_model * action_scale + default_angles
                # policy action order used for next step
                action_policy_prev[:] = action_policy

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

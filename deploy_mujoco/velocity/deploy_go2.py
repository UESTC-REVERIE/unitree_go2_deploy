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
        policy_decimation = config["policy_decimation"]
        control_decimation = config["control_decimation"]
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

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
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
    
    action_policy_prev = np.zeros(num_actions, dtype=np.float32)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.azimuth   = 0
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.5
        viewer.cam.lookat[:] = d.qpos[:3]
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        step_control = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            if config["lock_camera"]:
                viewer.cam.lookat[:] = d.qpos[:3] # lock camera focus on the robot base
            
            counter += 1
            if counter % control_decimation == 0:
                # if counter % policy_decimation == 0:
                policy_start = time.time()
                # Apply control signal here.
                cmd = update_command(d, cmd, heading_stiffness, heading_target, heading_command)
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

                if config["num_obs"] == 48:
                    obs[:3] = lin_vel
                    obs[3:6] = ang_vel
                    obs[6:9] = gravity_orientation
                    obs[9:12] = cmd * cmd_scale
                    obs[12 : 12 + num_actions] = qj
                    obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                    obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action_policy_prev
                elif config["num_obs"] == 45:
                    obs[:3] = ang_vel
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:21] = qj
                    obs[21:33] = dqj
                    obs[33:45] = action_policy_prev
                else:
                    raise ValueError(f"Unsupported number of observations: {config.num_obs}")
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action_policy = policy(obs_tensor).detach().numpy().squeeze()
                
                # mapping policy action order to model action order
                action_model = np.empty_like(action_policy)
                action_model[policy2model] = action_policy

                # model action order
                target_dof_pos = action_model * action_scale + default_angles
                # policy action order used for next step
                action_policy_prev[:] = action_policy
                # print(f"policy resume time {time.time()-policy_start_t} s")
                # time.sleep(0.02)
                for _ in range(4):
                    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

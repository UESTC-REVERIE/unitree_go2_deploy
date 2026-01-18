import numpy as np
import yaml

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.joint2motor_idx = config["joint2motor_idx"]
            
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.cloud_topic = config["cloud_topic"]
            
            self.policy_path = config["policy_path"]

            self.kps = np.array(config["kps"],dtype=np.float32)
            self.kds = np.array(config["kds"],dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.obs_scales_ang_vel = config["obs_scales_ang_vel"]
            self.obs_scales_dof_pos = config["obs_scales_dof_pos"]
            self.obs_scales_dof_vel = config["obs_scales_dof_vel"]

            self.command_scale = config["command_scale"]
            self.action_scale = config["action_scale"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"] 
            
            self.history_length = config["history_length"]
            self.proprio_obs_dim = config["proprio_obs_dim"]
            self.full_frame_min_pts = config.get("full_frame_min_pts", 16000)
            self.pc_cache_len = config.get("pc_cache_len", 4)
            
            self.save_image = config.get("save_image", False)  # Default to False if not specified
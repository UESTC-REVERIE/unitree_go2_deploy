import numpy as np
import time
import torch
import torch.nn.functional as F
import sys
import os


from typing import Tuple
from utils import *
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

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
        pts,
        fields: Tuple[str, ...] = ("x", "y", "z"),
) -> torch.Tensor:
    # 读取点云数据 CloudPoint2 格式到 numpy (N,3)
    # pts = cloud_to_xyz_numpy(msg) 
    # 无限数置0，与 hit_vec[torch.isinf(hit_vec)] = 0.0 hit_vec[torch.isnan(hit_vec)] = 0.0 逻辑对应
    pts[~np.isfinite(pts)] = 0.0

    # np.save("deploy/deploy_real/cloud_points.npy", pts)
    # save_pointcloud_2_5D_view(pts, "deploy/deploy_real/cloud_points.jpg")
    
    # pts[:,0] = pts[:,0]*-1
    # save_pointcloud_2_5D_view(pts, "deploy/deploy_real/cloud_points-x.jpg")

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
        

if __name__ == "__main__":
    # pts = np.load("deploy/deploy_real/isaac_cloud_points.npy")
    # # pts = pts[:, [1, 0, 2]]
    # save_pointcloud_2_5D_view(pts, "deploy/deploy_real/isaac_cloud_points.jpg")

    # heightmap = pointcloud2_to_heightmap(pts)
    # heightmap_np = heightmap.squeeze().cpu().numpy()
    
    # save_heightmap_visualization(heightmap_np,"deploy/deploy_real/isaac_heightmap_ours.jpg")
    # heightmap_np = np.load("deploy/deploy_real/isaac_heightmap.npy")
    # save_heightmap_visualization(heightmap_np,"deploy/deploy_real/isaac_heightmap.jpg")
    
    # # pts[:,0] = pts[:,0]*-1
    # # save_pointcloud_2_5D_view(pts, "deploy/deploy_real/test-x_yx.jpg")
    
    # ----------------------------------------
    # 可视化从真实环境保存下来的雷达数据
    index = 4
    pts = np.load(f"deploy/deploy_real/cloud_points_full/cloud_points_{index}.npy")
    save_path = f"deploy/deploy_real/data_full/{index}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # 雷达数据的转换，与 isaaclab 中的数据对齐
    pts[:, 0] = pts[:, 0] * -1
    pts = pts[:, [1, 0, 2]]
    # pts[:, 2] = pts[:, 2] * -1
    theta = - np.pi / 6
    # theta = - np.pi / 5.5
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    pts = pts @ Rz.T
    
    save_pointcloud_2_5D_view(pts, f"{save_path}/real_cloud_points.jpg")
    heightmap = pointcloud2_to_heightmap(pts)
    torch.save(heightmap,f"{save_path}/heightmap.pt")
    np.save(f"{save_path}/heightmap.npy",heightmap.cpu().numpy())
    heightmap_np = heightmap.squeeze().cpu().numpy()
    
    save_heightmap_visualization(heightmap_np,f"{save_path}/real_heightmap.jpg")
    
    
    

# from unitree_sdk2py.core.channel import ChannelFactoryInitialize
# from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2  
import numpy as np
import os
def convert_image(data, save_path = None):
    """_summary_

    Args:
        data (_type_): the original image data from the video stream
        save_path (_type_): path to save the image, if None, the numpy image will be returned without saving

    Returns:
        cv2.typing.MatLike: numpy image
    """
    # Convert to numpy image
    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Save the image
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        cv2.imwrite(save_path, image)
        # print(f"Front camera image saved to {save_path}")

    return image

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

import matplotlib.pyplot as plt

def save_pointcloud_2_5D_view(xyz: np.ndarray, filename: str):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # 2.5D 斜视角 scatter
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.8)
    ax.view_init(elev=45, azim=-60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('2.5D oblique view of point cloud')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"2.5D oblique view image saved to {filename}")

    fig_topdown = plt.figure(figsize=(8, 8))
    ax_topdown = fig_topdown.add_subplot(111)
    sc = ax_topdown.scatter(x, y, c=z, s=1, cmap='viridis', alpha=0.8)
    ax_topdown.set_xlabel('X')
    ax_topdown.set_ylabel('Y')
    ax_topdown.set_title('Top-down view of point cloud')
    ax_topdown.axis('equal')
    plt.colorbar(sc, ax=ax_topdown, label='Z height')
    ax_topdown.grid(True)
    plt.tight_layout()

    topdown_filename = filename.replace('.jpg', '_topdown.jpg')
    plt.savefig(topdown_filename, dpi=300)
    plt.close()
    print(f"Top-down view image saved to {topdown_filename}")
    

def save_heightmap_visualization(heightmap, save_path):
    # voxel_size_xy = 0.06
    # range_x = [-0.8, 0.2 + 1e-9]
    # range_y = [-0.8, 0.8 + 1e-9]

    H = 17
    W = 27

    print(f"Loaded heightmap shape: {heightmap.shape}, expected H={H}, W={W}")

    # 重新 reshape 为 2D
    heightmap_2d = heightmap.reshape(H, W)

    # 可视化
    plt.figure(figsize=(8, 8))
    img = plt.imshow(heightmap_2d, cmap='viridis', origin='lower')
    plt.colorbar(img, label='Height value')
    plt.title('Heightmap Visualization (env0)')
    plt.xlabel('X bins')
    plt.ylabel('Y bins')
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved heightmap visualization to {save_path}")

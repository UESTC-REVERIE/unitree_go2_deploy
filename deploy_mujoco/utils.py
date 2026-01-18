# HARD LINK between navigation and velocity
# shared utility functions for navigation and velocity tasks

# math
import numpy as np
def wrap_to_pi(angle):
    """mapping angle in (-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_to_heading_w(quat: np.ndarray) -> float:
    """calculate the heading angle (yaw) in world frame, equivalent to Isaac Lab's `heading_w`.

    Args:
        quat: [qw, qx, qy, qz] 

    Returns:
        float: yaw angle in radians, in the range (-pi, pi]
    """
    qw, qx, qy, qz = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)

# refer to source/isaaclab/isaaclab/utils/math.py
def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z).
        v: The vector in (x, y, z). 

    Returns:
        The rotated vector in (x, y, z). 
    """
    q_w = q[0]
    q_vec = q[1:4]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes a given input vector to unit length.

    Args:
        x: Input vector.
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized vector.
    """
    norm = max(np.linalg.norm(x), eps)
    return x / norm
def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z).

    Returns:
        A quaternion with only yaw component.
    """
    qw, qx, qy, qz = quat
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    quat_yaw = np.zeros_like(quat)
    quat_yaw[0] = np.cos(yaw / 2.0)
    quat_yaw[3] = np.sin(yaw / 2.0) 
    return normalize(quat_yaw)

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
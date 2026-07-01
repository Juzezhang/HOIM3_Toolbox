"""
Camera utilities for multi-view mask validation.
Handles camera parameter loading, projection, and coordinate transformations.
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_camera_params(
    calibration_path: str,
    view_indices: List[int],
    scale_factor: float = 2.0
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load camera parameters for selected views and scale K to match mask resolution.

    Args:
        calibration_path: Path to calibration.json file
        view_indices: List of view indices to load (e.g., [0, 2, 5, 6, 7, 8, 10, 11, 14, 15])
        scale_factor: Scale factor to convert K from calibration resolution to mask resolution.
                     Default 2.0 (4K to 1080p)

    Returns:
        Dictionary mapping view_index to {'K': 3x3, 'R': 3x3, 'T': 3x1}
    """
    with open(calibration_path, 'r') as f:
        calib_data = json.load(f)

    cameras = {}
    for view_idx in view_indices:
        view_key = str(view_idx)
        if view_key not in calib_data:
            raise ValueError(f"View {view_idx} not found in calibration file")

        cam = calib_data[view_key]

        # Load intrinsic matrix K and scale for mask resolution
        K = np.array(cam['K']).reshape(3, 3).astype(np.float64)
        K_scaled = K / scale_factor
        K_scaled[2, 2] = 1.0  # Keep homogeneous coordinate

        # Load extrinsic matrix RT
        RT = np.array(cam['RT']).astype(np.float64)
        if len(RT) == 16:
            RT = RT.reshape(4, 4)
        else:
            RT = RT.reshape(3, 4)

        R = RT[:3, :3]
        T = RT[:3, 3].reshape(3, 1)

        cameras[view_idx] = {
            'K': K_scaled,
            'R': R,
            'T': T,
            'K_original': K,  # Keep original for reference
        }

    return cameras


def project_points_to_image(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        T: (3, 1) translation vector

    Returns:
        points_2d: (N, 2) array of 2D points in image coordinates (x, y)
        depths: (N,) array of depths (z in camera coordinates)
    """
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 3)

    # Transform to camera coordinates: P_cam = R @ P_world + T
    points_cam = (R @ points_3d.T + T).T  # (N, 3)

    # Get depths
    depths = points_cam[:, 2]

    # Project to image coordinates: p = K @ P_cam
    points_homo = (K @ points_cam.T).T  # (N, 3)

    # Normalize by depth
    points_2d = points_homo[:, :2] / (points_homo[:, 2:3] + 1e-8)

    return points_2d, depths


def get_camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Get camera center in world coordinates.

    The camera center C in world coordinates satisfies: R @ C + T = 0
    Therefore: C = -R^T @ T

    Args:
        R: (3, 3) rotation matrix
        T: (3, 1) translation vector

    Returns:
        center: (3,) camera center in world coordinates
    """
    center = -R.T @ T
    return center.flatten()


def get_ray_direction(
    pixel: np.ndarray,
    K: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    Get ray direction from camera through a pixel in world coordinates.

    Args:
        pixel: (2,) pixel coordinates (x, y)
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix

    Returns:
        direction: (3,) normalized ray direction in world coordinates
    """
    # Convert pixel to homogeneous coordinates
    pixel_homo = np.array([pixel[0], pixel[1], 1.0])

    # Back-project to camera coordinates
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ pixel_homo

    # Transform to world coordinates (rotation only, as direction)
    ray_world = R.T @ ray_cam

    # Normalize
    ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-8)

    return ray_world


def triangulate_point_from_rays(
    camera_centers: List[np.ndarray],
    ray_directions: List[np.ndarray]
) -> np.ndarray:
    """
    Triangulate a 3D point from multiple camera rays using least squares.

    Solves for point P that minimizes sum of squared distances to all rays.
    Each ray is defined as: C + t * d, where C is camera center and d is direction.

    Args:
        camera_centers: List of (3,) camera centers
        ray_directions: List of (3,) normalized ray directions

    Returns:
        point_3d: (3,) triangulated 3D point
    """
    n_views = len(camera_centers)
    if n_views < 2:
        raise ValueError("Need at least 2 views for triangulation")

    # Build linear system: (I - d @ d.T) @ P = (I - d @ d.T) @ C
    # Stacking these gives: A @ P = b
    A = np.zeros((3 * n_views, 3))
    b = np.zeros(3 * n_views)

    for i, (C, d) in enumerate(zip(camera_centers, ray_directions)):
        d = d.reshape(3, 1)
        I_minus_ddT = np.eye(3) - d @ d.T
        A[3*i:3*(i+1), :] = I_minus_ddT
        b[3*i:3*(i+1)] = (I_minus_ddT @ C.reshape(3, 1)).flatten()

    # Solve least squares
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d


def get_calibration_date_for_sequence(
    dataset_info_path: str,
    seq_name: str
) -> Optional[str]:
    """
    Get the calibration date for a given sequence name.

    Args:
        dataset_info_path: Path to dataset_information.json
        seq_name: Sequence name (e.g., "bedroom_data01")

    Returns:
        Calibration date string (e.g., "20230912") or None if not found
    """
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    for date, sequences in dataset_info.items():
        if seq_name in sequences:
            return date

    return None


def is_point_in_image(
    point_2d: np.ndarray,
    img_width: int,
    img_height: int,
    margin: int = 0
) -> bool:
    """
    Check if a 2D point is within image bounds.

    Args:
        point_2d: (2,) or (N, 2) point coordinates
        img_width: Image width
        img_height: Image height
        margin: Optional margin to shrink valid region

    Returns:
        Boolean or boolean array indicating if point(s) are in image
    """
    if point_2d.ndim == 1:
        x, y = point_2d
        return (margin <= x < img_width - margin) and (margin <= y < img_height - margin)
    else:
        x, y = point_2d[:, 0], point_2d[:, 1]
        return (x >= margin) & (x < img_width - margin) & (y >= margin) & (y < img_height - margin)

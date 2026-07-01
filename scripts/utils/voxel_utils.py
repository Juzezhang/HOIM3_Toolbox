"""
Voxel utilities for multi-view mask validation.
Handles 3D voxel grid creation, visibility computation, and mask sampling.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage


def get_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box of non-zero region in mask.

    Args:
        mask: (H, W) binary mask

    Returns:
        (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)


def get_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Get centroid of non-zero region in mask.

    Args:
        mask: (H, W) binary mask

    Returns:
        (x, y) centroid or None if mask is empty
    """
    if not np.any(mask):
        return None

    y_coords, x_coords = np.where(mask > 0)
    return float(np.mean(x_coords)), float(np.mean(y_coords))


def estimate_3d_bbox_from_masks(
    masks: Dict[int, np.ndarray],
    cameras: Dict[int, Dict[str, np.ndarray]],
    img_size: Tuple[int, int],
    padding_ratio: float = 0.3,
    min_bbox_size: float = 0.5,
    max_bbox_size: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate 3D bounding box from multi-view mask projections.

    Args:
        masks: Dictionary mapping view_idx to (H, W) binary mask
        cameras: Dictionary mapping view_idx to {'K', 'R', 'T'}
        img_size: (width, height) of masks
        padding_ratio: Ratio to pad the estimated bbox
        min_bbox_size: Minimum bbox size in meters
        max_bbox_size: Maximum bbox size in meters

    Returns:
        bbox_min: (3,) minimum corner of 3D bbox
        bbox_max: (3,) maximum corner of 3D bbox
    """
    from .camera_utils import get_camera_center, get_ray_direction, triangulate_point_from_rays

    img_width, img_height = img_size

    # Collect valid mask centroids and bboxes
    view_indices = []
    centroids_2d = []
    bboxes_2d = []

    for view_idx, mask in masks.items():
        centroid = get_mask_centroid(mask)
        bbox = get_mask_bbox(mask)

        if centroid is not None and bbox is not None:
            view_indices.append(view_idx)
            centroids_2d.append(centroid)
            bboxes_2d.append(bbox)

    if len(view_indices) < 2:
        # Fallback to default human-sized bbox at origin
        return np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 2.0])

    # Triangulate centroid
    camera_centers = []
    ray_directions = []

    for view_idx, centroid in zip(view_indices, centroids_2d):
        cam = cameras[view_idx]
        center = get_camera_center(cam['R'], cam['T'])
        direction = get_ray_direction(np.array(centroid), cam['K'], cam['R'])
        camera_centers.append(center)
        ray_directions.append(direction)

    center_3d = triangulate_point_from_rays(camera_centers, ray_directions)

    # Estimate size from 2D bbox sizes
    # Use the median bbox diagonal as reference
    bbox_diagonals = []
    for bbox in bboxes_2d:
        x_min, y_min, x_max, y_max = bbox
        diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        bbox_diagonals.append(diagonal)

    median_diagonal = np.median(bbox_diagonals)

    # Estimate 3D size based on typical projection
    # Assuming average distance ~5m and focal length ~3000px (at 4K) / 2 = 1500px at 1080p
    # Size ~= depth * pixel_size / focal_length
    # We use a heuristic: larger masks -> larger objects
    avg_focal = 1500  # Approximate focal length at 1080p
    estimated_depth = 5.0  # meters

    estimated_size = (median_diagonal / avg_focal) * estimated_depth
    estimated_size = np.clip(estimated_size, min_bbox_size, max_bbox_size)

    # Add padding
    half_size = estimated_size * (1 + padding_ratio) / 2

    bbox_min = center_3d - np.array([half_size, half_size, half_size * 0.5])
    bbox_max = center_3d + np.array([half_size, half_size, half_size * 1.5])

    # Ensure minimum size
    for i in range(3):
        if bbox_max[i] - bbox_min[i] < min_bbox_size:
            mid = (bbox_max[i] + bbox_min[i]) / 2
            bbox_min[i] = mid - min_bbox_size / 2
            bbox_max[i] = mid + min_bbox_size / 2

    return bbox_min, bbox_max


def create_voxel_grid(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int = 128
) -> Tuple[np.ndarray, float]:
    """
    Create 3D voxel grid centers within a bounding box.

    Args:
        bbox_min: (3,) minimum corner of bbox
        bbox_max: (3,) maximum corner of bbox
        resolution: Number of voxels along each dimension

    Returns:
        voxel_centers: (resolution^3, 3) array of voxel center coordinates
        voxel_size: Size of each voxel (assuming uniform grid)
    """
    bbox_size = bbox_max - bbox_min
    voxel_size = np.max(bbox_size) / resolution

    # Create grid coordinates
    x = np.linspace(bbox_min[0] + voxel_size/2, bbox_max[0] - voxel_size/2, resolution)
    y = np.linspace(bbox_min[1] + voxel_size/2, bbox_max[1] - voxel_size/2, resolution)
    z = np.linspace(bbox_min[2] + voxel_size/2, bbox_max[2] - voxel_size/2, resolution)

    # Create meshgrid and flatten
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxel_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    return voxel_centers, voxel_size


def compute_visibility_table(
    voxel_centers: np.ndarray,
    cameras: Dict[int, Dict[str, np.ndarray]],
    img_size: Tuple[int, int],
    depth_check: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute visibility table: which voxels are visible from which cameras.

    A voxel is visible if:
    1. Its projection is within image bounds
    2. Its depth is positive (in front of camera)

    Args:
        voxel_centers: (N, 3) voxel center coordinates
        cameras: Dictionary mapping view_idx to {'K', 'R', 'T'}
        img_size: (width, height) of image
        depth_check: Whether to check depth positivity

    Returns:
        visibility: Dictionary mapping view_idx to (N,) boolean array
    """
    from .camera_utils import project_points_to_image, is_point_in_image

    img_width, img_height = img_size
    visibility = {}

    for view_idx, cam in cameras.items():
        points_2d, depths = project_points_to_image(
            voxel_centers, cam['K'], cam['R'], cam['T']
        )

        # Check if within image bounds
        in_bounds = is_point_in_image(points_2d, img_width, img_height)

        # Check depth positivity
        if depth_check:
            positive_depth = depths > 0
            visible = in_bounds & positive_depth
        else:
            visible = in_bounds

        visibility[view_idx] = visible

    return visibility


def sample_mask_at_projections(
    mask: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    voxel_centers: np.ndarray,
    interpolation: str = 'nearest'
) -> np.ndarray:
    """
    Sample mask values at projected voxel locations.

    Args:
        mask: (H, W) binary mask (values 0 or 255)
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        T: (3, 1) translation vector
        voxel_centers: (N, 3) voxel center coordinates
        interpolation: 'nearest' or 'bilinear'

    Returns:
        values: (N,) mask values at projected locations (0 or 1)
    """
    from .camera_utils import project_points_to_image

    H, W = mask.shape
    points_2d, depths = project_points_to_image(voxel_centers, K, R, T)

    # Convert mask to float [0, 1]
    mask_float = (mask > 127).astype(np.float32)

    if interpolation == 'nearest':
        # Round to nearest pixel
        x = np.clip(np.round(points_2d[:, 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(points_2d[:, 1]).astype(int), 0, H - 1)
        values = mask_float[y, x]
    else:
        # Bilinear interpolation using scipy
        from scipy.ndimage import map_coordinates
        coords = np.stack([points_2d[:, 1], points_2d[:, 0]], axis=0)
        values = map_coordinates(mask_float, coords, order=1, mode='constant', cval=0)

    # Mark out-of-bounds as 0
    out_of_bounds = (
        (points_2d[:, 0] < 0) | (points_2d[:, 0] >= W) |
        (points_2d[:, 1] < 0) | (points_2d[:, 1] >= H) |
        (depths <= 0)
    )
    values[out_of_bounds] = 0

    return values


def project_voxels_to_mask(
    occupancy: np.ndarray,
    voxel_centers: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    img_size: Tuple[int, int],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Project occupied voxels back to create an ideal mask.

    Args:
        occupancy: (N,) occupancy probability for each voxel
        voxel_centers: (N, 3) voxel center coordinates
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        T: (3, 1) translation vector
        img_size: (width, height) of output mask
        threshold: Occupancy threshold for considering a voxel occupied

    Returns:
        ideal_mask: (H, W) binary mask
    """
    from .camera_utils import project_points_to_image

    img_width, img_height = img_size
    ideal_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Get occupied voxels
    occupied = occupancy > threshold
    if not np.any(occupied):
        return ideal_mask

    occupied_centers = voxel_centers[occupied]
    points_2d, depths = project_points_to_image(occupied_centers, K, R, T)

    # Filter valid projections
    valid = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height) &
        (depths > 0)
    )

    if not np.any(valid):
        return ideal_mask

    x = np.clip(np.round(points_2d[valid, 0]).astype(int), 0, img_width - 1)
    y = np.clip(np.round(points_2d[valid, 1]).astype(int), 0, img_height - 1)

    ideal_mask[y, x] = 255

    # Dilate to fill gaps (voxel projection may be sparse)
    from scipy.ndimage import binary_dilation
    struct = np.ones((5, 5), dtype=bool)
    ideal_mask = (binary_dilation(ideal_mask > 0, structure=struct) * 255).astype(np.uint8)

    return ideal_mask


def compute_mask_metrics(
    raw_mask: np.ndarray,
    ideal_mask: np.ndarray,
    all_raw_areas: List[float]
) -> Tuple[float, float]:
    """
    Compute precision and area ratio metrics for mask validation.

    Args:
        raw_mask: (H, W) original mask
        ideal_mask: (H, W) ideal mask from 3D projection
        all_raw_areas: List of raw mask areas from all views (for median calculation)

    Returns:
        precision: Intersection / Raw_Area (how much of raw mask is valid)
        area_ratio: Raw_Area / Median_Area (relative size check)
    """
    raw_binary = raw_mask > 127
    ideal_binary = ideal_mask > 127

    raw_area = np.sum(raw_binary)
    intersection = np.sum(raw_binary & ideal_binary)

    # Precision
    precision = intersection / (raw_area + 1e-6)

    # Area ratio
    median_area = np.median(all_raw_areas) if all_raw_areas else raw_area
    area_ratio = raw_area / (median_area + 1e-6)

    return float(precision), float(area_ratio)

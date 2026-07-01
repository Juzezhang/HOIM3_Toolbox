"""
Multi-View Mask Sanity Check (GPU-Batched)

Validates mask quality using multi-view geometry:
1. Build visual hull via iterative reweighted voting in 3D voxel space
2. Project occupied voxels to 2D bounding box per view
3. Check if mask overlaps with the projected bbox

All view and frame operations are batched on GPU for maximum throughput.
Data loading uses threaded prefetch pipeline to overlap I/O with GPU compute.

Usage:
    python scripts/multi_view_mask_check.py \
        --root_path "/simurgh/group/juze/datasets/HOI-M3" \
        --seq_name "bedroom_data01" \
        --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity"
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import concurrent.futures
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from os.path import join

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.camera_utils import (
    load_camera_params,
    get_calibration_date_for_sequence,
)
from scripts.utils.mask_io import (
    SequenceShardReaders,
    load_frame_masks_shard,
    load_frame_masks_shard_full,
    load_frame_masks_npz,
    detect_mask_format,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-view mask sanity check')

    # Required arguments
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to HOI-M3 dataset')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='Sequence name (e.g., bedroom_data01)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for validity results')

    # View selection
    parser.add_argument('--views', type=int, nargs='+',
                        default=[0, 2, 5, 6, 7, 8, 10, 11, 14, 15],
                        help='View indices to process')
    parser.add_argument('--all_views', action='store_true',
                        help='Use all available views from calibration (up to 42, overrides --views)')

    # Algorithm parameters
    parser.add_argument('--voxel_res', type=int, default=48,
                        help='Voxel grid resolution (lower=faster, default: 48)')
    parser.add_argument('--max_iters', type=int, default=2,
                        help='Maximum iterations for reweighting (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of frames to process in parallel on GPU (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for parallel data loading (default: 4)')

    # Threshold parameters
    parser.add_argument('--thresh_init', type=float, default=0.5,
                        help='Initial occupancy threshold (default: 0.5)')
    parser.add_argument('--thresh_expand', type=float, default=0.35,
                        help='Expanded occupancy threshold for later iterations (default: 0.35)')
    parser.add_argument('--prec_penalty', type=float, default=0.3,
                        help='Precision threshold for weight penalty (default: 0.3)')
    parser.add_argument('--area_penalty', type=float, default=0.05,
                        help='Area ratio threshold for weight penalty (default: 0.05)')
    parser.add_argument('--bbox_padding', type=float, default=0.30,
                        help='Padding ratio for projected bbox (default: 0.30)')
    parser.add_argument('--min_overlap', type=float, default=0.025,
                        help='Min mask-bbox overlap ratio for validity (default: 0.025)')

    # Mask format
    parser.add_argument('--mask_format', type=str, default='auto',
                        choices=['auto', 'npz', 'shard'],
                        help='Mask storage format (default: auto-detect)')
    parser.add_argument('--mask_root', type=str, default=None,
                        help='Root directory for shard masks (contains mask_shards/)')

    # Frame range
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame (0 for auto)')
    parser.add_argument('--end_frame', type=int, default=0,
                        help='End frame (0 for auto)')

    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    return parser.parse_args()


def _load_frame_masks_npz(mask_npz_path: str, frame_file: str, view_indices: List[int]) -> Dict:
    """Load a single frame's mask data from NPZ.

    Runs in worker thread. Uses .copy() on view slices so the full
    (42, H, W) array can be garbage-collected immediately.
    """
    return load_frame_masks_npz(mask_npz_path, frame_file, view_indices)


def _load_batch_shard(
    seq_readers: SequenceShardReaders, frame_ids: List[int], view_indices: List[int]
) -> List[Dict]:
    """Load a batch of frames from shard format, sequentially.

    Each frame internally parallelizes across objects (LZ4 releases GIL).
    Runs in a single background thread for prefetch overlap with GPU.
    """
    return [load_frame_masks_shard(seq_readers, fid, view_indices) for fid in frame_ids]


def _load_batch_shard_full(
    seq_readers: SequenceShardReaders, frame_ids: List[int]
) -> List[Dict[str, np.ndarray]]:
    """Load batch as full (V, H, W) arrays per object — no per-view copy.

    Returns list of {obj_name: ndarray(V, H, W)}.
    """
    return [load_frame_masks_shard_full(seq_readers, fid) for fid in frame_ids]


def _save_frame_results(output_file: str, validity_results: Dict):
    """Save validity results for a single frame. Runs in worker thread."""
    np.savez_compressed(output_file, **validity_results)


class MultiViewMaskValidator:
    """GPU-batched multi-view mask validator.

    Batches all view operations into tensor ops and processes multiple
    frames in parallel for maximum GPU throughput.
    """

    def __init__(
        self,
        cameras: Dict[int, Dict[str, np.ndarray]],
        view_indices: List[int],
        img_size: Tuple[int, int],
        voxel_res: int = 48,
        device: str = 'cuda',
        thresh_init: float = 0.5,
        thresh_expand: float = 0.35,
        prec_penalty: float = 0.3,
        area_penalty: float = 0.05,
        max_iters: int = 2,
        bbox_padding: float = 0.30,
        min_overlap: float = 0.025,
    ):
        self.cameras = cameras
        self.view_indices = view_indices
        self.img_size = img_size
        self.voxel_res = voxel_res
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.thresh_init = thresh_init
        self.thresh_expand = thresh_expand
        self.prec_penalty = prec_penalty
        self.area_penalty = area_penalty
        self.max_iters = max_iters
        self.bbox_padding = bbox_padding
        self.min_overlap = min_overlap

        # Per-view camera params (for bbox estimation)
        self.cam_centers = {}
        for view_idx in view_indices:
            cam = cameras[view_idx]
            self.cam_centers[view_idx] = -cam['R'].T @ cam['T']

        # Pre-stack camera matrices for batched GPU ops: (V, 3, 3) / (V, 3, 1)
        V = len(view_indices)
        K_list, R_list, T_list = [], [], []
        for v in view_indices:
            cam = cameras[v]
            K_list.append(torch.tensor(cam['K'], dtype=torch.float32))
            R_list.append(torch.tensor(cam['R'], dtype=torch.float32))
            T_list.append(torch.tensor(cam['T'], dtype=torch.float32))

        self.K_batch = torch.stack(K_list).to(self.device)  # (V, 3, 3)
        self.R_batch = torch.stack(R_list).to(self.device)  # (V, 3, 3)
        self.T_batch = torch.stack(T_list).to(self.device)  # (V, 3, 1)

    def _adaptive_min_overlap(self, mask_area):
        """Keep overlap threshold strict; only tiny masks get very small relaxation."""
        area_ref = 100.0
        if torch.is_tensor(mask_area):
            area_scale = torch.sqrt(torch.clamp(mask_area / area_ref, min=0.0, max=1.0))
            # Very strict mode: at most 5% relaxation for tiny masks.
            return self.min_overlap * (0.95 + 0.05 * area_scale)

        area = float(mask_area)
        area_scale = np.sqrt(min(1.0, max(0.0, area / area_ref)))
        return self.min_overlap * (0.95 + 0.05 * area_scale)

    def _compute_view_stats(self, masks: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, Tuple[float, float]], List[float], np.ndarray
    ]:
        """Vectorized computation of centroids, bbox diagonals, and areas for all views.

        Uses moment-based centroid (avoids np.where which materializes all indices).
        Accepts either uint8 or pre-binarized bool masks.
        Returns: (centroids_dict, bbox_diagonals, areas_array)
        """
        view_list = sorted(masks.keys())
        V = len(view_list)
        H, W = next(iter(masks.values())).shape

        # Stack all masks → (V, H, W), binarize if needed
        masks_stack = np.stack([masks[v] for v in view_list])
        if masks_stack.dtype != bool:
            masks_stack = masks_stack > 127

        # Areas: (V,)
        areas = masks_stack.reshape(V, -1).sum(axis=1)

        # Row/col projections for bbox
        row_any = masks_stack.any(axis=2)  # (V, H) — any nonzero in each row
        col_any = masks_stack.any(axis=1)  # (V, W) — any nonzero in each col

        # Moment-based centroids: (V,)
        row_sums = masks_stack.sum(axis=2).astype(np.float64)  # (V, H)
        col_sums = masks_stack.sum(axis=1).astype(np.float64)  # (V, W)
        y_range = np.arange(H, dtype=np.float64)
        x_range = np.arange(W, dtype=np.float64)
        cy_all = (row_sums * y_range[None, :]).sum(axis=1) / (areas + 1e-8)  # (V,)
        cx_all = (col_sums * x_range[None, :]).sum(axis=1) / (areas + 1e-8)  # (V,)

        centroids = {}
        bbox_diags = []
        for i, v in enumerate(view_list):
            if areas[i] <= 0:
                continue
            centroids[v] = (float(cx_all[i]), float(cy_all[i]))
            # Bbox from first/last nonzero row/col
            rows_nz = np.where(row_any[i])[0]
            cols_nz = np.where(col_any[i])[0]
            if len(rows_nz) > 0 and len(cols_nz) > 0:
                dy = float(rows_nz[-1] - rows_nz[0])
                dx = float(cols_nz[-1] - cols_nz[0])
                bbox_diags.append(np.sqrt(dx * dx + dy * dy))

        return centroids, bbox_diags, areas

    def _triangulate_center(self, centroids: Dict[int, Tuple[float, float]]) -> np.ndarray:
        """Triangulate 3D center from 2D centroids using batched least squares."""
        view_list = list(centroids.keys())
        if len(view_list) < 2:
            return np.array([0.0, 0.0, 1.0])

        # Pre-stack K_inv, R, C for all views with centroids
        n = len(view_list)
        A = np.zeros((3 * n, 3))
        b = np.zeros(3 * n)

        for i, view_idx in enumerate(view_list):
            cx, cy = centroids[view_idx]
            K = self.cameras[view_idx]['K']
            R = self.cameras[view_idx]['R']
            C = self.cam_centers[view_idx].flatten()

            K_inv = np.linalg.inv(K)
            ray_cam = K_inv @ np.array([cx, cy, 1.0])
            ray_world = R.T @ ray_cam
            ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-8)

            d = ray_world.reshape(3, 1)
            I_minus_ddT = np.eye(3) - d @ d.T
            A[3*i:3*i+3] = I_minus_ddT
            b[3*i:3*i+3] = (I_minus_ddT @ C.reshape(3, 1)).flatten()

        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return center

    def _estimate_bbox(self, masks: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate 3D bounding box from multi-view masks (vectorized).

        Uses all provided views for estimation.
        """
        centroids, bbox_diags, _ = self._compute_view_stats(masks)

        if len(centroids) < 2:
            return np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 2.0])

        center_3d = self._triangulate_center(centroids)

        median_diagonal = np.median(bbox_diags) if bbox_diags else 200
        avg_focal = 1500
        estimated_depth = 5.0
        estimated_size = (median_diagonal / avg_focal) * estimated_depth
        estimated_size = min(estimated_size, 5.0)
        # Keep strictly positive extent for numerical stability.
        estimated_size = max(estimated_size, 1e-6)

        # Do not apply any fixed scale expansion on bbox size.
        half_size = estimated_size / 2
        bbox_min = center_3d - np.array([half_size, half_size, half_size * 0.5])
        bbox_max = center_3d + np.array([half_size, half_size, half_size * 1.5])

        return bbox_min, bbox_max

    def _create_voxel_grid(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> torch.Tensor:
        """Create voxel grid centers."""
        res = self.voxel_res
        x = torch.linspace(bbox_min[0], bbox_max[0], res, device=self.device)
        y = torch.linspace(bbox_min[1], bbox_max[1], res, device=self.device)
        z = torch.linspace(bbox_min[2], bbox_max[2], res, device=self.device)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        voxels = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        return voxels

    def validate(
        self, masks: Dict[int, np.ndarray], verbose: bool = False
    ) -> Tuple[Dict[int, int], Dict[int, float]]:
        """Single-frame validation (convenience wrapper)."""
        return self.validate_batch([masks], verbose=verbose)[0]

    def validate_batch(
        self, batch_masks: List[Dict[int, np.ndarray]], verbose: bool = False
    ) -> List[Tuple[Dict[int, int], Dict[int, float]]]:
        """
        Batch-validate mask consistency across views for multiple frames.

        All view and frame operations use batched GPU tensor ops.
        The voxel grid is shared across frames (union bbox).

        Args:
            batch_masks: list of B dicts, each {view_idx: np.ndarray(H, W)}
        Returns:
            list of B (validity_dict, precision_dict) tuples
        """
        B = len(batch_masks)
        V = len(self.view_indices)
        img_w, img_h = self.img_size

        # 1. Mask areas (fused >127 + count avoids extra memory write)
        areas_np = np.zeros((B, V), dtype=np.float32)
        for b in range(B):
            for vi, v in enumerate(self.view_indices):
                areas_np[b, vi] = np.count_nonzero(batch_masks[b][v] > 127)
        mask_areas = torch.from_numpy(areas_np).to(self.device)

        non_empty_counts = (mask_areas > 0).sum(dim=1)  # (B,)

        # 2. Estimate union bbox — subsample frames (consecutive frames barely move)
        bbox_step = max(1, B // 8)
        bbox_mins, bbox_maxs = [], []
        for b in range(0, B, bbox_step):
            if non_empty_counts[b] < 2:
                continue
            try:
                bmin, bmax = self._estimate_bbox(batch_masks[b])
                bbox_mins.append(bmin)
                bbox_maxs.append(bmax)
            except Exception:
                pass

        if not bbox_mins:
            # Fallback: all non-empty masks valid
            results = []
            for b in range(B):
                v_dict = {v: (0 if mask_areas[b, vi] <= 0 else 1)
                          for vi, v in enumerate(self.view_indices)}
                results.append((v_dict, {v: 0.5 for v in self.view_indices}))
            return results

        bbox_min = np.min(bbox_mins, axis=0)
        bbox_max = np.max(bbox_maxs, axis=0)

        # 3. Shared voxel grid
        voxels = self._create_voxel_grid(bbox_min, bbox_max)  # (N, 3)
        N = len(voxels)

        # 4. Batched projection to all views: (V, 3, N) → (V, N, 2)
        voxels_t = voxels.T.unsqueeze(0).expand(V, -1, -1)  # (V, 3, N)
        points_cam = torch.bmm(self.R_batch, voxels_t) + self.T_batch  # (V, 3, N)
        depths = points_cam[:, 2, :]  # (V, N)

        points_homo = torch.bmm(self.K_batch, points_cam)  # (V, 3, N)
        pts_2d_x = points_homo[:, 0, :] / (points_homo[:, 2, :] + 1e-8)  # (V, N)
        pts_2d_y = points_homo[:, 1, :] / (points_homo[:, 2, :] + 1e-8)  # (V, N)

        # 5. Visibility: (V, N)
        visibility = (
            (pts_2d_x >= 0) & (pts_2d_x < img_w) &
            (pts_2d_y >= 0) & (pts_2d_y < img_h) &
            (depths > 0)
        )

        # 6. Sampling indices — compute on CPU for CPU-side sampling
        x_idx = pts_2d_x.long().clamp(0, img_w - 1)  # (V, N)
        y_idx = pts_2d_y.long().clamp(0, img_h - 1)  # (V, N)
        linear_idx = y_idx * img_w + x_idx  # (V, N)
        linear_idx_cpu = linear_idx.cpu().numpy()  # (V, N) int64
        visibility_cpu = visibility.cpu().numpy()  # (V, N) bool

        # 7. CPU-side mask sampling: per-mask fancy indexing (16x faster than stack+take)
        sampled_np = np.zeros((B, V, N), dtype=np.float32)
        for b in range(B):
            for vi, v in enumerate(self.view_indices):
                sampled_np[b, vi] = (batch_masks[b][v] > 127).ravel()[linear_idx_cpu[vi]]
        # Zero out-of-bounds on CPU, then single transfer
        sampled_np[:, ~visibility_cpu] = 0
        sampled = torch.from_numpy(sampled_np).to(self.device)

        # 8. Batched iterative reweighting
        weights = torch.ones(B, V, device=self.device) / V
        vis_f = visibility.unsqueeze(0).float()  # (1, V, N)

        median_areas = mask_areas.median(dim=1).values  # (B,)
        area_ratios = mask_areas / (median_areas.unsqueeze(1) + 1e-6)  # (B, V)

        for iteration in range(self.max_iters):
            w = weights.unsqueeze(2)  # (B, V, 1)

            # Voting: (B, N)
            vote_num = (w * sampled * vis_f).sum(dim=1)  # (B, N)
            vote_den = (w * vis_f).sum(dim=1) + 1e-6  # (B, N)
            occupancy = vote_num / vote_den  # (B, N)

            thresh = self.thresh_init if iteration == 0 else self.thresh_expand
            occupied = (occupancy > thresh).unsqueeze(1).float()  # (B, 1, N)

            # Voxel-level precision (replaces convex hull): (B, V)
            prec_num = (sampled * occupied * vis_f).sum(dim=2)  # (B, V)
            prec_den = (sampled * vis_f).sum(dim=2) + 1e-6  # (B, V)
            precision = prec_num / prec_den  # (B, V)

            # Weight update
            new_w = weights.clone()
            new_w[precision < self.prec_penalty] *= 0.7
            new_w[area_ratios < self.area_penalty] *= 0.3

            # Normalize per frame
            w_sum = new_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            new_w = (new_w / w_sum).clamp(min=0.01)
            new_w = new_w / new_w.sum(dim=1, keepdim=True)
            weights = new_w

        # 9. Final bbox overlap check — all on CPU to avoid per-element GPU sync
        occupied_final = occupancy > self.thresh_expand  # (B, N)

        # Transfer to CPU once (eliminates 1344 individual .item() GPU syncs)
        pts_2d_x_cpu = pts_2d_x.cpu().numpy()  # (V, N)
        pts_2d_y_cpu = pts_2d_y.cpu().numpy()  # (V, N)
        depths_cpu = depths.cpu().numpy()  # (V, N)
        occupied_cpu = occupied_final.cpu().numpy()  # (B, N)
        precision_cpu = precision.cpu().numpy()  # (B, V)
        mask_areas_cpu = areas_np  # already on CPU

        results = []
        pad = self.bbox_padding
        for b in range(B):
            validity = {}
            prec_dict = {v: float(precision_cpu[b, vi]) for vi, v in enumerate(self.view_indices)}

            occ_b = occupied_cpu[b]  # (N,) bool

            if not occ_b.any():
                # No occupied voxels — all non-empty masks valid
                for vi, view_idx in enumerate(self.view_indices):
                    validity[view_idx] = 0 if mask_areas_cpu[b, vi] <= 0 else 1
                    if verbose:
                        status = "INVALID (empty mask)" if mask_areas_cpu[b, vi] <= 0 else "VALID (fallback)"
                        print(f"    view {view_idx}: {status}")
                results.append((validity, prec_dict))
                continue

            # Vectorized: extract occupied points for all views at once
            occ_x = pts_2d_x_cpu[:, occ_b]  # (V, M)
            occ_y = pts_2d_y_cpu[:, occ_b]  # (V, M)
            occ_d = depths_cpu[:, occ_b]    # (V, M)

            # Validity mask: (V, M)
            valid_pts = (
                (occ_x >= 0) & (occ_x < img_w) &
                (occ_y >= 0) & (occ_y < img_h) &
                (occ_d > 0)
            )
            has_valid = valid_pts.any(axis=1)  # (V,)

            # Compute bbox for all views using masked min/max
            # Views with no valid points get NaN → handled by has_valid check below
            occ_x_m = np.where(valid_pts, occ_x, np.inf)
            occ_y_m = np.where(valid_pts, occ_y, np.inf)
            bx1_all = np.min(occ_x_m, axis=1)  # (V,) — inf for no-valid views
            by1_all = np.min(occ_y_m, axis=1)
            occ_x_m2 = np.where(valid_pts, occ_x, -np.inf)
            occ_y_m2 = np.where(valid_pts, occ_y, -np.inf)
            bx2_all = np.max(occ_x_m2, axis=1) + 1
            by2_all = np.max(occ_y_m2, axis=1) + 1

            # Padding (vectorized) — only valid for has_valid views
            w_box = bx2_all - bx1_all
            h_box = by2_all - by1_all
            px_all = w_box * pad
            py_all = h_box * pad
            # Fill invalid views with 0 to avoid inf→int issues
            bx1_all = np.where(has_valid, np.maximum(0, bx1_all - px_all), 0).astype(int)
            by1_all = np.where(has_valid, np.maximum(0, by1_all - py_all), 0).astype(int)
            bx2_all = np.where(has_valid, np.minimum(img_w, bx2_all + px_all), 0).astype(int)
            by2_all = np.where(has_valid, np.minimum(img_h, by2_all + py_all), 0).astype(int)

            # Per-view overlap (requires variable-size slice per view)
            for vi, view_idx in enumerate(self.view_indices):
                if mask_areas_cpu[b, vi] <= 0:
                    validity[view_idx] = 0
                    if verbose:
                        print(f"    view {view_idx}: area=0 -> INVALID (empty mask)")
                    continue

                if not has_valid[vi]:
                    validity[view_idx] = 1
                    if verbose:
                        print(f"    view {view_idx}: no projection -> VALID (fallback)")
                    continue

                y1, y2 = by1_all[vi], by2_all[vi]
                x1, x2 = bx1_all[vi], bx2_all[vi]
                mask_in_bbox = np.sum(batch_masks[b][view_idx][y1:y2, x1:x2] > 127)
                total = int(mask_areas_cpu[b, vi])
                overlap = mask_in_bbox / (total + 1e-6)

                min_overlap_eff = self._adaptive_min_overlap(total)
                is_valid = overlap >= min_overlap_eff
                validity[view_idx] = 1 if is_valid else 0

                if verbose:
                    print(f"    view {view_idx}: overlap={overlap:.3f} "
                          f"bbox=({x1},{y1},{x2},{y2}) "
                          f"mask_area={total} min_overlap={min_overlap_eff:.4f} "
                          f"-> {'VALID' if is_valid else 'INVALID'}")

            results.append((validity, prec_dict))

        return results

    def validate_batch_stacked(
        self, batch_full: List[np.ndarray], verbose: bool = False
    ) -> List[Tuple[Dict[int, int], Dict[int, float]]]:
        """GPU-accelerated validation using view-streaming.

        Streams one view at a time to GPU (~64MB per view instead of ~3GB total),
        computing areas and sampling in a single pass per view.

        Args:
            batch_full: list of B arrays, each (total_views, H, W) uint8.
                        Indexed by view_indices to select views.
        """
        B = len(batch_full)
        V = len(self.view_indices)
        img_w, img_h = self.img_size
        H, W = img_h, img_w

        # 1. Bbox estimation on CPU — subsample frames
        bbox_step = max(1, B // 8)
        bbox_views = self.view_indices

        bbox_mins, bbox_maxs = [], []
        for b in range(0, B, bbox_step):
            # Quick non-empty check (avoid full _compute_view_stats)
            ne = 0
            for v in bbox_views:
                if batch_full[b][v].any():
                    ne += 1
                    if ne >= 2:
                        break
            if ne < 2:
                continue
            try:
                # Pass all requested views for bbox estimation
                frame_dict = {v: batch_full[b][v] for v in bbox_views}
                bmin, bmax = self._estimate_bbox(frame_dict)
                bbox_mins.append(bmin)
                bbox_maxs.append(bmax)
            except Exception:
                pass

        if not bbox_mins:
            results = []
            for b in range(B):
                v_dict = {}
                for vi, v in enumerate(self.view_indices):
                    v_dict[v] = 0 if not batch_full[b][v].any() else 1
                results.append((v_dict, {v: 0.5 for v in self.view_indices}))
            return results

        bbox_min = np.min(bbox_mins, axis=0)
        bbox_max = np.max(bbox_maxs, axis=0)

        # 2. Voxel grid + projection (on GPU)
        voxels = self._create_voxel_grid(bbox_min, bbox_max)  # (N, 3)
        N = len(voxels)

        voxels_t = voxels.T.unsqueeze(0).expand(V, -1, -1)
        points_cam = torch.bmm(self.R_batch, voxels_t) + self.T_batch
        depths = points_cam[:, 2, :]
        points_homo = torch.bmm(self.K_batch, points_cam)
        pts_2d_x = points_homo[:, 0, :] / (points_homo[:, 2, :] + 1e-8)
        pts_2d_y = points_homo[:, 1, :] / (points_homo[:, 2, :] + 1e-8)

        visibility = (
            (pts_2d_x >= 0) & (pts_2d_x < img_w) &
            (pts_2d_y >= 0) & (pts_2d_y < img_h) &
            (depths > 0)
        )

        x_idx = pts_2d_x.long().clamp(0, img_w - 1)
        y_idx = pts_2d_y.long().clamp(0, img_h - 1)
        linear_idx = y_idx * img_w + x_idx  # (V, N) int64

        # 3. GPU view-streaming with chunking: areas + sampling
        #    Processes 8 views per GPU transfer to reduce Python overhead.
        #    Peak GPU: ~512MB (B, chunk, H*W) + ~225MB indices
        chunk_size = 8
        mask_areas = torch.zeros(B, V, device=self.device)
        sampled = torch.zeros(B, V, N, device=self.device)

        for vi_start in range(0, V, chunk_size):
            vi_end = min(vi_start + chunk_size, V)
            C = vi_end - vi_start
            chunk_views = self.view_indices[vi_start:vi_end]

            # Stack (B, C, H*W) — contiguous for fast H2D
            chunk_np = np.empty((B, C, H * W), dtype=np.uint8)
            for ci, v in enumerate(chunk_views):
                for b in range(B):
                    chunk_np[b, ci] = batch_full[b][v].ravel()

            chunk_gpu = torch.from_numpy(chunk_np).to(self.device)  # (B, C, H*W)
            binary = chunk_gpu > 127  # (B, C, H*W) bool
            mask_areas[:, vi_start:vi_end] = binary.sum(dim=2).float()

            idx_chunk = linear_idx[vi_start:vi_end].unsqueeze(0).expand(B, -1, -1)  # (B, C, N)
            gathered = torch.gather(chunk_gpu, 2, idx_chunk)  # (B, C, N) uint8
            vis_chunk = visibility[vi_start:vi_end].unsqueeze(0).float()  # (1, C, N)
            sampled[:, vi_start:vi_end] = (gathered > 127).float() * vis_chunk

        del chunk_gpu, binary, gathered, chunk_np
        areas_np = mask_areas.cpu().numpy()

        # 4. Iterative reweighting (all on GPU)
        weights = torch.ones(B, V, device=self.device) / V
        vis_f = visibility.unsqueeze(0).float()
        median_areas = mask_areas.median(dim=1).values
        area_ratios = mask_areas / (median_areas.unsqueeze(1) + 1e-6)

        for iteration in range(self.max_iters):
            w = weights.unsqueeze(2)
            vote_num = (w * sampled * vis_f).sum(dim=1)
            vote_den = (w * vis_f).sum(dim=1) + 1e-6
            occupancy = vote_num / vote_den
            thresh = self.thresh_init if iteration == 0 else self.thresh_expand
            occupied = (occupancy > thresh).unsqueeze(1).float()
            prec_num = (sampled * occupied * vis_f).sum(dim=2)
            prec_den = (sampled * vis_f).sum(dim=2) + 1e-6
            precision = prec_num / prec_den
            new_w = weights.clone()
            new_w[precision < self.prec_penalty] *= 0.7
            new_w[area_ratios < self.area_penalty] *= 0.3
            w_sum = new_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            new_w = (new_w / w_sum).clamp(min=0.01)
            new_w = new_w / new_w.sum(dim=1, keepdim=True)
            weights = new_w

        # 5. GPU overlap check: use sampled values at occupied voxels
        #    "What fraction of visible occupied voxels project onto the mask?"
        #    Fully batched on GPU — no per-frame/per-view Python loop.
        occupied_final = (occupancy > self.thresh_expand).unsqueeze(1)  # (B, 1, N)
        vis_mask = visibility.unsqueeze(0)  # (1, V, N)
        occ_vis = occupied_final & vis_mask  # (B, V, N) — occupied AND visible
        occ_vis_f = occ_vis.float()

        # Hits: occupied+visible voxels that project onto mask
        hits = (sampled * occ_vis_f).sum(dim=2)  # (B, V)
        # Total visible occupied voxels per view
        total_occ_vis = occ_vis_f.sum(dim=2)  # (B, V)
        # Overlap: fraction of visible occupied voxels on the mask
        has_occ = total_occ_vis > 0
        overlap_ratio = torch.where(
            has_occ, hits / (total_occ_vis + 1e-6), torch.ones_like(hits)
        )  # default 1.0 (valid) when no occupied voxels project

        # Build validity: empty masks (area==0) → 0, otherwise check overlap
        is_empty = mask_areas <= 0  # (B, V)
        min_overlap_eff = self._adaptive_min_overlap(mask_areas)  # (B, V)
        is_valid_gpu = (~is_empty) & (overlap_ratio >= min_overlap_eff)
        validity_np = is_valid_gpu.cpu().numpy().astype(np.uint8)  # (B, V)
        precision_cpu = precision.cpu().numpy()

        results = []
        for b in range(B):
            validity = {v: int(validity_np[b, vi])
                        for vi, v in enumerate(self.view_indices)}
            prec_dict = {v: float(precision_cpu[b, vi])
                         for vi, v in enumerate(self.view_indices)}

            if verbose:
                for vi, view_idx in enumerate(self.view_indices):
                    if areas_np[b, vi] <= 0:
                        print(f"    view {view_idx}: area=0 -> INVALID (empty mask)")
                    else:
                        ov = float(overlap_ratio[b, vi])
                        v_str = 'VALID' if validity[view_idx] else 'INVALID'
                        print(f"    view {view_idx}: overlap={ov:.3f} -> {v_str}")

            results.append((validity, prec_dict))

        return results


def process_sequence(args):
    """Process a single sequence with prefetch pipeline + batched GPU ops.

    Data loading pipeline:
        Thread pool loads batch N+1 while GPU processes batch N.
        Within each batch, all frames are loaded in parallel across threads.
        Results are saved asynchronously in background threads.
    """
    root_path = args.root_path
    seq_name = args.seq_name
    output_path = args.output_path

    # Setup paths
    dataset_info_path = join(root_path, 'dataset_information.json')
    mask_npz_path = join(root_path, 'mask_npz', seq_name)

    # Detect mask format
    mask_format = args.mask_format
    if mask_format == 'auto':
        mask_format = detect_mask_format(root_path, seq_name, args.mask_root)
    print(f"Mask format: {mask_format}")

    # Get calibration date and load camera params
    calib_date = get_calibration_date_for_sequence(dataset_info_path, seq_name)
    if calib_date is None:
        print(f"Error: Could not find calibration date for sequence {seq_name}")
        return

    calibration_path = join(root_path, 'calibration', calib_date, 'calibration.json')
    print(f"Loading cameras from {calibration_path}")

    # Determine view indices
    if args.all_views:
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        view_indices = sorted([
            int(k) for k in calib_data if k.isdigit() and int(k) < 42
            and len(calib_data[k].get('K', [])) > 0
        ])
        print(f"--all_views: using {len(view_indices)} views from calibration")
    else:
        view_indices = args.views

    cameras = load_camera_params(calibration_path, view_indices, scale_factor=2.0)
    print(f"Loaded {len(cameras)} camera views: {list(cameras.keys())}")

    # Mask resolution
    img_size = (1920, 1080)

    # Create validator
    validator = MultiViewMaskValidator(
        cameras=cameras,
        view_indices=view_indices,
        img_size=img_size,
        voxel_res=args.voxel_res,
        device=args.device,
        thresh_init=args.thresh_init,
        thresh_expand=args.thresh_expand,
        prec_penalty=args.prec_penalty,
        area_penalty=args.area_penalty,
        max_iters=args.max_iters,
        bbox_padding=args.bbox_padding,
        min_overlap=args.min_overlap,
    )

    V = len(view_indices)
    N = args.voxel_res ** 3
    batch_size = args.batch_size
    num_workers = args.num_workers

    print(f"Using device: {validator.device}")
    print(f"Parameters: voxel_res={args.voxel_res}, max_iters={args.max_iters}, "
          f"batch_size={batch_size}, num_workers={num_workers}")
    print(f"Thresholds: thresh_init={args.thresh_init}, thresh_expand={args.thresh_expand}")
    print(f"           prec_penalty={args.prec_penalty}, area_penalty={args.area_penalty}")
    print(f"           bbox_padding={args.bbox_padding}, min_overlap={args.min_overlap}")

    # Memory estimate
    mask_mem_per_frame = V * 1080 * 1920 * 4 / (1024**3)  # GB
    sampled_mem = batch_size * V * N * 4 / (1024**3)  # GB
    total_est = mask_mem_per_frame + sampled_mem + 0.1
    print(f"Est. GPU memory per batch: ~{total_est:.1f} GB "
          f"(masks: {mask_mem_per_frame:.1f}GB + sampled: {sampled_mem:.1f}GB)")

    # Get frame list and open shard readers if needed
    seq_readers = None
    if mask_format == 'shard':
        shard_root = args.mask_root if args.mask_root else join(root_path, 'mask_shards')
        seq_readers = SequenceShardReaders(join(shard_root, seq_name))
        frame_list = seq_readers.frame_ids_list  # list of ints
        if args.start_frame > 0 or args.end_frame > 0:
            start = args.start_frame
            end = args.end_frame if args.end_frame > 0 else max(frame_list) + 1
            frame_list = [f for f in frame_list if start <= f < end]
        # For output filenames
        frame_output_names = [f"{fid:06d}.npz" for fid in frame_list]
        print(f"Shard root: {shard_root}")
        print(f"Objects: {seq_readers.objects}")
    else:
        frame_files = sorted([f for f in os.listdir(mask_npz_path) if f.endswith('.npz')])
        if args.start_frame > 0 or args.end_frame > 0:
            start = args.start_frame
            end = args.end_frame if args.end_frame > 0 else len(frame_files)
            frame_files = frame_files[start:end]
        frame_list = frame_files
        frame_output_names = frame_files

    total_frames = len(frame_list)
    print(f"Processing {total_frames} frames in batches of {batch_size}")

    # Create output directory
    output_seq_path = join(output_path, seq_name)
    os.makedirs(output_seq_path, exist_ok=True)

    # Split into batches
    batches = []
    batch_output_names = []
    for i in range(0, total_frames, batch_size):
        batches.append(frame_list[i:i + batch_size])
        batch_output_names.append(frame_output_names[i:i + batch_size])

    if not batches:
        print("No frames to process")
        if seq_readers:
            seq_readers.close()
        return

    # Thread pools for loading and saving
    load_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    save_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    save_futures = []

    use_stacked = (mask_format == 'shard')

    if mask_format == 'shard':
        # Shard: load full (V, H, W) arrays — no per-view copy
        def submit_batch_load(batch_items):
            return load_pool.submit(
                _load_batch_shard_full, seq_readers, batch_items
            )

        current_load_future = submit_batch_load(batches[0])
    else:
        # NPZ: parallel per-frame loading
        def submit_batch_load_npz(batch_files):
            return [
                load_pool.submit(_load_frame_masks_npz, mask_npz_path, ff, view_indices)
                for ff in batch_files
            ]

        current_npz_futures = submit_batch_load_npz(batches[0])

    with tqdm(total=total_frames, desc=f"Processing {seq_name}") as pbar:
        for batch_idx in range(len(batches)):
            actual_B = len(batches[batch_idx])

            # Collect current batch
            if mask_format == 'shard':
                batch_data = current_load_future.result()
                if batch_idx + 1 < len(batches):
                    current_load_future = submit_batch_load(batches[batch_idx + 1])
            else:
                batch_data = [f.result() for f in current_npz_futures]
                if batch_idx + 1 < len(batches):
                    current_npz_futures = submit_batch_load_npz(batches[batch_idx + 1])

            # --- GPU processing ---
            object_keys = list(batch_data[0].keys())
            batch_results = {}

            for obj_key in object_keys:
                if use_stacked:
                    # Shard path: pass full (V, H, W) arrays for GPU view-streaming
                    obj_full = [batch_data[b][obj_key] for b in range(actual_B)]
                    if args.verbose:
                        print(f"  [{obj_key}]")
                    batch_results[obj_key] = validator.validate_batch_stacked(
                        obj_full, verbose=args.verbose
                    )
                else:
                    # NPZ path: dict-based
                    obj_batch_masks = []
                    enough_views = True
                    for data in batch_data:
                        masks = data[obj_key]
                        if len(masks) < 2:
                            enough_views = False
                        obj_batch_masks.append(masks)

                    if not enough_views:
                        batch_results[obj_key] = [
                            ({v: 1 for v in view_indices}, {v: 0.5 for v in view_indices})
                            for _ in range(actual_B)
                        ]
                    else:
                        if args.verbose:
                            print(f"  [{obj_key}]")
                        batch_results[obj_key] = validator.validate_batch(
                            obj_batch_masks, verbose=args.verbose
                        )

                if args.verbose:
                    for b in range(actual_B):
                        validity = batch_results[obj_key][b][0]
                        valid_count = sum(1 for v in validity.values() if v == 1)
                        out_name = batch_output_names[batch_idx][b]
                        frame_idx = int(out_name.replace('.npz', ''))
                        print(f"    frame {frame_idx} {obj_key}: "
                              f"{valid_count}/{len(view_indices)} valid views")

            # Save results asynchronously
            for b in range(actual_B):
                frame_file = batch_output_names[batch_idx][b]
                validity_results = {}
                for obj_key in object_keys:
                    validity = batch_results[obj_key][b][0]
                    validity_arr = np.array(
                        [validity.get(v, 1) for v in view_indices], dtype=np.uint8
                    )
                    validity_results[f'{obj_key}_validity'] = validity_arr

                output_file = join(output_seq_path, frame_file)
                save_futures.append(
                    save_pool.submit(_save_frame_results, output_file, validity_results)
                )

            # Free batch data before next iteration
            del batch_data
            pbar.update(actual_B)

    # Wait for all saves to finish
    for f in save_futures:
        f.result()  # raises any exceptions from save threads

    load_pool.shutdown(wait=False)
    save_pool.shutdown(wait=False)
    if seq_readers:
        seq_readers.close()
    print(f"Done! Results saved to {output_seq_path}")


def main():
    args = parse_args()
    process_sequence(args)


if __name__ == '__main__':
    main()

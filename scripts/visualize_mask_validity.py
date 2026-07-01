"""
Visualize Mask Validity Results

Overlays validity results on video frames with color-coded masks:
- Green: valid mask (validity=1)
- Red: invalid mask (validity=0)

Outputs MP4 videos for easy review.

Usage:
    python scripts/visualize_mask_validity.py \
        --root_path "/simurgh/group/juze/datasets/HOI-M3" \
        --seq_name "bedroom_data01" \
        --validity_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity" \
        --views 0 2 5 6 7 8 10 11 14 15 \
        --output_path "/simurgh/group/juze/datasets/HOI-M3/mask_validity_vis"
"""
import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize mask validity results')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to HOI-M3 dataset')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='Sequence name (e.g., bedroom_data01)')
    parser.add_argument('--validity_path', type=str, required=True,
                        help='Path to mask_validity results')
    parser.add_argument('--views', type=int, nargs='+',
                        default=[0, 2, 5, 6, 7, 8, 10, 11, 14, 15],
                        help='View indices to visualize')
    parser.add_argument('--all_views', action='store_true',
                        help='Use all 42 views (overrides --views)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for visualization videos')
    parser.add_argument('--object_name', type=str, default=None,
                        help='Specific object to visualize (default: all)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame number (0 = from beginning)')
    parser.add_argument('--end_frame', type=int, default=0,
                        help='End frame number (0 = to the end)')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame step size (e.g., 10 = every 10th frame)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Mask overlay alpha (0-1)')
    parser.add_argument('--combined', action='store_true',
                        help='Create combined multi-view video')
    return parser.parse_args()


def preload_video_frames(
    video_path: str,
    frame_indices: List[int],
    target_size: Optional[Tuple[int, int]] = None,
    seek_threshold: int = 50
) -> Dict[int, np.ndarray]:
    """
    Read specific frames from video efficiently.

    Uses sequential reading for nearby frames (fast) and seeking only for
    large gaps. This is orders of magnitude faster than random seeking for
    each frame, because compressed video (H.264/H.265) requires decoding
    from the nearest keyframe.

    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to extract
        target_size: (width, height) to resize frames, or None for original
        seek_threshold: Skip sequential reading if gap exceeds this
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    frames = {}
    sorted_indices = sorted(set(frame_indices))
    if not sorted_indices:
        cap.release()
        return frames

    current_pos = 0

    for target in sorted_indices:
        gap = target - current_pos

        if gap > seek_threshold or gap < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current_pos = target
        else:
            # Sequential skip (grab without decode) — much faster than seeking
            while current_pos < target:
                cap.grab()
                current_pos += 1

        ret, frame = cap.read()
        if ret:
            if target_size:
                frame = cv2.resize(frame, target_size)
            frames[target] = frame
        current_pos = target + 1

    cap.release()
    return frames


def overlay_mask_on_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    is_valid: bool,
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay mask on frame with color based on validity."""
    result = frame.copy()
    mask_bool = mask > 127

    if not np.any(mask_bool):
        return result

    # Color: Green (valid) or Red (invalid)
    if is_valid:
        color = np.array([0, 255, 0], dtype=np.uint8)  # BGR: Green
    else:
        color = np.array([0, 0, 255], dtype=np.uint8)  # BGR: Red

    # Create overlay
    overlay = np.zeros_like(result)
    overlay[mask_bool] = color

    # Blend
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)

    # Draw contours for better visibility
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_color = (0, 200, 0) if is_valid else (0, 0, 200)
    cv2.drawContours(result, contours, -1, contour_color, 2)

    return result


def add_text_overlay(
    frame: np.ndarray,
    frame_idx: int,
    object_name: str,
    view_idx: int,
    is_valid: bool
) -> np.ndarray:
    """Add text overlay showing frame info and validity status."""
    result = frame.copy()

    # Background for text
    cv2.rectangle(result, (10, 10), (400, 100), (0, 0, 0), -1)

    # Text info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(result, f"Frame: {frame_idx}", (20, 35),
                font, font_scale, (255, 255, 255), thickness)
    cv2.putText(result, f"View: {view_idx}", (20, 60),
                font, font_scale, (255, 255, 255), thickness)
    cv2.putText(result, f"Object: {object_name}", (20, 85),
                font, font_scale, (255, 255, 255), thickness)

    # Validity status
    status_text = "VALID" if is_valid else "INVALID"
    status_color = (0, 255, 0) if is_valid else (0, 0, 255)
    cv2.putText(result, status_text, (250, 60),
                font, 1.0, status_color, 2)

    return result


def resolve_view_validity(
    validity_arr: np.ndarray,
    view_idx: int,
    view_idx_in_list: int,
    view_indices: List[int],
) -> bool:
    """Resolve validity for one view from either full-view or subset-format arrays."""
    arr = np.asarray(validity_arr).reshape(-1)
    n = arr.size
    if n == 0:
        return True

    # Format A: full 42-view array indexed by absolute view id.
    if n == 42 and 0 <= view_idx < n:
        return bool(arr[view_idx])

    # Format B: subset array aligned with current view list order.
    if n == len(view_indices) and 0 <= view_idx_in_list < n:
        return bool(arr[view_idx_in_list])

    # Fallback for mixed/legacy cases.
    if 0 <= view_idx < n:
        return bool(arr[view_idx])
    if 0 <= view_idx_in_list < n:
        return bool(arr[view_idx_in_list])
    return True


def create_video_for_object_view(
    args,
    object_name: str,
    view_idx: int,
    frame_files: List[str],
    frame_cache: Dict[int, np.ndarray],
    validity_data: Dict[int, dict],
    mask_data_cache: Dict[int, object],
    view_idx_in_list: int
) -> str:
    """Create video for a specific object and view."""
    output_dir = join(args.output_path, args.seq_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = join(output_dir, f"{object_name}_view{view_idx}.mp4")

    out_width = 1920
    out_height = 1080

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, args.fps, (out_width, out_height))

    for frame_file in tqdm(frame_files, desc=f"{object_name} view {view_idx}", leave=False):
        frame_idx = int(frame_file.replace('.npz', ''))

        # Get cached frame
        frame = frame_cache.get(frame_idx)
        if frame is None:
            continue

        # Get mask
        if frame_idx not in mask_data_cache:
            continue
        all_masks = mask_data_cache[frame_idx]
        if object_name not in all_masks.files:
            continue
        mask = all_masks[object_name][view_idx]

        # Get validity
        validity_key = f'{object_name}_validity'
        if frame_idx in validity_data and validity_key in validity_data[frame_idx]:
            validity_arr = validity_data[frame_idx][validity_key]
            is_valid = resolve_view_validity(
                validity_arr, view_idx, view_idx_in_list, args.views
            )
        else:
            is_valid = True

        # Overlay mask
        frame_with_mask = overlay_mask_on_frame(frame, mask, is_valid, args.alpha)

        # Add text overlay
        frame_final = add_text_overlay(frame_with_mask, frame_idx, object_name, view_idx, is_valid)

        writer.write(frame_final)

    writer.release()
    return output_file


def create_combined_video(
    args,
    object_name: str,
    frame_files: List[str],
    video_frame_caches: Dict[int, Dict[int, np.ndarray]],
    validity_data: Dict[int, dict],
    mask_data_cache: Dict[int, object]
) -> str:
    """Create combined multi-view video from pre-loaded frame caches."""
    output_dir = join(args.output_path, args.seq_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = join(output_dir, f"{object_name}_combined.mp4")

    n_views = len(args.views)
    # Dynamic grid layout based on view count
    import math
    n_cols = min(n_views, 7)
    n_rows = math.ceil(n_views / n_cols)

    # Scale down view size for large grids to keep output manageable
    if n_views <= 10:
        view_width, view_height = 384, 216
    elif n_views <= 21:
        view_width, view_height = 320, 180
    else:
        view_width, view_height = 256, 144

    out_width = view_width * n_cols
    out_height = view_height * n_rows

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, args.fps, (out_width, out_height))

    for frame_file in tqdm(frame_files, desc=f"{object_name} combined", leave=False):
        frame_idx = int(frame_file.replace('.npz', ''))

        combined_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        for i, view_idx in enumerate(args.views):
            row = i // n_cols
            col = i % n_cols

            # Get cached frame (already resized to view_width x view_height)
            cache = video_frame_caches.get(view_idx, {})
            frame = cache.get(frame_idx)
            if frame is None:
                continue

            # Get mask
            if frame_idx in mask_data_cache and object_name in mask_data_cache[frame_idx].files:
                mask = mask_data_cache[frame_idx][object_name][view_idx]
                mask = cv2.resize(mask, (view_width, view_height), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((view_height, view_width), dtype=np.uint8)

            # Get validity
            validity_key = f'{object_name}_validity'
            if frame_idx in validity_data and validity_key in validity_data[frame_idx]:
                validity_arr = validity_data[frame_idx][validity_key]
                is_valid = resolve_view_validity(
                    validity_arr, view_idx, i, args.views
                )
            else:
                is_valid = True

            # Overlay mask
            frame_with_mask = overlay_mask_on_frame(frame, mask, is_valid, args.alpha)

            # Add view index
            cv2.putText(frame_with_mask, f"V{view_idx}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            status_color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.circle(frame_with_mask, (view_width - 15, 15), 8, status_color, -1)

            # Place in combined frame
            y1 = row * view_height
            y2 = y1 + view_height
            x1 = col * view_width
            x2 = x1 + view_width
            combined_frame[y1:y2, x1:x2] = frame_with_mask

        # Add frame counter
        cv2.putText(combined_frame, f"Frame: {frame_idx}", (10, out_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(combined_frame)

    writer.release()
    return output_file


def process_sequence(args):
    """Process a single sequence for visualization."""
    root_path = args.root_path
    seq_name = args.seq_name
    view_indices = args.views

    # Setup paths
    videos_path = join(root_path, 'videos', seq_name, 'videos')
    mask_npz_path = join(root_path, 'mask_npz', seq_name)
    validity_seq_path = join(args.validity_path, seq_name)

    # Check paths exist
    if not os.path.exists(validity_seq_path):
        print(f"Error: Validity results not found at {validity_seq_path}")
        print("Please run multi_view_mask_check.py first.")
        return

    # Get frame list from validity results, filter by frame number range
    frame_files = sorted([f for f in os.listdir(validity_seq_path) if f.endswith('.npz')])
    if args.start_frame > 0 or args.end_frame > 0:
        frame_files = [
            f for f in frame_files
            if (args.start_frame <= int(f.replace('.npz', '')))
            and (int(f.replace('.npz', '')) < args.end_frame or args.end_frame == 0)
        ]
    if args.step > 1:
        frame_files = frame_files[::args.step]

    # Resolve --all_views
    if args.all_views:
        view_indices = list(range(42))
        args.views = view_indices
        print(f"--all_views: using {len(view_indices)} views")

    print(f"Processing {len(frame_files)} frames")

    # Load validity data
    print("Loading validity data...")
    validity_data = {}
    for frame_file in tqdm(frame_files, desc="Loading validity"):
        frame_idx = int(frame_file.replace('.npz', ''))
        validity_path = join(validity_seq_path, frame_file)
        validity_data[frame_idx] = dict(np.load(validity_path))

    # Load mask data (lazy-loaded npz — arrays are read on demand)
    print("Loading mask data...")
    mask_data_cache = {}
    for frame_file in tqdm(frame_files, desc="Loading masks"):
        frame_idx = int(frame_file.replace('.npz', ''))
        mask_path = join(mask_npz_path, frame_file)
        if os.path.exists(mask_path):
            mask_data_cache[frame_idx] = np.load(mask_path)

    # Get object names from first frame
    first_frame = int(frame_files[0].replace('.npz', ''))
    if first_frame in validity_data:
        object_names = [k.replace('_validity', '') for k in validity_data[first_frame].keys()]
    else:
        print("Error: Could not determine object names")
        return

    if args.object_name:
        if args.object_name in object_names:
            object_names = [args.object_name]
        else:
            print(f"Error: Object '{args.object_name}' not found. Available: {object_names}")
            return

    print(f"Objects to process: {object_names}")

    # Inspect one validity array to report indexing mode.
    sample_validity = None
    for obj in object_names:
        key = f"{obj}_validity"
        if first_frame in validity_data and key in validity_data[first_frame]:
            sample_validity = np.asarray(validity_data[first_frame][key]).reshape(-1)
            break
    if sample_validity is not None:
        if sample_validity.size == 42:
            print("Validity format: full 42-view array (index by view id)")
        elif sample_validity.size == len(view_indices):
            print("Validity format: subset array (index by current --views order)")
        else:
            print(f"Validity format: ambiguous length={sample_validity.size}, using fallback indexing")

    # Determine target frame size based on mode
    frame_indices = [int(f.replace('.npz', '')) for f in frame_files]
    if args.combined:
        n_views = len(view_indices)
        if n_views <= 10:
            target_size = (384, 216)
        elif n_views <= 21:
            target_size = (320, 180)
        else:
            target_size = (256, 144)
    else:
        target_size = (1920, 1080)

    # Pre-load video frames for all views (sequential read, no random seeking)
    print("Pre-loading video frames (sequential read)...")
    video_frame_caches = {}
    for view_idx in tqdm(view_indices, desc="Loading videos"):
        video_file = join(videos_path, f"{view_idx}.mp4")
        if os.path.exists(video_file):
            video_frame_caches[view_idx] = preload_video_frames(
                video_file, frame_indices, target_size
            )
        else:
            print(f"Warning: Video not found: {video_file}")

    if not video_frame_caches:
        print("Error: No video files found")
        return

    # Create output directory
    output_dir = join(args.output_path, seq_name)
    os.makedirs(output_dir, exist_ok=True)

    # Process each object
    for object_name in object_names:
        print(f"\nProcessing object: {object_name}")

        if args.combined:
            output_file = create_combined_video(
                args, object_name, frame_files,
                video_frame_caches, validity_data, mask_data_cache
            )
            print(f"  Created: {output_file}")
        else:
            for i, view_idx in enumerate(view_indices):
                if view_idx not in video_frame_caches:
                    continue

                output_file = create_video_for_object_view(
                    args, object_name, view_idx, frame_files,
                    video_frame_caches[view_idx], validity_data,
                    mask_data_cache, i
                )
                print(f"  Created: {output_file}")

    print(f"\nDone! Videos saved to {output_dir}")


def main():
    args = parse_args()
    process_sequence(args)


if __name__ == '__main__':
    main()

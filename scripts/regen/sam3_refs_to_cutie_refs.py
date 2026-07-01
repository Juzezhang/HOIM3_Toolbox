"""Convert SAM3 cross-view per_view_masks.npz → Cutie reference dir structure.

Reads:
    /simurgh2/datasets/HOI-M3/sam3_xv_probe/<seq>__<obj>__refv<V>_f<F>/
        per_view_masks.npz     (vK: (720, 1280) uint8)
        per_view_ref_frame.json ({view: frame_id})

Writes (matches cutie_track_one_view.py expected layout):
    /simurgh2/datasets/HOI-M3/cutie_refs_recovery_<obj>/<seq>/masks/
        <view>.npy             (indexed (1080, 1920) uint8; bg=0, obj=1)
        <view>_names.json      ({"mask_names": [<obj>]})
        <view>_start_frame.txt (anchor frame_id)
"""
import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--sam3_dir", required=True,
                    help="e.g. .../sam3_xv_probe/office_data32__shredder__refv35_f10000")
    ap.add_argument("--out_root", default=None,
                    help="Default: /simurgh2/datasets/HOI-M3/cutie_refs_recovery_<obj>")
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = f"/simurgh2/datasets/HOI-M3/cutie_refs_recovery_{args.obj}"

    masks_npz = osp.join(args.sam3_dir, "per_view_masks.npz")
    ref_frame_json = osp.join(args.sam3_dir, "per_view_ref_frame.json")
    assert osp.isfile(masks_npz), f"missing {masks_npz}"
    assert osp.isfile(ref_frame_json), f"missing {ref_frame_json}"

    data = np.load(masks_npz)
    rf = json.load(open(ref_frame_json))

    out_dir = osp.join(args.out_root, args.seq, "masks")
    os.makedirs(out_dir, exist_ok=True)

    TARGET_H, TARGET_W = 1080, 1920
    n_written = 0
    for view_key in data.keys():
        view = int(view_key[1:])  # "v17" → 17
        m720 = data[view_key]
        if m720.shape != (TARGET_H, TARGET_W):
            m = cv2.resize(m720.astype(np.uint8), (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        else:
            m = m720
        # Indexed: 0=bg, 1=our object
        indexed = (m > 0).astype(np.uint8)

        np.save(osp.join(out_dir, f"{view}.npy"), indexed)
        json.dump({"mask_names": [args.obj]}, open(osp.join(out_dir, f"{view}_names.json"), "w"))
        with open(osp.join(out_dir, f"{view}_start_frame.txt"), "w") as f:
            f.write(str(int(rf.get(str(view), 0))))
        n_written += 1

    print(f"[done] wrote {n_written} view refs → {out_dir}")
    print(f"       run Cutie via:")
    print(f"         cutie_track_one_view.py --seq {args.seq} --view <V> "
          f"--ref_root {args.out_root} --output_root .../cutie_tracking_recovery_{args.obj}")


if __name__ == "__main__":
    main()

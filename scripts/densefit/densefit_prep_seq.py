"""Generalized DenseFit prep for a full HOI-M3 seq (any date/views/persons/frames).
Writes calib (scaled to image res) + capture descriptor + cam_XX->view DIR
symlinks (cheap: one symlink per view, NOT per frame). Uses calib_with_distortion
so DenseFit output lands in the smplx_with_distortion world frame.
Usage: densefit_prep_seq.py <seq>
"""
import json, os, sys, glob
import numpy as np
import cv2

D = '/simurgh2/datasets/HOI-M3'
M = os.environ.get('DENSEFIT_ROOT', '/path/to/densefit_workspace')
INFO = json.load(open('/simurgh/group/juze/datasets/HOI-M3/dataset_information.json'))


def date_of(seq):
    for dt, ss in INFO.items():
        if seq in ss:
            return dt
    raise ValueError(f'no date for {seq}')


def prep(seq):
    dt = date_of(seq)
    try:
        meta = json.load(open(f'{D}/mask_shards/{seq}/meta.json'))
        npers = len([o for o in meta['objects'] if str(o).startswith('person')])
        nf = int(meta['num_frames'])
    except FileNotFoundError:
        # shards mid-rebuild (finish-cpu rm+convert race): fall back to
        # cutie_refs names (persons) + images dir count (frames)
        names = json.load(open(f'{D}/cutie_refs/{seq}/masks/0_names.json'))['mask_names']
        npers = len([n for n in names if n.startswith('person')])
        nf = len([f for f in os.listdir(f'{D}/images/{seq}/0') if f.endswith('.jpg')])
    calib = json.load(open(f'{D}/calib_with_distortion/{dt}/calibration.json'))
    # CANONICAL 37-view fitting set (excludes 30/31/32/33/40 = the unreliable/
    # per-date-nulled cameras) — MUST match mhr_withdist/smplx_with_distortion.
    # NOT a K-presence filter (that wrongly includes bad-but-present cams).
    CANON37 = [v for v in range(42) if v not in (30, 31, 32, 33, 40)]
    views = [v for v in CANON37 if calib.get(str(v)) and calib[str(v)].get('K')]
    assert len(views) == 37, f'{seq}: expected 37 canonical views, got {len(views)}: {views}'
    # image resolution (from first valid view's first frame)
    im = cv2.imread(sorted(glob.glob(f'{D}/images/{seq}/{views[0]}/*.jpg'))[0])
    ih, iw = im.shape[:2]
    scale = ih / 2160.0  # calib K is @4K (2160)

    out_cal = {}
    for v in views:
        c = calib[str(v)]
        K = np.array(c['K'], float).reshape(3, 3).copy(); K[:2, :] *= scale
        rt = np.array(c['RT'], float)
        ext = rt.reshape(4, 4)[:3] if rt.size == 16 else rt.reshape(3, 4)
        out_cal[f'cam_{v:02d}'] = {'intrinsic_matrix': K.tolist(),
                                   'extrinsics_matrix': ext.tolist(),
                                   'distortions': list(c.get('distCoeff') or [0]*5),
                                   'image_size': [iw, ih]}
    os.makedirs(f'{M}/configs/examples/calib', exist_ok=True)
    json.dump(out_cal, open(f'{M}/configs/examples/calib/hoim3_{seq}_full.json', 'w'), indent=1)

    # cam_XX -> view dir symlinks (frame k == 00000k.jpg already, full range)
    img_root = f'{M}/data/hoim3_full/{seq}/images'
    os.makedirs(img_root, exist_ok=True)
    for v in views:
        link = f'{img_root}/cam_{v:02d}'
        tgt = f'{D}/images/{seq}/{v}'
        if os.path.islink(link) or os.path.exists(link):
            if os.path.realpath(link) != os.path.realpath(tgt):
                os.remove(link); os.symlink(tgt, link)
        else:
            os.symlink(tgt, link)

    cams = [f'cam_{v:02d}' for v in views]
    cap = {'capture_root': f'{M}/data/hoim3_full',
           'calib': f'{M}/configs/examples/calib/hoim3_{seq}_full.json',
           'cam_fps': 30, 'videos_subdir': 'images', 'cams': cams,
           'sequences': {'000': {'name': seq}}}
    json.dump(cap, open(f'{M}/configs/examples/captures/hoim3_{seq}_full.json', 'w'), indent=1)
    print(f'{seq}: date={dt} persons={npers} frames={nf} views={len(views)} '
          f'imgHxW={ih}x{iw} scale={scale:.4f}')
    return dict(seq=seq, date=dt, persons=npers, frames=nf, views=views, ih=ih, iw=iw)


if __name__ == '__main__':
    r = prep(sys.argv[1])
    json.dump(r, open(f'/simurgh2/users/juze/calibjoint/prepinfo_{sys.argv[1]}.json', 'w'))

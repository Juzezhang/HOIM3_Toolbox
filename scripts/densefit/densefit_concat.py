"""Concatenate a seq's per-window DenseFit outputs → full-seq → convert to
smplx_with_distortion format → BACK UP original MHR files → merge into
/simurgh2/datasets/HOI-M3/smplx_with_distortion/.
body_id-0i is consistent across windows (masks use person_i→body_id i+1).
Usage: densefit_concat.py <seq> <win_size> [--commit]
  without --commit: writes to a staging dir + reports (dry run, no merge).
"""
import json, os, sys, glob
import numpy as np

M = os.environ.get('DENSEFIT_ROOT', '/path/to/densefit_workspace')
SMPLXWD = '/simurgh2/datasets/HOI-M3/smplx_with_distortion'
BAK = '/simurgh2/datasets/HOI-M3/smplx_with_distortion_mhr_bak'


def split_pose(pose):  # (N,165) -> dict of SMPL-X parts
    return dict(global_orient=pose[:, 0:3], body_pose=pose[:, 3:66],
               jaw_pose=pose[:, 66:69], leye_pose=pose[:, 69:72],
               reye_pose=pose[:, 72:75], left_hand_pose=pose[:, 75:120],
               right_hand_pose=pose[:, 120:165])


def main():
    seq, W = sys.argv[1], int(sys.argv[2])
    commit = '--commit' in sys.argv
    info = json.load(open(f'/simurgh2/users/juze/calibjoint/prepinfo_{seq}.json'))
    nf = info['frames']
    nwin = (nf + W - 1) // W
    # collect per-window dirs in order
    persons = {}   # pid -> list of (win, npz path)
    missing = []
    for w in range(nwin):
        d = glob.glob(f'{M}/output/ma_3d/{seq}_full_w{w:04d}/**/', recursive=True)
        found = False
        for dd in d:
            fs = glob.glob(f'{dd}/smplx_params_body_id-*.npz')
            if fs:
                for f in fs:
                    pid = int(f.split('body_id-')[1].split('.')[0])
                    persons.setdefault(pid, {})[w] = f
                found = True
        if not found:
            missing.append(w)
    if missing:
        print(f'{seq}: MISSING windows {missing[:10]}{"..." if len(missing)>10 else ""} '
              f'({len(missing)}/{nwin}) — NOT ready to concat'); sys.exit(2)

    stage = f'/simurgh2/users/juze/calibjoint/smplx_densefit_staged/{seq}'
    os.makedirs(stage, exist_ok=True)
    written = []
    for pid, wmap in sorted(persons.items()):
        parts = {k: [] for k in ['global_orient','body_pose','jaw_pose','leye_pose',
                                  'reye_pose','left_hand_pose','right_hand_pose',
                                  'transl','betas']}
        fids = []
        for w in range(nwin):
            z = np.load(wmap[w], allow_pickle=True)
            pose = z['smplx_pose'].astype(np.float32); n = pose.shape[0]
            sp = split_pose(pose)
            for k in sp: parts[k].append(sp[k])
            parts['transl'].append(z['smplx_translation'].astype(np.float32))
            parts['betas'].append(np.repeat(z['smplx_betas'].astype(np.float32), n, axis=0))  # (n,16)
            start = w * W
            fids.append(np.arange(start, start + n, dtype=np.int32))
        out = {k: np.concatenate(v, 0) for k, v in parts.items()}
        # Light 'touch' savgol (w=7,poly2) to damp per-window-fit boundary jumps
        # WITHOUT altering interior motion (interior step 0.2cm unchanged; boundary
        # 12cm->4cm; median per-frame displacement 0.02cm). transl + AA poses.
        from scipy.signal import savgol_filter
        for k in ('transl', 'global_orient', 'body_pose'):
            if out[k].shape[0] >= 7:
                out[k] = savgol_filter(out[k], 7, 2, axis=0).astype(np.float32)
        out['frame_ids'] = np.concatenate(fids)
        N = out['frame_ids'].shape[0]
        out['expression'] = np.zeros((N, 10), np.float32)
        out['fitting_errors'] = np.zeros((N,), np.float32)
        np.savez(f'{stage}/{seq}_person{pid}.npz', **out)
        json.dump({'seq': seq, 'person': pid, 'n_frames': int(N), 'n_betas': 16,
                   'source': 'DenseFit (cutie-fixed masks reuse)', 'calib': 'calib_with_distortion'},
                  open(f'{stage}/{seq}_person{pid}_meta.json', 'w'))
        written.append((pid, N))
        print(f'  {seq}_person{pid}: {N} frames, betas{out["betas"].shape[1]}')
    print(f'{seq}: staged {len(written)} persons ({nf} frames expected)')

    if commit:
        os.makedirs(BAK, exist_ok=True)
        for pid, _ in written:
            for suf in ['.npz', '_meta.json']:
                orig = f'{SMPLXWD}/{seq}_person{pid}{suf}'
                if os.path.exists(orig) and not os.path.exists(f'{BAK}/{seq}_person{pid}{suf}'):
                    import shutil; shutil.copy2(orig, f'{BAK}/{seq}_person{pid}{suf}')  # backup MHR
                import shutil; shutil.copy2(f'{stage}/{seq}_person{pid}{suf}', f'{SMPLXWD}/{seq}_person{pid}{suf}')
        print(f'{seq}: COMMITTED to {SMPLXWD} (MHR originals backed up to {BAK})')
    else:
        print(f'{seq}: DRY RUN (staged only). Re-run with --commit to backup+merge.')


if __name__ == '__main__':
    main()

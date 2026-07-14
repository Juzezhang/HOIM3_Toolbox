"""Convert DenseFit ma_3d output -> smplx_with_distortion format.
DenseFit smplx_pose (N,165) = SMPL-X full pose concat:
  [0:3]=global_orient [3:66]=body_pose(21x3) [66:69]=jaw [69:72]=leye
  [72:75]=reye [75:120]=left_hand(15x3) [120:165]=right_hand(15x3)
Output per person: <seq>_person<i>.npz + _meta.json, keys IDENTICAL to
/simurgh2/datasets/HOI-M3/smplx_with_distortion/. body_id-0i -> person i
(reinject maps person_i -> body_id i+1 -> DenseFit stack index i).
Same world frame (calib_with_distortion), so drop-in compatible.
"""
import numpy as np, json, glob, os, sys

REF_KEYS = ['frame_ids', 'jaw_pose', 'leye_pose', 'reye_pose', 'fitting_errors',
            'transl', 'global_orient', 'body_pose', 'left_hand_pose',
            'right_hand_pose', 'betas', 'expression']


def convert(densefit_dir, seq, frame_ids, out_dir, time_s=0.0):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f'{densefit_dir}/smplx_params_body_id-*.npz'))
    n_written = 0
    for f in files:
        pid = int(f.split('body_id-')[1].split('.')[0])  # 0-indexed person
        z = np.load(f, allow_pickle=True)
        pose = z['smplx_pose'].astype(np.float32)          # (N,165)
        N = pose.shape[0]
        transl = z['smplx_translation'].astype(np.float32)  # (N,3)
        betas16 = z['smplx_betas'].astype(np.float32)       # (1,16)
        betas = np.repeat(betas16, N, axis=0)               # (N,16) — KEEP all 16 (user decision 2026-07-12)
        fids = np.asarray(frame_ids[:N], dtype=np.int32)
        out = {
            'frame_ids': fids,
            'global_orient': pose[:, 0:3],
            'body_pose': pose[:, 3:66],
            'jaw_pose': pose[:, 66:69],
            'leye_pose': pose[:, 69:72],
            'reye_pose': pose[:, 72:75],
            'left_hand_pose': pose[:, 75:120],
            'right_hand_pose': pose[:, 120:165],
            'transl': transl,
            'betas': betas,
            'expression': np.zeros((N, 10), np.float32),   # DenseFit: no expression
            'fitting_errors': np.zeros((N,), np.float32),  # placeholder (DenseFit has no per-frame err)
        }
        assert set(out) == set(REF_KEYS), set(out) ^ set(REF_KEYS)
        np.savez(f'{out_dir}/{seq}_person{pid}.npz', **out)
        json.dump({'seq': seq, 'person': pid, 'n_frames': int(N),
                   'source': 'DenseFit (cutie-fixed masks reuse)',
                   'fitting_error_mean_m': None, 'time_s': float(time_s),
                   'calib': 'calib_with_distortion'},
                  open(f'{out_dir}/{seq}_person{pid}_meta.json', 'w'))
        n_written += 1
    return n_written


if __name__ == '__main__':
    # test on data03 reuse-cutie (60-frame window, frames 6000-6059)
    seq = sys.argv[1] if len(sys.argv) > 1 else 'livingroom_data03'
    densefit_dir = sys.argv[2] if len(sys.argv) > 2 else \
        f'/path/to/densefit_workspace/output/ma_3d/hoim3_reuse_cutie/hoim3_livingroom_data03_37/{seq}'
    fids = list(range(6000, 6060))
    out = sys.argv[3] if len(sys.argv) > 3 else '/simurgh2/users/juze/calibjoint/smplx_densefit_test'
    n = convert(densefit_dir, seq, fids, out)
    print(f'wrote {n} person files to {out}')
    # verify format matches reference
    ref = np.load('/simurgh2/datasets/HOI-M3/smplx_with_distortion/bedroom_data01_person0.npz', allow_pickle=True)
    new = np.load(f'{out}/{seq}_person0.npz', allow_pickle=True)
    print('KEYS match:', set(ref.files) == set(new.files))
    for k in ref.files:
        rs, ns = np.asarray(ref[k]).shape[1:], np.asarray(new[k]).shape[1:]
        print(f'  {k}: ref per-frame {rs} vs new {ns}  {"OK" if rs==ns else "MISMATCH"}')

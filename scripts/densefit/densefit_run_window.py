"""Run DenseFit reuse on ONE frame-window of a seq (for parallel full-length runs).
Injects that window's Cutie-fixed masks (relative-indexed 0..W-1), writes a
per-window config, invokes DenseFit. Called by the SLURM array (one task per window).
Usage: densefit_run_window.py <seq> <win_id> <win_size>
"""
import json, os, sys, subprocess, glob
import numpy as np, cv2

M = os.environ.get('DENSEFIT_ROOT', '/path/to/densefit_workspace')
CTROOT = '/simurgh2/datasets/HOI-M3/cutie_tracking'
# Cutie index: object order [....persons at the end]; person i -> body_id i+1.
# Person cutie index computed per-seq from ref names below.


def person_cutie_indices(seq):
    """cutie 'mask' index for person0..N-1 = position (1-based) in ref names."""
    names = json.load(open(f'/simurgh2/datasets/HOI-M3/cutie_refs/{seq}/masks/0_names.json'))['mask_names']
    return [names.index(f'person{i}') + 1 for i in range(len([n for n in names if n.startswith('person')]))]


def inject_window(seq, start, end, views, pidx, win_dir):
    """Write masks mask_{rel:04d}_{bid:02d}.png (bid=person i+1) for abs frames
    start..end-1. PARALLEL over views (NFS-latency bound; serial took ~30 min/
    window and was the pipeline bottleneck; cv2/np release the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    def _one_view(v):
        od = f'{win_dir}/{seq}/cam_{v:02d}/masks'
        os.makedirs(od, exist_ok=True)
        k = 0
        for rel, absf in enumerate(range(start, end)):
            p = f'{CTROOT}/{seq}/{v}/{absf:06d}.npz'
            m = np.load(p, allow_pickle=True)['mask'] if os.path.exists(p) else None
            for i, ci in enumerate(pidx):
                bid = i + 1
                if m is None:
                    a = np.zeros((720, 1280), np.uint8)
                else:
                    a = (m == ci).astype(np.uint8) * 255
                    if a.shape != (720, 1280):
                        a = cv2.resize(a, (1280, 720), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'{od}/mask_{rel:04d}_{bid:02d}.png', a)
                k += 1
        return k

    with ThreadPoolExecutor(max_workers=12) as pool:
        return sum(pool.map(_one_view, views))


def main():
    seq, win_id, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    info = json.load(open(f'/simurgh2/users/juze/calibjoint/prepinfo_{seq}.json'))
    nf, views = info['frames'], info['views']
    start, end = win_id * W, min((win_id + 1) * W, nf)
    if start >= nf:
        print(f'{seq} win{win_id}: start {start} >= nf {nf}, nothing to do'); return
    tag = f'{seq}_full_w{win_id:04d}'
    out3d = f'{M}/output/ma_3d/{tag}'
    if os.path.exists(out3d) and any('smplx_params' in f for _, _, fs in os.walk(out3d) for f in fs):
        print(f'{seq} win{win_id}: already done'); return

    # REUSE FALLBACK: if this window's ma_2d already has all 37 good cams, skip
    # ma_cap+ma_2d (the expensive part) and run ONLY ma_3d on the existing ma_2d.
    # Otherwise run the full pipeline (ma_cap + batched ma_2d + ma_3d).
    good_cams = [f'cam_{v:02d}' for v in views]  # views == the canonical 37
    ma2d_have = {os.path.basename(f)[:6]
                 for f in glob.glob(f'{M}/output/ma_2d/{tag}/**/cam_*.npz', recursive=True)}
    reuse_ma2d = all(g in ma2d_have for g in good_cams)
    macap_have = bool(glob.glob(f'{M}/output/ma_cap/{tag}/**/cam_*', recursive=True))

    win_dir = f'{M}/data/injected_full/{tag}'          # ma_masks_dir (parent; runner appends seq)
    if reuse_ma2d:
        print(f'{seq} win{win_id}: REUSE existing ma_2d (37/37 cams) -> ma_3d only', flush=True)
    else:
        pidx = person_cutie_indices(seq)
        nmask = inject_window(seq, start, end, views, pidx, win_dir)
        print(f'{seq} win{win_id}: full run; injected {nmask} masks [{start},{end})', flush=True)

    # per-window config
    cfg = {
        'global': {'version': 1.0, 'username': '', 'out_dir': 'output',
                   'jobs_log_dir': 'output/logs/jobs', 'bind': [], 'conda_env': 'densefit',
                   'start_frame': start, 'end_frame': end},
        'ma_cap': {'engine': 'conda', 'enabled': (not reuse_ma2d) or (not macap_have),
                   'dependencies': [], 'script': 'run_ma_cap.py', 'repo_path': 'capture',
                   'images_root_dir': f'{M}/data/hoim3_full/{seq}/images', 'flags': []},
        'ma_masks': {'engine': 'conda', 'enabled': False, 'dependencies': ['ma_cap'],
                     'script': 'run_ma_masks.py', 'repo_path': 'segmentation',
                     'flags': ['--sam_version sam2'], 'undistort': False},
        'ma_2d': {'engine': 'conda', 'enabled': not reuse_ma2d, 'dependencies': ['ma_cap'],
                  'script': 'run_ma_2d.py', 'repo_path': 'landmarks',
                  'config_path': 'configs/train/models_2d/config_densefitnet_mask_512.yaml',
                  'weights': f'{M}/data/weights/ma_2d/densefit_mask_full_cvpr.ckpt', 'flags': [],
                  'undistort': False, 'ma_masks_dir': win_dir},
        'ma_3d': {'engine': 'conda', 'enabled': True, 'dependencies': [] if reuse_ma2d else ['ma_2d'],
                  'script': 'run_ma_3d.py', 'repo_path': 'optimization',
                  'config_file': 'config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml',
                  'flags': []},
        'ma_vis': {'engine': 'conda', 'enabled': False, 'dependencies': ['ma_2d', 'ma_3d'],
                   'script': 'run_ma_vis.py', 'repo_path': 'visualization', 'flags': [], 'undistort': False},
    }
    import yaml
    cfgp = f'{M}/configs/examples/presets/{tag}.yaml'
    yaml.safe_dump(cfg, open(cfgp, 'w'))
    capp = f'{M}/configs/examples/captures/hoim3_{seq}_full.json'
    cmd = ['python', '-m', 'inference', 'run', '--cfg', cfgp, '--capture', capp, '--out-tag', tag, '-v']
    print('RUN', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=M)
    # cleanup masks to reclaim inodes/disk
    if r.returncode == 0:
        subprocess.run(['rm', '-rf', win_dir])
    print(f'{seq} win{win_id}: exit {r.returncode}', flush=True)
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()

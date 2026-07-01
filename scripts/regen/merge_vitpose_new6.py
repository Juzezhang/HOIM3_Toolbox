#!/usr/bin/env python3
"""
Merge new-6-view ViTPose cache (vitpose_new6/) into legacy 10-view cache (vitpose/).

For each seq's person dir, concatenate along the view axis so legacy goes from
shape (n_frames, 10, 23, 3) → (n_frames, 16, 23, 3) and meta.npz['views'] grows
from 10 names to 16 names.

Layout:
  Legacy: /scr/juze/datasets/HOI-M3/vitpose/<seq>/<person>/{keypoints_coco23.npy, meta.npz}
          legacy views = ['0','2','5','6','7','8','10','11','14','15']
  New:    /scr/juze/datasets/HOI-M3/vitpose_new6/<seq>/<person>/{keypoints_coco23.npy, meta.npz}
          new views = ['17','19','21','22','23','24']

After merge:
  16 views = ['0','2','5','6','7','8','10','11','14','15','17','19','21','22','23','24']

Idempotent:
  - If legacy already has 16 views, skip (already merged).
  - If new dir for a seq/person is missing, skip that person.
  - If frame counts mismatch between legacy/new, log warning and skip.

Atomic:
  - Writes to .tmp then os.replace() so a crashed merge never leaves the legacy
    cache half-written.

Usage:
  python merge_vitpose_new6.py                     # all seqs, 4 workers
  python merge_vitpose_new6.py --workers 8         # more parallelism
  python merge_vitpose_new6.py --seq bedroom_data01  # single seq
  python merge_vitpose_new6.py --dry-run           # print what would happen
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


LEGACY_ROOT = Path('/scr/juze/datasets/HOI-M3/vitpose')
NEW_ROOT = Path('/scr/juze/datasets/HOI-M3/vitpose_new6')
LEGACY_VIEWS = ['0', '2', '5', '6', '7', '8', '10', '11', '14', '15']
NEW_VIEWS = ['17', '19', '21', '22', '23', '24']
FINAL_VIEWS = LEGACY_VIEWS + NEW_VIEWS  # 16 views
EXPECTED_LEGACY_NVIEWS = len(LEGACY_VIEWS)  # 10
EXPECTED_NEW_NVIEWS = len(NEW_VIEWS)  # 6
EXPECTED_FINAL_NVIEWS = len(FINAL_VIEWS)  # 16


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    # np.save auto-appends .npy when given a path string. We must give it a
    # path that already ends with .npy so the actual file on disk matches
    # what we pass to os.replace. Otherwise tmp would be "foo.npy.tmp" but
    # np.save would write "foo.npy.tmp.npy", and os.replace would fail.
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)  # e.g. keypoints_coco23.tmp.npy
    np.save(str(tmp), arr)
    os.replace(str(tmp), str(path))


def _atomic_save_npz(path: Path, **kwargs) -> None:
    # Same trick for .npz: np.savez auto-appends .npz.
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)  # e.g. meta.tmp.npz
    np.savez(str(tmp), **kwargs)
    os.replace(str(tmp), str(path))


def merge_person(legacy_dir: Path, new_dir: Path, dry_run: bool = False) -> tuple[str, str]:
    """Returns (status, msg)."""
    legacy_kp_path = legacy_dir / 'keypoints_coco23.npy'
    legacy_meta_path = legacy_dir / 'meta.npz'
    new_kp_path = new_dir / 'keypoints_coco23.npy'
    new_meta_path = new_dir / 'meta.npz'

    if not legacy_kp_path.exists() or not legacy_meta_path.exists():
        return 'skip_no_legacy', f'no legacy at {legacy_dir}'
    if not new_kp_path.exists() or not new_meta_path.exists():
        return 'skip_no_new', f'no new at {new_dir}'

    legacy_meta = np.load(legacy_meta_path, allow_pickle=True)
    legacy_views = [str(v) for v in legacy_meta['views']]

    # Idempotency check.
    if len(legacy_views) == EXPECTED_FINAL_NVIEWS:
        return 'skip_already_merged', f'legacy already has {EXPECTED_FINAL_NVIEWS} views'

    if len(legacy_views) != EXPECTED_LEGACY_NVIEWS:
        return 'warn_unexpected_legacy_nviews', (
            f'legacy has {len(legacy_views)} views, expected {EXPECTED_LEGACY_NVIEWS}: {legacy_views}'
        )

    new_meta = np.load(new_meta_path, allow_pickle=True)
    new_views = [str(v) for v in new_meta['views']]
    if new_views != NEW_VIEWS:
        return 'warn_unexpected_new_views', f'new views={new_views} expected {NEW_VIEWS}'

    legacy_kp = np.load(legacy_kp_path)  # (n_frames, 10, 23, 3)
    new_kp = np.load(new_kp_path)        # (n_frames, 6, 23, 3)

    if legacy_kp.ndim != 4 or new_kp.ndim != 4:
        return 'warn_bad_ndim', f'legacy ndim={legacy_kp.ndim}, new ndim={new_kp.ndim}'
    if legacy_kp.shape[0] != new_kp.shape[0]:
        return 'warn_frame_mismatch', (
            f'legacy n_frames={legacy_kp.shape[0]} != new n_frames={new_kp.shape[0]}'
        )
    if legacy_kp.shape[1] != EXPECTED_LEGACY_NVIEWS:
        return 'warn_bad_legacy_shape', f'legacy shape={legacy_kp.shape}'
    if new_kp.shape[1] != EXPECTED_NEW_NVIEWS:
        return 'warn_bad_new_shape', f'new shape={new_kp.shape}'

    merged_kp = np.concatenate([legacy_kp, new_kp], axis=1).astype(legacy_kp.dtype)
    merged_views = legacy_views + new_views
    if merged_views != FINAL_VIEWS:
        return 'warn_view_order_mismatch', f'merged_views={merged_views}'

    # frame_indices: keep legacy's (they should match anyway since both indexed
    # mono frame numbers identically).
    legacy_frame_indices = np.asarray(legacy_meta['frame_indices'])

    if dry_run:
        return 'dry_run_ok', (
            f'would write {merged_kp.shape} dtype={merged_kp.dtype} '
            f'views=[{len(merged_views)}]'
        )

    _atomic_save_npy(legacy_kp_path, merged_kp)
    _atomic_save_npz(
        legacy_meta_path,
        views=np.array(merged_views),
        frame_indices=legacy_frame_indices,
    )
    return 'ok', f'merged {merged_kp.shape}'


def merge_seq(seq: str, dry_run: bool = False) -> dict:
    """Returns dict with per-person status, plus seq-level summary."""
    legacy_seq = LEGACY_ROOT / seq
    new_seq = NEW_ROOT / seq

    result = {
        'seq': seq,
        'persons': {},
        'n_merged': 0,
        'n_skipped': 0,
        'n_warned': 0,
        'error': None,
    }

    if not legacy_seq.is_dir():
        result['error'] = f'no legacy seq {legacy_seq}'
        return result
    if not new_seq.is_dir():
        result['error'] = f'no new seq {new_seq}'
        return result

    persons = sorted(p.name for p in new_seq.iterdir() if p.is_dir() and p.name.startswith('person'))
    if not persons:
        result['error'] = f'no persons under {new_seq}'
        return result

    for p in persons:
        legacy_pd = legacy_seq / p
        new_pd = new_seq / p
        try:
            status, msg = merge_person(legacy_pd, new_pd, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001
            status, msg = 'error', f'exception: {e!r}'
        result['persons'][p] = (status, msg)
        if status == 'ok' or status == 'dry_run_ok':
            result['n_merged'] += 1
        elif status.startswith('skip'):
            result['n_skipped'] += 1
        else:
            result['n_warned'] += 1

    return result


def _worker(args):
    seq, dry_run = args
    return merge_seq(seq, dry_run=dry_run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seq', type=str, default=None, help='single seq instead of all')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.seq is not None:
        seqs = [args.seq]
    else:
        seqs = sorted(p.name for p in NEW_ROOT.iterdir() if p.is_dir())

    print(f'Found {len(seqs)} candidate seqs in {NEW_ROOT}', flush=True)
    print(f'workers={args.workers}, dry_run={args.dry_run}', flush=True)

    t0 = time.time()
    total = {'merged': 0, 'skipped': 0, 'warned': 0, 'errored': 0}
    warn_details = []

    if args.workers <= 1:
        for s in seqs:
            res = merge_seq(s, dry_run=args.dry_run)
            _accumulate(res, total, warn_details)
            print(_fmt_result(res), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, (s, args.dry_run)): s for s in seqs}
            for fut in as_completed(futs):
                res = fut.result()
                _accumulate(res, total, warn_details)
                print(_fmt_result(res), flush=True)

    dt = time.time() - t0
    print(f'\n=== SUMMARY (elapsed={dt:.0f}s) ===', flush=True)
    print(f'  seqs processed:      {len(seqs)}', flush=True)
    print(f'  persons merged:      {total["merged"]}', flush=True)
    print(f'  persons skipped:     {total["skipped"]}', flush=True)
    print(f'  persons warned:      {total["warned"]}', flush=True)
    print(f'  seq-level errors:    {total["errored"]}', flush=True)
    if warn_details:
        print('\nWarnings (up to 50):', flush=True)
        for w in warn_details[:50]:
            print(f'  {w}', flush=True)
    return 0


def _accumulate(res, total, warn_details):
    if res.get('error'):
        total['errored'] += 1
        warn_details.append(f'{res["seq"]}: SEQ_ERROR {res["error"]}')
        return
    total['merged'] += res['n_merged']
    total['skipped'] += res['n_skipped']
    total['warned'] += res['n_warned']
    for p, (status, msg) in res['persons'].items():
        if not (status == 'ok' or status == 'dry_run_ok' or status.startswith('skip')):
            warn_details.append(f'{res["seq"]}/{p}: {status} — {msg}')


def _fmt_result(res):
    if res.get('error'):
        return f'[{res["seq"]}] SEQ_ERROR: {res["error"]}'
    parts = [f'[{res["seq"]}]']
    parts.append(f'merged={res["n_merged"]}')
    parts.append(f'skipped={res["n_skipped"]}')
    if res['n_warned']:
        parts.append(f'WARNED={res["n_warned"]}')
    for p, (status, msg) in res['persons'].items():
        if not (status == 'ok' or status == 'dry_run_ok'):
            parts.append(f'{p}:{status}')
    return ' '.join(parts)


if __name__ == '__main__':
    sys.exit(main())

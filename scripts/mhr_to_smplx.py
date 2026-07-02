#!/usr/bin/env python3
"""HOI-M3 multi-view MHR fits -> standard SMPL-X, per-person packed.

Adapts hodome_mhr_official_convert.py to HOI-M3's MULTI-PERSON format. The canonical HOI-M3 fits
live at /simurgh2/datasets/HOI-M3/mhr_simplified_skel_v2_scale_pose/{seq}/mhr/{frame:06d}.json,
where each json is a LIST of persons, each:
  {id, Rh(3), Th(3), model_parameters(1,204), identity_coeffs(1), face_expr_coeffs(1)}
The MHR rig's model_parameters already encode the global transform (verified: params[3:6]≈Rh,
params[0:3]≈10*Th), so we run MHR's official Conversion.convert_mhr2smpl(method='pytorch',
single_identity=True) exactly as for HoDome, but once PER PERSON (grouped by id across frames).

Output: {output_dir}/{seq}_person{id}.npz with T-stacked SMPL-X arrays
  frame_ids (T,), transl (T,3), global_orient (T,3), body_pose (T,63),
  left_hand_pose (T,45), right_hand_pose (T,45), jaw/leye/reye_pose (T,3),
  betas (T,10), expression (T,10), fitting_errors (T,)
+ a {seq}_person{id}_meta.json sidecar.

Run in the `hodome` env (L40s) or the `ytfree` env (free-pool Pascal/Volta; needs the
pymomentum.solver dunder-safe stub — see tools/youtube_talking_mhr_to_smplx.py).
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

# MHR official conversion deps (same as the hodome/youtube converters)
MVBODYFIT = Path("/simurgh/u/juze/code/mv-bodyfit")
if str(MVBODYFIT) not in sys.path:
    sys.path.insert(0, str(MVBODYFIT))
MHR_REPO = Path("/simurgh/u/juze/code/MHR")
CONV_DIR = MHR_REPO / "tools" / "mhr_smpl_conversion"
if str(CONV_DIR) not in sys.path:
    sys.path.insert(0, str(CONV_DIR))
if str(MHR_REPO) not in sys.path:
    sys.path.insert(0, str(MHR_REPO))
os.chdir(CONV_DIR)  # conversion's file_assets.py resolves ./assets/ relative to cwd

import smplx as smplx_pkg
from mhr.mhr import MHR
# pymomentum.solver links a torch ABI that breaks under the free-pool torch downgrade; stub it if
# it can't load natively (the GPU 'pytorch' method never calls it). dunders must raise AttributeError
# so torch._dynamo's lazy import (inspect.getsourcefile) doesn't splitext a fake __file__.
import types as _types
try:
    import pymomentum.solver  # noqa: F401
except Exception:
    class _AnyAttr:
        def __getattr__(self, _n): return _AnyAttr()
        def __call__(self, *_a, **_k): return _AnyAttr()
    _stub = _types.ModuleType("pymomentum.solver")
    def _ga(_name):
        if _name.startswith("__") and _name.endswith("__"): raise AttributeError(_name)
        return _AnyAttr()
    _stub.__getattr__ = _ga
    sys.modules["pymomentum.solver"] = _stub
from conversion import Conversion

DEFAULT_HOIM3_ROOT = "/simurgh2/datasets/HOI-M3/mhr_simplified_skel_v2_scale_pose"
DEFAULT_OUTPUT_DIR = "/simurgh2/datasets/HOI-M3/smplx_from_mhr"
DEFAULT_SMPLX_DIR = "/simurgh2/users/juze/smplx_models"  # smplx.create expects {dir}/smplx/SMPLX_NEUTRAL.npz


def load_seq_persons(seq, root, frame_step=1):
    """Read all mhr/{frame}.json (list-of-persons) for a seq, group by person id.
    Returns dict pid -> (frame_ids(N,), ic(N,45), mp(N,204), fe(N,72)) and a reason on failure."""
    mhr_dir = Path(root) / seq / "mhr"
    if not mhr_dir.is_dir():
        return None, f"no {seq}/mhr dir"
    frame_files = sorted(mhr_dir.glob("*.json"))[::frame_step]
    if not frame_files:
        return None, "empty mhr dir"
    per = collections.defaultdict(lambda: {"fid": [], "ic": [], "mp": [], "fe": []})
    for fp in frame_files:
        try:
            data = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            data = [data]
        try:
            fid = int(fp.stem.lstrip("0") or "0")
        except ValueError:
            continue
        for rec in data:
            if not isinstance(rec, dict) or "model_parameters" not in rec:
                continue
            pid = int(rec.get("id", 0))
            mp = np.asarray(rec.get("model_parameters", []), np.float32).reshape(-1)
            ic = np.asarray(rec.get("identity_coeffs", []), np.float32).reshape(-1)
            fe = np.asarray(rec.get("face_expr_coeffs", []), np.float32).reshape(-1)
            def fit(a, n): return (np.concatenate([a, np.zeros(n - a.size, np.float32)]) if a.size < n else a[:n])
            mp, ic, fe = fit(mp, 204), fit(ic, 45), fit(fe, 72)
            if not (np.isfinite(mp).all() and np.isfinite(ic).all() and np.isfinite(fe).all()):
                continue
            per[pid]["fid"].append(fid); per[pid]["mp"].append(mp)
            per[pid]["ic"].append(ic); per[pid]["fe"].append(fe)
    out = {}
    for pid, d in per.items():
        if not d["fid"]:
            continue
        out[pid] = (np.array(d["fid"], np.int32), np.stack(d["ic"]), np.stack(d["mp"]), np.stack(d["fe"]))
    if not out:
        return None, "no valid persons"
    return out, None


class Worker:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        print(f"[gpu={args.gpu}] loading MHR + SMPL-X + Conversion...", flush=True)
        self.mhr_model = MHR.from_files(lod=1, device=self.device)
        self.smplx_model = smplx_pkg.create(
            args.smplx_dir, model_type="smplx", gender="neutral",
            use_pca=False, flat_hand_mean=True, num_betas=10, num_expression_coeffs=10).to(self.device)
        self.converter = Conversion(mhr_model=self.mhr_model, smpl_model=self.smplx_model,
                                    method="pytorch", batch_size=args.batch_size)
        print(f"[gpu={args.gpu}] ready.", flush=True)

    def process_seq(self, seq):
        persons, reason = load_seq_persons(seq, self.args.hoim3_root, self.args.frame_step)
        if persons is None:
            return [("blocked", seq, reason, 0)]
        results = []
        for pid, (fid, ic, mp, fe) in sorted(persons.items()):
            out_npz = Path(self.args.output_dir) / f"{seq}_person{pid}.npz"
            if out_npz.exists() and not self.args.overwrite:
                results.append(("skipped", f"{seq}_p{pid}", "exists", len(fid))); continue
            T = len(fid)
            t0 = time.perf_counter()
            # HOI-M3 seqs are LONG (~21k frames); converting all at once OOMs the 12GB free-pool cards.
            # Chunk: C frames per convert_mhr2smpl call, accumulate on CPU, concat. (single_identity is
            # per-chunk — fine here, one person per id, betas near-identical across chunks.)
            C = self.args.max_frames_per_call
            SMPL_KEYS = {"transl": 3, "global_orient": 3, "body_pose": 63,
                         "left_hand_pose": 45, "right_hand_pose": 45, "betas": 10, "expression": 10}
            acc = {k: [] for k in SMPL_KEYS}; acc_err = []
            def to_np(t, sh): return (np.zeros(sh, np.float32) if t is None else t.detach().cpu().numpy().astype(np.float32))
            try:
                for s in range(0, T, C):
                    e = min(s + C, T); n = e - s
                    res = self.converter.convert_mhr2smpl(
                        mhr_parameters={
                            "lbs_model_params": torch.from_numpy(mp[s:e]).to(self.device),
                            "identity_coeffs": torch.from_numpy(ic[s:e]).to(self.device),
                            "face_expr_coeffs": torch.from_numpy(fe[s:e]).to(self.device)},
                        single_identity=True, return_smpl_parameters=True, return_smpl_meshes=False,
                        return_smpl_vertices=False, return_fitting_errors=True, batch_size=self.args.batch_size)
                    pr = res.result_parameters
                    for k, d in SMPL_KEYS.items():
                        v = to_np(pr.get(k), (n, d))
                        if v.shape[0] != n: v = np.broadcast_to(v[0:1], (n, d)).copy()
                        acc[k].append(v)
                    acc_err.append(res.result_errors.astype(np.float32) if res.result_errors is not None else np.zeros((n,), np.float32))
                    del res; torch.cuda.empty_cache()
            except Exception as ex:
                results.append(("failed", f"{seq}_p{pid}", f"{type(ex).__name__}: {ex}", T)); continue
            o = {"frame_ids": fid.astype(np.int32),
                 "jaw_pose": np.zeros((T, 3), np.float32), "leye_pose": np.zeros((T, 3), np.float32),
                 "reye_pose": np.zeros((T, 3), np.float32), "fitting_errors": np.concatenate(acc_err)}
            for k in SMPL_KEYS:
                o[k] = np.concatenate(acc[k])
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_npz, **o)
            dt = time.perf_counter() - t0
            json.dump({"seq": seq, "person": pid, "n_frames": T, "fitting_error_mean_m": float(o["fitting_errors"].mean()),
                       "time_s": dt, "fps": T / max(dt, 1e-6)}, open(str(out_npz)[:-4] + "_meta.json", "w"), indent=2)
            results.append(("ok", f"{seq}_p{pid}", f"err={o['fitting_errors'].mean():.4f}m", T))
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoim3-root", default=DEFAULT_HOIM3_ROOT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--smplx-dir", default=DEFAULT_SMPLX_DIR)
    ap.add_argument("--seqs", default=None, help="comma-sep seq names (default: all)")
    ap.add_argument("--shard", default=None, help="i/N")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per seq (quick test)")
    ap.add_argument("--max-frames-per-call", type=int, default=2048,
                    help="chunk size: frames per convert_mhr2smpl call (bounds GPU mem on 12GB free-pool cards)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seqs:
        seqs = args.seqs.split(",")
    else:
        seqs = sorted(d.name for d in Path(args.hoim3_root).iterdir() if (d / "mhr").is_dir())
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/")); seqs = seqs[i::n]
    print(f"{len(seqs)} seqs to convert -> {args.output_dir}", flush=True)
    w = Worker(args)
    for seq in seqs:
        for status, name, info, n in w.process_seq(seq):
            print(f"  [{status}] {name} ({n} fr) {info}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared-variance ratio per trait per training stage, from vectors or activations dirs.

Usage: olmo_stage_analysis.py --layer 15 --stage base=/path/base/vectors --stage sft=/path/sft/vectors \
         --stage sft_raw=/path/sft_rawfmt/caa_activations ...
A dir containing <persona>_<trait>.pt is read as vectors (n_layers, hidden); a dir containing
<persona>_<trait>_pos.pt / _neg.pt is read as activations and vectors are formed as mean(pos)-mean(neg).
"""
import argparse, json
from pathlib import Path
import torch
import pathlib

import json as _json
def _lidx(path, L):
    """Map an absolute layer index to the stored index, via <dir>/layers.json when activations were layer-subset."""
    sc = pathlib.Path(path).parent / "layers.json"
    if sc.exists():
        layers = _json.loads(sc.read_text())["layers"]
        if L not in layers: raise KeyError(f"layer {L} not stored in {sc.parent} (stored: {layers})")
        return layers.index(L)
    return L
TRAITS = ["assertiveness","confidence","deference","empathy","honesty","impulsivity","risk_taking","warmth"]
PERSONAS = ["con_artist","drill_sergeant","farmer","kindergarten_teacher","politician","professor","street_hustler","surgeon","tech_ceo","therapist"]

def load_vec(d: Path, p, t, L):
    f = d / f"{p}_{t}.pt"
    if f.exists():
        v = torch.load(f, map_location="cpu", weights_only=True)
        v = v["vector"] if isinstance(v, dict) and "vector" in v else v
        return torch.nan_to_num(v[L].float())
    P = torch.load(d / f"{p}_{t}_pos.pt", map_location="cpu", weights_only=True)
    N = torch.load(d / f"{p}_{t}_neg.pt", map_location="cpu", weights_only=True)
    li = _lidx(d / f"{p}_{t}_pos.pt", L)
    P = torch.stack([torch.nan_to_num(x[li].float()) for x in P.values()]); N = torch.stack([torch.nan_to_num(x[li].float()) for x in N.values()])
    return P.mean(0) - N.mean(0)

def shared_variance(vecs):
    V = torch.stack(vecs); U = V / V.norm(dim=1, keepdim=True)
    s = U.mean(0); s = s / s.norm()
    proj = V @ s
    return (proj.pow(2).sum() / V.pow(2).sum()).item()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--stage", action="append", required=True, help="label=dir"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); res = {}
    for spec in a.stage:
        label, d = spec.split("=", 1); d = Path(d); res[label] = {}
        for t in TRAITS:
            try: res[label][t] = shared_variance([load_vec(d, p, t, a.layer) for p in PERSONAS])
            except FileNotFoundError as e: res[label][t] = None
        vals = [v for v in res[label].values() if v is not None]
        print(f"{label:14s} mean={sum(vals)/len(vals):.3f}  " + " ".join(f"{t[:5]}={res[label][t]:.3f}" if res[label][t] is not None else f"{t[:5]}=NA" for t in TRAITS))
    if a.out: json.dump({"layer": a.layer, "shared_variance": res}, open(a.out, "w"), indent=1)
if __name__ == "__main__": main()

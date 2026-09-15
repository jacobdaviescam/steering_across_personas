#!/usr/bin/env python3
"""Paired question-bootstrap CI for cos(v_ctx, v_null) - cos(v_nonsense, v_null), per trait, at one layer.
Same redraw of question ids applied to context, nonsense and null files (they share question ids)."""
import argparse, json, csv
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
def load(path, L):
    d = torch.load(path, map_location="cpu", weights_only=True)
    keys = sorted(d, key=lambda k: int(k[1:]))
    li = _lidx(path, L)
    return torch.stack([torch.nan_to_num(d[k][li].float()) for k in keys])
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
ap = argparse.ArgumentParser(); ap.add_argument("--acts", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--layer", type=int, default=22); ap.add_argument("--n-boot", type=int, default=200)
ap.add_argument("--contexts", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); acts, ref, L = Path(a.acts), Path(a.ref), a.layer
g = torch.Generator().manual_seed(0); rows = []
for t in TRAITS:
    if not (ref/f"null_{t}_pos.pt").exists(): continue  # skip missing trait
    Pn, Nn = load(ref/f"null_{t}_pos.pt", L), load(ref/f"null_{t}_neg.pt", L)
    Pz, Nz = load(ref/f"nonsense_{t}_pos.pt", L), load(ref/f"nonsense_{t}_neg.pt", L)
    n = min(len(Pn), len(Pz))
    for c in a.contexts:
        src = acts if (acts/f"{c}_{t}_pos.pt").exists() else ref
        Pc, Nc = load(src/f"{c}_{t}_pos.pt", L), load(src/f"{c}_{t}_neg.pt", L)
        m = min(n, len(Pc)); point = cos(Pc[:m].mean(0)-Nc[:m].mean(0), Pn[:m].mean(0)-Nn[:m].mean(0)) - cos(Pz[:m].mean(0)-Nz[:m].mean(0), Pn[:m].mean(0)-Nn[:m].mean(0))
        diffs = []
        for _ in range(a.n_boot):
            i = torch.randint(m, (m,), generator=g)
            vn = Pn[i].mean(0)-Nn[i].mean(0); vc = Pc[i].mean(0)-Nc[i].mean(0); vz = Pz[i].mean(0)-Nz[i].mean(0)
            diffs.append(cos(vc, vn) - cos(vz, vn))
        d = torch.tensor(diffs); lo, hi = d.quantile(0.025).item(), d.quantile(0.975).item()
        rows.append({"context": c, "trait": t, "ctx_minus_nonsense": point, "ci_lo": lo, "ci_hi": hi, "clear": "below" if hi < 0 else ("above" if lo > 0 else "overlaps 0")})
        print(f"{c:18s} {t:13s} {point:+.3f} [{lo:+.3f}, {hi:+.3f}] {rows[-1]['clear']}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out/"paired_boot.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(rows, open(out/"paired_boot.json", "w"), indent=1)

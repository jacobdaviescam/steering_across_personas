#!/usr/bin/env python3
"""Assistant-axis analysis at one layer.
For each context c: u_c = mean(all activations under c) - mean(under null); report the signed projection of u_c on the
unit axis, and the fraction of ||u_c||^2 that lies along the axis.
For each (c, trait): cos(v_c, v_null) before and after projecting the axis component out of both vectors.
Then the correlation across cells between |axis projection of u_c| and (1 - cos to null)."""
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
    li = _lidx(path, L)
    return torch.stack([torch.nan_to_num(v[li].float()) for v in d.values()])
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
ap = argparse.ArgumentParser(); ap.add_argument("--acts", nargs="+", required=True, help="dirs to search for <ctx>_<trait>_<dir>.pt")
ap.add_argument("--ref", required=True); ap.add_argument("--axis", required=True); ap.add_argument("--layer", type=int, default=22)
ap.add_argument("--contexts", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); L = a.layer; ref = Path(a.ref)
axis = torch.load(a.axis, map_location="cpu", weights_only=True)[L].float(); axis = axis / axis.norm()
def find(c, t, d):
    for base in [Path(x) for x in a.acts] + [ref]:
        p = base / f"{c}_{t}_{d}.pt"
        if p.exists(): return p
    raise FileNotFoundError(f"{c}_{t}_{d}")
null_all, null_vec = {}, {}
for t in TRAITS:
    Pn, Nn = load(find("null", t, "pos"), L), load(find("null", t, "neg"), L)
    null_all[t] = torch.cat([Pn, Nn]).mean(0); null_vec[t] = Pn.mean(0) - Nn.mean(0)
ctx_rows, cell_rows = [], []
for c in a.contexts:
    us, cells = [], []
    for t in TRAITS:
        Pc, Nc = load(find(c, t, "pos"), L), load(find(c, t, "neg"), L)
        u = torch.cat([Pc, Nc]).mean(0) - null_all[t]; us.append(u)
        v = Pc.mean(0) - Nc.mean(0); vn = null_vec[t]
        v_o = v - (v @ axis) * axis; vn_o = vn - (vn @ axis) * axis
        cells.append({"context": c, "trait": t, "cos_to_null": cos(v, vn), "cos_to_null_axis_removed": cos(v_o, vn_o),
                      "axis_frac_of_trait_vec": ((v @ axis) ** 2 / (v @ v)).item(),
                      "u_axis_proj": (u @ axis).item(), "u_norm": u.norm().item(), "u_axis_frac": ((u @ axis) ** 2 / (u @ u)).item()})
    cell_rows += cells; U = torch.stack(us).mean(0)
    ctx_rows.append({"context": c, "u_axis_proj": (U @ axis).item(), "u_norm": U.norm().item(), "u_axis_frac": ((U @ axis) ** 2 / (U @ U)).item(),
                     "mean_cos_to_null": sum(x["cos_to_null"] for x in cells) / 8, "mean_cos_axis_removed": sum(x["cos_to_null_axis_removed"] for x in cells) / 8})
    r = ctx_rows[-1]; print(f"{c:18s} axis_proj={r['u_axis_proj']:+8.1f} |u|={r['u_norm']:7.1f} frac_on_axis={r['u_axis_frac']:.3f}  cos_null={r['mean_cos_to_null']:.3f} -> axis_removed={r['mean_cos_axis_removed']:.3f}")
x = torch.tensor([abs(r["u_axis_proj"]) for r in cell_rows]); y = torch.tensor([1 - r["cos_to_null"] for r in cell_rows])
rho = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
xs = torch.tensor([r["u_norm"] for r in cell_rows]); rho_norm = torch.corrcoef(torch.stack([xs, y]))[0, 1].item()
print(f"cells={len(cell_rows)}  corr(|axis proj|, 1-cos)={rho:+.3f}   corr(|u|, 1-cos)={rho_norm:+.3f}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump({"layer": L, "contexts": ctx_rows, "cells": cell_rows, "corr_axisproj_vs_rotation": rho, "corr_unorm_vs_rotation": rho_norm}, open(out/"axis_projection.json", "w"), indent=1)
with open(out/"axis_cells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cell_rows[0])); w.writeheader(); w.writerows(cell_rows)

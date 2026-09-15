#!/usr/bin/env python3
"""Result C: dispersion of trait directions across evaluation contexts, against three floors.

Inputs: a caa_activations dir with <ctx>_<trait>_<dir>.pt (500 q, keys q{id}) and companion
<ctx>_<trait>_<dir>.variants.pt (keys v{i}_q{id}); plus a dir holding null_/nonsense_ activations
from the same model and code. Everything is computed at --layer from raw activations.

Outputs (JSON + CSV): per (ctx, trait) cosine-to-null; per-trait paraphrase floor; bootstrap floor;
nonsense; matched-pair contrasts (auto_grader vs human_evaluator, simulated_env vs real_deployment,
under_evaluation vs unobserved).
"""
import argparse, json, csv, itertools
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

PAIRS = [("auto_grader", "human_evaluator"), ("simulated_env", "real_deployment"), ("under_evaluation", "unobserved")]
TRAITS = ["assertiveness", "confidence", "deference", "empathy", "honesty", "impulsivity", "risk_taking", "warmth"]

def load_layer(path, layer):
    d = torch.load(path, map_location="cpu", weights_only=True)
    keys = sorted(d.keys(), key=lambda k: (k.split("_")[0] if "_" in k else "", int(k.split("q")[-1])))
    li = _lidx(path, layer)
    X = torch.stack([torch.nan_to_num(d[k][li].float()) for k in keys])
    return keys, X

def vec(P, N): return P.mean(0) - N.mean(0)
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def bootstrap_floor(P, N, n_boot=50, seed=0):
    g = torch.Generator().manual_seed(seed); vs = []
    for _ in range(n_boot):
        ip = torch.randint(len(P), (len(P),), generator=g); iN = torch.randint(len(N), (len(N),), generator=g)
        vs.append(vec(P[ip], N[iN]))
    V = torch.stack(vs); V = V / V.norm(dim=1, keepdim=True)
    C = V @ V.T; iu = torch.triu_indices(len(vs), len(vs), 1)
    return C[iu[0], iu[1]].mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--ref", required=True, help="dir with null_/nonsense_ activations")
    ap.add_argument("--layer", type=int, default=22); ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", nargs="*", default=[c for p in PAIRS for c in p])
    ap.add_argument("--n-boot", type=int, default=50)
    a = ap.parse_args(); acts, ref, L = Path(a.acts), Path(a.ref), a.layer
    rows, floors = [], {}
    V = {}  # (ctx, trait) -> vector
    for t in TRAITS:
        if not (ref / f"null_{t}_pos.pt").exists(): continue  # skip missing trait
        _, Pn = load_layer(ref / f"null_{t}_pos.pt", L); _, Nn = load_layer(ref / f"null_{t}_neg.pt", L)
        vnull = vec(Pn, Nn); V[("null", t)] = vnull
        if (ref / f"nonsense_{t}_pos.pt").exists():
            _, Pz = load_layer(ref / f"nonsense_{t}_pos.pt", L); _, Nz = load_layer(ref / f"nonsense_{t}_neg.pt", L)
            V[("nonsense", t)] = vec(Pz, Nz)
        for c in a.contexts:
            pp, nn = acts / f"{c}_{t}_pos.pt", acts / f"{c}_{t}_neg.pt"
            if not pp.exists(): continue
            _, P = load_layer(pp, L); _, N = load_layer(nn, L)
            v = vec(P, N); V[(c, t)] = v
            row = {"context": c, "trait": t, "cos_to_null": cos(v, vnull), "norm": v.norm().item(),
                   "norm_ratio_to_null": (v.norm() / vnull.norm()).item(),
                   "bootstrap_floor": bootstrap_floor(P, N, a.n_boot)}
            # paraphrase floor: v0 restricted to first 100 q vs v1..v4 on the same 100 q
            vp = acts / f"{c}_{t}_pos.variants.pt"
            if vp.exists():
                kp, Pv = load_layer(vp, L); kn, Nv = load_layer(acts / f"{c}_{t}_neg.variants.pt", L)
                variants = sorted({k.split("_")[0] for k in kp}); qids = sorted({k.split("_")[1] for k in kp})
                k0 = [f"q{i}" for i in range(len(qids))]
                subs = []
                # variant 0 on the same first-K questions
                keysP, _ = load_layer(pp, L); Pfull = load_layer(pp, L)[1]; Nfull = load_layer(nn, L)[1]
                idx = [i for i, k in enumerate(keysP) if k in set(kp_i.split("_")[1] for kp_i in kp)]
                subs.append(vec(Pfull[idx], Nfull[idx]))
                for vi in variants:
                    ip = [i for i, k in enumerate(kp) if k.startswith(vi + "_")]; iN = [i for i, k in enumerate(kn) if k.startswith(vi + "_")]
                    subs.append(vec(Pv[ip], Nv[iN]))
                pair_cos = [cos(x, y) for x, y in itertools.combinations(subs, 2)]
                row["paraphrase_floor"] = sum(pair_cos) / len(pair_cos); row["n_paraphrases"] = len(subs)
                row["paraphrase_cos_to_null_mean"] = sum(cos(x, vnull) for x in subs) / len(subs)
            rows.append(row)
        if ("nonsense", t) in V:
            floors[t] = {"nonsense_cos_to_null": cos(V[("nonsense", t)], vnull)}
            vz = ref / f"nonsense_{t}_pos.variants.pt"
            if vz.exists():
                kz, Pzv = load_layer(vz, L); knz, Nzv = load_layer(ref / f"nonsense_{t}_neg.variants.pt", L)
                kP, Pz0 = load_layer(ref / f"nonsense_{t}_pos.pt", L); kN, Nz0 = load_layer(ref / f"nonsense_{t}_neg.pt", L)
                qset = {k.split("_")[1] for k in kz}
                i0 = [i for i, k in enumerate(kP) if k in qset]
                band = [cos(vec(Pz0[i0], Nz0[i0]), vnull)]
                for vi in sorted({k.split("_")[0] for k in kz}):
                    ip = [i for i, k in enumerate(kz) if k.startswith(vi + "_")]; iN = [i for i, k in enumerate(knz) if k.startswith(vi + "_")]
                    band.append(cos(vec(Pzv[ip], Nzv[iN]), vnull))
                floors[t]["nonsense_band"] = band; floors[t]["nonsense_band_mean"] = sum(band) / len(band); floors[t]["nonsense_band_max"] = max(band); floors[t]["nonsense_band_min"] = min(band)
                for r in rows:
                    if r["trait"] == t:
                        r["below_band_mean"] = r["cos_to_null"] < floors[t]["nonsense_band_mean"]; r["below_band_all"] = r["cos_to_null"] < floors[t]["nonsense_band_min"]
    # matched-pair contrasts
    pairs = []
    for (c1, c2) in PAIRS:
        for t in TRAITS:
            if (c1, t) in V and (c2, t) in V:
                pairs.append({"pair": f"{c1}~{c2}", "trait": t, "cos_between": cos(V[(c1, t)], V[(c2, t)]),
                              "cos_c1_null": cos(V[(c1, t)], V[("null", t)]), "cos_c2_null": cos(V[(c2, t)], V[("null", t)])})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"layer": L, "cells": rows, "nonsense": floors, "pairs": pairs}, open(out / "evalctx_summary.json", "w"), indent=1)
    with open(out / "evalctx_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    with open(out / "evalctx_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]) if pairs else ["pair"]); w.writeheader(); w.writerows(pairs)
    # console summary
    print(f"layer {L}: {len(rows)} cells")
    for t in TRAITS:
        cells = [r for r in rows if r["trait"] == t]
        if not cells: continue
        s = " ".join(f"{r['context'][:6]}={r['cos_to_null']:.3f}" for r in cells)
        nz = floors.get(t, {}).get("nonsense_cos_to_null", float("nan"))
        pf = [r.get("paraphrase_floor") for r in cells if "paraphrase_floor" in r]
        bd = floors.get(t, {}).get("nonsense_band"); bs = f" band=[{min(bd):.3f},{max(bd):.3f}]" if bd else ""
        print(f"{t:13s} {s} | nonsense={nz:.3f}{bs} boot={sum(r['bootstrap_floor'] for r in cells)/len(cells):.3f}"
              + (f" para={sum(pf)/len(pf):.3f}" if pf else ""))
    for p in pairs: print(f"{p['pair']:34s} {p['trait']:13s} between={p['cos_between']:.3f}  c1~null={p['cos_c1_null']:.3f} c2~null={p['cos_c2_null']:.3f}")
if __name__ == "__main__": main()

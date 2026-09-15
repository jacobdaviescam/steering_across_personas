#!/usr/bin/env python3
"""Thin full-layer activation dirs in place: write <stem>.means.pt (all-layer mean, n) and keep per-question
activations only at --keep layers, with a layers.json sidecar. Verifies the thinned file loads before replacing.
Skips dirs that already have layers.json."""
import argparse, json, os, sys, torch
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--keep", required=True)
a = ap.parse_args(); d = Path(a.dir); keep = [int(x) for x in a.keep.split(",")]
if (d / "layers.json").exists(): print("already thinned:", d); sys.exit(0)
files = sorted(p for p in d.glob("*.pt") if not p.name.endswith(".means.pt")); freed = 0
for p in files:
    x = torch.load(p, map_location="cpu", weights_only=True)
    keys = list(x.keys()); stack = torch.stack([x[k] for k in keys])           # (n, L, d) fp16
    L = stack.shape[1]
    if L <= len(keep):  # already small
        continue
    mp = p.with_name(p.stem + ".means.pt")
    if not mp.exists():
        torch.save({"mean": stack.float().mean(0).half(), "n": stack.shape[0]}, mp)
    thin = {k: x[k][keep].clone() for k in keys}
    tmp = p.with_suffix(".tmp"); torch.save(thin, tmp)
    chk = torch.load(tmp, map_location="cpu", weights_only=True); assert len(chk) == len(thin) and chk[keys[0]].shape[0] == len(keep)
    before = p.stat().st_size; os.replace(tmp, p); freed += before - p.stat().st_size
(d / "layers.json").write_text(json.dumps({"layers": keep, "note": "per-question activations kept at these absolute layers; <stem>.means.pt holds all-layer per-cell means"}))
print(f"{d}: {len(files)} files, freed {freed/1e9:.1f} GB")

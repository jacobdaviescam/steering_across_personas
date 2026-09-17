#!/usr/bin/env python3
"""Cross-checkpoint summary for the reward-hacking trajectory.
For each checkpoint dir <out>/<step>/caa_activations_evalctx (with null_/nonsense_ controls and layers.json):
run the per-cell analysis (cos to null, floors, gibberish band) at --layer, then tabulate rotation per context and
per trait against RL step, plus the gate report's behavioural log-odds. Writes trajectory_summary.json/.csv."""
import argparse, json, csv, subprocess, sys
from pathlib import Path
CTX = ["auto_grader","human_evaluator","simulated_env","real_deployment","under_evaluation","unobserved","coding_harness"]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=18)
ap.add_argument("--steps", nargs="+", default=["step-0","step-216","step-496","step-752","step-952"]); ap.add_argument("--analyzer", default="/workspace/iclr2026/analyze_evalctx.py")
a = ap.parse_args(); out = Path(a.out); rows = []
gate = json.load(open(out / "gate_report.json")) if (out / "gate_report.json").exists() else {}
for st in a.steps:
    d = out / st / "caa_activations_evalctx"; ad = out / st / f"analysis_L{a.layer}"
    if not (ad / "evalctx_summary.json").exists():
        subprocess.run([sys.executable, a.analyzer, "--acts", str(d), "--ref", str(d), "--layer", str(a.layer), "--contexts", *CTX, "--out", str(ad), "--n-boot", "50"], check=True)
    s = json.load(open(ad / "evalctx_summary.json")); band = {t: v.get("nonsense_band", [v["nonsense_cos_to_null"]]) for t, v in s["nonsense"].items()}
    for r in s["cells"]:
        b = band[r["trait"]]
        rows.append({"step": int(st.split("-")[1]), "context": r["context"], "trait": r["trait"], "rotation": 1 - r["cos_to_null"],
                     "gibberish_rotation_mean": 1 - sum(b)/len(b), "gibberish_rotation_min": 1 - max(b), "gibberish_rotation_max": 1 - min(b),
                     "bootstrap_rotation": 1 - r["bootstrap_floor"], "paraphrase_rotation": 1 - r["paraphrase_floor"] if "paraphrase_floor" in r else None,
                     "behaviour_logodds_hack": gate.get(st, {}).get("behaviour_logodds_hack"), "mean_cos_to_base": gate.get(st, {}).get("mean_cos_to_base")})
with open(out / f"trajectory_summary_L{a.layer}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(rows, open(out / f"trajectory_summary_L{a.layer}.json", "w"), indent=1)
# console: per step, mean rotation per context (over traits) vs gibberish
steps = sorted({r["step"] for r in rows})
print(f"layer {a.layer}: rotation = 1 - cos to null, mean over 8 traits; [gib] = gibberish band mean")
print("step     " + " ".join(f"{c[:8]:>9s}" for c in CTX) + "   [gib]  behaviour")
for st in steps:
    rs = [r for r in rows if r["step"] == st]; gib = sum(r["gibberish_rotation_mean"] for r in rs)/len(rs); beh = rs[0]["behaviour_logodds_hack"]
    print(f"{st:<8d} " + " ".join(f"{sum(r['rotation'] for r in rs if r['context']==c)/8:9.3f}" for c in CTX) + f"   {gib:.3f}  {beh if beh is None else round(beh,2)}")

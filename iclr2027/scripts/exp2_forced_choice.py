#!/usr/bin/env python3
"""Experiment 2: forced-choice behavioural readout under evaluation contexts.

Reads the per-question log-odds companions written by
`pipeline/2c_caa_activations.py --logits-only` (files `<persona>_<trait>_<dir>.logits.pt`,
dict q{id} -> log p(A) - log p(B) at the position predicting the answer letter) and computes,
per model and per (context, trait) cell:

  * mean log-odds toward the trait-positive option (sign-corrected with a_is_positive),
  * the shift versus the null context (paired over questions, with a 95% CI and Cohen's d),
  * the fraction of questions on which the positive option is preferred,

and, across the 7 evaluation-context x 8 trait cells, the Pearson and Spearman correlation of
the behavioural shift (signed and absolute) with the representational rotation
(cosine to null, `evalctx_cells.csv` from analyze_evalctx.py) at each analysed layer.

The pos and neg files of a cell share the same prefix up to the answer letter, so their
log-odds must agree; the maximum |pos - neg| per model is reported as a consistency check.

Usage:
    python iclr2027/scripts/exp2_forced_choice.py --root /root/work/outputs --out /root/work/analysis/exp2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EVAL_CTX = ["auto_grader", "human_evaluator", "simulated_env", "real_deployment",
            "under_evaluation", "unobserved", "coding_harness"]
CONTROLS = ["null", "nonsense"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty", "confidence",
          "deference", "warmth", "impulsivity"]
# Representational analyses on the results branch, per model short name.
CELLS = {
    "gemma-2-27b-it": {"L22": "iclr2027/analysis/evalctx_L22_v3ref/evalctx_cells.csv"},
    "Llama-3.1-8B-Instruct": {"L15": "iclr2027/analysis/Llama-3.1-8B-Instruct_evalctx_L15/evalctx_cells.csv",
                              "L20": "iclr2027/analysis/Llama-3.1-8B-Instruct_evalctx_L20/evalctx_cells.csv"},
    "Qwen3-8B": {"L18": "iclr2027/analysis/Qwen3-8B_evalctx_L18/evalctx_cells.csv",
                 "L24": "iclr2027/analysis/Qwen3-8B_evalctx_L24/evalctx_cells.csv"},
}


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal markdown table (avoids the tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def a_is_positive(trait: str) -> dict[str, bool]:
    d = json.loads((ROOT / "data/prompts/caa" / f"{trait}.json").read_text())
    return {f"q{q['id']}": bool(q["a_is_positive"]) for q in d["questions"]}


def load_cell(d: Path, ctx: str, trait: str, sign: dict[str, bool]) -> tuple[pd.Series, float]:
    """Signed log-odds toward the positive option per question, plus max |pos - neg| discrepancy."""
    pos = torch.load(d / f"{ctx}_{trait}_pos.logits.pt")
    negf = d / f"{ctx}_{trait}_neg.logits.pt"
    disc = float("nan")
    if negf.exists():
        neg = torch.load(negf)
        disc = max(abs(pos[k] - neg[k]) for k in pos if k in neg)
    s = pd.Series({k: (v if sign[k] else -v) for k, v in pos.items()}).sort_index()
    return s, disc


def analyse_model(model_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    short = model_dir.parent.name
    lines = [f"## {short}", ""]
    rows, disc_max = [], 0.0
    per_q: dict[tuple[str, str], pd.Series] = {}
    for trait in TRAITS:
        sign = a_is_positive(trait)
        for ctx in EVAL_CTX + CONTROLS:
            f = model_dir / f"{ctx}_{trait}_pos.logits.pt"
            if not f.exists():
                continue
            s, disc = load_cell(model_dir, ctx, trait, sign)
            per_q[(ctx, trait)] = s
            if disc == disc:
                disc_max = max(disc_max, disc)
    for (ctx, trait), s in per_q.items():
        null = per_q.get(("null", trait))
        row = dict(model=short, context=ctx, trait=trait, n=len(s),
                   mean_logodds=s.mean(), sd_logodds=s.std(ddof=1), frac_positive=(s > 0).mean())
        if null is not None and ctx != "null":
            diff = (s - null.reindex(s.index)).dropna()
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            row.update(null_mean=null.mean(), shift_vs_null=diff.mean(), shift_se=se,
                       shift_ci_lo=diff.mean() - 1.96 * se, shift_ci_hi=diff.mean() + 1.96 * se,
                       cohen_d=diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan,
                       frac_shift=(s > 0).mean() - (null > 0).mean(),
                       p_paired_t=stats.ttest_rel(s, null.reindex(s.index)).pvalue)
        rows.append(row)
    cells = pd.DataFrame(rows)
    pd.concat([s.rename("logodds_toward_positive").rename_axis("qid").reset_index().assign(context=c, trait=t)
               for (c, t), s in per_q.items()])[["context", "trait", "qid", "logodds_toward_positive"]].to_csv(
        out / f"exp2_per_question_{short}.csv", index=False, float_format="%.4f")
    lines.append(f"Max |pos - neg| log-odds discrepancy (same prefix, must be ~0): {disc_max:.4f}")
    lines.append("")

    # Shift matrix: context x trait, mean log-odds toward the positive option minus null.
    ev = cells[cells.context != "null"]
    mat = ev.pivot(index="context", columns="trait", values="shift_vs_null").reindex(EVAL_CTX + ["nonsense"])[TRAITS]
    mat["mean_abs"] = mat[TRAITS].abs().mean(axis=1)
    lines.append("Shift vs null in mean log-odds toward the trait-positive option (rows: context; last column: mean |shift|):")
    lines.append("")
    lines.append(md(mat))
    lines.append("")
    nullrow = cells[cells.context == "null"].set_index("trait").reindex(TRAITS)
    lines.append("Null context: mean log-odds toward positive / fraction positive per trait:")
    lines.append("")
    lines.append(md(pd.DataFrame({"null_mean_logodds": nullrow.mean_logodds, "null_frac_positive": nullrow.frac_positive}).T))
    lines.append("")
    sig = ev[(ev.context != "nonsense") & (ev.p_paired_t < 0.05 / 56)]
    lines.append(f"Cells with a shift significant at Bonferroni 0.05/56 (paired t over questions): {len(sig)}/56; "
                 f"largest |shift| cells: " + "; ".join(
                     f"{r.context}/{r.trait} {r.shift_vs_null:+.2f} [{r.shift_ci_lo:+.2f}, {r.shift_ci_hi:+.2f}]"
                     for r in ev[ev.context != "nonsense"].reindex(ev[ev.context != "nonsense"]["shift_vs_null"].abs().sort_values(ascending=False).index).head(6).itertuples()))
    lines.append("")

    # Correlation with rotation across the 56 evaluation-context cells.
    corr_rows = []
    for layer, rel in CELLS.get(short, {}).items():
        p = ROOT / rel
        if not p.exists():
            lines.append(f"(no rotation file for {layer}: {rel})")
            continue
        rc = pd.read_csv(p)[["context", "trait", "cos_to_null"]]
        m = ev[ev.context.isin(EVAL_CTX)].merge(rc, on=["context", "trait"], how="inner")
        m["rotation"] = 1 - m.cos_to_null
        m["abs_shift"] = m["shift_vs_null"].abs()
        m["abs_d"] = m.cohen_d.abs()
        cells.loc[cells.context.isin(EVAL_CTX), f"cos_to_null_{layer}"] = cells[cells.context.isin(EVAL_CTX)].merge(
            rc, on=["context", "trait"], how="left").cos_to_null.values
        for yname in ["abs_shift", "shift_vs_null", "abs_d"]:
            pr = stats.pearsonr(m.rotation, m[yname]); sp = stats.spearmanr(m.rotation, m[yname])
            corr_rows.append(dict(model=short, layer=layer, n_cells=len(m), behaviour=yname, against="rotation (1 - cos to null)",
                                  pearson_r=pr[0], pearson_p=pr[1], spearman_rho=sp[0], spearman_p=sp[1]))
        # Context-level: mean rotation vs mean |shift| over traits (7 points).
        g = m.groupby("context")[["rotation", "abs_shift"]].mean()
        sp = stats.spearmanr(g.rotation, g.abs_shift)
        corr_rows.append(dict(model=short, layer=layer, n_cells=len(g), behaviour="abs_shift (context means)",
                              against="rotation (context means)", pearson_r=stats.pearsonr(g.rotation, g.abs_shift)[0],
                              pearson_p=stats.pearsonr(g.rotation, g.abs_shift)[1], spearman_rho=sp[0], spearman_p=sp[1]))
        lines.append(f"Context means at {layer} (rotation = 1 - cos to null; |shift| in log-odds):")
        lines.append("")
        lines.append(md(g.assign(mean_shift=m.groupby("context")["shift_vs_null"].mean()).reindex(EVAL_CTX)))
        lines.append("")
    corr = pd.DataFrame(corr_rows)
    if len(corr):
        lines.append("Correlation across the 56 evaluation-context cells (behaviour vs rotation):")
        lines.append("")
        lines.append(md(corr.drop(columns=["model"]), index=False))
        lines.append("")
    return cells, corr, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="outputs root containing <Model>/exp2_logits/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subdir", default="exp2_logits")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    all_cells, all_corr, doc = [], [], ["# Experiment 2: forced-choice behavioural readout", ""]
    for d in sorted(Path(a.root).glob(f"*/{a.subdir}")):
        if not any(d.glob("*_pos.logits.pt")):
            continue
        cells, corr, lines = analyse_model(d, out)
        all_cells.append(cells); all_corr.append(corr); doc += lines
    pd.concat(all_cells).to_csv(out / "exp2_cells.csv", index=False)
    if any(len(c) for c in all_corr):
        pd.concat(all_corr).to_csv(out / "exp2_corr.csv", index=False)
    (out / "exp2_summary.md").write_text("\n".join(doc))
    print("\n".join(doc))


if __name__ == "__main__":
    main()

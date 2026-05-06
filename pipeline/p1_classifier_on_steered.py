#!/usr/bin/env python3
"""P1: Run the X1 SBERT classifier on cross-steered generations.

For each (target persona, source persona, trait) cell in the existing
steering eval (steered_responses_alpha2/.jsonl), encode the response with
the SBERT model from X1 and apply the trained linear head. We record:

  P(target_persona | response)   — does the response still 'sound like' target?
  P(source_persona | response)   — does cross-steering leak source identity?
  P(any other persona | response)— diffuse leakage

Headline plot for the persona-residue story:
  For each (source != target) cross-steering condition, plot
    Δ P(source) = P(source | cross-steered) - P(source | baseline_target)
  vs
    Δ P(target) = P(target | cross-steered) - P(target | baseline_target)

  Self-steering should produce big positive Δ P(target) and ≈ 0 Δ P(source).
  Cross-steering should produce smaller Δ P(target) and *positive* Δ P(source)
  — the source persona's vector leaves a recognisable fingerprint in the
  generated text, even though the model is system-prompted as the target.

We also dump a per-cell table for use in the paper.

Usage:
    python pipeline/p1_classifier_on_steered.py \
        --steered-dir   outputs/gemma-2-27b-it/steered_responses_alpha2 \
        --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
        --output-dir     outputs/gemma-2-27b-it/v2/persona_residue
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PERSONAS = [
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
]
TRAITS = [
    "assertiveness", "empathy", "risk_taking", "honesty",
    "confidence", "deference", "warmth", "impulsivity",
]
TRAIT_LABEL = {t: t.replace("_", " ").title() for t in TRAITS}
PERSONA_LABEL = {p: p.replace("_", " ").title() for p in PERSONAS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steered-dir", required=True)
    p.add_argument("--classifier-dir", required=True,
                   help="Output dir of pipeline/x1_context_classifier.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def load_classifier(classifier_dir: Path):
    head_data = torch.load(classifier_dir / "head.pt",
                           map_location="cpu", weights_only=False)
    contexts = head_data["contexts"]
    sbert_model = head_data.get("sbert_model", "all-mpnet-base-v2")
    head = torch.nn.Linear(head_data["in_dim"], head_data["n_classes"])
    state = {k.removeprefix("fc."): v for k, v in head_data["state_dict"].items()}
    head.load_state_dict(state)
    head.eval()
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_model)
    return head, sbert, contexts


def truncate(text: str, max_tokens: int, tokenizer) -> str:
    toks = tokenizer.tokenize(text)[:max_tokens]
    return tokenizer.convert_tokens_to_string(toks)


def parse_filename(stem: str) -> tuple[str | None, str, str] | None:
    """Returns (source_persona_or_None, target_persona, trait) or None."""
    if stem.startswith("baseline_"):
        rest = stem[len("baseline_"):]
        for t in TRAITS:
            if rest.endswith("_" + t):
                tgt = rest[:-(len(t) + 1)]
                if tgt in PERSONAS:
                    return (None, tgt, t)
        return None
    for t in TRAITS:
        if stem.endswith("_" + t):
            base = stem[:-(len(t) + 1)]
            for src in PERSONAS:
                if base.startswith(src + "_"):
                    tgt = base[len(src) + 1:]
                    if tgt in PERSONAS:
                        return (src, tgt, t)
    return None


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steered_dir = Path(args.steered_dir)

    head, sbert, contexts = load_classifier(Path(args.classifier_dir))
    tokenizer = sbert.tokenizer
    ctx_idx = {c: i for i, c in enumerate(contexts)}

    # ---- collect responses ----
    items = []  # (src, tgt, trait, text)
    for f in sorted(steered_dir.glob("*.jsonl")):
        parsed = parse_filename(f.stem)
        if not parsed:
            continue
        src, tgt, trait = parsed
        with open(f) as fh:
            for line in fh:
                e = json.loads(line)
                resp = e.get("response", "")
                if not resp:
                    continue
                items.append((src, tgt, trait, truncate(resp, args.max_tokens, tokenizer)))
    if not items:
        print("No items found.")
        return

    print(f"Encoding {len(items)} responses ...")
    texts = [t for *_, t in items]
    X = sbert.encode(texts, batch_size=args.batch_size,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    with torch.no_grad():
        logits = head(torch.tensor(X)).numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    # ---- aggregate per (src, tgt, trait) ----
    per_cell = defaultdict(list)
    for (src, tgt, trait, _), prow in zip(items, probs):
        per_cell[(src, tgt, trait)].append(prow)

    cell_summary = {}
    for (src, tgt, trait), rows in per_cell.items():
        P = np.stack(rows).mean(axis=0)
        cell_summary[f"{src or 'baseline'}->{tgt}|{trait}"] = {
            "src": src, "tgt": tgt, "trait": trait,
            "n": len(rows),
            "P_target": float(P[ctx_idx[tgt]]) if tgt in ctx_idx else None,
            "P_source": float(P[ctx_idx[src]]) if (src and src in ctx_idx) else None,
            "P_full": {c: float(P[ctx_idx[c]]) for c in contexts if c in ctx_idx},
        }
    (out_dir / "cell_summary.json").write_text(json.dumps(cell_summary, indent=2))
    print(f"Saved cell_summary.json with {len(cell_summary)} cells")

    # ---- residue scatter ----
    rows_residue = []
    for trait in TRAITS:
        for tgt in PERSONAS:
            base_key = f"baseline->{tgt}|{trait}"
            if base_key not in cell_summary:
                continue
            base = cell_summary[base_key]
            for src in PERSONAS:
                if src == tgt:
                    continue
                cross_key = f"{src}->{tgt}|{trait}"
                self_key = f"{tgt}->{tgt}|{trait}"
                if cross_key not in cell_summary or self_key not in cell_summary:
                    continue
                cross = cell_summary[cross_key]
                self_ = cell_summary[self_key]
                rows_residue.append({
                    "trait": trait, "src": src, "tgt": tgt,
                    "delta_P_source_under_cross": cross["P_full"][src] - base["P_full"][src],
                    "delta_P_target_under_cross": cross["P_target"] - base["P_target"],
                    "delta_P_target_under_self":  self_["P_target"] - base["P_target"],
                    "delta_P_source_under_self":  self_["P_full"][src] - base["P_full"][src],
                    "P_target_baseline": base["P_target"],
                })

    if rows_residue:
        (out_dir / "residue_pairs.json").write_text(json.dumps(rows_residue, indent=2))

        # plot 1: scatter Δ P(source under cross) vs Δ P(source under self)
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("tab10")
        color = {t: cmap(i % 10) for i, t in enumerate(TRAITS)}
        xs, ys = [], []
        for r in rows_residue:
            x = r["delta_P_source_under_self"]
            y = r["delta_P_source_under_cross"]
            xs.append(x); ys.append(y)
            ax.scatter(x, y, color=color[r["trait"]], s=24, alpha=0.6,
                       edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.plot([min(xs+ys), max(xs+ys)], [min(xs+ys), max(xs+ys)],
                "k--", lw=0.5, alpha=0.5)
        ax.set_xlabel(r"$\Delta$ P(source | self-steered) — should be ~0")
        ax.set_ylabel(r"$\Delta$ P(source | cross-steered)")
        ax.set_title("Persona residue under cross-steering: does steering with\n"
                     "source's vector make the response read as source?")
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="", color=color[t],
                          markersize=6, label=TRAIT_LABEL[t]) for t in TRAITS]
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  frameon=False, ncol=2)
        ax.grid(alpha=0.25, ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / "p1_residue_scatter.pdf")
        fig.savefig(out_dir / "p1_residue_scatter.png", dpi=180)
        plt.close(fig)

        # plot 2: per-trait mean residue (Δ P(source) under cross)
        fig, ax = plt.subplots(figsize=(8, 4))
        per_trait = defaultdict(list)
        for r in rows_residue:
            per_trait[r["trait"]].append(r["delta_P_source_under_cross"])
        traits_sorted = sorted(per_trait, key=lambda t: np.mean(per_trait[t]))
        means = [np.mean(per_trait[t]) for t in traits_sorted]
        stds  = [np.std(per_trait[t])  / np.sqrt(len(per_trait[t])) for t in traits_sorted]
        ax.bar(np.arange(len(traits_sorted)), means, yerr=stds, capsize=3,
               color=[color[t] for t in traits_sorted], alpha=0.85,
               edgecolor="white")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(np.arange(len(traits_sorted)))
        ax.set_xticklabels([TRAIT_LABEL[t] for t in traits_sorted], rotation=20, ha="right")
        ax.set_ylabel(r"mean $\Delta$ P(source) under cross-steering")
        ax.set_title("Persona residue per trait (averaged over 90 src≠tgt pairs)")
        ax.grid(axis="y", alpha=0.25, ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / "p1_residue_per_trait.pdf")
        fig.savefig(out_dir / "p1_residue_per_trait.png", dpi=180)
        plt.close(fig)

        # quick console summary
        for t, vals in per_trait.items():
            print(f"  {t:14s}  mean ΔP(source|cross) = {np.mean(vals):+.3f}  "
                  f"se = {np.std(vals)/np.sqrt(len(vals)):.3f}  n = {len(vals)}")


if __name__ == "__main__":
    main()

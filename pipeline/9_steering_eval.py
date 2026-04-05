#!/usr/bin/env python3
"""Score steered responses with Claude judge, build behavioural transfer matrices,
correlate with geometric transfer, and generate figures.

Usage:
    python pipeline/9_steering_eval.py \
        --steered-dir outputs/gemma-2-27b-it/steered_responses \
        --analysis-dir outputs/gemma-2-27b-it/analysis \
        --output-dir outputs/gemma-2-27b-it/eval

    # Score only (skip figures):
    python pipeline/9_steering_eval.py \
        --steered-dir outputs/gemma-2-27b-it/steered_responses \
        --output-dir outputs/gemma-2-27b-it/eval \
        --score-only

    # Figures only (from existing scores):
    python pipeline/9_steering_eval.py \
        --steered-dir outputs/gemma-2-27b-it/steered_responses \
        --analysis-dir outputs/gemma-2-27b-it/analysis \
        --output-dir outputs/gemma-2-27b-it/eval \
        --figures-only
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_steering.config import Trait, PERSONA_SLUGS, TARGET_LAYER
from persona_steering.evaluation import LLMJudge
from persona_steering.utils import log, save_json, load_json
from persona_steering.wandb_utils import log_metrics as wb_log_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score steered responses and analyse steering transfer"
    )
    parser.add_argument(
        "--steered-dir", type=str, required=True,
        help="Directory with steered response JSONL files from step 8",
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=None,
        help="Directory with geometric analysis from step 4 (for correlation)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for scores and figures",
    )
    parser.add_argument(
        "--judge-model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model for LLM judge",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Score responses only (no figures)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Generate figures only (from existing scores)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_steered_responses(steered_dir: Path, output_dir: Path, judge_model: str) -> dict:
    """Score all steered response files with LLM judge.

    Returns dict: {trait: {"source->target": {"mean_score": ..., "scores": [...], "baseline": ...}}}
    """
    judge = LLMJudge(model=judge_model)
    scores_file = output_dir / "steering_transfer_scores.json"

    # Load existing scores for resume
    if scores_file.exists():
        all_scores = load_json(scores_file)
        log.info("Loaded existing scores from %s", scores_file)
    else:
        all_scores = {}

    # Discover all JSONL files
    jsonl_files = sorted(steered_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No JSONL files found in %s", steered_dir)
        return all_scores

    trait_values = {t.value for t in Trait}
    files_scored = 0
    total_to_score = len(jsonl_files)

    for jsonl_file in jsonl_files:
        stem = jsonl_file.stem  # e.g. "farmer_therapist_assertiveness" or "baseline_farmer_assertiveness"

        # Parse filename to identify source, target, trait
        source, target, trait_name = _parse_steered_filename(stem, trait_values)
        if trait_name is None:
            log.warning("Cannot parse filename: %s, skipping", stem)
            continue

        # Build score key
        if source is None:
            score_key = f"baseline_{target}"
        else:
            score_key = f"{source}->{target}"

        # Check if already scored
        trait_scores = all_scores.setdefault(trait_name, {})
        if score_key in trait_scores:
            log.info("Skipping %s/%s (already scored)", trait_name, score_key)
            continue

        # Load responses
        responses = []
        with open(jsonl_file) as f:
            for line in f:
                responses.append(json.loads(line))

        if not responses:
            continue

        # Score each response
        trait_enum = Trait(trait_name)
        scores = []
        for entry in responses:
            result = judge.score_trait(entry["response"], trait_enum)
            scores.append(result.score)

        trait_scores[score_key] = {
            "mean_score": float(np.mean(scores)),
            "scores": scores,
            "n": len(scores),
        }

        log.info("Scored %s/%s: mean=%.3f (n=%d)",
                 trait_name, score_key, np.mean(scores), len(scores))
        files_scored += 1
        wb_log_metrics({
            "scoring/files_done": files_scored,
            "scoring/files_total": total_to_score,
            f"scoring/{trait_name}/{score_key}": float(np.mean(scores)),
        })

        # Save incrementally for resume
        save_json(all_scores, scores_file)

    log.info("All scoring complete. Saved to %s", scores_file)
    return all_scores


def _parse_steered_filename(
    stem: str, trait_values: set[str]
) -> tuple[str | None, str | None, str | None]:
    """Parse a steered response filename into (source, target, trait).

    Filenames:
        baseline_{target}_{trait}      -> (None, target, trait)
        {source}_{target}_{trait}      -> (source, target, trait)

    Multi-word slugs (e.g. "con_artist") are handled by matching known trait
    suffixes, then splitting the remainder.
    """
    # Match trait suffix
    trait_name = None
    prefix = None
    for tv in trait_values:
        if stem.endswith(f"_{tv}"):
            trait_name = tv
            prefix = stem[:-(len(tv) + 1)]
            break

    if trait_name is None or prefix is None:
        return None, None, None

    # Baseline case
    if prefix.startswith("baseline_"):
        target = prefix[len("baseline_"):]
        return None, target, trait_name

    # Steered case: need to split source and target
    # Both can be multi-word slugs (e.g. con_artist, drill_sergeant)
    # Use known persona slugs to disambiguate
    for slug in sorted(PERSONA_SLUGS, key=len, reverse=True):
        if prefix.startswith(slug + "_"):
            source = slug
            target = prefix[len(slug) + 1:]
            return source, target, trait_name

    # Fallback: unknown source slug, shouldn't happen with canonical personas
    return None, None, None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_behavioural_transfer(
    all_scores: dict, trait_name: str, personas: list[str]
) -> tuple[np.ndarray, dict[str, float]]:
    """Build 10x10 behavioural transfer matrix for a single trait.

    Entry (i, j) = mean trait score when source_i's vector is applied to target_j.

    Returns:
        (matrix, baselines) where baselines maps target -> baseline score.
    """
    n = len(personas)
    matrix = np.full((n, n), np.nan)
    baselines = {}

    trait_scores = all_scores.get(trait_name, {})

    for j, target in enumerate(personas):
        # Baseline
        bl_key = f"baseline_{target}"
        if bl_key in trait_scores:
            baselines[target] = trait_scores[bl_key]["mean_score"]

        for i, source in enumerate(personas):
            key = f"{source}->{target}"
            if key in trait_scores:
                matrix[i, j] = trait_scores[key]["mean_score"]

    return matrix, baselines


def correlate_geometric_behavioural(
    geometric_matrix: np.ndarray,
    behavioural_matrix: np.ndarray,
) -> dict:
    """Correlate geometric transfer (cosine sim) with behavioural transfer (trait scores).

    Uses off-diagonal entries only (cross-persona transfer).
    """
    n = geometric_matrix.shape[0]
    geo_vals = []
    beh_vals = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g = geometric_matrix[i, j]
            b = behavioural_matrix[i, j]
            if not (np.isnan(g) or np.isnan(b)):
                geo_vals.append(g)
                beh_vals.append(b)

    if len(geo_vals) < 3:
        return {"pearson_r": None, "n_pairs": len(geo_vals)}

    from scipy import stats
    r, p = stats.pearsonr(geo_vals, beh_vals)
    rho, p_rho = stats.spearmanr(geo_vals, beh_vals)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "n_pairs": len(geo_vals),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(
    all_scores: dict,
    personas: list[str],
    output_dir: Path,
    analysis_dir: Path | None = None,
) -> None:
    """Generate all steering transfer figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    traits_with_data = [t for t in all_scores if all_scores[t]]
    if not traits_with_data:
        log.warning("No scored data available for figures.")
        return

    # --- Figure 1: Behavioural transfer heatmaps (per trait) ---
    for trait_name in traits_with_data:
        matrix, baselines = build_behavioural_transfer(all_scores, trait_name, personas)

        if np.all(np.isnan(matrix)):
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(personas)))
        ax.set_xticklabels(personas, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas, fontsize=8)
        ax.set_xlabel("Target persona")
        ax.set_ylabel("Source persona")
        ax.set_title(f"Behavioural transfer: {trait_name}")
        plt.colorbar(im, ax=ax, label="Trait score (0-1)")

        # Annotate cells
        for i in range(len(personas)):
            for j in range(len(personas)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if 0.3 < val < 0.7 else "white")

        plt.tight_layout()
        fig.savefig(figures_dir / f"behavioural_transfer_{trait_name}.png", dpi=150)
        plt.close(fig)
        log.info("Saved behavioural transfer heatmap for %s", trait_name)

    # --- Figure 1b: Average behavioural transfer heatmap ---
    all_matrices = []
    for trait_name in traits_with_data:
        m, _ = build_behavioural_transfer(all_scores, trait_name, personas)
        if not np.all(np.isnan(m)):
            all_matrices.append(m)

    if all_matrices:
        avg_matrix = np.nanmean(np.stack(all_matrices), axis=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(personas)))
        ax.set_xticklabels(personas, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas, fontsize=8)
        ax.set_xlabel("Target persona")
        ax.set_ylabel("Source persona")
        ax.set_title("Behavioural transfer (averaged across traits)")
        plt.colorbar(im, ax=ax, label="Mean trait score")

        for i in range(len(personas)):
            for j in range(len(personas)):
                val = avg_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if 0.3 < val < 0.7 else "white")

        plt.tight_layout()
        fig.savefig(figures_dir / "behavioural_transfer_avg.png", dpi=150)
        plt.close(fig)
        log.info("Saved average behavioural transfer heatmap")

    # --- Figure 2: Geometric vs behavioural transfer scatter ---
    if analysis_dir is not None:
        correlation_results = {}
        geo_all = []
        beh_all = []

        for trait_name in traits_with_data:
            geo_path = analysis_dir / f"transfer_{trait_name}.npy"
            if not geo_path.exists():
                log.warning("No geometric transfer matrix for %s at %s", trait_name, geo_path)
                continue

            geo_matrix = np.load(geo_path)

            # Need to align persona order: geometric matrix uses analysis-time order
            meta_path = analysis_dir / "transfer_meta.json"
            if meta_path.exists():
                meta = load_json(meta_path)
                geo_personas = meta.get("personas", personas)
            else:
                geo_personas = personas

            beh_matrix, _ = build_behavioural_transfer(all_scores, trait_name, geo_personas)

            corr = correlate_geometric_behavioural(geo_matrix, beh_matrix)
            correlation_results[trait_name] = corr
            log.info("  %s: Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (n=%d)",
                     trait_name,
                     corr.get("pearson_r", 0) or 0,
                     corr.get("pearson_p", 1) or 1,
                     corr.get("spearman_rho", 0) or 0,
                     corr["n_pairs"])

            # Collect scatter points
            n = geo_matrix.shape[0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    g = geo_matrix[i, j]
                    b = beh_matrix[i, j]
                    if not (np.isnan(g) or np.isnan(b)):
                        geo_all.append(g)
                        beh_all.append(b)

        save_json(correlation_results, output_dir / "geometric_vs_behavioural.json")

        if geo_all:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(geo_all, beh_all, alpha=0.4, s=20)
            ax.set_xlabel("Geometric similarity (cosine)")
            ax.set_ylabel("Behavioural transfer (trait score)")
            ax.set_title("Geometric vs behavioural transfer")

            # Add trend line
            if len(geo_all) > 2:
                from scipy import stats
                slope, intercept, r, p, se = stats.linregress(geo_all, beh_all)
                xs = np.linspace(min(geo_all), max(geo_all), 100)
                ax.plot(xs, slope * xs + intercept, "r--", alpha=0.7,
                        label=f"r={r:.3f}, p={p:.3f}")
                ax.legend()

            plt.tight_layout()
            fig.savefig(figures_dir / "geometric_vs_behavioural.png", dpi=150)
            plt.close(fig)
            log.info("Saved geometric vs behavioural scatter")

    # --- Figure 3: Self vs cross steering bar chart ---
    self_scores_all = []
    cross_scores_all = []
    per_trait_self_cross = {}

    for trait_name in traits_with_data:
        trait_data = all_scores[trait_name]
        self_scores = []
        cross_scores = []

        for key, val in trait_data.items():
            if "->" not in key:
                continue
            source, target = key.split("->")
            score = val["mean_score"]
            if source == target:
                self_scores.append(score)
            else:
                cross_scores.append(score)

        if self_scores and cross_scores:
            per_trait_self_cross[trait_name] = {
                "self_mean": float(np.mean(self_scores)),
                "cross_mean": float(np.mean(cross_scores)),
                "self_std": float(np.std(self_scores)),
                "cross_std": float(np.std(cross_scores)),
            }
            self_scores_all.extend(self_scores)
            cross_scores_all.extend(cross_scores)

    if per_trait_self_cross:
        save_json(per_trait_self_cross, output_dir / "self_vs_cross.json")

        trait_names = list(per_trait_self_cross.keys())
        x = np.arange(len(trait_names))
        self_means = [per_trait_self_cross[t]["self_mean"] for t in trait_names]
        cross_means = [per_trait_self_cross[t]["cross_mean"] for t in trait_names]
        self_stds = [per_trait_self_cross[t]["self_std"] for t in trait_names]
        cross_stds = [per_trait_self_cross[t]["cross_std"] for t in trait_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        ax.bar(x - width / 2, self_means, width, yerr=self_stds, label="Self-steering",
               capsize=3, color="#4CAF50", alpha=0.8)
        ax.bar(x + width / 2, cross_means, width, yerr=cross_stds, label="Cross-steering",
               capsize=3, color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(trait_names, rotation=45, ha="right")
        ax.set_ylabel("Mean trait score")
        ax.set_title("Self-steering vs cross-persona steering")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(figures_dir / "self_vs_cross.png", dpi=150)
        plt.close(fig)
        log.info("Saved self vs cross steering chart")

    # --- Figure 4: Baseline vs steered per target persona ---
    for trait_name in traits_with_data:
        trait_data = all_scores[trait_name]
        matrix, baselines = build_behavioural_transfer(all_scores, trait_name, personas)

        targets_with_baseline = [p for p in personas if p in baselines]
        if not targets_with_baseline:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(targets_with_baseline))

        # Baseline bars
        bl_vals = [baselines[p] for p in targets_with_baseline]
        ax.bar(x - 0.3, bl_vals, 0.2, label="Baseline", color="#9E9E9E", alpha=0.8)

        # Self-steered bars
        self_vals = []
        for j, target in enumerate(targets_with_baseline):
            key = f"{target}->{target}"
            if key in trait_data:
                self_vals.append(trait_data[key]["mean_score"])
            else:
                self_vals.append(np.nan)
        ax.bar(x - 0.1, self_vals, 0.2, label="Self-steered", color="#4CAF50", alpha=0.8)

        # Mean cross-steered bars
        cross_vals = []
        for j, target in enumerate(targets_with_baseline):
            cross = []
            for source in personas:
                if source == target:
                    continue
                key = f"{source}->{target}"
                if key in trait_data:
                    cross.append(trait_data[key]["mean_score"])
            cross_vals.append(float(np.mean(cross)) if cross else np.nan)
        ax.bar(x + 0.1, cross_vals, 0.2, label="Cross-steered (mean)", color="#2196F3", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(targets_with_baseline, rotation=45, ha="right")
        ax.set_ylabel("Trait score")
        ax.set_title(f"Baseline vs steered: {trait_name}")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(figures_dir / f"baseline_vs_steered_{trait_name}.png", dpi=150)
        plt.close(fig)
        log.info("Saved baseline vs steered chart for %s", trait_name)

    log.info("All figures saved to %s", figures_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from persona_steering.wandb_utils import (
        init_run, finish_run, log_metrics, log_images, log_artifact, ensure_dir,
    )

    args = parse_args()

    steered_dir = Path(args.steered_dir)
    short = steered_dir.parent.name
    steered_dir = ensure_dir(f"{short}-steered-responses", steered_dir, "*.jsonl")
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else None
    if analysis_dir:
        analysis_dir = ensure_dir(f"{short}-analysis", analysis_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine persona order from available files
    personas = _discover_personas(steered_dir)
    if not personas:
        personas = PERSONA_SLUGS
    log.info("Personas: %s", personas)

    # W&B tracking (init early for live progress)
    init_run("step9_steering_eval", short, config=vars(args))

    if not args.figures_only:
        # Score
        all_scores = score_steered_responses(steered_dir, output_dir, args.judge_model)
    else:
        scores_file = output_dir / "steering_transfer_scores.json"
        if not scores_file.exists():
            log.error("No scores file found at %s", scores_file)
            return
        all_scores = load_json(scores_file)

    if not args.score_only:
        # Figures
        generate_figures(all_scores, personas, output_dir, analysis_dir)

    log.info("Done. Results in %s", output_dir)

    # Log final W&B metrics
    geo_beh_path = output_dir / "geometric_vs_behavioural.json"
    if geo_beh_path.exists():
        geo_beh = load_json(geo_beh_path)
        wb_metrics = {}
        for trait_name, data in geo_beh.items():
            wb_metrics[f"steering/{trait_name}/pearson_r"] = data["pearson_r"]
            wb_metrics[f"steering/{trait_name}/pearson_p"] = data["pearson_p"]
        log_metrics(wb_metrics)
    # Log self vs cross metrics
    svc_path = output_dir / "self_vs_cross.json"
    if svc_path.exists():
        svc = load_json(svc_path)
        wb_metrics = {}
        for trait_name, data in svc.items():
            wb_metrics[f"self_vs_cross/{trait_name}/self_mean"] = data["self_mean"]
            wb_metrics[f"self_vs_cross/{trait_name}/cross_mean"] = data["cross_mean"]
        log_metrics(wb_metrics)
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        log_images(figures_dir, prefix="steering")
    log_artifact(f"{short}-steering-eval", "evaluation", output_dir)
    finish_run()


def _discover_personas(steered_dir: Path) -> list[str]:
    """Discover persona slugs from baseline filenames."""
    trait_values = {t.value for t in Trait}
    personas = set()

    for f in steered_dir.glob("baseline_*.jsonl"):
        stem = f.stem
        for tv in trait_values:
            if stem.endswith(f"_{tv}"):
                slug = stem[len("baseline_"):-(len(tv) + 1)]
                personas.add(slug)
                break

    return sorted(personas)


if __name__ == "__main__":
    main()

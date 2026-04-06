#!/usr/bin/env python3
"""Analyse oracle results: compute trait/persona classification accuracy
and compare instruction-variant vs CAA vectors.

Usage:
    python pipeline/11_oracle_analysis.py
    python pipeline/11_oracle_analysis.py --v2-dir outputs/gemma-2-27b-it/oracle_v2 --caa-dir outputs/gemma-2-27b-it/oracle_caa
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from persona_steering.config import PERSONA_SLUGS, Trait, OUTPUTS_DIR
from persona_steering.utils import log

# Canonical labels for matching
TRAITS = [t.value for t in Trait]
TRAIT_ALIASES = {
    "assertiveness": ["assertive", "assertiveness", "ambition", "ambitious"],
    "empathy": ["empathy", "empathetic", "compassion", "compassionate"],
    "risk_taking": ["risk-taking", "risk taking", "risk", "boldness", "bold", "adventurous"],
    "honesty": ["honesty", "honest", "truthful", "integrity", "stubbornness"],
    "confidence": ["confidence", "confident", "self-assured", "discipline", "disciplined"],
    "deference": ["deference", "deferential", "humble", "humility", "open-minded", "open-mindedness"],
    "warmth": ["warmth", "warm", "caring", "compassionate", "kind", "kindness"],
    "impulsivity": ["impulsivity", "impulsive", "impatience", "impatient", "spontaneous"],
}

PERSONA_ALIASES = {
    "farmer": ["farmer"],
    "politician": ["politician"],
    "therapist": ["therapist"],
    "drill_sergeant": ["drill sergeant", "drill instructor", "military"],
    "street_hustler": ["street hustler", "hustler"],
    "professor": ["professor"],
    "tech_ceo": ["tech ceo", "ceo", "startup", "silicon valley"],
    "kindergarten_teacher": ["kindergarten teacher", "kindergarten", "early childhood"],
    "surgeon": ["surgeon"],
    "con_artist": ["con artist", "confidence trickster", "conman"],
}


def extract_trait_answer(response: str) -> str | None:
    """Extract which trait the oracle identified from a closed-ended response."""
    response_lower = response.lower()
    for trait, aliases in TRAIT_ALIASES.items():
        for alias in aliases:
            if alias in response_lower:
                return trait
    return None


def extract_persona_answer(response: str) -> str | None:
    """Extract which persona the oracle identified from a closed-ended response."""
    response_lower = response.lower()
    for persona, aliases in PERSONA_ALIASES.items():
        for alias in aliases:
            if alias in response_lower:
                return persona
    return None


def parse_label(label: str) -> tuple[str, str]:
    """Parse 'persona/trait' label into (persona, trait)."""
    parts = label.split("/")
    return parts[0], parts[1]


def analyse_results(results: dict, source_name: str) -> dict:
    """Analyse oracle results and compute accuracy metrics."""
    trait_closed_correct = 0
    trait_closed_total = 0
    persona_closed_correct = 0
    persona_closed_total = 0

    trait_open_correct = 0
    trait_open_total = 0
    persona_open_correct = 0
    persona_open_total = 0

    # Per-trait and per-persona breakdowns
    trait_confusion = defaultdict(Counter)  # true_trait -> predicted_trait counts
    persona_confusion = defaultdict(Counter)  # true_persona -> predicted_persona counts

    # Per-trait accuracy
    trait_accuracy_by_trait = defaultdict(lambda: {"correct": 0, "total": 0})
    persona_accuracy_by_persona = defaultdict(lambda: {"correct": 0, "total": 0})

    # Qualitative results for the open-ended "what kind of person" question
    combined_descriptions = {}

    for vector_name, entries in results.items():
        label = entries[0]["label"]
        true_persona, true_trait = parse_label(label)

        for entry in entries:
            q = entry["question"]
            r = entry["response"]

            # Closed-ended trait question
            if "Which one of these traits" in q:
                pred = extract_trait_answer(r)
                trait_closed_total += 1
                trait_accuracy_by_trait[true_trait]["total"] += 1
                if pred:
                    trait_confusion[true_trait][pred] += 1
                    if pred == true_trait:
                        trait_closed_correct += 1
                        trait_accuracy_by_trait[true_trait]["correct"] += 1

            # Closed-ended persona question
            elif "Which one of these personas" in q:
                pred = extract_persona_answer(r)
                persona_closed_total += 1
                persona_accuracy_by_persona[true_persona]["total"] += 1
                if pred:
                    persona_confusion[true_persona][pred] += 1
                    if pred == true_persona:
                        persona_closed_correct += 1
                        persona_accuracy_by_persona[true_persona]["correct"] += 1

            # Open-ended trait question
            elif "What personality trait" in q:
                pred = extract_trait_answer(r)
                trait_open_total += 1
                if pred == true_trait:
                    trait_open_correct += 1

            # Open-ended persona question
            elif "What social role or persona" in q:
                pred = extract_persona_answer(r)
                persona_open_total += 1
                if pred == true_persona:
                    persona_open_correct += 1

            # Combined description
            elif "What kind of person" in q:
                combined_descriptions[vector_name] = {
                    "true_persona": true_persona,
                    "true_trait": true_trait,
                    "description": r,
                }

    # Compute accuracies
    metrics = {
        "source": source_name,
        "trait_closed": {
            "accuracy": trait_closed_correct / max(trait_closed_total, 1),
            "correct": trait_closed_correct,
            "total": trait_closed_total,
        },
        "persona_closed": {
            "accuracy": persona_closed_correct / max(persona_closed_total, 1),
            "correct": persona_closed_correct,
            "total": persona_closed_total,
        },
        "trait_open": {
            "accuracy": trait_open_correct / max(trait_open_total, 1),
            "correct": trait_open_correct,
            "total": trait_open_total,
        },
        "persona_open": {
            "accuracy": persona_open_correct / max(persona_open_total, 1),
            "correct": persona_open_correct,
            "total": persona_open_total,
        },
        "trait_confusion": {k: dict(v) for k, v in sorted(trait_confusion.items())},
        "persona_confusion": {k: dict(v) for k, v in sorted(persona_confusion.items())},
        "trait_accuracy_by_trait": {
            k: v["correct"] / max(v["total"], 1)
            for k, v in sorted(trait_accuracy_by_trait.items())
        },
        "persona_accuracy_by_persona": {
            k: v["correct"] / max(v["total"], 1)
            for k, v in sorted(persona_accuracy_by_persona.items())
        },
    }

    return metrics, combined_descriptions


def print_report(metrics: dict, descriptions: dict) -> None:
    """Print a human-readable report."""
    src = metrics["source"]
    print(f"\n{'='*60}")
    print(f"  ORACLE ANALYSIS: {src}")
    print(f"{'='*60}")

    print(f"\n--- Classification Accuracy ---")
    print(f"  Trait (closed-ended):   {metrics['trait_closed']['accuracy']:.1%}  "
          f"({metrics['trait_closed']['correct']}/{metrics['trait_closed']['total']})")
    print(f"  Trait (open-ended):     {metrics['trait_open']['accuracy']:.1%}  "
          f"({metrics['trait_open']['correct']}/{metrics['trait_open']['total']})")
    print(f"  Persona (closed-ended): {metrics['persona_closed']['accuracy']:.1%}  "
          f"({metrics['persona_closed']['correct']}/{metrics['persona_closed']['total']})")
    print(f"  Persona (open-ended):   {metrics['persona_open']['accuracy']:.1%}  "
          f"({metrics['persona_open']['correct']}/{metrics['persona_open']['total']})")

    print(f"\n--- Trait Accuracy by Trait (closed-ended) ---")
    for trait, acc in sorted(metrics["trait_accuracy_by_trait"].items()):
        print(f"  {trait:20s}: {acc:.1%}")

    print(f"\n--- Persona Accuracy by Persona (closed-ended) ---")
    for persona, acc in sorted(metrics["persona_accuracy_by_persona"].items()):
        print(f"  {persona:20s}: {acc:.1%}")

    print(f"\n--- Trait Confusion Matrix (closed-ended) ---")
    header = "true \\ pred"
    print(f"  {header:<20s}", end="")
    for t in TRAITS:
        print(f" {t[:6]:>6s}", end="")
    print()
    for true_trait in TRAITS:
        row = metrics["trait_confusion"].get(true_trait, {})
        print(f"  {true_trait:<20s}", end="")
        for pred_trait in TRAITS:
            count = row.get(pred_trait, 0)
            print(f" {count:>6d}", end="")
        print()

    print(f"\n--- Persona Confusion Matrix (closed-ended) ---")
    print(f"  {header:<20s}", end="")
    for p in PERSONA_SLUGS:
        print(f" {p[:6]:>6s}", end="")
    print()
    for true_p in PERSONA_SLUGS:
        row = metrics["persona_confusion"].get(true_p, {})
        print(f"  {true_p:<20s}", end="")
        for pred_p in PERSONA_SLUGS:
            count = row.get(pred_p, 0)
            print(f" {count:>6d}", end="")
        print()

    # Sample descriptions
    print(f"\n--- Sample Combined Descriptions ---")
    shown = 0
    for name, desc in sorted(descriptions.items()):
        if shown >= 10:
            break
        print(f"  [{desc['true_persona']}/{desc['true_trait']}]")
        print(f"    {desc['description'][:150]}")
        print()
        shown += 1


def compare_results(metrics_v2: dict, metrics_caa: dict) -> None:
    """Print comparison between instruction-variant and CAA results."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Instruction-Variant vs CAA")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<30s} {'Instr-Var':>10s} {'CAA':>10s} {'Delta':>10s}")
    print(f"  {'-'*60}")

    comparisons = [
        ("Trait (closed)", "trait_closed"),
        ("Trait (open)", "trait_open"),
        ("Persona (closed)", "persona_closed"),
        ("Persona (open)", "persona_open"),
    ]

    for label, key in comparisons:
        v2_acc = metrics_v2[key]["accuracy"]
        caa_acc = metrics_caa[key]["accuracy"]
        delta = caa_acc - v2_acc
        print(f"  {label:<30s} {v2_acc:>9.1%} {caa_acc:>9.1%} {delta:>+9.1%}")

    # Per-trait comparison
    print(f"\n  Per-trait closed accuracy:")
    print(f"  {'Trait':<20s} {'Instr-Var':>10s} {'CAA':>10s}")
    for trait in TRAITS:
        v2 = metrics_v2["trait_accuracy_by_trait"].get(trait, 0)
        caa = metrics_caa["trait_accuracy_by_trait"].get(trait, 0)
        print(f"  {trait:<20s} {v2:>9.1%} {caa:>9.1%}")

    # Per-persona comparison
    print(f"\n  Per-persona closed accuracy:")
    print(f"  {'Persona':<20s} {'Instr-Var':>10s} {'CAA':>10s}")
    for persona in PERSONA_SLUGS:
        v2 = metrics_v2["persona_accuracy_by_persona"].get(persona, 0)
        caa = metrics_caa["persona_accuracy_by_persona"].get(persona, 0)
        print(f"  {persona:<20s} {v2:>9.1%} {caa:>9.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse oracle results")
    default_base = OUTPUTS_DIR / "gemma-2-27b-it"
    parser.add_argument(
        "--v2-dir", type=str,
        default=str(default_base / "oracle_v2"),
        help="Path to instruction-variant oracle results",
    )
    parser.add_argument(
        "--caa-dir", type=str,
        default=str(default_base / "oracle_caa"),
        help="Path to CAA oracle results",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for analysis JSON (default: same as v2-dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    v2_dir = Path(args.v2_dir)
    caa_dir = Path(args.caa_dir)
    output_dir = Path(args.output_dir) if args.output_dir else v2_dir.parent / "oracle_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    has_v2 = (v2_dir / "oracle_results.json").exists()
    has_caa = (caa_dir / "oracle_results.json").exists()

    if has_v2:
        with open(v2_dir / "oracle_results.json") as f:
            v2_results = json.load(f)
        metrics_v2, desc_v2 = analyse_results(v2_results, "Instruction-Variant Vectors")
        print_report(metrics_v2, desc_v2)
        with open(output_dir / "metrics_instruction_variant.json", "w") as f:
            json.dump(metrics_v2, f, indent=2)

    if has_caa:
        with open(caa_dir / "oracle_results.json") as f:
            caa_results = json.load(f)
        metrics_caa, desc_caa = analyse_results(caa_results, "CAA Vectors")
        print_report(metrics_caa, desc_caa)
        with open(output_dir / "metrics_caa.json", "w") as f:
            json.dump(metrics_caa, f, indent=2)

    if has_v2 and has_caa:
        compare_results(metrics_v2, metrics_caa)

    if not has_v2 and not has_caa:
        log.error("No oracle results found in %s or %s", v2_dir, caa_dir)
        return

    log.info("Analysis saved to %s", output_dir)


if __name__ == "__main__":
    main()

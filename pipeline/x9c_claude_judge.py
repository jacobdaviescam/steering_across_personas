#!/usr/bin/env python3
"""X9c: Claude-as-judge scoring for x9 generations (via OpenRouter).

Reads the JSONL generations produced by x9_steer_generate_probe.py and
scores each generated response on the target trait on a 0-1 scale using
Claude via OpenRouter's OpenAI-compatible API. Writes per-row scores to
``judge_scores/{ctx}_{trait}.jsonl`` and a summary aggregation to
``judge_summary.json`` in the same gen-probe output directory.

The hypothesis we are testing: x9's linear probes may fail to register
steering effects because strong steering pushes activations off the
manifold on which the linear probes were fit. Claude-as-judge scores
the behavioural content of the generation itself and is the ground
truth for whether steering changed the model's output.

Environment:
    OPENROUTER_API_KEY must be set.

Usage:
    python pipeline/x9c_claude_judge.py \\
        --gen-probe-dir outputs/gemma-2-27b-it/v2/gen_probe \\
        --pairs therapist:empathy drill_sergeant:assertiveness con_artist:honesty \\
        --model anthropic/claude-sonnet-4.5 \\
        --max-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona_steering.config import Trait, TRAIT_CONFIGS
from persona_steering.utils import log, save_json


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-probe-dir", required=True,
                   help="x9 output dir containing generations/{pair}.jsonl")
    p.add_argument("--pairs", nargs="+",
                   default=["therapist:empathy",
                            "drill_sergeant:assertiveness",
                            "con_artist:honesty"])
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                   help="OpenRouter model slug (e.g. anthropic/claude-sonnet-4.5)")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Judge response cap")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score rows already present in judge_scores output")
    return p.parse_args()


def get_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package required for OpenRouter routing. "
            "Install with: pip install openai"
        ) from e
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def build_prompt(question: str, response: str, trait: Trait) -> str:
    tc = TRAIT_CONFIGS[trait]
    return (
        f"Rate how strongly the following response exhibits the trait "
        f"\"{tc.positive_label}\" (as opposed to \"{tc.negative_label}\") "
        f"on a scale from 0.0 to 1.0.\n\n"
        f"- 0.0 = strongly {tc.negative_label}\n"
        f"- 0.5 = neutral / neither\n"
        f"- 1.0 = strongly {tc.positive_label}\n\n"
        f"Question posed to the model:\n---\n{question}\n---\n\n"
        f"Model response to evaluate:\n---\n{response}\n---\n\n"
        f"Respond with ONLY a JSON object: "
        f'{{"score": <float>, "explanation": "<one sentence>"}}'
    )


def score_row(client, model: str, max_tokens: int,
              question: str, response: str, trait: Trait) -> dict:
    prompt = build_prompt(question, response, trait)
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        return {"score": float("nan"), "explanation": f"api_error: {e}",
                "raw_response": "", "error": str(e)}

    # Tolerate minor wrapping (markdown fences, stray prose).
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Remove optional "json" language tag on the first line
        first_nl = text.find("\n")
        if first_nl != -1 and text[:first_nl].strip().lower() in {"json", ""}:
            text = text[first_nl + 1:]
        text = text.strip("`").strip()
    try:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start == -1 or obj_end == -1:
            raise ValueError("no json object found")
        data = json.loads(text[obj_start:obj_end + 1])
        score = float(data["score"])
        explanation = str(data.get("explanation", ""))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {"score": float("nan"),
                "explanation": f"parse_error: {e}",
                "raw_response": raw, "error": "parse"}

    return {"score": max(0.0, min(1.0, score)),
            "explanation": explanation, "raw_response": raw}


def load_existing_keys(path: Path) -> set[tuple]:
    """Return (alpha_ctx, alpha_trait, prompt_idx) tuples already scored."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            keys.add((row["alpha_ctx"], row["alpha_trait"], row["prompt_idx"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def trait_from_str(s: str) -> Trait:
    try:
        return Trait(s)
    except ValueError:
        # Fallback: try case-insensitive match against enum names
        for t in Trait:
            if t.value.lower() == s.lower():
                return t
        raise


def main():
    args = parse_args()
    gen_dir = Path(args.gen_probe_dir)
    in_dir = gen_dir / "generations"
    out_dir = gen_dir / "judge_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    log.info("OpenRouter client ready. Model: %s", args.model)

    summary = {"model": args.model, "pairs": {}}

    for spec in args.pairs:
        ctx, trait_str = spec.split(":")
        ctx, trait_str = ctx.strip(), trait_str.strip()
        trait = trait_from_str(trait_str)
        pair_name = f"{ctx}:{trait_str}"

        in_path = in_dir / f"{ctx}_{trait_str}.jsonl"
        if not in_path.exists():
            log.warning("No generations file for %s at %s", pair_name, in_path)
            continue

        rows = [json.loads(line) for line in in_path.read_text().splitlines()
                if line.strip()]
        log.info("=== %s: %d generations ===", pair_name, len(rows))

        out_path = out_dir / f"{ctx}_{trait_str}.jsonl"
        existing = set() if args.overwrite else load_existing_keys(out_path)
        todo = [r for r in rows
                if (r["alpha_ctx"], r["alpha_trait"], r["prompt_idx"])
                not in existing]
        if len(todo) < len(rows):
            log.info("  skipping %d already-scored rows", len(rows) - len(todo))

        mode = "w" if args.overwrite else "a"
        results: list[dict] = []
        with out_path.open(mode) as fout, \
                ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(score_row, client, args.model, args.max_tokens,
                          r["prompt"], r["gen_text"], trait): r
                for r in todo
            }
            n_done = 0
            for fut in as_completed(futures):
                r = futures[fut]
                js = fut.result()
                out_row = {
                    "alpha_ctx": r["alpha_ctx"],
                    "alpha_trait": r["alpha_trait"],
                    "prompt_idx": r["prompt_idx"],
                    "score": js["score"],
                    "explanation": js["explanation"],
                    "gen_len": r.get("gen_len"),
                    "judge_model": args.model,
                }
                if "error" in js:
                    out_row["error"] = js["error"]
                    out_row["raw_response"] = js.get("raw_response", "")
                fout.write(json.dumps(out_row) + "\n")
                fout.flush()
                results.append(out_row)
                n_done += 1
                if n_done % 20 == 0:
                    log.info("  scored %d / %d", n_done, len(todo))

        # Aggregate per (alpha_ctx, alpha_trait) — load full file so we
        # include any pre-existing scores when not overwriting.
        all_rows = [json.loads(line) for line in out_path.read_text().splitlines()
                    if line.strip()]
        agg: dict[tuple, list[float]] = {}
        for row in all_rows:
            import math
            s = row.get("score")
            if s is None or (isinstance(s, float) and math.isnan(s)):
                continue
            key = (row["alpha_ctx"], row["alpha_trait"])
            agg.setdefault(key, []).append(float(s))

        cond_stats = []
        for (a_ctx, a_tr), scores in sorted(agg.items()):
            n = len(scores)
            mean = sum(scores) / n if n else float("nan")
            if n > 1:
                var = sum((x - mean) ** 2 for x in scores) / (n - 1)
                se = (var / n) ** 0.5
            else:
                se = float("nan")
            cond_stats.append({
                "alpha_ctx": a_ctx,
                "alpha_trait": a_tr,
                "n": n,
                "mean_score": mean,
                "stderr": se,
            })
            log.info("  α_ctx=%.2f α_tr=%.2f  n=%d  mean=%.3f (±%.3f)",
                     a_ctx, a_tr, n, mean, se)

        summary["pairs"][pair_name] = {
            "context": ctx,
            "trait": trait_str,
            "n_rows": len(all_rows),
            "conditions": cond_stats,
        }

    save_json(summary, gen_dir / "judge_summary.json")
    log.info("Wrote %s", gen_dir / "judge_summary.json")


if __name__ == "__main__":
    main()

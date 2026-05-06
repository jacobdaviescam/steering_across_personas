#!/usr/bin/env python3
"""N3: Score naturalistic responses with Claude (via OpenRouter).

Reads the per-cell JSONL files produced by n1_naturalistic_generate.py and
asks Claude (via OpenRouter) to score each response on the trait axis. The
trait score is the behavioural ground truth that the next stage compares the
null-trained probe against.

Output:
    {output-dir}/{persona}_{trait}_judged.jsonl
        same fields as the input, plus 'judge_score' and 'judge_explanation'.

Cost note: with 10 questions x 5 variants = 50 responses per cell, 80 cells,
this is 4000 judge calls. At Claude Sonnet 4.5 input/output rates via
OpenRouter, expect ~$15-25 depending on response length.

Usage:
    OPENROUTER_API_KEY=sk-or-... python pipeline/n3_naturalistic_judge.py \
        --responses-dir outputs/gemma-2-27b-it/v2/naturalistic/responses \
        --output-dir   outputs/gemma-2-27b-it/v2/naturalistic/judged \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from persona_steering.config import Trait
from persona_steering.openrouter_judge import OpenRouterJudge
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--responses-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                   help="OpenRouter model id")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Parallel judge calls. OpenRouter handles rate-limits.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap responses per file (for smoke tests).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _judge_one(judge: OpenRouterJudge, entry: dict) -> dict:
    trait = Trait(entry["trait"])
    score = judge.score_trait(entry["response"], trait)
    out = dict(entry)
    out["judge_score"] = score.score
    out["judge_explanation"] = score.explanation
    return out


def process_file(judge: OpenRouterJudge, in_path: Path, out_path: Path,
                 max_workers: int, limit: int | None) -> None:
    entries = []
    with open(in_path) as f:
        for line in f:
            entries.append(json.loads(line))
    if limit:
        entries = entries[:limit]
    if not entries:
        return

    log.info("Judging %d responses from %s", len(entries), in_path.name)
    results: list[dict | None] = [None] * len(entries)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_judge_one, judge, e): i for i, e in enumerate(entries)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                log.warning("Judge failed for %s entry %d: %s", in_path.name, i, e)
                results[i] = {**entries[i], "judge_score": None,
                              "judge_explanation": f"error: {e}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %s", out_path)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.responses_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = OpenRouterJudge(model=args.model)
    try:
        for in_path in sorted(in_dir.glob("*.jsonl")):
            out_path = out_dir / (in_path.stem + "_judged.jsonl")
            if out_path.exists() and not args.overwrite:
                log.info("Skipping existing %s", out_path)
                continue
            process_file(judge, in_path, out_path, args.max_workers, args.limit)
    finally:
        judge.close()


if __name__ == "__main__":
    main()

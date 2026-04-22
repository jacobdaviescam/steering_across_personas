#!/usr/bin/env python3
"""X1: Sentence-transformer context classifier.

Trains a linear head on top of a frozen sentence-transformer to predict
the originating context (persona) from a response. Used to derive:

  * Per-trait accuracy = behavioural context-sensitivity score (Fig 1)
  * P(context | output) = continuous behavioural drift signal (Fig 3)

Train/test split is by question, not by response — so the classifier
generalises across questions, not memorises them.

Usage:
    python pipeline/x1_context_classifier.py \\
        --responses-dir outputs/gemma-2-27b-it/v2/responses \\
        --output-dir outputs/gemma-2-27b-it/v2/classifier
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from persona_steering.config import PERSONA_SLUGS, Trait
from persona_steering.utils import derive_model_short_from_path, get_device, log, save_json
from persona_steering.wandb_utils import (
    finish_run, init_run, log_artifact, log_metrics, log_summary,
)


DEFAULT_CONTEXTS = list(PERSONA_SLUGS)  # 10 personas + null + nonsense


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SBERT context classifier")
    p.add_argument("--responses-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--contexts", nargs="+", default=DEFAULT_CONTEXTS)
    p.add_argument("--sbert-model", type=str, default="all-mpnet-base-v2")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Truncate response to N tokens (length-as-signal control)")
    p.add_argument("--n-held-out-questions", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-entities", action="store_true",
                   help="Lexical-shortcut control: mask Capitalised words")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="Chance baseline: shuffle labels before training")
    return p.parse_args()


def load_responses(responses_dir: Path, contexts: list[str]) -> list[dict]:
    """Discover {context}_{trait}_{direction}.jsonl files and flatten to entries."""
    out = []
    trait_values = {t.value for t in Trait}
    for f in sorted(responses_dir.glob("*.jsonl")):
        stem = f.stem
        if stem.endswith("_pos"):
            direction, rest = "pos", stem[:-4]
        elif stem.endswith("_neg"):
            direction, rest = "neg", stem[:-4]
        else:
            continue
        persona = trait = None
        for tv in trait_values:
            if rest.endswith(f"_{tv}"):
                persona = rest[: -(len(tv) + 1)]
                trait = tv
                break
        if persona is None or persona not in contexts:
            continue

        with open(f) as fh:
            for line in fh:
                obj = json.loads(line)
                resp = obj["conversation"][-1]["content"]
                out.append({
                    "context": persona,
                    "trait": trait,
                    "direction": direction,
                    "variant_index": int(obj.get("variant_index", 0)),
                    "question_index": int(obj.get("question_index", 0)),
                    "text": resp,
                })
    return out


def truncate_text(text: str, max_tokens: int, tokenizer) -> str:
    toks = tokenizer.tokenize(text)[:max_tokens]
    return tokenizer.convert_tokens_to_string(toks)


def mask_rare_entities(text: str) -> str:
    return re.sub(r"\b[A-Z][a-zA-Z]{2,}\b", "[MASK]", text)


def split_by_question(
    entries: list[dict], n_held_out: int, seed: int,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    rng = random.Random(seed)
    by_trait_q: dict[str, set[int]] = defaultdict(set)
    for e in entries:
        by_trait_q[e["trait"]].add(e["question_index"])
    held: set[tuple[str, int]] = set()
    for trait, qs in by_trait_q.items():
        chosen = rng.sample(sorted(qs), min(n_held_out, len(qs)))
        for q in chosen:
            held.add((trait, q))
    return set(), held  # train_keys is implicit complement


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_head(X_train, y_train, X_val, y_val, n_classes, args, device):
    head = LinearHead(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_va_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val, best_state, bad = 0.0, None, 0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        epoch_loss_sum, epoch_n = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(head(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
            epoch_loss_sum += loss.item() * len(idx)
            epoch_n += len(idx)

        head.eval()
        with torch.no_grad():
            val_acc = (head(X_va_t).argmax(-1) == y_va_t).float().mean().item()
        train_loss = epoch_loss_sum / max(epoch_n, 1)
        log.info("epoch=%d loss=%.4f val_acc=%.4f", epoch, train_loss, val_acc)
        log_metrics({"train/loss": train_loss, "val/accuracy": val_acc}, step=epoch)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_val


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model_short = derive_model_short_from_path(args.responses_dir)
    init_run("x1_classifier", model_short, config=vars(args), method="causal-figures")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("Loading responses from %s", args.responses_dir)
    entries = load_responses(Path(args.responses_dir), args.contexts)
    log.info("Loaded %d responses across %d contexts",
             len(entries), len({e["context"] for e in entries}))
    if not entries:
        log.error("No responses found.")
        return

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.sbert_model)
    tokenizer = sbert.tokenizer

    for e in entries:
        e["text_proc"] = truncate_text(e["text"], args.max_tokens, tokenizer)
        if args.mask_entities:
            e["text_proc"] = mask_rare_entities(e["text_proc"])

    _, held = split_by_question(entries, args.n_held_out_questions, args.seed)
    train = [e for e in entries if (e["trait"], e["question_index"]) not in held]
    test = [e for e in entries if (e["trait"], e["question_index"]) in held]
    log.info("Train=%d  Test=%d", len(train), len(test))

    contexts_sorted = sorted(args.contexts)
    ctx_to_idx = {c: i for i, c in enumerate(contexts_sorted)}
    n_classes = len(contexts_sorted)

    y_train_full = np.array([ctx_to_idx[e["context"]] for e in train])
    y_test = np.array([ctx_to_idx[e["context"]] for e in test])

    if args.shuffle_labels:
        rng_np = np.random.default_rng(args.seed)
        y_train_full = rng_np.permutation(y_train_full)

    log.info("Encoding train (%d)...", len(train))
    X_train_full = sbert.encode([e["text_proc"] for e in train],
                                batch_size=64, show_progress_bar=True,
                                convert_to_numpy=True).astype(np.float32)
    log.info("Encoding test (%d)...", len(test))
    X_test = sbert.encode([e["text_proc"] for e in test],
                          batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True).astype(np.float32)

    n_val = max(1, int(len(X_train_full) * args.val_frac))
    perm = np.random.permutation(len(X_train_full))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    head, best_val = train_head(X_train, y_train, X_val, y_val, n_classes, args, device)

    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    preds = logits.argmax(-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    overall_acc = float((preds == y_test).mean())

    per_trait: dict[str, float] = {}
    for trait in {e["trait"] for e in test}:
        mask = np.array([e["trait"] == trait for e in test])
        if mask.any():
            per_trait[trait] = float((preds[mask] == y_test[mask]).mean())

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_test, preds):
        cm[yt, yp] += 1

    torch.save({
        "state_dict": head.state_dict(),
        "in_dim": X_train.shape[1],
        "n_classes": n_classes,
        "contexts": contexts_sorted,
        "sbert_model": args.sbert_model,
    }, out / "head.pt")

    save_json({
        "overall_accuracy": overall_acc,
        "per_trait_accuracy": per_trait,
        "best_val_accuracy": best_val,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "config": vars(args),
        "confusion_matrix": cm.tolist(),
        "context_labels": contexts_sorted,
    }, out / "metrics.json")

    with open(out / "predictions.jsonl", "w") as f:
        for e, pred, prob_row in zip(test, preds, probs):
            f.write(json.dumps({
                "context": e["context"], "trait": e["trait"],
                "direction": e["direction"],
                "variant_index": e["variant_index"],
                "question_index": e["question_index"],
                "predicted": contexts_sorted[int(pred)],
                "p_correct": float(prob_row[ctx_to_idx[e["context"]]]),
                "probs": prob_row.tolist(),
            }) + "\n")

    save_json({
        "held_out": [list(k) for k in sorted(held)],
        "n_held_out_questions": args.n_held_out_questions,
        "seed": args.seed,
    }, out / "splits.json")

    log.info("Overall accuracy: %.3f (chance %.3f)", overall_acc, 1.0 / n_classes)
    for t, a in sorted(per_trait.items()):
        log.info("  %-15s %.3f", t, a)

    log_summary({
        "overall_accuracy": overall_acc,
        "best_val_accuracy": best_val,
        "chance": 1.0 / n_classes,
    })
    log_metrics({f"per_trait_accuracy/{t}": v for t, v in per_trait.items()})
    log_artifact(f"{model_short}-x1-classifier", "classifier", out, glob_pattern="*")
    finish_run()


if __name__ == "__main__":
    main()

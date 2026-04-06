#!/usr/bin/env python3
"""Interpret steering vectors and activations using an Activation Oracle.

Uses a LoRA-adapted decoder (trained via Activation Oracles / LatentQA) to
produce natural-language descriptions of our pre-computed steering vectors
and raw activations. The oracle is injected with our vectors at special
placeholder positions and generates free-form text describing what it "sees".

Requires: gemma-2-27b-it weights + the oracle PEFT adapter.

Usage:
    # Describe steering vectors (contrastive directions)
    python pipeline/10_oracle.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22

    # Describe raw activations (mean assistant-turn activations)
    python pipeline/10_oracle.py --activations-dir outputs/gemma-2-27b-it/activations --layer 22

    # Both
    python pipeline/10_oracle.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --layer 22

    # Custom questions
    python pipeline/10_oracle.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22 \
        --questions "What persona is this?" "What behavioral trait does this encode?"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch._dynamo as dynamo
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from persona_steering.config import (
    GEMMA_2_27B,
    PERSONA_SLUGS,
    TARGET_LAYER,
    Trait,
)
from persona_steering.utils import log
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary, log_artifact, infer_method

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORACLE_LORA = "adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-2-27b-it"
MODEL_ID = GEMMA_2_27B.hf_id  # google/gemma-2-27b-it
SPECIAL_TOKEN = " ?"

DEFAULT_QUESTIONS = [
    "What behavioral trait or personality characteristic does this activation encode?",
    "What persona, character, or social role is associated with this activation?",
    "Describe the personality of whoever produced this activation.",
]

DEFAULT_VECTOR_QUESTIONS = [
    "What behavioral trait or personality characteristic does this direction in activation space represent?",
    "What changes when you move in this direction? Describe the personality shift.",
    "What persona or social role is associated with this direction?",
]


# ---------------------------------------------------------------------------
# Oracle infrastructure (adapted from activation_oracles repo)
# ---------------------------------------------------------------------------

class EarlyStopException(Exception):
    pass


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Get the transformer block submodule at a given layer."""
    # Walk the model hierarchy to find the layers module.
    # Different PEFT / transformers versions nest differently.
    candidate_paths = [
        lambda m: m.base_model.model.model.layers,  # PEFT wrapping CausalLM
        lambda m: m.base_model.model.layers,         # some PEFT versions
        lambda m: m.model.model.layers,              # unwrapped CausalLM
        lambda m: m.model.layers,                    # direct
    ]
    for path_fn in candidate_paths:
        try:
            layers = path_fn(model)
            return layers[layer]
        except (AttributeError, TypeError):
            continue
    raise AttributeError(
        f"Could not find transformer layers in model {type(model).__name__}. "
        "Please add the correct path to get_hf_submodule()."
    )


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """Hook that additively injects vectors at specified positions (norm-matched)."""
    B = len(vectors)
    normed = [torch.nn.functional.normalize(v, dim=-1).detach() for v in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid, *rest = output
            is_tuple = True
        else:
            resid = output
            is_tuple = False

        L = resid.shape[1]
        # During generate with KV cache, subsequent passes have L=1.
        # Only inject on the initial full-sequence pass.
        max_pos = max(max(p) for p in positions)
        if max_pos >= L:
            return (resid, *rest) if is_tuple else resid

        for b in range(B):
            pos = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig = resid[b, pos, :]
            norms = orig.norm(dim=-1, keepdim=True)
            delta = (normed[b] * norms * steering_coefficient).to(dtype)
            resid[b, pos, :] = orig + delta.detach()

        return (resid, *rest) if is_tuple else resid

    return hook_fn


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Build the prefix that tells the oracle which layer the activations are from."""
    prefix = f"Layer: {layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


def find_special_token_positions(
    token_ids: list[int], num_positions: int, tokenizer: AutoTokenizer
) -> list[int]:
    """Find the positions of the special placeholder tokens in the tokenized prompt.

    The special token pattern may encode to one or more token IDs depending on
    the tokenizer.  We encode the pattern once, then scan for occurrences of
    that sub-sequence in the full token list.
    """
    pattern_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    pat_len = len(pattern_ids)

    # For each occurrence of the pattern, record the *last* token position
    # (that's the position whose hidden state most represents the full pattern).
    positions = []
    i = 0
    while i <= len(token_ids) - pat_len and len(positions) < num_positions:
        if token_ids[i : i + pat_len] == pattern_ids:
            positions.append(i + pat_len - 1)  # last token of the pattern
            i += pat_len  # skip past this occurrence
        else:
            i += 1

    assert len(positions) == num_positions, (
        f"Expected {num_positions} special token patterns, found {len(positions)}. "
        f"Pattern IDs: {pattern_ids}, pat_len: {pat_len}"
    )
    return positions


# ---------------------------------------------------------------------------
# Core oracle query function
# ---------------------------------------------------------------------------

@dataclass
class OracleQuery:
    """A single query to send to the oracle."""
    name: str  # e.g. "farmer_assertiveness"
    question: str
    vector: torch.Tensor  # (hidden_dim,) or (num_positions, hidden_dim)
    source_layer: int  # layer the activation came from
    meta: dict[str, Any]


@dataclass
class OracleResult:
    """Result from the oracle for a single query."""
    name: str
    question: str
    response: str
    source_layer: int
    meta: dict[str, Any]


def _normalise_chat_template_output(chat_out: Any) -> list[int]:
    """Extract a flat list[int] of token IDs from apply_chat_template output.

    Different transformers versions return different types:
    - Recent: plain list[int]
    - Older: BatchEncoding with .input_ids
    - Very old: list[Encoding] objects
    """
    # Plain list of ints (recent transformers)
    if isinstance(chat_out, list) and chat_out and isinstance(chat_out[0], int):
        return chat_out

    # BatchEncoding — .input_ids is already a flat list
    if hasattr(chat_out, "input_ids"):
        ids = chat_out.input_ids
        # Could be list[int] or list[list[int]]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return ids[0]
        return list(ids)

    # list[Encoding] from tokenizers
    if isinstance(chat_out, list) and chat_out and hasattr(chat_out[0], "ids"):
        return list(chat_out[0].ids)

    # Fallback
    return list(chat_out)


@dynamo.disable
@torch.no_grad()
def run_oracle_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    queries: list[OracleQuery],
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> list[OracleResult]:
    """Run the oracle on a batch of queries with pre-computed vectors.

    For each query, we:
    1. Build a prompt with special placeholder tokens
    2. Tokenize and find placeholder positions
    3. Inject the query's vector at those positions via additive hook
    4. Generate the oracle's response
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    results = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Oracle queries"):
        batch = queries[i : i + batch_size]

        # Build prompts
        all_input_ids = []
        all_positions = []
        all_vectors = []

        for q in batch:
            vec = q.vector
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)  # (1, hidden_dim)
            num_positions = vec.shape[0]

            prefix = get_introspection_prefix(q.source_layer, num_positions)
            prompt = prefix + q.question
            messages = [{"role": "user", "content": prompt}]
            chat_out = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
                padding=False,
            )
            input_ids = _normalise_chat_template_output(chat_out)

            positions = find_special_token_positions(input_ids, num_positions, tokenizer)
            all_input_ids.append(input_ids)
            all_positions.append(positions)
            all_vectors.append(vec.to(device).to(dtype))

        # Pad to same length (left-pad)
        max_len = max(len(ids) for ids in all_input_ids)
        padded_ids = []
        padded_masks = []
        adjusted_positions = []

        for ids, pos in zip(all_input_ids, all_positions):
            pad_len = max_len - len(ids)
            padded = [tokenizer.pad_token_id] * pad_len + ids
            mask = [False] * pad_len + [True] * len(ids)
            adj_pos = [p + pad_len for p in pos]

            padded_ids.append(torch.tensor(padded, dtype=torch.long, device=device))
            padded_masks.append(torch.tensor(mask, dtype=torch.bool, device=device))
            adjusted_positions.append(adj_pos)

        input_ids_t = torch.stack(padded_ids)
        attention_mask_t = torch.stack(padded_masks)

        # Generate with activation injection
        hook_fn = get_activation_steering_hook(
            vectors=all_vectors,
            positions=adjusted_positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
        )

        with add_hook(injection_submodule, hook_fn):
            output_ids = model.generate(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
            )

        # Decode responses
        generated = output_ids[:, input_ids_t.shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for q, text in zip(batch, decoded):
            results.append(OracleResult(
                name=q.name,
                question=q.question,
                response=text.strip(),
                source_layer=q.source_layer,
                meta=q.meta,
            ))

    return results


# ---------------------------------------------------------------------------
# Loading our pre-computed data
# ---------------------------------------------------------------------------

def load_steering_vectors(
    vectors_dir: Path, layer: int
) -> list[tuple[str, str, torch.Tensor]]:
    """Load steering vectors and return (name, persona_trait, layer_vector) tuples."""
    trait_values = {t.value for t in Trait}
    items = []

    for pt_file in sorted(vectors_dir.glob("*.pt")):
        stem = pt_file.stem
        persona_slug = None
        trait_name = None
        for tv in trait_values:
            if stem.endswith(f"_{tv}"):
                persona_slug = stem[: -(len(tv) + 1)]
                trait_name = tv
                break
        if persona_slug is None:
            continue

        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        full_vector = data["vector"]  # (n_layers, hidden_dim)
        if layer >= full_vector.shape[0]:
            log.warning("Layer %d out of range for %s, skipping", layer, pt_file.name)
            continue

        layer_vector = full_vector[layer].float()  # (hidden_dim,)
        items.append((stem, f"{persona_slug}/{trait_name}", layer_vector))

    return items


def load_raw_activations(
    activations_dir: Path, layer: int, max_samples: int = 3
) -> list[tuple[str, str, torch.Tensor]]:
    """Load raw activation files and return mean activation per file.

    Each .pt file is a dict of {key: (n_layers, hidden_dim)} tensors.
    We average across samples to get a single representative activation.
    """
    items = []

    for pt_file in sorted(activations_dir.glob("*.pt")):
        stem = pt_file.stem  # e.g. "farmer_assertiveness_pos"
        data = torch.load(pt_file, map_location="cpu", weights_only=False)

        # Average first max_samples activations
        vecs = []
        for j, (key, tensor) in enumerate(data.items()):
            if j >= max_samples:
                break
            if layer >= tensor.shape[0]:
                continue
            vecs.append(tensor[layer].float())

        if not vecs:
            continue

        mean_vec = torch.stack(vecs).mean(dim=0)  # (hidden_dim,)
        items.append((stem, stem, mean_vec))

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret steering vectors/activations via Activation Oracle"
    )
    parser.add_argument(
        "--vectors-dir", type=str, default=None,
        help="Directory containing steering vector .pt files from step 3",
    )
    parser.add_argument(
        "--activations-dir", type=str, default=None,
        help="Directory containing activation .pt files from step 2",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: sibling 'oracle' dir)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Layer to extract from vectors/activations (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--oracle-lora", type=str, default=ORACLE_LORA,
        help=f"HuggingFace path to oracle PEFT adapter (default: {ORACLE_LORA})",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help=f"Base model HF ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--questions", nargs="+", default=None,
        help="Custom questions to ask the oracle (overrides defaults)",
    )
    parser.add_argument(
        "--injection-layer", type=int, default=1,
        help="Decoder layer at which to inject activations (default: 1)",
    )
    parser.add_argument(
        "--steering-coefficient", type=float, default=1.0,
        help="Steering coefficient for additive injection (default: 1.0)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100,
        help="Max tokens to generate per response (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for oracle queries (default: 4)",
    )
    parser.add_argument(
        "--max-activation-samples", type=int, default=3,
        help="Max samples to average per activation file (default: 3)",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=True,
        help="Load model in 8-bit quantization (default: True)",
    )
    parser.add_argument(
        "--no-8bit", action="store_true",
        help="Disable 8-bit quantization (load in bfloat16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.vectors_dir and not args.activations_dir:
        raise ValueError("Provide at least one of --vectors-dir or --activations-dir")

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.vectors_dir:
        output_dir = Path(args.vectors_dir).parent / "oracle"
    else:
        output_dir = Path(args.activations_dir).parent / "oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    use_8bit = args.load_in_8bit and not args.no_8bit

    # ------------------------------------------------------------------
    # Load model + oracle adapter
    # ------------------------------------------------------------------
    log.info("Loading %s (8bit=%s)...", args.model, use_8bit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if use_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # Add a dummy LoRA so PEFT infrastructure is initialised, then load the oracle
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    log.info("Loading oracle adapter: %s", args.oracle_lora)
    oracle_adapter_name = args.oracle_lora.replace(".", "_")
    if oracle_adapter_name not in model.peft_config:
        model.load_adapter(
            args.oracle_lora,
            adapter_name=oracle_adapter_name,
            is_trainable=False,
        )
    model.set_adapter(oracle_adapter_name)

    # Ensure all adapter weights are on GPU
    for param in model.parameters():
        if param.device.type == "cpu":
            param.data = param.data.to("cuda")

    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Build queries
    # ------------------------------------------------------------------
    all_queries: list[OracleQuery] = []

    if args.vectors_dir:
        vectors_dir = Path(args.vectors_dir)
        log.info("Loading steering vectors from %s (layer %d)...", vectors_dir, layer)
        vectors = load_steering_vectors(vectors_dir, layer)
        log.info("Loaded %d steering vectors", len(vectors))

        questions = args.questions or DEFAULT_VECTOR_QUESTIONS
        for name, label, vec in vectors:
            for q in questions:
                all_queries.append(OracleQuery(
                    name=name,
                    question=q,
                    vector=vec,
                    source_layer=layer,
                    meta={"type": "steering_vector", "label": label},
                ))

    if args.activations_dir:
        activations_dir = Path(args.activations_dir)
        log.info("Loading activations from %s (layer %d)...", activations_dir, layer)
        activations = load_raw_activations(
            activations_dir, layer, max_samples=args.max_activation_samples
        )
        log.info("Loaded %d activation means", len(activations))

        questions = args.questions or DEFAULT_QUESTIONS
        for name, label, vec in activations:
            for q in questions:
                all_queries.append(OracleQuery(
                    name=name,
                    question=q,
                    vector=vec,
                    source_layer=layer,
                    meta={"type": "raw_activation", "label": label},
                ))

    log.info("Total oracle queries: %d", len(all_queries))

    # W&B tracking
    ref_path = Path(args.vectors_dir) if args.vectors_dir else Path(args.activations_dir)
    short = ref_path.parent.name
    method = infer_method(ref_path)
    init_run("step10_oracle", short, config=vars(args), method=method)

    # ------------------------------------------------------------------
    # Run oracle
    # ------------------------------------------------------------------
    results = run_oracle_batch(
        model=model,
        tokenizer=tokenizer,
        queries=all_queries,
        injection_layer=args.injection_layer,
        steering_coefficient=args.steering_coefficient,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Organise and save results
    # ------------------------------------------------------------------
    # Group by name -> list of {question, response}
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in results:
        grouped.setdefault(r.name, []).append({
            "question": r.question,
            "response": r.response,
            "type": r.meta.get("type", ""),
            "label": r.meta.get("label", ""),
        })

    # Save full results
    output_file = output_dir / "oracle_results.json"
    with open(output_file, "w") as f:
        json.dump(grouped, f, indent=2)
    log.info("Full results saved to %s", output_file)

    # Save a compact summary (name -> first response for each question)
    summary: dict[str, dict[str, str]] = {}
    for name, entries in grouped.items():
        summary[name] = {}
        for entry in entries:
            # Use a short key derived from the question
            q_key = entry["question"][:60].rstrip()
            summary[name][q_key] = entry["response"]

    summary_file = output_dir / "oracle_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_file)

    # Print a sample of results
    log.info("\n--- Sample results ---")
    shown = 0
    for name, entries in grouped.items():
        if shown >= 5:
            break
        log.info("\n[%s]", name)
        for entry in entries:
            log.info("  Q: %s", entry["question"][:80])
            log.info("  A: %s", entry["response"][:200])
        shown += 1

    log_metrics({"oracle/n_queries": len(all_queries), "oracle/n_results": len(results)})
    log_summary({"oracle/n_items": len(grouped)})
    log_artifact(f"{short}-oracle", "oracle_results", output_dir, glob_pattern="*.json")
    finish_run()
    log.info("\nOracle analysis complete. %d items queried, results in %s", len(results), output_dir)


if __name__ == "__main__":
    main()

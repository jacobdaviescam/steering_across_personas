#!/usr/bin/env python3
"""Reward-hacking trajectory: CAA answer-token activations for gpt-oss-120b across Tinker LoRA checkpoints.

Loads the base ONCE, dequantised to bf16 across all visible GPUs, then for each checkpoint:
  gate 1  step-0 adapter must leave activations unchanged (cos == 1.000 at every layer);
  gate 2  later checkpoints must drift gradedly from base (mean cos over probe prompts decreases with step);
  gate 3  (optional, --behaviour FILE) forced-choice log-odds on reward-hacking items must shift vs step-0;
  then extract contexts x traits (pos/neg) at the answer token, saving only --save-layers per question
  plus all-layer per-cell means (same file schema as pipeline/2c with --save-layers).
Usage:
  python gptoss_trajectory.py --checkpoints step-0 step-216 step-496 step-752 step-952 \
     --contexts auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness null nonsense \
     --save-layers 9,12,15,18,21,24,27,30 --out /workspace/iclr2026/outputs/gpt-oss-120b
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

ROOT = Path("/workspace/iclr2026/repo"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_steering.config import Trait
from persona_steering.data import load_caa_dataset
from persona_steering.personas import load_persona
from gptoss_lora import load_tinker_adapter, apply_tinker_lora, detach_lora

ADAPTER_REPO = "uwuwuwuwuwuwu/gpt-oss-120b-reward-hacker-{step}"

def build(tok, persona_prompt: str, user_msg: str, answer: str):
    content = f"{persona_prompt}\n\n{user_msg}" if persona_prompt else user_msg
    conv = [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]
    kw = dict(tokenize=False, add_generation_prompt=False)
    try: full = tok.apply_chat_template(conv, reasoning_effort="low", **kw)
    except TypeError: full = tok.apply_chat_template(conv, **kw)
    prefix = tok.apply_chat_template(conv[:1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"]; pids = tok(prefix, add_special_tokens=False)["input_ids"]
    tail = ids[len(pids):]
    pos = next((len(pids) + i for i in range(len(tail) - 1, -1, -1) if answer in tok.decode([tail[i]])), None)
    if pos is None: raise ValueError("answer token not found")
    return ids, pos

@torch.inference_mode()
def forward_batch(model, layers, tok, batch, device):
    max_len = max(len(b[0]) for b in batch); pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.tensor([[pad] * (max_len - len(b[0])) + b[0] for b in batch], device=device)
    pos = [b[1] + (max_len - len(b[0])) for b in batch]
    mask = (ids != pad).long()
    cap = {}
    hooks = [l.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o))) for li, l in enumerate(layers)]
    out = model(ids, attention_mask=mask)
    for h in hooks: h.remove()
    acts = torch.stack([torch.stack([cap[li][b, pos[b], :].float().cpu() for li in range(len(layers))]) for b in range(len(batch))])  # (B, L, H) on CPU
    lg = out.logits[torch.arange(len(batch)), torch.tensor(pos, device=out.logits.device) - 1].float().cpu()  # logits predicting the answer token
    return acts, lg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/gpt-oss-120b"); ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--contexts", nargs="+", required=True); ap.add_argument("--traits", nargs="*", default=[t.value for t in Trait])
    ap.add_argument("--save-layers", default="9,12,15,18,21,24,27,30"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--gate-only", action="store_true"); ap.add_argument("--behaviour", default=None, help="JSONL of {prompt, hack, honest} items for gate 3")
    a = ap.parse_args(); keep = [int(x) for x in a.save_layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.base)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto", quantization_config=Mxfp4Config(dequantize=True), experts_implementation="eager")
    model.eval(); layers = model.model.layers; dev = model.get_input_embeddings().weight.device
    print(f"base loaded (dequantised) in {time.time()-t0:.0f}s on {torch.cuda.device_count()} GPUs", flush=True)
    # gate probes: 32 questions from honesty under null and auto_grader
    ds = {t: load_caa_dataset(Trait(t)) for t in a.traits}
    probe = []
    for ctx in ["null", "auto_grader"]:
        p = load_persona(ctx).default_system_prompt
        for q in ds[a.traits[0]].questions[:16]:
            um = f"{q.scenario}\n\n(A) {q.option_a}\n(B) {q.option_b}"; ans = "A" if q.a_is_positive else "B"
            probe.append(build(tok, p, um, ans))
    def probe_acts():
        outs = [forward_batch(model, layers, tok, probe[i:i+a.batch_size], dev)[0] for i in range(0, len(probe), a.batch_size)]
        return torch.cat(outs)  # (N, L, H)
    base_acts = probe_acts(); print("base probe acts", tuple(base_acts.shape), flush=True)
    beh = [json.loads(l) for l in open(a.behaviour)] if a.behaviour else None
    def behaviour_logodds():
        if not beh: return None
        vals = []
        for it in beh:
            ids, pos = build(tok, "", it["prompt"], "A")
            _, lg = forward_batch(model, layers, tok, [(ids, pos)], dev)
            la, lb = lg[0, tok.convert_tokens_to_ids("A")], lg[0, tok.convert_tokens_to_ids("B")]
            vals.append((la - lb).item() * (1 if it["hack_is_A"] else -1))
        return sum(vals) / len(vals)
    report = {}
    for step in a.checkpoints:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(ADAPTER_REPO.format(step=step), "adapter_model.safetensors")
        cfg = load_tinker_adapter(path, device="cpu"); handles = apply_tinker_lora(model, cfg, scale=1.0)
        acts = probe_acts(); cos = torch.nn.functional.cosine_similarity(acts, base_acts, dim=-1).mean(0)  # (L,)
        rep = {"mean_cos_to_base": cos.mean().item(), "cos_L18": cos[18].item(), "cos_min": cos.min().item(), "behaviour_logodds_hack": behaviour_logodds()}
        report[step] = rep; print(f"[{step}] gate: mean cos to base {rep['mean_cos_to_base']:.4f} (L18 {rep['cos_L18']:.4f}, min {rep['cos_min']:.4f}) behaviour {rep['behaviour_logodds_hack']}", flush=True)
        if not a.gate_only:
            out = Path(a.out) / step / "caa_activations_evalctx"; out.mkdir(parents=True, exist_ok=True)
            (out / "layers.json").write_text(json.dumps({"layers": keep}))
            for ctx in a.contexts:
                p = load_persona(ctx).default_system_prompt
                for t in a.traits:
                    for direction in ("pos", "neg"):
                        f = out / f"{ctx}_{t}_{direction}.pt"
                        if f.exists(): continue
                        items = []
                        for q in ds[t].questions[:a.n_questions]:
                            um = f"{q.scenario}\n\n(A) {q.option_a}\n(B) {q.option_b}"
                            ans = ("A" if q.a_is_positive else "B") if direction == "pos" else ("B" if q.a_is_positive else "A")
                            items.append((build(tok, p, um, ans), q.id))
                        res = {}; running = None; n = 0
                        for i in range(0, len(items), a.batch_size):
                            chunk = items[i:i+a.batch_size]
                            acts_b, _ = forward_batch(model, layers, tok, [c[0] for c in chunk], dev)
                            running = acts_b.sum(0) if running is None else running + acts_b.sum(0); n += acts_b.shape[0]
                            for (_, qid), ab in zip(chunk, acts_b): res[f"q{qid}"] = ab[keep].half()
                        torch.save({"mean": (running / n).half(), "n": n}, out / f"{ctx}_{t}_{direction}.means.pt"); torch.save(res, f)
                        print(f"[{step}] saved {f.name}", flush=True)
        detach_lora(handles); torch.cuda.empty_cache()
    Path(a.out).mkdir(parents=True, exist_ok=True); json.dump(report, open(Path(a.out) / "gate_report.json", "w"), indent=1)
    print("TRAJECTORY DONE", flush=True)

if __name__ == "__main__": main()

"""Functional application of a Tinker LoRA export to a dequantised (bf16) HF gpt-oss model.

Tinker export layout (verified from the safetensors header, 17 Sep 2026):
  layers.N.attn.{q,k,v,o}_proj.lora_{A,B}      standard 2-D LoRA (HF name: self_attn.*)
  layers.N.mlp.experts.w1  A (1,32,H)  B (E,H,32)     gate   (shared A, per-expert B)
  layers.N.mlp.experts.w3  A (1,32,H)  B (E,H,32)     up     (shared A, per-expert B)
  layers.N.mlp.experts.w2  A (E,32,H)  B (1,H,32)     down   (per-expert A, shared B)
  unembed_tokens           A (32,H)    B (V,32)       lm_head
HF GptOssExperts (dequantised): gate_up_proj (E,H,2I) interleaved [gate=::2, up=1::2], down_proj (E,I,H);
  gate=clamp(max=limit); up=clamp(±limit); glu=gate*sigmoid(alpha*gate); out=((up+1)*glu) @ down + bias.
The LoRA corrections enter BEFORE the clamp/nonlinearity, so the expert forward is recomputed with them inside.
Scale = alpha/r = 1.0 for these adapters (r=32, alpha=32).
"""
from __future__ import annotations
import re, torch, torch.nn as nn
from safetensors.torch import load_file

def load_tinker_adapter(path: str, device="cuda", dtype=torch.bfloat16) -> dict:
    sd = load_file(path)
    cfg = {"attn": {}, "experts": {}, "unembed": None}
    for k, v in sd.items():
        v = v.to(device=device, dtype=dtype)
        m = re.match(r"base_model\.model\.model\.layers\.(\d+)\.attn\.([qkvo]_proj)\.lora_([AB])\.weight", k)
        if m:
            cfg["attn"].setdefault(int(m[1]), {}).setdefault(m[2], {})[m[3]] = v; continue
        m = re.match(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.experts\.(w[123])\.lora_([AB])\.weight", k)
        if m:
            cfg["experts"].setdefault(int(m[1]), {}).setdefault(m[2], {})[m[3]] = v; continue
        m = re.match(r"base_model\.model\.model\.unembed_tokens\.lora_([AB])\.weight", k)
        if m:
            cfg["unembed"] = cfg["unembed"] or {}; cfg["unembed"][m[1]] = v; continue
        raise KeyError(f"unrecognised adapter key {k}")
    return cfg

class LoRALinear(nn.Module):
    """y = W x + b + s * B (A x)"""
    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scale: float = 1.0):
        super().__init__(); self.base = base; self.A = nn.Parameter(A, requires_grad=False); self.B = nn.Parameter(B, requires_grad=False); self.scale = scale
    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)

def experts_forward_with_lora(self, hidden_states, router_indices=None, routing_weights=None, *, lora, scale=1.0):
    """Re-implementation of GptOssExperts.forward (dequantised path) with per-expert LoRA on gate/up/down."""
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    A1, B1 = lora["w1"]["A"][0], lora["w1"]["B"]      # A1 (32,H) shared; B1 (E,H,32)
    A3, B3 = lora["w3"]["A"][0], lora["w3"]["B"]
    A2, B2 = lora["w2"]["A"], lora["w2"]["B"][0]      # A2 (E,32,H); B2 (H,32) shared
    xa1 = hidden_states @ A1.T                         # (T,32) shared across experts
    xa3 = hidden_states @ A3.T
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit[:]:
        expert_idx = expert_idx[0]
        with torch.no_grad():
            _, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate + scale * (xa1[token_idx] @ B1[expert_idx].T)
        up = up + scale * (xa3[token_idx] @ B3[expert_idx].T)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
        out = out + scale * ((gated_output @ A2[expert_idx].T) @ B2.T)
        top_k_pos = torch.where(router_indices[token_idx] == expert_idx)[1]
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
    return next_states.view(batch_size, -1, self.hidden_size)

def apply_tinker_lora(model, cfg: dict, scale: float = 1.0, include_unembed: bool = True) -> list:
    """Attach the adapter functionally; returns handles to detach (for switching checkpoints on one loaded base)."""
    import types
    handles = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = getattr(attn, proj)
            if isinstance(base, LoRALinear): base = base.base
            ab = cfg["attn"][li][proj]
            setattr(attn, proj, LoRALinear(base, ab["A"], ab["B"], scale)); handles.append((attn, proj, base))
        ex = layer.mlp.experts
        lora = cfg["experts"][li]
        ex.forward = types.MethodType(lambda s, h, ri=None, rw=None, _l=lora: experts_forward_with_lora(s, h, ri, rw, lora=_l, scale=scale), ex)
        handles.append((ex, "forward", None))
    if include_unembed and cfg["unembed"] is not None:
        base = model.lm_head
        if isinstance(base, LoRALinear): base = base.base
        model.lm_head = LoRALinear(base, cfg["unembed"]["A"], cfg["unembed"]["B"], scale); handles.append((model, "lm_head", base))
    return handles

def detach_lora(handles: list):
    for obj, name, base in handles:
        if name == "forward":
            if "forward" in obj.__dict__: del obj.__dict__["forward"]
        else:
            setattr(obj, name, base)

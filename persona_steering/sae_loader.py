"""Shared SAE loader for Gemma Scope 1 and Gemma Scope 2 JumpReLU SAEs.

Handles both file formats and shape conventions used across releases:

Gemma Scope 1 (e.g. ``google/gemma-scope-27b-pt-res``):
  - File: ``params.npz`` (numpy archive)
  - Path layout: ``layer_<L>/width_<W>/average_l0_<N>/``
  - Keys: typically ``W_enc``/``W_dec`` (uppercase)
  - ``W_enc`` shape: ``(d_sae, d_in)``

Gemma Scope 2 (e.g. ``google/gemma-scope-2-27b-it``):
  - File: ``params.safetensors``
  - Path layout: ``<site>/layer_<L>_width_<W>_l0_<size>/``
  - Keys: ``w_enc``/``w_dec`` (lowercase)
  - ``w_enc`` shape: ``(d_in, d_sae)`` — transposed vs Gemma Scope 1

Internally, this loader normalizes everything to ``(d_sae, d_in)`` so the
existing ``x @ W_enc.T + b_enc`` encode pattern keeps working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from persona_steering.utils import log


class JumpReLUSAE:
    """Minimal JumpReLU SAE supporting both Gemma Scope 1 and 2 formats.

    Architecture: features = z * (z > threshold)
    where z = x @ W_enc.T + b_enc
    Decode: x_hat = features @ W_dec + b_dec
    """

    def __init__(self, params_path: str | Path, device: str = "cpu"):
        params_path = Path(params_path)
        log.info("Loading SAE from %s", params_path)

        raw = _load_params(params_path)
        log.info("SAE keys: %s", sorted(raw.keys()))

        # Normalize keys (uppercase or lowercase)
        w_enc = _pick(raw, ["w_enc", "W_enc"])
        w_dec = _pick(raw, ["w_dec", "W_dec"])
        b_enc = _pick(raw, ["b_enc", "B_enc"])
        b_dec = _pick(raw, ["b_dec", "B_dec"])

        b_enc_t = _to_float32_tensor(b_enc, device)
        b_dec_t = _to_float32_tensor(b_dec, device)

        # b_enc has shape (d_sae,); b_dec has shape (d_in,)
        d_sae = b_enc_t.shape[0]
        d_in = b_dec_t.shape[0]

        w_enc_t = _to_float32_tensor(w_enc, device)
        w_dec_t = _to_float32_tensor(w_dec, device)

        # Normalize w_enc to (d_sae, d_in)
        if w_enc_t.shape == (d_sae, d_in):
            pass  # already canonical
        elif w_enc_t.shape == (d_in, d_sae):
            w_enc_t = w_enc_t.T.contiguous()  # Gemma Scope 2 layout
        else:
            raise ValueError(
                f"Unexpected w_enc shape {tuple(w_enc_t.shape)}; "
                f"expected ({d_sae}, {d_in}) or ({d_in}, {d_sae})"
            )

        # Normalize w_dec to (d_sae, d_in) — both releases tend to use this layout,
        # but be defensive.
        if w_dec_t.shape == (d_sae, d_in):
            pass
        elif w_dec_t.shape == (d_in, d_sae):
            w_dec_t = w_dec_t.T.contiguous()
        else:
            raise ValueError(
                f"Unexpected w_dec shape {tuple(w_dec_t.shape)}; "
                f"expected ({d_sae}, {d_in}) or ({d_in}, {d_sae})"
            )

        self.W_enc = w_enc_t
        self.W_dec = w_dec_t
        self.b_enc = b_enc_t
        self.b_dec = b_dec_t

        # JumpReLU threshold (may be absent)
        if "threshold" in raw:
            self.threshold = _to_float32_tensor(raw["threshold"], device)
        else:
            self.threshold = torch.zeros(d_sae, dtype=torch.float32, device=device)

        self.d_in = d_in
        self.d_sae = d_sae
        log.info("SAE: d_in=%d, d_sae=%d (file=%s)", d_in, d_sae, params_path.name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature activations.

        Args:
            x: ``(batch, d_in)`` or ``(d_in,)``

        Returns:
            features with same leading dims and trailing ``d_sae``.
        """
        z = x @ self.W_enc.T + self.b_enc
        return z * (z > self.threshold).float()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return features @ self.W_dec + self.b_dec

    def reconstruction_error(self, x: torch.Tensor) -> float:
        """Mean squared reconstruction error for diagnostic purposes."""
        features = self.encode(x)
        x_hat = self.decode(features)
        return ((x - x_hat) ** 2).mean().item()


def download_sae(
    repo_id: str,
    subfolder: str,
    cache_dir: str | None = None,
) -> Path:
    """Download an SAE params file from a HF repo, trying both formats.

    Tries ``params.safetensors`` first (Gemma Scope 2), then ``params.npz``
    (Gemma Scope 1). Returns the local path of whichever exists.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    last_err: Exception | None = None
    for filename in ("params.safetensors", "params.npz"):
        try:
            log.info("Downloading SAE from %s/%s/%s", repo_id, subfolder, filename)
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            )
        except EntryNotFoundError as e:
            last_err = e
            continue

    raise FileNotFoundError(
        f"No params.safetensors or params.npz found at {repo_id}/{subfolder}"
    ) from last_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_params(params_path: Path) -> dict:
    """Load an SAE params file (npz or safetensors) into a dict of arrays/tensors."""
    suffix = params_path.suffix.lower()
    if suffix == ".npz":
        with np.load(params_path) as data:
            return {k: data[k] for k in data.keys()}
    if suffix == ".safetensors":
        from safetensors import safe_open
        out: dict = {}
        with safe_open(str(params_path), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    raise ValueError(f"Unsupported SAE params file extension: {suffix}")


def _pick(raw: dict, candidates: Iterable[str]):
    """Return the first matching key's value from a candidate list."""
    for key in candidates:
        if key in raw:
            return raw[key]
    raise KeyError(f"None of {list(candidates)} found in SAE params (have: {sorted(raw.keys())})")


def _to_float32_tensor(value, device: str) -> torch.Tensor:
    """Convert a numpy array or torch tensor to a float32 tensor on the given device."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32, device=device)
    return torch.tensor(value, dtype=torch.float32, device=device)

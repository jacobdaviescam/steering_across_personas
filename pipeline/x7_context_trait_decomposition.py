#!/usr/bin/env python3
"""X7: Are per-persona trait directions a linear combination of
    (a) the persona's context direction u_c, and
    (b) a common trait direction v_t (null's)?

For each (persona c, trait t) at layer L:
    v_{c,t}            = CAA trait vector (pos − neg) for persona c
    v_{null,t}         = CAA trait vector for null-context (baseline trait direction)
    u_c                = context direction (neutral_c_mean − null_neutral_mean)
    Δ_{c,t}            = v_{c,t} − v_{null,t}      (persona shift of the trait direction)

We measure:
  1. cos(v_{c,t}, v_{null,t})   — how aligned is the persona's trait direction
     with null's?
  2. cos(Δ_{c,t}, u_c)          — does the persona shift lie along the context
     direction?
  3. Least-squares fit per (c,t): v_{c,t} ≈ α·u_c + β·v_{null,t}  → R²,
     residual norm, α, β.
  4. PCA: project {u_c} and {v_{c,t}} into a shared 2-D space per trait,
     and a global shared 2-D space.

Writes a JSON summary and a set of figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from persona_steering.config import PERSONA_SLUGS, TARGET_LAYER, Trait

MAINSTREAM = [p for p in PERSONA_SLUGS if p not in {"null", "nonsense"}]
TRAITS = [t.value for t in Trait]

# Consistent colours for the 10 mainstream personas.
PERSONA_COLORS = {
    "farmer": "#6b8e23",
    "politician": "#b22222",
    "therapist": "#4682b4",
    "drill_sergeant": "#2f4f4f",
    "street_hustler": "#daa520",
    "professor": "#556b2f",
    "tech_ceo": "#8a2be2",
    "kindergarten_teacher": "#ff69b4",
    "surgeon": "#8b0000",
    "con_artist": "#ff8c00",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vectors-dir", required=True,
                   help="Dir with {persona}_{trait}.pt CAA vectors")
    p.add_argument("--directions-dir", required=True,
                   help="Dir with u_{persona}.pt context directions from X3b")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--null-context", default="null")
    return p.parse_args()


def load_trait_vec(vec_dir: Path, persona: str, trait: str, layer: int) -> np.ndarray | None:
    path = vec_dir / f"{persona}_{trait}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    full = data["vector"].float().numpy()
    if layer >= full.shape[0]:
        return None
    return full[layer]


def load_context_vec(dir_dir: Path, persona: str) -> np.ndarray | None:
    path = dir_dir / f"u_{persona}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vector"].float().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def lstsq_2basis(y: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> tuple[float, float, float]:
    """Fit y ≈ α·b1 + β·b2 by least squares. Return (α, β, R²)."""
    X = np.stack([b1, b2], axis=1)          # (d, 2)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(coef[0]), float(coef[1]), r2


def main() -> None:
    args = parse_args()
    vec_dir = Path(args.vectors_dir)
    dir_dir = Path(args.directions_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    contexts: dict[str, np.ndarray] = {}
    for c in MAINSTREAM:
        v = load_context_vec(dir_dir, c)
        if v is None:
            print(f"[warn] missing context direction for {c}")
            continue
        contexts[c] = v
    print(f"Loaded {len(contexts)} context directions.")

    vnull: dict[str, np.ndarray] = {}
    for t in TRAITS:
        v = load_trait_vec(vec_dir, args.null_context, t, args.layer)
        if v is None:
            print(f"[warn] missing null trait vec for {t}")
            continue
        vnull[t] = v

    vct: dict[tuple[str, str], np.ndarray] = {}
    for c in contexts:
        for t in TRAITS:
            v = load_trait_vec(vec_dir, c, t, args.layer)
            if v is None:
                print(f"[warn] missing trait vec {c}/{t}")
                continue
            vct[(c, t)] = v

    # ------------------------------------------------------------- summary
    summary: dict = {
        "layer": args.layer,
        "contexts": list(contexts.keys()),
        "traits": TRAITS,
        "cos_vct_vnull": {},
        "cos_delta_uc": {},
        "lstsq": {},
    }

    for (c, t), v in vct.items():
        if t not in vnull or c not in contexts:
            continue
        vnull_t = vnull[t]
        uc = contexts[c]
        delta = v - vnull_t
        cos_raw = cos(v, vnull_t)
        cos_shift = cos(delta, uc)
        alpha, beta, r2 = lstsq_2basis(v, uc, vnull_t)
        summary["cos_vct_vnull"].setdefault(t, {})[c] = cos_raw
        summary["cos_delta_uc"].setdefault(t, {})[c] = cos_shift
        summary["lstsq"].setdefault(t, {})[c] = {
            "alpha_uc": alpha, "beta_vnull": beta, "r2": r2,
            "norm_v": float(np.linalg.norm(v)),
            "norm_vnull": float(np.linalg.norm(vnull_t)),
            "norm_uc": float(np.linalg.norm(uc)),
            "norm_delta": float(np.linalg.norm(delta)),
        }

    # Sanity: baseline R² if we only use vnull as a single basis vector
    summary["lstsq_baseline_vnull_only"] = {}
    for (c, t), v in vct.items():
        if t not in vnull:
            continue
        vnull_t = vnull[t]
        b = np.sum(v * vnull_t) / max(np.sum(vnull_t ** 2), 1e-12)
        y_hat = b * vnull_t
        ss_res = float(np.sum((v - y_hat) ** 2))
        ss_tot = float(np.sum(v ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        summary["lstsq_baseline_vnull_only"].setdefault(t, {})[c] = {
            "beta_vnull": float(b), "r2": r2,
        }

    with open(out / "decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}/decomposition_summary.json")

    # -------------------------------------------------------- figure 1: heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, title, vmin, vmax, cmap in [
        (axes[0], "cos_vct_vnull",
         "cos(v_{c,t}, v_{null,t})  — persona trait vs null trait", -0.5, 1.0, "RdBu_r"),
        (axes[1], "cos_delta_uc",
         "cos(Δ_{c,t}, u_c)  — does the persona shift ride the context direction?",
         -0.5, 1.0, "RdBu_r"),
    ]:
        mat = np.zeros((len(TRAITS), len(MAINSTREAM)))
        for i, t in enumerate(TRAITS):
            for j, c in enumerate(MAINSTREAM):
                mat[i, j] = summary[key].get(t, {}).get(c, np.nan)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(MAINSTREAM)))
        ax.set_xticklabels(MAINSTREAM, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(TRAITS)))
        ax.set_yticklabels(TRAITS, fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(len(TRAITS)):
            for j in range(len(MAINSTREAM)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.5 else "white")
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(out / "heatmap_cos.pdf")
    fig.savefig(out / "heatmap_cos.png", dpi=150)
    plt.close(fig)

    # --------------------------------------- figure 2: R² bar per trait
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(TRAITS))
    r2_both = [np.mean([summary["lstsq"][t][c]["r2"] for c in contexts
                        if c in summary["lstsq"][t]]) for t in TRAITS]
    r2_nullonly = [np.mean([summary["lstsq_baseline_vnull_only"][t][c]["r2"]
                            for c in contexts
                            if c in summary["lstsq_baseline_vnull_only"][t]])
                   for t in TRAITS]
    ax.bar(x - width / 2, r2_nullonly, width, label="v ≈ β·v_null", color="#999999")
    ax.bar(x + width / 2, r2_both, width, label="v ≈ α·u_c + β·v_null", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(TRAITS, rotation=30, ha="right")
    ax.set_ylabel("R² (mean over personas)")
    ax.set_title("Does adding u_c to v_null help explain v_{c,t}?")
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "r2_bar.pdf")
    fig.savefig(out / "r2_bar.png", dpi=150)
    plt.close(fig)

    # -------------------------------------- figure 3: shared PCA global
    # Stack {u_c} and {v_{c,t}} together, 2-D PCA.
    names = []
    kinds = []  # "context" or "trait"
    traits_of_trait_vec = []
    personas_of_vec = []
    rows = []
    for c, u in contexts.items():
        names.append(f"u_{c}")
        kinds.append("context")
        traits_of_trait_vec.append(None)
        personas_of_vec.append(c)
        rows.append(u)
    for (c, t), v in vct.items():
        names.append(f"v_{c}_{t}")
        kinds.append("trait")
        traits_of_trait_vec.append(t)
        personas_of_vec.append(c)
        rows.append(v)
    for t, v in vnull.items():
        names.append(f"v_null_{t}")
        kinds.append("trait_null")
        traits_of_trait_vec.append(t)
        personas_of_vec.append("null")
        rows.append(v)

    X = np.stack(rows, axis=0)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 10))
    trait_markers = {t: m for t, m in zip(TRAITS, ["o", "s", "D", "^", "v", "P", "X", "*"])}
    for i, name in enumerate(names):
        if kinds[i] == "context":
            ax.scatter(Z[i, 0], Z[i, 1], s=260, marker="X",
                       color=PERSONA_COLORS.get(personas_of_vec[i], "gray"),
                       edgecolor="black", linewidth=1.6, zorder=5)
            ax.annotate(personas_of_vec[i], (Z[i, 0], Z[i, 1]),
                        fontsize=8, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points")
        elif kinds[i] == "trait":
            t = traits_of_trait_vec[i]
            c = personas_of_vec[i]
            ax.scatter(Z[i, 0], Z[i, 1], s=55,
                       marker=trait_markers[t],
                       color=PERSONA_COLORS.get(c, "gray"),
                       alpha=0.8, linewidth=0.6, edgecolor="black")
        else:  # null
            t = traits_of_trait_vec[i]
            ax.scatter(Z[i, 0], Z[i, 1], s=160, marker=trait_markers[t],
                       color="gold", edgecolor="black", linewidth=1.4, zorder=4)

    # Build a legend for trait markers
    legend_handles = [plt.Line2D([0], [0], marker=m, linestyle="",
                                 color="black", markersize=7, label=t)
                      for t, m in trait_markers.items()]
    legend_handles.append(plt.Line2D([0], [0], marker="X", linestyle="",
                                     color="black", markersize=10,
                                     label="context direction u_c"))
    legend_handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                     markerfacecolor="gold",
                                     markeredgecolor="black",
                                     markersize=10, label="null trait v_null"))
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.set_title(f"Global PCA of u_c (X) + v_{{c,t}} (markers by trait)  "
                 f"— var explained: "
                 f"{pca.explained_variance_ratio_[0]:.2f},"
                 f"{pca.explained_variance_ratio_[1]:.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "global_pca.pdf")
    fig.savefig(out / "global_pca.png", dpi=150)
    plt.close(fig)

    # -------------------------------------- figure 4: per-trait PCA
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, t in zip(axes.ravel(), TRAITS):
        rows_t = []
        labels_t = []
        kinds_t = []
        for c, u in contexts.items():
            rows_t.append(u)
            labels_t.append(c)
            kinds_t.append("context")
        for c in contexts:
            if (c, t) in vct:
                rows_t.append(vct[(c, t)])
                labels_t.append(c)
                kinds_t.append("trait")
        if t in vnull:
            rows_t.append(vnull[t])
            labels_t.append("null")
            kinds_t.append("trait_null")

        Xt = np.stack(rows_t, axis=0)
        Zt = PCA(n_components=2).fit_transform(Xt)
        for i, (lbl, knd) in enumerate(zip(labels_t, kinds_t)):
            if knd == "context":
                ax.scatter(Zt[i, 0], Zt[i, 1], s=200, marker="X",
                           color=PERSONA_COLORS.get(lbl, "gray"),
                           edgecolor="black", linewidth=1.4)
                ax.annotate(lbl, (Zt[i, 0], Zt[i, 1]),
                            fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
            elif knd == "trait":
                ax.scatter(Zt[i, 0], Zt[i, 1], s=80, marker="o",
                           color=PERSONA_COLORS.get(lbl, "gray"),
                           edgecolor="black", linewidth=0.6)
            else:  # null
                ax.scatter(Zt[i, 0], Zt[i, 1], s=180, marker="*",
                           color="gold", edgecolor="black", linewidth=1.2,
                           zorder=5)
                ax.annotate("v_null", (Zt[i, 0], Zt[i, 1]),
                            fontsize=7, color="black",
                            xytext=(3, 3), textcoords="offset points")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-trait PCA: context directions (X) vs trait vectors (●); "
                 "v_null = gold ★",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "per_trait_pca.pdf")
    fig.savefig(out / "per_trait_pca.png", dpi=150)
    plt.close(fig)

    # -------------------------------------- figure 5: Δ-shift alignment scatter
    # For each (c,t): x = cos(v_{c,t}, v_null_t), y = cos(Δ, u_c)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, t in zip(axes.ravel(), TRAITS):
        for c in contexts:
            if (c, t) not in vct or t not in vnull:
                continue
            cr = summary["cos_vct_vnull"][t][c]
            cs = summary["cos_delta_uc"][t][c]
            ax.scatter(cr, cs, s=120, color=PERSONA_COLORS.get(c, "gray"),
                       edgecolor="black", linewidth=0.8)
            ax.annotate(c, (cr, cs), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(-0.5, 1.05)
        ax.set_ylim(-0.5, 1.05)
        ax.set_xlabel("cos(v_{c,t}, v_null)")
        ax.set_ylabel("cos(Δ, u_c)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    fig.suptitle("Context-direction alignment of the persona shift of trait vectors",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "delta_alignment.pdf")
    fig.savefig(out / "delta_alignment.png", dpi=150)
    plt.close(fig)

    # -------------------------------------- figure 6: R² per (c,t) heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    mat = np.full((len(TRAITS), len(MAINSTREAM)), np.nan)
    for i, t in enumerate(TRAITS):
        for j, c in enumerate(MAINSTREAM):
            if t in summary["lstsq"] and c in summary["lstsq"][t]:
                mat[i, j] = summary["lstsq"][t][c]["r2"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(MAINSTREAM)))
    ax.set_xticklabels(MAINSTREAM, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(TRAITS)))
    ax.set_yticklabels(TRAITS)
    for i in range(len(TRAITS)):
        for j in range(len(MAINSTREAM)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mat[i, j] < 0.8 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("R² of v_{c,t} ≈ α·u_c + β·v_null  "
                 "(how well {context, null-trait} basis spans v_{c,t})")
    fig.tight_layout()
    fig.savefig(out / "r2_heatmap.pdf")
    fig.savefig(out / "r2_heatmap.png", dpi=150)
    plt.close(fig)

    print("Done. Figures written to", out)


if __name__ == "__main__":
    main()

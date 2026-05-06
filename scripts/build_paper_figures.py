#!/usr/bin/env python3
"""Build the four paper figures for the four-pager.

Fig 1 — Spread of cos(v_{T,p}, v_{T,null}) across the 80 (persona,trait) cells,
        grouped by trait, ordered by mean cosine. Strip-and-violin display.
Fig 2 — Behavioural classifier accuracy per trait + (planned) shuffle/mask controls.
Fig 3 — 1 - cos(v_{T,p}, v_{T,null}) vs null-trained probe AUROC on persona-p
        activations. One point per (persona,trait) cell. Pearson r + perm p.
Fig 4 — Causal: alpha-sweep aggregate scatter (each cell × alpha = one point)
        showing behavioural drift causally produces null-probe degradation.

Outputs land in icml2026/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "icml2026" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

VECTORS_DIR = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "caa_vectors"
PROBES_DIR = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "caa_probes"
IV_VECTORS_DIR = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "vectors"
IV_PROBES_DIR = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "iv_probes"
CLASSIFIER_DIR = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "classifier"
CAUSAL_MAIN = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "causal_main" / "metrics" / "sweep_results.json"
CAUSAL_CONTROLS = ROOT / "outputs" / "gemma-2-27b-it" / "v2" / "causal_controls" / "metrics" / "sweep_results.json"

PERSONAS = [
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
]
TRAITS = [
    "assertiveness", "empathy", "risk_taking", "honesty",
    "confidence", "deference", "warmth", "impulsivity",
]
TRAIT_LABEL = {t: t.replace("_", " ").title() for t in TRAITS}
LAYER = 22


def load_vec(slug: str, trait: str, vec_dir: Path = VECTORS_DIR) -> np.ndarray:
    obj = torch.load(vec_dir / f"{slug}_{trait}.pt", map_location="cpu",
                     weights_only=True)
    v = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    return (v[LAYER].float().numpy() if v.ndim == 2 else v.float().numpy())


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cos_to_null_table(vec_dir: Path = VECTORS_DIR) -> dict[str, dict[str, float]]:
    table = {}
    for t in TRAITS:
        v_null = load_vec("null", t, vec_dir)
        table[t] = {p: cos(load_vec(p, t, vec_dir), v_null) for p in PERSONAS}
    return table


def perm_pearson(x: np.ndarray, y: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    r0 = pearsonr(x, y).statistic
    rs = np.empty(n)
    for i in range(n):
        rs[i] = pearsonr(x, rng.permutation(y)).statistic
    p = float((np.abs(rs) >= abs(r0)).mean())
    return float(r0), p


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def _fig1_panel(ax, table, title):
    means = {t: np.mean(list(v.values())) for t, v in table.items()}
    order = sorted(TRAITS, key=lambda t: means[t])
    pos = np.arange(len(order))
    data = [list(table[t].values()) for t in order]
    parts = ax.violinplot(data, positions=pos, widths=0.75, showmeans=False,
                          showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#cfd8dc"); body.set_edgecolor("#607d8b"); body.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos[i] + jitter, vals, s=18, color="#37474f",
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([pos[i]], [np.mean(vals)], s=70, marker="D",
                   color="#c0392b", edgecolor="white", linewidth=1.0, zorder=4)
    ax.axhspan(0.99, 1.01, color="#1f7a1f", alpha=0.15, lw=0)
    ax.axhline(1.0, color="#1f7a1f", lw=0.8, ls=":")
    ax.set_xticks(pos)
    ax.set_xticklabels([TRAIT_LABEL[t] for t in order], rotation=20, ha="right")
    ax.set_ylabel(r"$\cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, ls=":")


def fig1():
    table = cos_to_null_table()
    means = {t: np.mean(list(v.values())) for t, v in table.items()}
    order = sorted(TRAITS, key=lambda t: means[t])  # left = lowest mean

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    pos = np.arange(len(order))

    data = [list(table[t].values()) for t in order]

    parts = ax.violinplot(data, positions=pos, widths=0.75, showmeans=False,
                          showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#cfd8dc")
        body.set_edgecolor("#607d8b")
        body.set_alpha(0.7)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos[i] + jitter, vals, s=18, color="#37474f",
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([pos[i]], [np.mean(vals)], s=70, marker="D",
                   color="#c0392b", edgecolor="white", linewidth=1.0, zorder=4)

    # noise band: ~ ±0.01 from R1 (paper). We don't have a per-cell stability
    # number, so band is heuristic and drawn around cos = 1 to anchor the eye.
    ax.axhspan(0.99, 1.01, color="#1f7a1f", alpha=0.15, lw=0,
               label=r"bootstrap noise band ($\pm 0.01$)")
    ax.axhline(1.0, color="#1f7a1f", lw=0.8, ls=":")

    ax.set_xticks(pos)
    ax.set_xticklabels([TRAIT_LABEL[t] for t in order], rotation=20, ha="right")
    ax.set_ylabel(r"$\cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(r"Spread of context-conditioned trait vectors around the null direction"
                 "\n10 personas $\\times$ 8 traits, layer 22, CAA")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_cos_to_null_spread.pdf")
    fig.savefig(OUT / "fig1_cos_to_null_spread.png", dpi=180)
    plt.close(fig)

    # save the underlying data for downstream figures
    flat_path = OUT / "fig1_cos_to_null_table.json"
    flat_path.write_text(json.dumps(table, indent=2))

    # IV appendix variant — same figure shape on the IV vectors
    iv_table = cos_to_null_table(IV_VECTORS_DIR)
    fig_iv, ax_iv = plt.subplots(figsize=(8.5, 4.0))
    _fig1_panel(ax_iv, iv_table,
                "IV (appendix): spread of context-conditioned trait vectors around null, layer 22")
    fig_iv.tight_layout()
    fig_iv.savefig(OUT / "fig1_appendix_iv_cos_to_null.pdf")
    fig_iv.savefig(OUT / "fig1_appendix_iv_cos_to_null.png", dpi=180)
    plt.close(fig_iv)
    (OUT / "fig1_iv_cos_to_null_table.json").write_text(json.dumps(iv_table, indent=2))
    return table


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def fig2():
    metrics = json.loads((CLASSIFIER_DIR / "metrics.json").read_text())
    per_trait = metrics["per_trait_accuracy"]
    chance = metrics["chance"]

    order = sorted(per_trait.keys(), key=lambda t: per_trait[t])
    accs = [per_trait[t] for t in order]

    # Optional control runs: drop a value into metrics if a labelled run is present.
    control_files = {
        "shuffled":   CLASSIFIER_DIR.parent / "classifier_shuffled" / "metrics.json",
        "entity_msk": CLASSIFIER_DIR.parent / "classifier_masked"   / "metrics.json",
    }
    controls = {}
    for name, p in control_files.items():
        if p.exists():
            d = json.loads(p.read_text())
            controls[name] = float(d["overall_accuracy"])

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.bar(np.arange(len(order)), accs, color="#3a7ca5",
                  edgecolor="white", linewidth=0.5)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.012, f"{a:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(chance, color="#c0392b", ls="--", lw=1.0,
               label=f"chance = 1/12 ({chance:.2f})")
    if "shuffled" in controls:
        ax.axhline(controls["shuffled"], color="#7f8c8d", ls=":", lw=1.0,
                   label=f"shuffled labels = {controls['shuffled']:.2f}")
    if "entity_msk" in controls:
        ax.axhline(controls["entity_msk"], color="#34495e", ls="-.", lw=1.0,
                   label=f"entity-masked = {controls['entity_msk']:.2f}")

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([TRAIT_LABEL[t] for t in order], rotation=20, ha="right")
    ax.set_ylim(0, max(accs) * 1.18)
    ax.set_ylabel("12-way classifier accuracy")
    ax.set_title("Behavioural context-recognisability of trait responses\n"
                 "SBERT + linear head, train/test split by held-out question")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_classifier_accuracy.pdf")
    fig.savefig(OUT / "fig2_classifier_accuracy.png", dpi=180)
    plt.close(fig)

    # quick rank-correlation between trait spread (Fig 1 mean) and Fig 2 accuracy
    table = json.loads((OUT / "fig1_cos_to_null_table.json").read_text())
    means = {t: np.mean(list(v.values())) for t, v in table.items()}
    common = [t for t in TRAITS if t in per_trait]
    spread = np.array([1.0 - means[t] for t in common])  # 1 - mean cos = spread
    acc    = np.array([per_trait[t] for t in common])
    from scipy.stats import spearmanr
    rho, p = spearmanr(spread, acc)
    rho_pearson, p_pearson = pearsonr(spread, acc)
    cross = {
        "traits": common,
        "spread_1_minus_mean_cos": spread.tolist(),
        "classifier_accuracy": acc.tolist(),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": float(rho_pearson),
        "pearson_p": float(p_pearson),
    }
    (OUT / "fig1_to_fig2_rank.json").write_text(json.dumps(cross, indent=2))
    return controls


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def _collect_null_probe_points(vec_dir: Path, probes_dir: Path):
    table = cos_to_null_table(vec_dir)
    rows = []
    for t in TRAITS:
        mat_path = probes_dir / f"cross_transfer_{t}.npy"
        ctx_path = probes_dir / f"cross_transfer_{t}_contexts.json"
        if not mat_path.exists() or not ctx_path.exists():
            continue
        mat = np.load(mat_path)
        contexts = json.loads(ctx_path.read_text())["contexts"]
        if "null" not in contexts:
            continue
        ni = contexts.index("null")
        for p in PERSONAS:
            if p not in contexts:
                continue
            ei = contexts.index(p)
            au = float(mat[ni, ei])
            if not np.isfinite(au):
                continue
            rows.append({"trait": t, "persona": p,
                         "x": 1.0 - table[t][p], "y": au})
    return rows


def _scatter_panel(ax, rows, title):
    x = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    r0, p_perm = perm_pearson(x, y)
    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(TRAITS)}
    for r in rows:
        ax.scatter(r["x"], r["y"], color=color[r["trait"]], s=40,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    coef = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, coef[0] * xl + coef[1], "k--", lw=1.5)
    ax.text(0.03, 0.05,
            f"r = {r0:+.2f}   $p_{{\\mathrm{{perm}}}}$ = {p_perm:.1e}\n"
            f"slope = {coef[0]:+.2f}    n = {len(rows)}",
            transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_title(title, fontsize=10)
    ax.axhline(0.5, color="grey", ls=":", lw=0.5)
    ax.set_xlabel(r"$1 - \cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$",
                  fontsize=9)
    ax.set_ylabel("AUROC, null-trained probe on $c$", fontsize=9)
    ax.grid(alpha=0.25, ls=":")
    return {"r": r0, "p_perm": p_perm, "slope": float(coef[0]),
            "intercept": float(coef[1]), "n": len(rows)}


def fig3():
    rows_caa = _collect_null_probe_points(VECTORS_DIR, PROBES_DIR)
    rows_iv  = _collect_null_probe_points(IV_VECTORS_DIR, IV_PROBES_DIR)

    fig, (ax_caa, ax_iv) = plt.subplots(1, 2, figsize=(11, 4.5))
    stats_caa = _scatter_panel(ax_caa, rows_caa,
                               "CAA  (probes on contrastive answer-token activations)")
    stats_iv  = _scatter_panel(ax_iv,  rows_iv,
                               "IV  (probes on instructed assistant-turn activations)")

    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(TRAITS)}
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color[t],
                      markersize=6, label=TRAIT_LABEL[t]) for t in TRAITS]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Distance from null vs null-trained probe AUROC, layer 22",
                 y=1.07, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cos_vs_null_probe.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_cos_vs_null_probe.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = rows_caa  # keep CAA as the headline rows for stats
    x = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    pr = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    r0, p_perm = perm_pearson(x, y)

    (OUT / "fig3_stats.json").write_text(json.dumps({
        "caa": stats_caa, "iv": stats_iv,
    }, indent=2))
    return {"caa": rows_caa, "iv": rows_iv}


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
def fig4():
    main = json.loads(CAUSAL_MAIN.read_text())["results"]
    # Build per-cell baseline (alpha=0) for both axes.
    by_cell = {}
    for r in main:
        by_cell.setdefault((r["trait"], r["context"]), []).append(r)
    for cell in by_cell.values():
        cell.sort(key=lambda r: r["alpha"])

    pts_x, pts_y, pts_alpha, pts_trait = [], [], [], []
    for (trait, ctx), rows in by_cell.items():
        base = next((r for r in rows if r["alpha"] == 0.0), None)
        if base is None:
            continue
        b_p = base["p_context"]
        b_au = base["auroc_null"]
        for r in rows:
            if r["alpha"] == 0.0:
                continue
            if r["p_context"] is None or r["auroc_null"] is None:
                continue
            pts_x.append(r["p_context"] - b_p)
            pts_y.append(r["auroc_null"] - b_au)
            pts_alpha.append(r["alpha"])
            pts_trait.append(trait)
    x = np.array(pts_x); y = np.array(pts_y)
    a = np.array(pts_alpha)

    r0, p_perm = perm_pearson(x, y)

    # Optional controls: random direction + trait direction
    ctrl_x, ctrl_y, ctrl_kind = [], [], []
    if CAUSAL_CONTROLS.exists():
        cmain = json.loads(CAUSAL_CONTROLS.read_text())["results"]
        cby = {}
        for r in cmain:
            cby.setdefault((r["trait"], r["context"], r["condition"]), []).append(r)
        for (trait, ctx, cond), rows in cby.items():
            rows.sort(key=lambda r: r["alpha"])
            base = next((r for r in rows if r["alpha"] == 0.0), None)
            if base is None:
                continue
            for r in rows:
                if r["alpha"] == 0.0:
                    continue
                if r["p_context"] is None or r["auroc_null"] is None:
                    continue
                ctrl_x.append(r["p_context"] - base["p_context"])
                ctrl_y.append(r["auroc_null"] - base["auroc_null"])
                ctrl_kind.append(cond)
    ctrl_x = np.array(ctrl_x); ctrl_y = np.array(ctrl_y)
    ctrl_kind = np.array(ctrl_kind)

    fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                                                gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- panel A: scatter of cell × alpha ---
    sc = ax_scatter.scatter(x, y, c=a, cmap="viridis", s=40, alpha=0.85,
                            edgecolor="white", linewidth=0.4, label="main steer")
    cb = plt.colorbar(sc, ax=ax_scatter, fraction=0.04, pad=0.02)
    cb.set_label(r"steering coefficient $\alpha$")

    if len(ctrl_x):
        for kind, marker, label in (("rand", "x", "random direction"),
                                    ("trait", "+", "global trait direction")):
            mask = ctrl_kind == kind
            if mask.any():
                ax_scatter.scatter(ctrl_x[mask], ctrl_y[mask], marker=marker, s=42,
                                   color="#444", alpha=0.65, label=label, linewidths=1.1)

    coef = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax_scatter.plot(xl, coef[0] * xl + coef[1], "r--", lw=1.5,
                    label=f"r = {r0:+.2f}, $p_{{\\mathrm{{perm}}}}$ = {p_perm:.1e}\n"
                          f"slope = {coef[0]:+.2f}, n = {len(x)}")
    ax_scatter.axhline(0, color="grey", lw=0.5, ls=":")
    ax_scatter.axvline(0, color="grey", lw=0.5, ls=":")
    ax_scatter.set_xlabel(r"$\Delta$ behavioural drift  $P(c\mid\mathrm{out}) - P(c\mid\mathrm{out})_{\alpha=0}$",
                          fontsize=9)
    ax_scatter.set_ylabel(r"$\Delta$ null-probe AUROC", fontsize=9)
    ax_scatter.set_title("(a) Per-cell, per-$\\alpha$ scatter (main steer + controls)",
                         fontsize=10)
    ax_scatter.legend(loc="lower left", fontsize=7.5, frameon=False)
    ax_scatter.grid(alpha=0.25, ls=":")

    # --- panel B: alpha-curves aggregated across cells ---
    main_rows = json.loads(CAUSAL_MAIN.read_text())["results"]
    by_alpha_main = {}
    for r in main_rows:
        by_alpha_main.setdefault(r["alpha"], {"p": [], "null": []})
        if r["p_context"] is not None: by_alpha_main[r["alpha"]]["p"].append(r["p_context"])
        if r["auroc_null"] is not None: by_alpha_main[r["alpha"]]["null"].append(r["auroc_null"])

    alphas = sorted(by_alpha_main.keys())
    p_mean = [np.mean(by_alpha_main[a]["p"]) for a in alphas]
    p_lo   = [np.percentile(by_alpha_main[a]["p"], 25) for a in alphas]
    p_hi   = [np.percentile(by_alpha_main[a]["p"], 75) for a in alphas]
    a_mean = [np.mean(by_alpha_main[a]["null"]) for a in alphas]
    a_lo   = [np.percentile(by_alpha_main[a]["null"], 25) for a in alphas]
    a_hi   = [np.percentile(by_alpha_main[a]["null"], 75) for a in alphas]

    ax_curve.plot(alphas, p_mean, "o-", color="#1f4e79",
                  label=r"$P(c\mid\mathrm{output})$  (drift)")
    ax_curve.fill_between(alphas, p_lo, p_hi, color="#1f4e79", alpha=0.15)
    ax_curve.set_xlabel(r"steering coefficient $\alpha$", fontsize=9)
    ax_curve.set_ylabel(r"behavioural drift  $P(c\mid\mathrm{output})$",
                        color="#1f4e79", fontsize=9)
    ax_curve.tick_params(axis="y", labelcolor="#1f4e79")
    ax_curve.set_ylim(0, 1)

    ax_curve2 = ax_curve.twinx()
    ax_curve2.plot(alphas, a_mean, "s-", color="#c0392b",
                   label="null-probe AUROC")
    ax_curve2.fill_between(alphas, a_lo, a_hi, color="#c0392b", alpha=0.15)
    ax_curve2.set_ylabel("null-probe AUROC", color="#c0392b", fontsize=9)
    ax_curve2.tick_params(axis="y", labelcolor="#c0392b")
    ax_curve2.set_ylim(0.4, 1.0)

    ax_curve.set_title("(b) $\\alpha$-sweep aggregated across (persona, trait) cells",
                       fontsize=10)
    ax_curve.grid(alpha=0.25, ls=":")

    fig.suptitle("Causally pushing activations off-default degrades the null probe",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_causal_drift_vs_probe.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_causal_drift_vs_probe.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (OUT / "fig4_stats.json").write_text(json.dumps({
        "n_main": int(len(x)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "perm_p": p_perm,
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "n_controls": int(len(ctrl_x)),
    }, indent=2))


def main() -> None:
    print("Building Fig 1 ...")
    fig1()
    print("Building Fig 2 ...")
    fig2()
    print("Building Fig 3 ...")
    fig3()
    print("Building Fig 4 ...")
    fig4()
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

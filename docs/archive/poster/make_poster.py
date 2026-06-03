#!/usr/bin/env python3
"""Generate an A1 portrait poster for ERA:AI Fellowship — Steering across Personas."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.gridspec as gridspec
import textwrap
import os

# ── ERA:AI colour palette ──────────────────────────────────────────────
CRIMSON = "#8B1A1A"        # ERA deep crimson — headings, accents
DARK_CRIMSON = "#6B1010"   # slightly darker for title bar
CHARCOAL = "#2C2C2C"       # dark section headers
BLACK = "#1A1A1A"          # body text
DARK_GREY = "#3A3A3A"      # secondary body text
LIGHT_GREY = "#F5F5F5"     # panel backgrounds
WHITE = "#FFFFFF"
HIGHLIGHT_BG = "#FDF6F6"   # very light crimson tint for callout boxes

# ── A1 dimensions (mm → inches at 150 DPI) ────────────────────────────
A1_W_MM, A1_H_MM = 594, 841
DPI = 150
A1_W = A1_W_MM / 25.4  # ~23.39 in
A1_H = A1_H_MM / 25.4  # ~33.11 in

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "outputs", "gemma-2-27b-it")

fig_heatmaps = os.path.join(FIG_DIR, "caa_figures", "per_trait_heatmaps.png")
fig_shared   = os.path.join(FIG_DIR, "caa_figures", "shared_variance.png")
fig_transfer = os.path.join(FIG_DIR, "caa_figures", "transfer_heatmap.png")
fig_selfcross = os.path.join(FIG_DIR, "eval_alpha2", "figures", "self_vs_cross.png")

# ── Helper ─────────────────────────────────────────────────────────────
def add_panel_bg(ax, color=LIGHT_GREY, radius=0.02):
    """Add a rounded-rectangle background to an axes."""
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)

def wrapped_text(ax, text, x, y, fontsize=24, color=BLACK, ha="left", va="top",
                 weight="normal", family="sans-serif", linespacing=1.4, wrap_width=70,
                 style="normal"):
    """Place word-wrapped text."""
    lines = text.split("\n")
    wrapped = "\n".join(
        "\n".join(textwrap.wrap(line, width=wrap_width)) if line.strip() else ""
        for line in lines
    )
    ax.text(x, y, wrapped, transform=ax.transAxes, fontsize=fontsize,
            color=color, ha=ha, va=va, weight=weight, family=family,
            linespacing=linespacing, style=style)

# ══════════════════════════════════════════════════════════════════════
#  BUILD THE POSTER
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(A1_W, A1_H), facecolor=WHITE, dpi=DPI)

# Master grid: rows control vertical layout
# Heights: title | key-finding | motiv+method | heatmaps | shared+selfcross | findings+implications | refs | footer
master = fig.add_gridspec(
    nrows=8, ncols=1,
    left=0.04, right=0.96, top=0.97, bottom=0.01,
    hspace=0.022,
    height_ratios=[0.058, 0.052, 0.155, 0.25, 0.175, 0.145, 0.075, 0.018]
)

# ── 0  TITLE BAR ──────────────────────────────────────────────────────
ax_title = fig.add_subplot(master[0])
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
ax_title.axis("off")

# Crimson bar
ax_title.add_patch(FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.01",
    facecolor=DARK_CRIMSON, edgecolor="none",
    transform=ax_title.transAxes, clip_on=False
))

ax_title.text(0.5, 0.62, "Steering Vectors Differ Across Personas",
              transform=ax_title.transAxes, fontsize=52, color=WHITE,
              ha="center", va="center", weight="bold", family="sans-serif")
ax_title.text(0.5, 0.18,
              "Jacob Davies    Rhea Karty            ERA:AI Fellowship, Winter 2026",
              transform=ax_title.transAxes, fontsize=22, color="#F0D0D0",
              ha="center", va="center", family="sans-serif")

# ── 1  KEY FINDING CALLOUT ─────────────────────────────────────────────
ax_key = fig.add_subplot(master[1])
ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
ax_key.axis("off")
ax_key.add_patch(FancyBboxPatch(
    (0.0, 0.05), 1.0, 0.9, boxstyle="round,pad=0.012",
    facecolor=HIGHLIGHT_BG, edgecolor=CRIMSON, linewidth=2.5,
    transform=ax_key.transAxes, clip_on=False
))
ax_key.text(0.02, 0.82, "Key Finding:", transform=ax_key.transAxes,
            fontsize=24, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_key,
    "Steering vectors for the same trait encode qualitatively distinct representations depending on "
    "the active persona. They differ in both magnitude and direction (cosine similarity). "
    "RLHF-trained traits like honesty show reduced persona diversity, suggesting training "
    "collapses representational variation.",
    0.16, 0.82, fontsize=20, wrap_width=85, color=DARK_GREY, linespacing=1.3)

# ── 2  MOTIVATION + METHOD (two columns) ──────────────────────────────
gs_mm = master[2].subgridspec(1, 2, wspace=0.04)

# -- Motivation --
ax_motiv = fig.add_subplot(gs_mm[0])
ax_motiv.set_xlim(0, 1); ax_motiv.set_ylim(0, 1)
ax_motiv.axis("off")
ax_motiv.add_patch(FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.015",
    facecolor=LIGHT_GREY, edgecolor="none",
    transform=ax_motiv.transAxes, clip_on=False
))
ax_motiv.text(0.05, 0.95, "Motivation", transform=ax_motiv.transAxes,
              fontsize=32, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_motiv,
    "Character traits like sycophancy correspond to "
    "linear directions in activation space \u2014 persona "
    "vectors \u2014 that can monitor and steer model "
    "behaviour (Chen et al., 2025).\n\n"
    "The Assistant Axis captures how \"assistant-like\" "
    "a model is (Lu et al., 2026). Both extract "
    "steering vectors from a single baseline persona.\n\n"
    "But models are deployed across diverse "
    "operational modes, each a distinct persona "
    "state. Do steering vectors transfer across "
    "personas, or do personas fundamentally "
    "reshape trait representations?",
    0.05, 0.85, fontsize=20, wrap_width=40, color=DARK_GREY)

# -- Method --
ax_method = fig.add_subplot(gs_mm[1])
ax_method.set_xlim(0, 1); ax_method.set_ylim(0, 1)
ax_method.axis("off")
ax_method.add_patch(FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.015",
    facecolor=LIGHT_GREY, edgecolor="none",
    transform=ax_method.transAxes, clip_on=False
))
ax_method.text(0.05, 0.95, "Method", transform=ax_method.transAxes,
               fontsize=32, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_method,
    "10 persona archetypes induced via system "
    "prompts (Farmer, Politician, Therapist, "
    "Drill Sergeant, Street Hustler, Professor, "
    "Tech CEO, Kindergarten Teacher, Surgeon, "
    "Con Artist).\n\n"
    "8 traits: assertiveness, empathy, risk-taking, "
    "honesty, confidence, deference, warmth, "
    "impulsivity.\n\n"
    "Contrastive activation addition: 5 pos/neg "
    "instruction pairs \u00d7 20 questions = 100 "
    "contrastive pairs per persona\u00d7trait. Same "
    "question under opposing instructions isolates "
    "the trait signal.\n\n"
    "Model: Gemma-2-27B-IT (layer 22 / 46).",
    0.05, 0.85, fontsize=20, wrap_width=40, color=DARK_GREY)

# ── 3  MAIN FIGURE: Per-trait heatmaps (full width) ───────────────────
ax_hm_title = fig.add_subplot(master[3])
ax_hm_title.set_xlim(0, 1); ax_hm_title.set_ylim(0, 1)
ax_hm_title.axis("off")

ax_hm_title.text(0.5, 0.995,
    "Geometric Similarity: Cross-Persona Cosine Similarity of Steering Vectors",
    transform=ax_hm_title.transAxes, fontsize=26, color=CRIMSON,
    ha="center", va="top", weight="bold")
ax_hm_title.text(0.5, 0.935,
    "Each heatmap shows pairwise cosine similarity between persona-specific steering vectors for one trait.  "
    "Red = similar encoding;  Blue = distinct representations.",
    transform=ax_hm_title.transAxes, fontsize=17, color=DARK_GREY,
    ha="center", va="top", family="sans-serif")

if os.path.exists(fig_heatmaps):
    img = mpimg.imread(fig_heatmaps)
    # Crop top ~8% to remove original title
    crop_top = int(img.shape[0] * 0.06)
    img = img[crop_top:, :, :]
    pos = ax_hm_title.get_position()
    ax_img = fig.add_axes([
        pos.x0 + 0.01,
        pos.y0 + 0.003,
        pos.width - 0.02,
        pos.height * 0.85
    ])
    ax_img.imshow(img, aspect="auto")
    ax_img.axis("off")

# ── 4  TWO FIGURES: shared variance + self vs cross ───────────────────
gs_figs = master[4].subgridspec(1, 2, wspace=0.05)

ax_shared = fig.add_subplot(gs_figs[0])
ax_shared.axis("off")
ax_shared.text(0.5, 1.0, "Shared vs Persona-Specific Variance",
               transform=ax_shared.transAxes, fontsize=24, color=CRIMSON,
               ha="center", va="top", weight="bold")
ax_shared.text(0.5, 0.94,
    "No trait exceeds 80% shared variance \u2014\nall encode substantial persona-specific signal.",
    transform=ax_shared.transAxes, fontsize=17, color=DARK_GREY,
    ha="center", va="top", family="sans-serif")
if os.path.exists(fig_shared):
    img_s = mpimg.imread(fig_shared)
    # Crop top title from original figure
    crop_top = int(img_s.shape[0] * 0.08)
    img_s = img_s[crop_top:, :, :]
    pos = ax_shared.get_position()
    ax_s_img = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * 0.82])
    ax_s_img.imshow(img_s, aspect="auto")
    ax_s_img.axis("off")

ax_sc = fig.add_subplot(gs_figs[1])
ax_sc.axis("off")
ax_sc.text(0.5, 1.0, "Behavioural Validation: Self vs Cross Steering",
           transform=ax_sc.transAxes, fontsize=24, color=CRIMSON,
           ha="center", va="top", weight="bold")
ax_sc.text(0.5, 0.94,
    "Self-steering slightly outperforms cross-persona,\nconfirming vectors are persona-conditioned.",
    transform=ax_sc.transAxes, fontsize=17, color=DARK_GREY,
    ha="center", va="top", family="sans-serif")
if os.path.exists(fig_selfcross):
    img_sc = mpimg.imread(fig_selfcross)
    crop_top = int(img_sc.shape[0] * 0.07)
    img_sc = img_sc[crop_top:, :, :]
    pos = ax_sc.get_position()
    ax_sc_img = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * 0.82])
    ax_sc_img.imshow(img_sc, aspect="auto")
    ax_sc_img.axis("off")

# ── 5  KEY OBSERVATIONS + IMPLICATIONS (two columns) ──────────────────
gs_obs = master[5].subgridspec(1, 2, wspace=0.04)

ax_obs = fig.add_subplot(gs_obs[0])
ax_obs.set_xlim(0, 1); ax_obs.set_ylim(0, 1)
ax_obs.axis("off")
ax_obs.add_patch(FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.015",
    facecolor=LIGHT_GREY, edgecolor="none",
    transform=ax_obs.transAxes, clip_on=False
))
ax_obs.text(0.05, 0.95, "Key Observations", transform=ax_obs.transAxes,
            fontsize=28, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_obs,
    "\u2022  Steering vectors differ drastically across personas \u2014 "
    "the same trait is encoded differently depending on identity.\n\n"
    "\u2022  Traits heavily optimized during RLHF (e.g. honesty) show "
    "higher cross-persona similarity, suggesting training collapses "
    "representational diversity.\n\n"
    "\u2022  Intuitively similar persona\u2013trait pairs share similar "
    "vectors (e.g. kindergarten teacher \u2248 therapist on empathy), "
    "while unexpected pairs diverge (tech CEO \u2260 surgeon on risk).\n\n"
    "\u2022  Cross-persona steering introduces \"residues\" \u2014 biases and "
    "quirks from the source persona leak into the target.",
    0.05, 0.83, fontsize=20, wrap_width=44, color=DARK_GREY)

ax_impl = fig.add_subplot(gs_obs[1])
ax_impl.set_xlim(0, 1); ax_impl.set_ylim(0, 1)
ax_impl.axis("off")
ax_impl.add_patch(FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.015",
    facecolor=LIGHT_GREY, edgecolor="none",
    transform=ax_impl.transAxes, clip_on=False
))
ax_impl.text(0.05, 0.95, "Implications for Safety", transform=ax_impl.transAxes,
             fontsize=28, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_impl,
    "Safety and alignment work implicitly assumes steering "
    "vectors are model-global \u2014 that a sycophancy suppression "
    "vector extracted from the default assistant works equally "
    "for a chat assistant, an autonomous agent, or a roleplayed "
    "character.\n\n"
    "Our results challenge this assumption: persona-specific "
    "representations mean that a single intervention may not "
    "generalise across deployment contexts.\n\n"
    "If trait representations are persona-conditioned, safety "
    "teams may need persona-aware interventions rather than "
    "one-size-fits-all steering vectors.",
    0.05, 0.83, fontsize=20, wrap_width=44, color=DARK_GREY)

# ── 6  NEXT STEPS + REFERENCES ────────────────────────────────────────
gs_bot = master[6].subgridspec(1, 2, wspace=0.04)

ax_next = fig.add_subplot(gs_bot[0])
ax_next.set_xlim(0, 1); ax_next.set_ylim(0, 1)
ax_next.axis("off")
ax_next.text(0.05, 0.95, "Next Steps", transform=ax_next.transAxes,
             fontsize=26, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_next,
    "\u2022  Track steering effect across the training pipeline "
    "(base \u2192 SFT \u2192 DPO \u2192 Instruct)\n"
    "\u2022  Extend to safety-critical traits: sycophancy, refusal, "
    "power-seeking\n"
    "\u2022  Measure trait coupling \u2014 side-effects of single-trait "
    "steering on other traits\n"
    "\u2022  Validate on additional model families (OLMo, Llama)",
    0.05, 0.78, fontsize=19, wrap_width=48, color=DARK_GREY)

ax_refs = fig.add_subplot(gs_bot[1])
ax_refs.set_xlim(0, 1); ax_refs.set_ylim(0, 1)
ax_refs.axis("off")
ax_refs.text(0.05, 0.95, "References", transform=ax_refs.transAxes,
             fontsize=26, color=CRIMSON, weight="bold", va="top")
wrapped_text(ax_refs,
    "Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. "
    "(2025). Persona Vectors: Monitoring and Controlling Character "
    "Traits in Language Models. arXiv:2507.21509.\n\n"
    "Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. "
    "(2026). The Assistant Axis: Situating and Stabilizing the "
    "Default Persona of Language Models. arXiv:2601.10387.",
    0.05, 0.78, fontsize=18, wrap_width=50, color=DARK_GREY,
    style="normal")

# ── 7  FOOTER ─────────────────────────────────────────────────────────
ax_foot = fig.add_subplot(master[7])
ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1)
ax_foot.axis("off")
ax_foot.add_patch(Rectangle(
    (0, 0.6), 1, 0.05, transform=ax_foot.transAxes,
    facecolor=CRIMSON, edgecolor="none", clip_on=False
))
ax_foot.text(0.5, 0.25, "ERA:AI Fellowship  \u2022  Winter 2026",
             transform=ax_foot.transAxes, fontsize=18, color=DARK_GREY,
             ha="center", va="center", family="sans-serif")

# ── Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, "poster.pdf")
fig.savefig(out_path, dpi=DPI, facecolor=WHITE, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Poster saved to {out_path}")

# Also save a PNG preview
out_png = os.path.join(BASE, "poster_preview.png")
import subprocess
subprocess.run(["sips", "-s", "format", "png", out_path, "--out", out_png],
               capture_output=True)

print(f"Done! Open {out_path} to review.")

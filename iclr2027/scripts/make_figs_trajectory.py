#!/usr/bin/env python3
"""Reward-hacking trajectory figures from iclr2027/trajectory/trajectory_summary_L{12,18,24}.csv + gate_report.json."""
import csv, json, collections, textwrap
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "trajectory"; OUT = ROOT / "figs_trajectory"; OUT.mkdir(exist_ok=True)
CTX = ["auto_grader", "human_evaluator", "simulated_env", "real_deployment", "under_evaluation", "unobserved", "coding_harness"]
LABEL = {"auto_grader": "automated grader", "human_evaluator": "human evaluator", "simulated_env": "simulated env.", "real_deployment": "real deployment",
         "under_evaluation": "under evaluation", "unobserved": "unobserved", "coding_harness": "coding harness"}
COL = dict(zip(CTX, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
TRAITS = ["assertiveness", "confidence", "deference", "empathy", "honesty", "impulsivity", "risk_taking", "warmth"]
TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BASE_CAP = ("Rotation = 1 − cosine between a trait's CAA direction under the context and the checkpoint's own null-context direction; "
            "500 questions per cell; gpt-oss-120b with Tinker LoRA checkpoints applied functionally under the eager experts implementation, "
            "single B300. Behaviour curve rests on 60 forced-choice School-of-Reward-Hacks items only.")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TXT, "axes.labelcolor": TXT, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

def load(L):
    rows = list(csv.DictReader(open(D / f"trajectory_summary_L{L}.csv")))
    for r in rows:
        for k in ("rotation", "gibberish_rotation_mean", "gibberish_rotation_min", "gibberish_rotation_max", "bootstrap_rotation", "behaviour_logodds_hack", "mean_cos_to_base"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["step"] = int(r["step"])
    return rows

def by_step_ctx(rows, key="rotation"):
    acc = collections.defaultdict(list)
    for r in rows: acc[(r["step"], r["context"])].append(r[key])
    return {k: sum(v) / len(v) for k, v in acc.items()}

def gib(rows):
    """Per step: (mean, min, max) over the 8 traits of the gibberish-control rotation.
    The CSV's per-cell min/mean/max coincide (one gibberish prompt per trait and checkpoint), so the band is the across-trait spread."""
    g = collections.defaultdict(list)
    for r in rows: g[r["step"]].append(r["gibberish_rotation_mean"])
    return {s: (sum(v) / len(v), min(v), max(v)) for s, v in g.items()}

def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def title(fig, text, width):
    fig.suptitle("\n".join(textwrap.wrap(text, width)), x=0.01, ha="left", fontsize=10.5, fontweight="bold", linespacing=1.3)

def repel(ys, gap):
    """Push label y positions apart (sorted order preserved) so adjacent labels are >= gap apart."""
    order = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out

def caption(fig, text, y=0.01):
    fig.text(0.01, y, text, fontsize=7.2, color=MUTED, ha="left", va="bottom", wrap=True)

def draw_context_lines(ax, rows, steps, contexts, faint=(), gibband=True, boot=True, label_end=True):
    m = by_step_ctx(rows); g = gib(rows)
    if gibband:
        ax.fill_between(steps, [g[s][1] for s in steps], [g[s][2] for s in steps], color="#c9c8c3", alpha=0.45, lw=0, label="gibberish band (min–max over 8 traits)")
        ax.plot(steps, [g[s][0] for s in steps], ls="--", color=MUTED, lw=1.3, label="gibberish mean")
    if boot:
        b = by_step_ctx(rows, "bootstrap_rotation"); bb = [sum(b[(s, c)] for c in CTX) / len(CTX) for s in steps]
        ax.plot(steps, bb, ls=":", color=MUTED, lw=1.1, label="bootstrap floor")
    for c in faint:
        ax.plot(steps, [m[(s, c)] for s in steps], color="#b8b7b2", lw=1.0, alpha=0.8, zorder=1)
    ends = []
    for c in contexts:
        ys = [m[(s, c)] for s in steps]
        ax.plot(steps, ys, color=COL[c], lw=2.0, marker="o", ms=4.5, mec="#fcfcfb", mew=1, label=LABEL[c], zorder=3)
        ends.append(ys[-1])
    if label_end:
        span = max(max(m[(s, c)] for s in steps for c in contexts), max(g[s][2] for s in steps)) - min(min(m[(s, c)] for s in steps for c in contexts), 0.015)
        for c, y0, y1 in zip(contexts, ends, repel(ends, 0.035 * span)):
            ax.annotate(LABEL[c], (steps[-1], y0), xytext=(steps[-1] + 22, y1), textcoords="data", fontsize=7.5, color=TXT, va="center",
                        arrowprops=dict(arrowstyle="-", color="#b8b7b2", lw=0.6, shrinkA=0, shrinkB=2) if abs(y1 - y0) > 1e-9 else None)
    ax.set_xticks(steps); ax.set_xlabel("RL step (reward-hacking training)"); ax.set_ylabel("rotation = 1 − cos to null (mean of 8 traits)")

steps = [0, 216, 496, 752, 952]
# ---------- t1 ----------
rows18 = load(18)
fig, ax = plt.subplots(figsize=(9, 5.2)); style(ax)
draw_context_lines(ax, rows18, steps, CTX)
ax.set_xlim(-30, 1120); ax.set_title("gpt-oss-120b, layer 18", loc="left", fontsize=9, color=MUTED)
title(fig, "RL toward reward hacking makes trait directions increasingly conditional on being graded by a script, not on being read by a person", 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=7.5)
caption(fig, BASE_CAP + " Lines: mean rotation over 8 traits per evaluation context. Gibberish control: one nonsense prompt per trait and checkpoint, so the band is the min–max across traits and the dashed line their mean. Bootstrap floor: mean over contexts of the within-cell resampling rotation.")
fig.subplots_adjust(top=0.84, bottom=0.3, left=0.08, right=0.86); fig.savefig(OUT / "t1_rotation_vs_step_L18.png", dpi=300); plt.close(fig)

# ---------- t2 ----------
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
m = collections.defaultdict(dict); g = collections.defaultdict(dict)
for r in rows18: m[r["trait"]][(r["step"], r["context"])] = r["rotation"]; g[r["trait"]][r["step"]] = r["gibberish_rotation_mean"]
for ax, t in zip(axes.flat, TRAITS):
    style(ax)
    for c in [x for x in CTX if x not in ("auto_grader", "human_evaluator", "coding_harness")]:
        ax.plot(steps, [m[t][(s, c)] for s in steps], color="#b8b7b2", lw=0.9, zorder=1)
    ax.plot(steps, [g[t][s] for s in steps], ls="--", color=MUTED, lw=1.2)
    for c in ("auto_grader", "human_evaluator", "coding_harness"):
        ax.plot(steps, [m[t][(s, c)] for s in steps], color=COL[c], lw=1.9, marker="o", ms=3.5, mec="#fcfcfb", mew=0.8, zorder=3)
    ax.set_title(t.replace("_", "-"), loc="left", fontsize=9); ax.set_xticks(steps); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
for ax in axes[1]: ax.set_xlabel("RL step")
for ax in axes[:, 0]: ax.set_ylabel("rotation")
handles = [Line2D([], [], color=COL[c], lw=2, label=LABEL[c]) for c in ("auto_grader", "human_evaluator", "coding_harness")] + \
          [Line2D([], [], color=MUTED, ls="--", lw=1.2, label="gibberish"), Line2D([], [], color="#b8b7b2", lw=1, label="other four contexts")]
fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.99, 0.94))
title(fig, "The grader split widens on all eight traits, most on warmth and empathy; human-evaluator framing stays near the gibberish control throughout", 125)
caption(fig, BASE_CAP + " Layer 18; one panel per trait; per-trait values, no averaging.")
fig.subplots_adjust(top=0.84, bottom=0.2, left=0.06, right=0.99, hspace=0.35, wspace=0.12); fig.savefig(OUT / "t2_rotation_by_trait_L18.png", dpi=300); plt.close(fig)

# ---------- t3 ----------
gate = json.load(open(D / "gate_report.json")); beh = [gate[f"step-{s}"]["behaviour_logodds_hack"] for s in steps]; cosb = [gate[f"step-{s}"]["mean_cos_to_base"] for s in steps]
m18 = by_step_ctx(rows18); ag = [m18[(s, "auto_grader")] for s in steps]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax in axes: style(ax); ax.set_xticks(steps); ax.set_xlabel("RL step")
axes[0].plot(steps, beh, color="#eb6834", lw=2, marker="o", ms=4.5, mec="#fcfcfb"); axes[0].axhline(0, color=MUTED, lw=0.8, ls=":")
axes[0].set_ylabel("log-odds toward the reward-hacking response"); axes[0].set_title("behaviour: preference for the hacking response", loc="left", fontsize=9)
axes[1].plot(steps, cosb, color="#4a3aa7", lw=2, marker="o", ms=4.5, mec="#fcfcfb"); axes[1].set_ylabel("mean cosine of activations to the base model"); axes[1].set_title("geometry: how far activations moved from base", loc="left", fontsize=9)
axes[2].plot(steps, ag, color=COL["auto_grader"], lw=2, marker="o", ms=4.5, mec="#fcfcfb"); axes[2].plot(steps, [m18[(s, "human_evaluator")] for s in steps], color=COL["human_evaluator"], lw=2, marker="o", ms=4.5, mec="#fcfcfb")
axes[2].plot(steps, [gib(rows18)[s][0] for s in steps], ls="--", color=MUTED, lw=1.2); axes[2].set_ylabel("rotation, mean of 8 traits (layer 18)"); axes[2].set_title("conditionality: grader vs human-evaluator framing", loc="left", fontsize=9)
axes[2].legend(handles=[Line2D([], [], color=COL["auto_grader"], lw=2, label="automated grader"), Line2D([], [], color=COL["human_evaluator"], lw=2, label="human evaluator"), Line2D([], [], color=MUTED, ls="--", label="gibberish")], frameon=False, fontsize=7.5, loc="upper left")
title(fig, "Behaviour moves monotonically toward hacking; activations reorganise by step 216 and then consolidate, while the grader split widens from step 216 on", 125)
caption(fig, BASE_CAP + " Middle panel: 32 probe prompts (honesty, null and grader contexts), all 36 layers; a zero adapter gives exactly 1.000.")
fig.subplots_adjust(top=0.8, bottom=0.26, left=0.06, right=0.99, wspace=0.32); fig.savefig(OUT / "t3_behaviour_and_geometry.png", dpi=300); plt.close(fig)

# ---------- t4 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
for ax, L in zip(axes, (12, 18, 24)):
    style(ax); rows = load(L)
    draw_context_lines(ax, rows, steps, ["auto_grader", "human_evaluator", "coding_harness"], faint=[c for c in CTX if c not in ("auto_grader", "human_evaluator", "coding_harness")], boot=False, label_end=False)
    ax.set_title(f"layer {L}", loc="left", fontsize=9); ax.set_xlim(-30, 1000)
axes[0].legend(handles=[Line2D([], [], color=COL[c], lw=2, label=LABEL[c]) for c in ("auto_grader", "human_evaluator", "coding_harness")] + [Line2D([], [], color=MUTED, ls="--", label="gibberish mean"), Line2D([], [], color="#b8b7b2", lw=1, label="other four contexts")], frameon=False, fontsize=7.5, loc="upper left")
for ax in axes[1:]: ax.set_ylabel("")
title(fig, "The widening grader split and the step-752 harness peak appear at layers 18 and 24 but not at layer 12", 125)
caption(fig, BASE_CAP + " Same quantity at layers 12, 18 and 24 of 36; shared y axis. Grey band: gibberish min–max across traits.")
fig.subplots_adjust(top=0.84, bottom=0.26, left=0.06, right=0.99, wspace=0.08); fig.savefig(OUT / "t4_layers_L12_L18_L24.png", dpi=300); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

#!/usr/bin/env python3
"""Deck v4 figures: the spread metric per trait and per context. rotation = 1 - cos(v_ctx, v_null)."""
import json, csv, os, textwrap, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D = "/private/tmp/claude-502/-Users-jacobdavies-Vaults-The-Borderlands-Atlas/2eee3d75-6a40-4549-8670-bdc9196d560e/deck_v3/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
CTX = ["auto_grader","human_evaluator","simulated_env","real_deployment","under_evaluation","unobserved","coding_harness"]
CTXL = {"auto_grader":"automated grader","human_evaluator":"human evaluator","simulated_env":"simulated env.","real_deployment":"real deployment","under_evaluation":"under evaluation","unobserved":"unobserved","coding_harness":"coding harness"}
COL = dict(zip(CTX, ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7"]))
TR = ["warmth","empathy","deference","assertiveness","impulsivity","confidence","honesty","risk_taking"]
TRL = {t:t.replace("_","-") for t in TR}
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size":9,"axes.edgecolor":BAND,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.titleweight":"bold","axes.titlesize":10,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False})
# occupational per-trait cosine means (Ignacio's report, verbatim) in TR order
OCC = {"evalctx_L22_v3ref":[0.641,0.681,0.688,0.694,0.712,0.718,0.748,0.769],
       "Llama-3.1-8B-Instruct_evalctx_L15":[0.703,0.682,0.680,0.684,0.749,0.758,0.757,0.774],
       "Llama-3.1-8B-Instruct_evalctx_L20":[0.496,0.530,0.458,0.520,0.593,0.514,0.664,0.555],
       "Qwen3-32B_evalctx_L42":[0.870,0.873,0.831,0.852,0.824,0.858,0.864,0.853]}
# gibberish bands measured on 4 traits (first 100 q) from the ledger, for runs analysed before the band patch
LEDGER_BAND = {"evalctx_L22_v3ref":{"honesty":[0.962,0.963,0.844,0.960,0.960],"risk_taking":[0.956,0.952,0.833,0.945,0.955],"deference":[0.933,0.934,0.768,0.930,0.931],"warmth":[0.956,0.955,0.842,0.954,0.952]},
  "Qwen3-32B_evalctx_L31":{"honesty":[0.791,0.830,0.801,0.840,0.873],"risk_taking":[0.821,0.867,0.841,0.860,0.895],"deference":[0.695,0.778,0.733,0.801,0.835],"warmth":[0.810,0.837,0.823,0.827,0.861]},
  "Qwen3-32B_evalctx_L42":{"honesty":[0.843,0.887,0.842,0.887,0.896],"risk_taking":[0.855,0.925,0.894,0.927,0.926],"deference":[0.850,0.895,0.879,0.923,0.915],"warmth":[0.929,0.946,0.935,0.947,0.953]},
  "Llama-3.1-8B-Instruct_evalctx_L15":{"honesty":[0.864,0.890,0.844,0.809,0.883],"risk_taking":[0.852,0.868,0.844,0.830,0.865],"deference":[0.708,0.738,0.678,0.666,0.737],"warmth":[0.845,0.861,0.843,0.806,0.863]}}
PANELS = [("evalctx_L22_v3ref","Gemma-2-27B L22, chat template"),("Llama-3.1-8B-Instruct_evalctx_L15","Llama-3.1-8B L15, template"),("Llama-3.1-8B-Instruct_evalctx_L20","Llama-3.1-8B L20, template"),
          ("Qwen3-32B_evalctx_L42","Qwen3-32B L42, template"),("Qwen3-32B-raw_evalctx_L31","Qwen3-32B L31, raw format"),("Qwen3-8B_evalctx_L18","Qwen3-8B L18, template"),
          ("Qwen3-8B-raw_evalctx_L18","Qwen3-8B L18, raw format"),("Qwen3-8B-Base-raw_evalctx_L18","Qwen3-8B-Base L18, raw"),("Qwen3-14B-raw_evalctx_L20","Qwen3-14B L20, raw"),
          ("Qwen3-14B-Base-raw_evalctx_L20","Qwen3-14B-Base L20, raw"),("Qwen2.5-32B-Instruct_evalctx_L42","Qwen2.5-32B L42, template")]
def load(d):
    s = json.load(open(f"{D}/{d}/evalctx_summary.json")); cells = {(c["context"], c["trait"]): c for c in s["cells"]}
    band = {}
    for t, v in s["nonsense"].items():
        b = v.get("nonsense_band"); 
        if b is None and d in LEDGER_BAND and t in LEDGER_BAND[d]: b = LEDGER_BAND[d][t]
        band[t] = {"single": v["nonsense_cos_to_null"], "band": b}
    pb = {(r["context"], r["trait"]): r for r in csv.DictReader(open(f"{D}/{d}/paired_boot.csv"))} if os.path.exists(f"{D}/{d}/paired_boot.csv") else {}
    return cells, band, pb
def rot(c): return 1 - c
def draw_panel(ax, d, title, show_occ=True, legend=False, ylim=None, occ_points=None):
    cells, band, pb = load(d); x = np.arange(len(TR)); w = 0.09
    for ti, t in enumerate(TR):
        b = band.get(t, {}); 
        if b.get("band"):
            lo, hi = rot(max(b["band"])), rot(min(b["band"])); ax.add_patch(plt.Rectangle((ti-0.42, lo), 0.84, hi-lo, color=BAND, alpha=0.45, lw=0, zorder=1))
            ax.plot([ti-0.42, ti+0.42], [rot(np.mean(b["band"]))]*2, color=INK2, lw=1.0, zorder=2)
        elif b.get("single") is not None:
            ax.plot([ti-0.42, ti+0.42], [rot(b["single"])]*2, color=INK2, lw=1.0, ls="--", zorder=2)
        c0 = cells.get((CTX[0], t))
        if c0:
            ax.plot([ti-0.42, ti+0.42], [rot(c0["bootstrap_floor"])]*2, color=MUTED, lw=0.7, ls=":", zorder=2)
            if "paraphrase_floor" in c0: ax.plot([ti-0.42, ti+0.42], [rot(c0["paraphrase_floor"])]*2, color=MUTED, lw=0.7, ls="-.", zorder=2)
        for ci, c in enumerate(CTX):
            cell = cells.get((c, t)); 
            if cell: ax.scatter(ti + (ci-3)*w, rot(cell["cos_to_null"]), s=22, color=COL[c], zorder=4, edgecolor="#fcfcfb", lw=0.5)
        if show_occ and d in OCC:
            if occ_points and t in occ_points: ax.scatter([ti+0.30]*len(occ_points[t]), [rot(v) for v in occ_points[t]], s=8, color="#0b0b0b", alpha=0.18, zorder=3)
            ax.scatter(ti+0.30, rot(OCC[d][ti]), marker="D", s=30, color="#0b0b0b", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([TRL[t] for t in TR], rotation=30, ha="right"); ax.set_title(title, loc="left"); ax.set_ylabel("rotation = 1 − cos(v_ctx, v_null)")
    if ylim: ax.set_ylim(*ylim)
    if legend:
        h = [Line2D([],[],marker="o",ls="",color=COL[c],label=CTXL[c]) for c in CTX] + [Line2D([],[],marker="D",ls="",color="#0b0b0b",label="10 occupational personas (mean; faint = each)"),
             plt.Rectangle((0,0),1,1,color=BAND,alpha=0.45,label="gibberish band (5 prompts); line = mean"), Line2D([],[],color=INK2,ls="--",label="gibberish (single prompt)"), Line2D([],[],color=MUTED,ls=":",label="bootstrap floor"), Line2D([],[],color=MUTED,ls="-.",label="paraphrase floor")]
        ax.legend(handles=h, ncol=3, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28))
# ---------- fig 1
occ_pts = {}
for r in csv.DictReader(open(f"{D}/evalctx_L22_v3ref/axis_cells.csv")):
    if r["context"] not in CTX + ["nonsense"]: occ_pts.setdefault(r["trait"], []).append(float(r["cos_to_null"]))
fig, ax = plt.subplots(figsize=(11, 5.6)); draw_panel(ax, "evalctx_L22_v3ref", "Gemma-2-27B-IT, layer 22", legend=True, occ_points=occ_pts, ylim=(-0.005, 0.72))
fig.suptitle("The coding harness rotates Gemma's confidence direction to persona scale; who-is-watching contexts sit near the floors", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.01, 0.005, "\n".join(textwrap.wrap("Measured: rotation of each trait's CAA direction under a context, 1 − cosine to the no-prompt direction (0 = identical). Persona text in the user message; 500 questions per cell. Floors: bootstrap (50 redraws), paraphrase (5 prompts, 100 q), gibberish band (4 traits from the ledger, 100 q; other traits single prompt). Caveat: floors and band computed on 100-question vectors for the band, 500 for cells.", int(fig.get_figwidth()*17))), fontsize=7.2, color=INK2)
fig.subplots_adjust(top=0.90, bottom=0.36); fig.savefig(f"{OUT}/s1_spread_by_trait_gemma.png", dpi=200); plt.close(fig)
# ---------- fig 2 small multiples
fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), sharey=True); axes = axes.ravel()
for ax, (d, t) in zip(axes, PANELS): draw_panel(ax, d, t, ylim=(-0.01, 0.50)); ax.set_ylabel("")
axes[-1].axis("off"); axes[0].set_ylabel("rotation = 1 − cos"); axes[4].set_ylabel("rotation = 1 − cos"); axes[8].set_ylabel("rotation = 1 − cos")
h = [Line2D([],[],marker="o",ls="",color=COL[c],label=CTXL[c]) for c in CTX] + [Line2D([],[],marker="D",ls="",color="#0b0b0b",label="occupational mean (where run)"), plt.Rectangle((0,0),1,1,color=BAND,alpha=0.45,label="gibberish band"), Line2D([],[],color=INK2,ls="--",label="gibberish single prompt"), Line2D([],[],color=MUTED,ls=":",label="bootstrap floor"), Line2D([],[],color=MUTED,ls="-.",label="paraphrase floor")]
axes[-1].legend(handles=h, loc="center", fontsize=8.5, frameon=False, ncol=1)
fig.suptitle("The chat template breaks Qwen3's gibberish floor; under a fixed raw format the floor is tight on every family", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("Same measure as the Gemma figure, one panel per model/layer/regime; shared y axis. Raw format = 'Context/Question/Answer' text with no chat template on every stage. Caveat: bands for the four pre-band runs (Gemma, Llama L15, Qwen3-32B template L31/L42) cover four traits only; other traits show the single gibberish prompt as a dashed line.", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.93, bottom=0.11, hspace=0.55, wspace=0.08); fig.savefig(f"{OUT}/s2_spread_by_trait_models.png", dpi=170); plt.close(fig)
# ---------- fig 3 per-cell intervals
fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), sharey=True); axes = axes.ravel()
for ax, (d, t) in zip(axes, PANELS):
    cells, band, pb = load(d); w = 0.1
    for ti, tr in enumerate(TR):
        for ci, c in enumerate(CTX):
            r = pb.get((c, tr)); 
            if not r: continue
            x = ti + (ci-3)*w; v = -float(r["ctx_minus_nonsense"]); lo, hi = -float(r["ci_hi"]), -float(r["ci_lo"])
            ax.plot([x, x], [lo, hi], color=COL[c], lw=1.1, alpha=0.9); ax.scatter(x, v, s=14, color=COL[c], zorder=4)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(len(TR))); ax.set_xticklabels([TRL[t] for t in TR], rotation=30, ha="right"); ax.set_title(t, loc="left")
axes[-1].axis("off"); axes[-1].legend(handles=[Line2D([],[],marker="o",ls="",color=COL[c],label=CTXL[c]) for c in CTX], loc="center", fontsize=9, frameon=False)
for i in (0,4,8): axes[i].set_ylabel("extra rotation vs gibberish prompt")
fig.suptitle("Every cell with its interval: context rotation minus the gibberish prompt's rotation (positive = the context rotates more than gibberish)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("Whiskers: 95% paired question-bootstrap intervals, 200 redraws, the same redraw applied to context, gibberish and null vectors. Above zero with the whisker clear of zero = the context reorganises the direction beyond what an arbitrary prompt does. Reference gibberish = the single 24-word prompt.", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.93, bottom=0.11, hspace=0.55, wspace=0.08); fig.savefig(f"{OUT}/s3_context_effect_ci.png", dpi=170); plt.close(fig)
# ---------- fig 4 heatmap: spread over 7 contexts minus gibberish (band mean or single)
rows = []; M = []
for d, t in PANELS:
    cells, band, _ = load(d); row = []
    for tr in TR:
        vals = [rot(cells[(c, tr)]["cos_to_null"]) for c in CTX if (c, tr) in cells]; b = band.get(tr, {})
        g = rot(np.mean(b["band"])) if b.get("band") else rot(b["single"])
        row.append(np.mean(vals) - g)
    rows.append(t); M.append(row)
M = np.array(M); fig, ax = plt.subplots(figsize=(11, 6.2))
vmax = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#1baf7a", "#f0efec", "#2a78d6"]), vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(TR))); ax.set_xticklabels([TRL[t] for t in TR], rotation=30, ha="right"); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=7.5, color=INK)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("spread over 7 evaluation contexts − gibberish rotation")
fig.suptitle("Honesty is the most context-invariant direction on every model; risk-taking and confidence the most sensitive", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("Cell = mean rotation over the seven evaluation contexts minus the gibberish rotation (band mean where five prompts exist, else single prompt). 0 = at the floor; positive = contexts reorganise more than gibberish; negative = gibberish reorganises more (the template regime on Qwen3).", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.90, bottom=0.16, left=0.22); fig.savefig(f"{OUT}/s4_trait_invariance_heatmap.png", dpi=200); plt.close(fig)
# ---------- fig 5 axis
a = json.load(open(f"{D}/evalctx_L22_v3ref/axis_projection.json")); ac = list(csv.DictReader(open(f"{D}/evalctx_L22_v3ref/axis_cells.csv")))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for r in a["contexts"]:
    c = r["context"]; occ = c not in CTX and c != "nonsense"; col = COL.get(c, "#0b0b0b" if occ else "#898781")
    ax1.scatter(r["u_axis_frac"], 1 - r["mean_cos_to_null"], s=40, color=col, marker="D" if occ else "o", zorder=3); ax1.annotate(CTXL.get(c, c.replace("_"," ")), (r["u_axis_frac"], 1 - r["mean_cos_to_null"]), fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK2)
ax1.set_xlabel("fraction of the context's activation shift lying on the assistant axis"); ax1.set_ylabel("mean rotation of trait directions (1 − cos)"); ax1.set_title("Where each context sits (diamonds = occupational personas)", loc="left")
xs = [1 - float(r["cos_to_null"]) for r in ac]; ys = [1 - float(r["cos_to_null_axis_removed"]) for r in ac]; cols = [COL.get(r["context"], "#0b0b0b") for r in ac]
ax2.scatter(xs, ys, s=12, c=cols, alpha=0.8); lim = max(xs + ys) * 1.05; ax2.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--"); ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("rotation, full vectors"); ax2.set_ylabel("rotation after projecting the axis out"); ax2.set_title("144 cells: removing the axis component changes rotation by ≤ 0.006", loc="left")
fig.suptitle("Trait rotation is orthogonal to displacement along the assistant axis", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("Gemma-2-27B-IT, layer 22, Lu et al. assistant axis. Left: occupational personas put 11–32% of their shift on the axis, evaluation contexts 0–2.3%. Right: each (context, trait) cell before vs after removing the axis component from both the context and null vectors.", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.86, bottom=0.17, wspace=0.28); fig.savefig(f"{OUT}/s5_axis.png", dpi=200); plt.close(fig)
# ---------- fig 6 OLMo persona spread (shared variance)
o = json.load(open(f"{D}/olmo_rawfmt_control_L15.json"))["shared_variance"]; stages = ["base","sft","dpo","instruct"]
fig, ax = plt.subplots(figsize=(9, 5))
for reg, col, lab in [("template", "#eb6834", "chat template (original)"), ("raw", "#2a78d6", "raw format held fixed")]:
    keys = ["base"] + [f"{s}_{reg}" for s in stages[1:]]
    for t in TR: ax.plot(range(4), [o[k][t] for k in keys], color=col, alpha=0.25, lw=0.9)
    ax.plot(range(4), [np.mean(list(o[k].values())) for k in keys], color=col, lw=2.4, marker="o", label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(["base","SFT","DPO","Instruct (RLVR)"]); ax.set_ylabel("shared variance across 10 personas (1 = one direction)"); ax.legend(frameon=False)
fig.suptitle("Three quarters of the reported SFT drop was the prompt-format switch; a small monotone decline remains", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("OLMo-2-1124-7B, layer 15. Faint lines: the eight traits; bold: their mean. Template: base 0.968 → SFT 0.848 → DPO 0.834 → Instruct 0.825. Raw: 0.968 → 0.940 → 0.934 → 0.929. Base has no chat template, so the original run switched format at SFT.", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.88, bottom=0.17); fig.savefig(f"{OUT}/s6_olmo_persona_spread.png", dpi=200); plt.close(fig)
# ---------- fig 7 OLMo context spread across stages with paired CIs
st = [("OLMo_base_evalctx_rawfmt_L15","base"),("OLMo_sft_evalctx_rawfmt_L15","SFT"),("OLMo_dpo_evalctx_rawfmt_L15","DPO"),("OLMo_instruct_evalctx_rawfmt_L15","Instruct (RLVR)")]
fig, ax = plt.subplots(figsize=(10, 5.4)); gib = []
series = {c: {"m": [], "lo": [], "hi": []} for c in CTX}
for d, lab in st:
    cells, band, pb = load(d); g = np.mean([rot(band[t]["single"]) for t in TR]); gib.append(g)
    for c in CTX:
        m = np.mean([rot(cells[(c, t)]["cos_to_null"]) for t in TR])
        if pb: lo = np.mean([g + (-float(pb[(c,t)]["ci_hi"])) for t in TR]); hi = np.mean([g + (-float(pb[(c,t)]["ci_lo"])) for t in TR])
        else: lo = hi = m
        series[c]["m"].append(m); series[c]["lo"].append(lo); series[c]["hi"].append(hi)
x = range(4); ax.plot(x, gib, color=INK2, lw=1.4, ls="--", marker="s", label="gibberish prompt")
for c in CTX:
    ax.fill_between(x, series[c]["lo"], series[c]["hi"], color=COL[c], alpha=0.12, lw=0); ax.plot(x, series[c]["m"], color=COL[c], lw=2, marker="o", label=CTXL[c])
ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in st]); ax.set_ylabel("mean rotation over 8 traits (1 − cos)"); ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1))
fig.suptitle("Post-training broadens which contexts reshape trait directions; DPO and RLVR add breadth, not depth", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("OLMo-2-1124-7B, layer 15, raw format on every stage. Lines: mean rotation per context; shaded: mean of the per-trait 95% paired question-bootstrap intervals (200 redraws), anchored to the gibberish prompt's rotation at that stage. The gibberish prompt's own rotation falls across stages (0.050 → 0.039).", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.88, bottom=0.27); fig.savefig(f"{OUT}/s7_olmo_context_spread.png", dpi=200); plt.close(fig)
# ---------- fig 8 Qwen template vs raw
pairs = [("Qwen3-8B_evalctx_L18","Qwen3-8B L18, chat template"),("Qwen3-8B-raw_evalctx_L18","Qwen3-8B L18, raw format"),("Qwen3-32B_evalctx_L31","Qwen3-32B L31, chat template"),("Qwen3-32B-raw_evalctx_L31","Qwen3-32B L31, raw format"),("Qwen3-32B_evalctx_L42","Qwen3-32B L42, chat template"),("Qwen3-32B-raw_evalctx_L42","Qwen3-32B L42, raw format")]
fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey=True); axes = axes.ravel()
for ax, (d, t) in zip(axes, pairs): draw_panel(ax, d, t, show_occ=False, ylim=(-0.01, 0.50)); ax.set_ylabel("")
for i in (0,2,4): axes[i].set_ylabel("rotation = 1 − cos")
axes[1].legend(handles=[Line2D([],[],marker="o",ls="",color=COL[c],label=CTXL[c]) for c in CTX] + [plt.Rectangle((0,0),1,1,color=BAND,alpha=0.45,label="gibberish band")], fontsize=7.5, frameon=False, loc="upper right", ncol=2)
fig.suptitle("The chat template widens Qwen3's gibberish band and swallows the contexts; the raw format restores a tight floor", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.006, "\n".join(textwrap.wrap("Same model and layer, left through its chat template, right as raw text. Template-regime bands on Qwen3-32B cover four traits (ledger); raw-regime bands cover all eight. Under raw format the who-evaluates contexts rotate most at 8B; at 32B only the harness rotates and no context clears the band.", int(fig.get_figwidth()*16))), fontsize=8, color=INK2)
fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.5, wspace=0.06); fig.savefig(f"{OUT}/s8_qwen_template_vs_raw.png", dpi=170); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))

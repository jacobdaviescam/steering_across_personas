"""Final figure set, 15 Sep 2026 (v3). Reads analysis CSV/JSON pulled from the pod into ./data. Writes ./figs/*.png.
Every number comes from the analysis files; nothing is typed in except the four-trait gibberish values for the two
older analysis dirs (Gemma L22, Qwen3-32B L42 template) whose summaries predate the band field, copied from the ledger."""
import csv, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(S, "data"); OUT = os.path.join(S, "figs"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})
TRAITS = ["assertiveness","confidence","deference","empathy","honesty","impulsivity","risk_taking","warmth"]
CTX = ["coding_harness","auto_grader","under_evaluation","real_deployment","simulated_env","human_evaluator","unobserved"]
LBL = {"coding_harness":"coding harness","auto_grader":"automated grader","under_evaluation":"under evaluation","real_deployment":"real deployment",
       "simulated_env":"simulated env.","human_evaluator":"human evaluator","unobserved":"unobserved","nonsense":"nonsense","null":"null"}
PERSONAS = ["farmer","politician","therapist","drill_sergeant","street_hustler","professor","tech_ceo","kindergarten_teacher","surgeon","con_artist"]
CTX_COL = {"coding_harness": ORANGE, "auto_grader": BLUE, "human_evaluator": AQUA, "real_deployment": YELLOW, "simulated_env": MAGENTA, "under_evaluation": "#4a3aa7", "unobserved": "#008300"}

def rows(p): return list(csv.DictReader(open(p)))
def have(d): return os.path.exists(f"{D}/{d}/evalctx_summary.json")
def cells(d):
    r = rows(f"{D}/{d}/evalctx_cells.csv"); s = json.load(open(f"{D}/{d}/evalctx_summary.json"))
    cos = {(x["context"], x["trait"]): float(x["cos_to_null"]) for x in r}
    boot = sum(float(x["bootstrap_floor"]) for x in r)/len(r)
    pf = [float(x["paraphrase_floor"]) for x in r if x.get("paraphrase_floor")]; para = sum(pf)/len(pf) if pf else None
    nz = {t: s["nonsense"][t]["nonsense_cos_to_null"] for t in TRAITS if t in s["nonsense"]}
    band = {t: s["nonsense"][t].get("nonsense_band") for t in TRAITS if t in s["nonsense"]}
    return cos, boot, para, nz, band
def save(fig, name, caption):
    fig.text(0.01, -0.04, caption, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# ---------- fig1: Gemma contexts ----------
def dotplot(ax, d, title, persona_ref=None, ylim=(0.55, 1.0), show_band=False, annotate=True, rot=0):
    cos, boot, para, nz, band = cells(d)
    order = sorted(CTX, key=lambda c: sum(cos[(c,t)] for t in TRAITS)/8)
    xs = list(range(len(order))); labels = [LBL[c].replace(" ", "\n") for c in order]
    if persona_ref is not None: xs.append(len(order)); labels.append("10 occupational\npersonas")
    for i, c in enumerate(order):
        ys = [cos[(c,t)] for t in TRAITS]
        ax.scatter([i]*8, ys, s=16, color=BLUE, alpha=0.65, zorder=3, linewidths=0); ax.hlines(sum(ys)/8, i-0.28, i+0.28, color=BLUE, lw=2.2, zorder=4)
    if persona_ref is not None:
        ax.scatter([len(order)]*len(persona_ref), persona_ref, s=16, color=ORANGE, alpha=0.65, zorder=3, linewidths=0)
        ax.hlines(sum(persona_ref)/len(persona_ref), len(order)-0.28, len(order)+0.28, color=ORANGE, lw=2.2, zorder=4)
    nzm = sum(nz.values())/len(nz)
    if show_band and all(band.values()):
        lo = sum(min(b) for b in band.values())/8; hi = sum(max(b) for b in band.values())/8
        ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0, zorder=1)
        if annotate: ax.annotate("gibberish band (5 prompts)", xy=(1.0, (lo+hi)/2), xycoords=("axes fraction","data"), xytext=(4,0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    else:
        ax.axhline(nzm, color=INK2, ls="--", lw=1)
        if annotate: ax.annotate("nonsense control", xy=(1.0, nzm), xycoords=("axes fraction","data"), xytext=(4,0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.axhline(boot, color=INK2, ls=":", lw=1)
    if annotate: ax.annotate("bootstrap floor", xy=(1.0, boot), xycoords=("axes fraction","data"), xytext=(4,0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    if para:
        ax.axhline(para, color=INK2, ls="-.", lw=1)
        if annotate: ax.annotate("paraphrase floor", xy=(1.0, para), xycoords=("axes fraction","data"), xytext=(4,0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=8 if rot else 9); ax.set_ylim(*ylim); ax.set_ylabel("cosine to null-context trait vector"); ax.set_title(title); ax.grid(axis="x", visible=False)
ax_cells = rows(f"{D}/evalctx_L22_v3ref/axis_cells.csv")
persona_cos = [float(x["cos_to_null"]) for x in ax_cells if x["context"] in PERSONAS]
fig, ax = plt.subplots(figsize=(10.2, 4.6)); dotplot(ax, "evalctx_L22_v3ref", "Coding harness rotates trait directions most; who-is-watching contexts barely move them", persona_cos)
ax.text(0.01, 0.03, "dots = 8 traits, bar = mean; blue = evaluation contexts, orange = occupational personas (same run)", transform=ax.transAxes, fontsize=7.5, color=INK2)
save(fig, "fig1_gemma_contexts.png", "Gemma-2-27B-IT, layer 22, CAA answer-token vectors, persona text in the user message. Measured: cosine between each context's trait vector and the null-context vector; 1 = identical direction. Caveat: nonsense line is one gibberish prompt; the five-prompt band is in fig 5.")

# ---------- fig1c v3: Qwen3-8B template vs raw vs base (the regime story) ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
for ax, (d, name) in zip(axes, [("Qwen3-8B_evalctx_L18","Qwen3-8B, chat template"),("Qwen3-8B-raw_evalctx_L18","Qwen3-8B, raw format"),("Qwen3-8B-Base-raw_evalctx_L18","Qwen3-8B-Base, raw format")]):
    dotplot(ax, d, name, ylim=(0.6, 1.0), show_band=True, annotate=(ax is axes[-1]), rot=35); ax.set_ylabel("")
axes[0].set_ylabel("cosine to null-context trait vector")
fig.suptitle("Qwen3's floor breaks only through its chat template; under a fixed format base and instruct behave alike", x=0.01, ha="left", fontsize=11, fontweight="bold"); fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.12)
save(fig, "fig1c_qwen_regimes.png", "Layer 18 of 36 in all three panels; same seven contexts and controls. Shaded band = the five gibberish prompts (first 100 questions). Under the template the band drops to 0.76-0.94 and swallows every context; under the raw format it sits at 0.94-0.95 and the grader and human-evaluator contexts fall below it on both base and instruct.")

# ---------- fig2: forest across regimes ----------
PANELS = [("evalctx_L22_v3ref","Gemma-2-27B L22\nchat template"),("Llama-3.1-8B-Instruct_evalctx_L15","Llama-3.1-8B L15\nchat template"),("Llama-3.1-8B-Instruct_evalctx_L20","Llama-3.1-8B L20\nchat template"),
          ("Qwen2.5-32B-Instruct_evalctx_L42","Qwen2.5-32B L42\nchat template"),("Qwen3-32B_evalctx_L42","Qwen3-32B L42\nchat template"),("Qwen3-32B-raw_evalctx_L31","Qwen3-32B L31\nraw format"),
          ("Qwen3-8B_evalctx_L18","Qwen3-8B L18\nchat template"),("Qwen3-8B-raw_evalctx_L18","Qwen3-8B L18\nraw format"),("Qwen3-8B-Base-raw_evalctx_L18","Qwen3-8B-Base L18\nraw format"),
          ("Qwen3-14B-raw_evalctx_L20","Qwen3-14B L20\nraw format"),("Qwen3-14B-Base-raw_evalctx_L20","Qwen3-14B-Base L20\nraw format")]
PANELS = [p for p in PANELS if os.path.exists(f"{D}/{p[0]}/paired_boot.csv")]
fig, axes = plt.subplots(1, len(PANELS), figsize=(1.55*len(PANELS)+2.5, 4.6), sharey=True, sharex=True)
for ax, (d, name) in zip(axes, PANELS):
    pb = rows(f"{D}/{d}/paired_boot.csv"); by = defaultdict(list)
    for r in pb: by[r["context"]].append(r)
    for yi, c in enumerate(CTX):
        rs = by[c]; vals = [float(r["ctx_minus_nonsense"]) for r in rs]; lo = min(float(r["ci_lo"]) for r in rs); hi = max(float(r["ci_hi"]) for r in rs)
        below = sum(1 for r in rs if r["clear"] == "below"); col = ORANGE if below >= 6 else (BLUE if below >= 3 else INK2)
        ax.hlines(yi, lo, hi, color=col, lw=1.2, alpha=0.5); ax.scatter(vals, [yi]*8, s=10, color=col, alpha=0.8, linewidths=0, zorder=3)
        ax.scatter([sum(vals)/8], [yi], s=46, marker="|", color=col, lw=2, zorder=4)
        ax.text(0.03, yi+0.28, f"{below}/8", fontsize=6.5, color=col, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.8); ax.set_title(name, fontsize=8.5); ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(CTX))); ax.set_yticklabels([LBL[c] for c in CTX]); ax.invert_yaxis(); ax.set_xlim(-0.3, 0.16); ax.set_xticks([-0.2, 0, 0.1]); ax.tick_params(axis="x", labelsize=7.5)
axes[len(axes)//2].set_xlabel("cosine to null: context minus the single gibberish prompt (negative = context rotates the trait direction more than gibberish)")
fig.suptitle("Under a fixed prompt format the gibberish floor is tight on every family; the chat template is what breaks Qwen3's floor", x=0.01, ha="left", fontsize=11, fontweight="bold"); fig.subplots_adjust(top=0.80)
save(fig, "fig2_paired_deltas.png", "Dots = 8 traits; tick = mean; bar = union of 95% paired question-bootstrap intervals (200 redraws); label = traits whose interval lies entirely below zero. Orange >= 6/8, blue 3-5/8, grey <= 2/8. Panels are labelled by prompt regime; raw format = Context/Question/Answer text with no chat template.")

# ---------- fig3: assistant axis (unchanged) ----------
aj = json.load(open(f"{D}/evalctx_L22_v3ref/axis_projection.json"))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4)); ax = axes[0]
for c in aj["contexts"]:
    grp = "persona" if c["context"] in PERSONAS else ("nonsense" if c["context"]=="nonsense" else "eval"); col = ORANGE if grp=="persona" else (INK2 if grp=="nonsense" else BLUE)
    ax.scatter(c["u_axis_frac"], 1-c["mean_cos_to_null"], s=34, color=col, linewidths=0, zorder=3)
    if grp != "eval" or c["context"] in ("coding_harness", "auto_grader"):
        off = (4, -9) if c["context"] == "tech_ceo" else (4, 3); ax.annotate(LBL.get(c["context"], c["context"].replace("_"," ")), (c["u_axis_frac"], 1-c["mean_cos_to_null"]), fontsize=7, color=INK, xytext=off, textcoords="offset points")
ax.set_xlabel("fraction of the context's activation shift lying on the assistant axis"); ax.set_ylabel("trait rotation  (1 - mean cosine to null)"); ax.set_title("Where each context sits"); ax.set_xlim(-0.01, 0.36); ax.set_ylim(-0.01, 0.56)
ax.text(0.98, 0.03, "orange = occupational personas\nblue = evaluation contexts (unlabelled: the five\nwho-is-watching contexts)\ngrey = nonsense", transform=ax.transAxes, fontsize=7.5, color=INK2, ha="right")
ax = axes[1]; ev = [x for x in ax_cells if x["context"] not in PERSONAS and x["context"]!="nonsense"]; pe = [x for x in ax_cells if x["context"] in PERSONAS]
ax.plot([0.3,1],[0.3,1], color=INK2, lw=0.8)
ax.scatter([float(x["cos_to_null"]) for x in pe], [float(x["cos_to_null_axis_removed"]) for x in pe], s=12, color=ORANGE, alpha=0.7, linewidths=0, zorder=3)
ax.scatter([float(x["cos_to_null"]) for x in ev], [float(x["cos_to_null_axis_removed"]) for x in ev], s=12, color=BLUE, alpha=0.7, linewidths=0, zorder=3)
ax.set_xlabel("cosine to null, original vectors"); ax.set_ylabel("cosine to null, axis component projected out"); ax.set_title("Axis removed: change <= 0.006 in 144 cells"); ax.set_xlim(0.3,1); ax.set_ylim(0.3,1)
fig.suptitle("Trait rotation is orthogonal to assistant-axis displacement", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig3_axis.png", "Gemma-2-27B-IT layer 22; axis from Lu et al. (2026). Shift = mean activation under context minus under null, pooled over both answers. Correlation |axis projection| vs rotation is +0.60 over all cells but is carried by the persona group.")

# ---------- fig4: OLMo stages shared variance (unchanged) ----------
oj = json.load(open(f"{D}/olmo_rawfmt_control_L15.json"))["shared_variance"]; stages = ["base","sft","dpo","instruct"]
def series(kind): return [oj["base" if s=="base" else f"{s}_{kind}"] for s in stages]
fig, ax = plt.subplots(figsize=(8, 4.4))
for kind, col, lab in [("template", ORANGE, "chat template (as reported)"), ("raw", BLUE, "raw format held fixed on every stage")]:
    ser = series(kind); means = [sum(d.values())/8 for d in ser]
    for i, d in enumerate(ser): ax.scatter([i]*8, list(d.values()), s=12, color=col, alpha=0.35, linewidths=0, zorder=2)
    ax.plot(range(4), means, color=col, lw=2.2, marker="o", ms=6, zorder=3, label=lab); ax.annotate(f"{means[-1]:.3f}", (3, means[-1]), xytext=(6,0), textcoords="offset points", fontsize=8, color=col, va="center")
ax.set_xticks(range(4)); ax.set_xticklabels(["base","SFT","DPO","Instruct (RLVR)"]); ax.set_ylabel("shared variance across 10 personas (1 = one direction)"); ax.set_ylim(0.72, 1.0)
ax.set_title("Three quarters of the reported SFT drop was the prompt-format switch"); ax.legend(frameon=False, loc="lower left"); ax.grid(axis="x", visible=False)
save(fig, "fig4_olmo_stages.png", "OLMo-2-1124-7B stages, layer 15, 10 occupational personas x 8 traits (light dots = traits). The base tokenizer has no chat template, so the reported trajectory switched format at SFT; the blue series uses the base's raw Context/Question/Answer format on every stage.")

# ---------- fig5: gibberish band, template vs raw ----------
LEDGER_NV = {"Gemma-2-27B L22\nchat template": {"honesty":[0.962,0.963,0.844,0.960,0.960],"risk_taking":[0.956,0.952,0.833,0.945,0.955],"deference":[0.933,0.934,0.768,0.930,0.931],"warmth":[0.956,0.955,0.842,0.954,0.952]},
             "Qwen3-32B L31\nchat template": {"honesty":[0.791,0.830,0.801,0.840,0.873],"risk_taking":[0.821,0.867,0.841,0.860,0.895],"deference":[0.695,0.778,0.733,0.801,0.835],"warmth":[0.810,0.837,0.823,0.827,0.861]}}
BAND_DIRS = [("Qwen3-32B-raw_evalctx_L31","Qwen3-32B L31\nraw format"),("Qwen3-8B_evalctx_L18","Qwen3-8B L18\nchat template"),("Qwen3-8B-raw_evalctx_L18","Qwen3-8B L18\nraw format"),
             ("Qwen3-8B-Base-raw_evalctx_L18","Qwen3-8B-Base L18\nraw format"),("Qwen2.5-32B-Instruct_evalctx_L42","Qwen2.5-32B L42\nchat template"),("Llama-3.1-8B-Instruct_evalctx_L15","Llama-3.1-8B L15\nchat template")]
panels = [(n, LEDGER_NV[n]) for n in LEDGER_NV]
for d, n in BAND_DIRS:
    if have(d):
        _, _, _, _, band = cells(d)
        if all(band.get(t) for t in TRAITS): panels.append((n, {t: band[t] for t in TRAITS}))
fig, axes = plt.subplots(1, len(panels), figsize=(1.9*len(panels)+1.5, 4.0), sharey=True)
for ax, (m, d) in zip(axes, panels):
    tr = [t for t in ["honesty","risk_taking","deference","warmth"] if t in d]
    for ti, t in enumerate(tr):
        for vi, val in enumerate(d[t]): ax.scatter(ti + (vi-2)*0.12, val, s=24 if vi==0 else 16, color=ORANGE if vi==0 else BLUE, linewidths=0, zorder=3)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels([t.replace("_","-")[:6] for t in tr], fontsize=7.5); ax.set_title(m, fontsize=8.5); ax.grid(axis="x", visible=False); ax.set_ylim(0.5, 1.0)
axes[0].set_ylabel("cosine to null (first 100 questions)")
fig.suptitle("The gibberish floor is a band, and the chat template widens it", x=0.01, ha="left", fontsize=11, fontweight="bold"); fig.subplots_adjust(top=0.78)
save(fig, "fig5_nonsense_band.png", "Orange = the 24-word control prompt; blue = the four alternative gibberish prompts. Same five prompts on every model. Under Qwen3's chat template the band spans 0.5 to 0.9 and one prompt sits far below the rest; under the raw format on the same weights it collapses to about 0.95, the Gemma level. Report the floor as a band and measure it in the same template regime as the contexts.")

# ---------- fig6: OLMo evaluation contexts across stages ----------
ST = [("OLMo_base_evalctx_rawfmt_L15","base"),("OLMo_sft_evalctx_rawfmt_L15","SFT"),("OLMo_dpo_evalctx_rawfmt_L15","DPO"),("OLMo_instruct_evalctx_rawfmt_L15","Instruct (RLVR)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios":[1.15,1]})
ax = axes[0]; means = {c: [] for c in CTX}; nzs = []; counts = {c: [] for c in CTX}; cis = {c: [] for c in CTX}
for d, _ in ST:
    cos, boot, para, nz, band = cells(d); nzm = sum(nz.values())/8; nzs.append(nzm)
    pbp = f"{D}/{d}/paired_boot.csv"; pb = defaultdict(list)
    if os.path.exists(pbp):
        for r in rows(pbp): pb[r["context"]].append(r)
    for c in CTX:
        ys = [cos[(c,t)] for t in TRAITS]; means[c].append(sum(ys)/8); counts[c].append(sum(1 for y in ys if y < nzm))
        if pb.get(c): cis[c].append((sum(float(r["ci_lo"]) for r in pb[c])/8, sum(float(r["ci_hi"]) for r in pb[c])/8))
x = range(4)
ax.plot(list(x), nzs, color=INK2, ls="--", lw=1.4, marker="s", ms=4, label="gibberish control", zorder=2)
for c in CTX:
    ax.plot(list(x), means[c], color=CTX_COL[c], lw=1.8, marker="o", ms=5, label=LBL[c], zorder=3)
    if len(cis[c]) == 4:  # mean of per-trait CI on (ctx - nonsense), shifted to absolute scale
        lo = [nzs[i] + cis[c][i][0] for i in x]; hi = [nzs[i] + cis[c][i][1] for i in x]; ax.fill_between(list(x), lo, hi, color=CTX_COL[c], alpha=0.10, lw=0)
ax.set_xticks(list(x)); ax.set_xticklabels([s for _, s in ST]); ax.set_ylabel("mean cosine to null (8 traits)"); ax.set_title("Where each context sits, by stage"); ax.grid(axis="x", visible=False); ax.set_ylim(0.915, 0.97)
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax = axes[1]; w = 0.11
for i, c in enumerate(CTX): ax.bar([xx + (i-3)*w for xx in x], counts[c], width=w, color=CTX_COL[c], label=LBL[c], linewidth=0)
ax.set_xticks(list(x)); ax.set_xticklabels([s for _, s in ST]); ax.set_ylabel("traits (of 8) below the gibberish control"); ax.set_ylim(0, 8.5); ax.set_title("How many traits each context moves past the control"); ax.grid(axis="x", visible=False)
fig.suptitle("Post-training broadens which contexts reshape trait directions; RL stages add breadth, not depth", x=0.01, ha="left", fontsize=11, fontweight="bold"); fig.subplots_adjust(bottom=0.24, top=0.85)
save(fig, "fig6_olmo_context_stages.png", "OLMo-2-1124-7B, layer 15, raw Context/Question/Answer format on every stage so the stages are comparable. Left: mean cosine to null per evaluation context; shaded = mean 95% paired question-bootstrap interval where available. Right: traits below the single gibberish control. The control itself drifts toward null across stages (0.950 to 0.961).")
print("figures written:", sorted(os.listdir(OUT)))

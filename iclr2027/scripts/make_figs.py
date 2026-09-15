import csv, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(__file__), "deck_data"); OUT = os.path.join(os.path.dirname(__file__), "figs"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})
TRAITS = ["assertiveness","confidence","deference","empathy","honesty","impulsivity","risk_taking","warmth"]
CTX = ["coding_harness","auto_grader","under_evaluation","real_deployment","simulated_env","human_evaluator","unobserved"]
LBL = {"coding_harness":"coding harness","auto_grader":"automated grader","under_evaluation":"under evaluation","real_deployment":"real deployment",
       "simulated_env":"simulated env.","human_evaluator":"human evaluator","unobserved":"unobserved","nonsense":"nonsense","null":"null"}
PERSONAS = ["farmer","politician","therapist","drill_sergeant","street_hustler","professor","tech_ceo","kindergarten_teacher","surgeon","con_artist"]
def rows(p): return list(csv.DictReader(open(p)))
def cells(d):
    r = rows(f"{D}/{d}/evalctx_cells.csv"); s = json.load(open(f"{D}/{d}/evalctx_summary.json"))
    cos = {(x["context"], x["trait"]): float(x["cos_to_null"]) for x in r}
    boot = sum(float(x["bootstrap_floor"]) for x in r)/len(r); para = sum(float(x["paraphrase_floor"]) for x in r if x.get("paraphrase_floor"))/len(r)
    nz = {t: s["nonsense"][t]["nonsense_cos_to_null"] for t in TRAITS}
    return cos, boot, para, nz
def save(fig, name, caption):
    fig.text(0.01, -0.04, caption, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# ---------- fig1: Gemma contexts dot plot (+ occupational reference from axis_cells) ----------
def dotplot(ax, d, title, persona_ref=None, ylim=(0.55, 1.0)):
    cos, boot, para, nz = cells(d)
    order = sorted(CTX, key=lambda c: sum(cos[(c,t)] for t in TRAITS)/8)
    xs = list(range(len(order))); labels = [LBL[c].replace(" ", "\n") for c in order]
    if persona_ref is not None:
        xs.append(len(order)); labels.append("10 occupational\npersonas")
    for i, c in enumerate(order):
        ys = [cos[(c,t)] for t in TRAITS]
        ax.scatter([i]*8, ys, s=16, color=BLUE, alpha=0.65, zorder=3, linewidths=0)
        ax.hlines(sum(ys)/8, i-0.28, i+0.28, color=BLUE, lw=2.2, zorder=4)
    if persona_ref is not None:
        ax.scatter([len(order)]*len(persona_ref), persona_ref, s=16, color=ORANGE, alpha=0.65, zorder=3, linewidths=0)
        ax.hlines(sum(persona_ref)/len(persona_ref), len(order)-0.28, len(order)+0.28, color=ORANGE, lw=2.2, zorder=4)
    nzm = sum(nz.values())/8
    ax.axhline(nzm, color=INK2, ls="--", lw=1); ax.axhline(boot, color=INK2, ls=":", lw=1); ax.axhline(para, color=INK2, ls="-.", lw=1)
    for yv, lab in [(nzm, "nonsense control"), (boot, "bootstrap floor"), (para, "paraphrase floor")]:
        ax.annotate(lab, xy=(1.0, yv), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center", ha="left")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(labels); ax.set_ylim(*ylim); ax.set_ylabel("cosine to null-context trait vector")
    ax.set_title(title); ax.grid(axis="x", visible=False)
ax_cells = rows(f"{D}/evalctx_L22_v3ref/axis_cells.csv")
persona_cos = [float(x["cos_to_null"]) for x in ax_cells if x["context"] in PERSONAS]
fig, ax = plt.subplots(figsize=(10.2, 4.6)); dotplot(ax, "evalctx_L22_v3ref", "Coding harness rotates trait directions most; who-is-watching contexts barely move them", persona_cos)
ax.text(0.01, 0.03, "dots = 8 traits, bar = mean; blue = evaluation contexts, orange = occupational personas (same run)", transform=ax.transAxes, fontsize=7.5, color=INK2)
save(fig, "fig1_gemma_contexts.png", "Gemma-2-27B-IT, layer 22, CAA answer-token vectors, persona text in the user message. Measured: cosine between each context's trait vector and the null-context vector; 1 = identical direction. Caveat: nonsense line is the mean of one gibberish prompt (see fig 5).")

# per-model dotplots for Llama and Qwen
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
dotplot(axes[0], "Llama-3.1-8B-Instruct_evalctx_L15", "Llama-3.1-8B-Instruct, layer 15 (depth-matched)", ylim=(0.4,1.0))
dotplot(axes[1], "Llama-3.1-8B-Instruct_evalctx_L20", "Llama-3.1-8B-Instruct, layer 20", ylim=(0.4,1.0)); axes[1].set_ylabel("")
fig.suptitle("Llama: the automated grader is the context that rotates trait directions", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig1b_llama_contexts.png", "Same extraction and placement as Gemma. Occupational-persona means for scale: 0.723 at L15, 0.541 at L20 (Ignacio's report). Llama's nonsense floor is lower than Gemma's under user-message placement.")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
dotplot(axes[0], "Qwen3-32B_evalctx_L31", "Qwen3-32B, layer 31 (depth-matched)", ylim=(0.6,1.0))
dotplot(axes[1], "Qwen3-32B_evalctx_L42", "Qwen3-32B, layer 42", ylim=(0.6,1.0)); axes[1].set_ylabel("")
fig.suptitle("Qwen: every evaluation context sits at or above the nonsense control", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig1c_qwen_contexts.png", "Occupational-persona mean for scale: 0.853 at L42 (Ignacio's report). Qwen's gibberish control sits low (0.70-0.94) and is length-confounded; its bootstrap and paraphrase floors are lower than Gemma's, so effects of Gemma's size would not be resolvable.")

# ---------- fig2: forest panels ----------
PANELS = [("evalctx_L22_v3ref","Gemma-2-27B L22"),("Llama-3.1-8B-Instruct_evalctx_L15","Llama-3.1-8B L15"),("Llama-3.1-8B-Instruct_evalctx_L20","Llama-3.1-8B L20"),("Qwen3-32B_evalctx_L31","Qwen3-32B L31"),("Qwen3-32B_evalctx_L42","Qwen3-32B L42")]
fig, axes = plt.subplots(1, 5, figsize=(12.5, 4.4), sharey=True, sharex=True)
for ax, (d, name) in zip(axes, PANELS):
    pb = rows(f"{D}/{d}/paired_boot.csv"); by = defaultdict(list)
    for r in pb: by[r["context"]].append(r)
    for yi, c in enumerate(CTX):
        rs = by[c]; vals = [float(r["ctx_minus_nonsense"]) for r in rs]; lo = min(float(r["ci_lo"]) for r in rs); hi = max(float(r["ci_hi"]) for r in rs)
        below = sum(1 for r in rs if r["clear"] == "below")
        col = ORANGE if below >= 6 else (BLUE if below >= 3 else INK2)
        ax.hlines(yi, lo, hi, color=col, lw=1.2, alpha=0.5)
        ax.scatter(vals, [yi]*8, s=12, color=col, alpha=0.8, linewidths=0, zorder=3)
        ax.scatter([sum(vals)/8], [yi], s=46, marker="|", color=col, lw=2, zorder=4)
        ax.text(0.03, yi+0.28, f"{below}/8", fontsize=7, color=col, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.8); ax.set_title(name, fontsize=9.5); ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(CTX))); ax.set_yticklabels([LBL[c] for c in CTX]); ax.invert_yaxis(); ax.set_xlim(-0.3, 0.12)
for ax in axes: ax.set_xlabel("")
axes[2].set_xlabel("cosine to null: context minus nonsense (negative = context rotates the trait direction more than gibberish)")
fig.suptitle("Automated grader splits Llama, coding harness splits Gemma, nothing resolves on Qwen", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig2_paired_deltas.png", "Dots = 8 traits; tick = mean; bar = union of 95% paired question-bootstrap intervals (200 redraws); label = traits whose interval lies entirely below zero. Orange >= 6/8, blue 3-5/8, grey <= 2/8.")

# ---------- fig3: assistant axis ----------
aj = json.load(open(f"{D}/evalctx_L22_v3ref/axis_projection.json"))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axes[0]
for c in aj["contexts"]:
    grp = "persona" if c["context"] in PERSONAS else ("nonsense" if c["context"]=="nonsense" else "eval")
    col = ORANGE if grp=="persona" else (INK2 if grp=="nonsense" else BLUE)
    ax.scatter(c["u_axis_frac"], 1-c["mean_cos_to_null"], s=34, color=col, linewidths=0, zorder=3)
    if grp != "eval" or c["context"] in ("coding_harness", "auto_grader"):
        off = (4, -9) if c["context"] == "tech_ceo" else (4, 3)
        ax.annotate(LBL.get(c["context"], c["context"].replace("_"," ")), (c["u_axis_frac"], 1-c["mean_cos_to_null"]), fontsize=7, color=INK, xytext=off, textcoords="offset points")
ax.set_xlabel("fraction of the context's activation shift lying on the assistant axis"); ax.set_ylabel("trait rotation  (1 - mean cosine to null)")
ax.set_title("Where each context sits"); ax.set_xlim(-0.01, 0.36); ax.set_ylim(-0.01, 0.56)
ax.text(0.98, 0.03, "orange = occupational personas\nblue = evaluation contexts (unlabelled: the five\nwho-is-watching contexts)\ngrey = nonsense", transform=ax.transAxes, fontsize=7.5, color=INK2, ha="right")
ax = axes[1]
ev = [x for x in ax_cells if x["context"] not in PERSONAS and x["context"]!="nonsense"]; pe = [x for x in ax_cells if x["context"] in PERSONAS]
ax.plot([0.3,1],[0.3,1], color=INK2, lw=0.8)
ax.scatter([float(x["cos_to_null"]) for x in pe], [float(x["cos_to_null_axis_removed"]) for x in pe], s=12, color=ORANGE, alpha=0.7, linewidths=0, zorder=3)
ax.scatter([float(x["cos_to_null"]) for x in ev], [float(x["cos_to_null_axis_removed"]) for x in ev], s=12, color=BLUE, alpha=0.7, linewidths=0, zorder=3)
ax.set_xlabel("cosine to null, original vectors"); ax.set_ylabel("cosine to null, axis component projected out"); ax.set_title("Axis removed: change <= 0.006 in 144 cells")
ax.set_xlim(0.3,1); ax.set_ylim(0.3,1)
fig.suptitle("Trait rotation is orthogonal to assistant-axis displacement", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig3_axis.png", "Gemma-2-27B-IT layer 22; axis from Lu et al. (2026). Shift = mean activation under context minus under null, pooled over both answers. Correlation |axis projection| vs rotation is +0.60 over all cells but is carried by the persona group.")

# ---------- fig4: OLMo stages ----------
oj = json.load(open(f"{D}/olmo_rawfmt_control_L15.json"))["shared_variance"]
stages = ["base","sft","dpo","instruct"]; x = range(4)
def series(kind):
    out = []
    for s in stages:
        k = "base" if s=="base" else f"{s}_{kind}"; out.append(oj[k])
    return out
fig, ax = plt.subplots(figsize=(8, 4.4))
for kind, col, lab in [("template", ORANGE, "chat template (as reported)"), ("raw", BLUE, "raw format held fixed on every stage")]:
    ser = series(kind); means = [sum(d.values())/8 for d in ser]
    for i, d in enumerate(ser): ax.scatter([i]*8, list(d.values()), s=12, color=col, alpha=0.35, linewidths=0, zorder=2)
    ax.plot(list(x), means, color=col, lw=2.2, marker="o", ms=6, zorder=3, label=lab)
    ax.annotate(f"{means[-1]:.3f}", (3, means[-1]), xytext=(6,0), textcoords="offset points", fontsize=8, color=col, va="center")
ax.set_xticks(list(x)); ax.set_xticklabels(["base","SFT","DPO","Instruct (RLVR)"]); ax.set_ylabel("shared variance across 10 personas (1 = one direction)"); ax.set_ylim(0.72, 1.0)
ax.set_title("Three quarters of the reported SFT drop was the prompt-format switch"); ax.legend(frameon=False, loc="lower left"); ax.grid(axis="x", visible=False)
save(fig, "fig4_olmo_stages.png", "OLMo-2-1124-7B stages, layer 15, 10 occupational personas x 8 traits (light dots = traits). The base tokenizer has no chat template, so the reported trajectory switched format at SFT; the blue series uses the base's raw Context/Question/Answer format on every stage.")

# ---------- fig5: nonsense variants ----------
NV = {"Gemma-2-27B L22": {"honesty":[0.962,0.963,0.844,0.960,0.960],"risk_taking":[0.956,0.952,0.833,0.945,0.955],"deference":[0.933,0.934,0.768,0.930,0.931],"warmth":[0.956,0.955,0.842,0.954,0.952]},
      "Qwen3-32B L31": {"honesty":[0.791,0.830,0.801,0.840,0.873],"risk_taking":[0.821,0.867,0.841,0.860,0.895],"deference":[0.695,0.778,0.733,0.801,0.835],"warmth":[0.810,0.837,0.823,0.827,0.861]},
      "Llama-3.1-8B L15": {"honesty":[0.864,0.890,0.844,0.809,0.883],"risk_taking":[0.852,0.868,0.844,0.830,0.865],"deference":[0.708,0.738,0.678,0.666,0.737],"warmth":[0.845,0.861,0.843,0.806,0.863]}}
fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
for ax, (m, d) in zip(axes, NV.items()):
    for ti, t in enumerate(["honesty","risk_taking","deference","warmth"]):
        v = d[t]
        for vi, val in enumerate(v):
            ax.scatter(ti + (vi-2)*0.12, val, s=26 if vi==0 else 18, color=ORANGE if vi==0 else BLUE, linewidths=0, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["honesty","risk-taking","deference","warmth"]); ax.set_title(m, fontsize=9.5); ax.grid(axis="x", visible=False); ax.set_ylim(0.6, 1.0)
axes[0].set_ylabel("cosine to null (first 100 questions)"); axes[0].text(0.02, 0.04, "orange = the 24-word prompt used as the control\nblue = the four alternative gibberish prompts", transform=axes[0].transAxes, fontsize=7.5, color=INK2)
fig.suptitle("The nonsense floor depends on which gibberish prompt you pick", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig5_nonsense_variants.png", "On Gemma one alternative prompt rotates directions as much as a persona; on Qwen the long control string sits lowest and shorter ones higher (token-length confound). Recommendation: floor = mean over several gibberish prompts.")
print("figures written:", sorted(os.listdir(OUT)))

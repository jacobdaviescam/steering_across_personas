#!/usr/bin/env python3
"""Deck v4: results around the spread metric. Structure per the empirical-research-slides advice."""
import os, yaml
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
H = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(H, "figs")
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BL = prs.slide_layouts[6]
INK = RGBColor(0x0b,0x0b,0x0b); INK2 = RGBColor(0x52,0x51,0x4e)
def tb(slide, x, y, w, h, text, size=14, bold=False, color=INK, bullets=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = ("• " + line) if bullets else line
        for r in p.runs: r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return box
def title(slide, text): tb(slide, 0.4, 0.25, 12.5, 0.9, text, 22, True)
def fig_slide(claim, png, caption):
    s = prs.slides.add_slide(BL); title(s, claim)
    s.shapes.add_picture(os.path.join(F, png), Inches(0.5), Inches(1.15), width=Inches(12.3)) if png else None
    pic = s.shapes[-1]
    if pic.height > Inches(5.6): pic.height, pic.width = Inches(5.6), int(pic.width * Inches(5.6) / pic.height)
    tb(s, 0.5, 6.85, 12.3, 0.6, caption, 10, color=INK2)
def text_slide(claim, lines, size=15):
    s = prs.slides.add_slide(BL); title(s, claim); tb(s, 0.6, 1.3, 12.1, 5.9, lines, size, bullets=True)
# 1 takeaways
text_slide("Takeaways: trait directions are conditional on the situation the model believes it is in, and post-training broadens it", [
 "Metric: rotation of a trait's contrastive direction under a context, 1 − cos(v_ctx, v_null); spread = mean rotation over a context set. Three floors: bootstrap, paraphrase, gibberish band. The chat template is itself a context.",
 "Gemma-2-27B: the coding harness rotates the confidence direction to occupational-persona scale (0.30 vs persona mean 0.28); who-is-watching contexts sit within 0.01–0.07 of the floors; 'unobserved' equals gibberish.",
 "The rotation is orthogonal to the assistant axis: removing the axis component changes rotation by ≤ 0.006 in all 144 cells; evaluation contexts put ≤ 2.3% of their shift on the axis.",
 "Across families the sensitive contexts differ (harness on Gemma; automated grader on Llama and Qwen3-8B/14B), but honesty is the most invariant direction everywhere and risk-taking and confidence the most sensitive.",
 "Qwen3's 'anomaly' is the chat template: under raw format the gibberish band is tight (0.94–0.95) and the model matches its base checkpoint; through the template the band spans 0.5–0.9 and swallows every context.",
 "OLMo-2 stages, format held fixed: persona conditionality rises 0.968 → 0.929 (not 0.825; the rest was the format switch); evaluation-context rotation exists at base, sharpens at SFT, and DPO/RLVR broaden it to more contexts while the gibberish prompt drifts toward null.",
 "Reward-hacking trajectory (gpt-oss-120b, Tinker adapters): feasible; adapters must be applied functionally inside the expert forward on a dequantised base (2× B200); code written, three validation gates defined, awaiting a pod."], 13)
# 2 agenda
text_slide("Agenda (7 result slides, then limitations and next steps)", [
 "Metric and floors (1 slide)", "Gemma: spread per trait and context (1)", "Cross-model small multiples and per-cell intervals (2)", "Trait invariance (1)", "Assistant axis (1)",
 "OLMo stages: persona spread and context spread (2)", "Qwen3 template vs raw (1)", "Limitations, next steps, questions (3)", "Appendix: prompts, extraction, code-drift checks"])
# 3 metric
text_slide("The metric: rotation of a trait direction under context, read against three floors", [
 "Trait vector under context c: v_{T,c} = mean(pos activations) − mean(neg activations) at the CAA answer token, 500 A/B questions, persona text in the user message.",
 "Rotation(T, c) = 1 − cos(v_{T,c}, v_{T,null}); 0 = identical direction. Spread(T, S) = mean over contexts c ∈ S of Rotation(T, c).",
 "Bootstrap floor: 1 − mean pairwise cosine over 50 question redraws (≈ 0.005–0.02). Paraphrase floor: rotation between five rewordings of the same context (≈ 0.02–0.05). Gibberish band: rotation of five length-matched nonsense prompts; a context counts only if it clears the band in the same template regime.",
 "Intervals: 95% paired question-bootstrap, 200 redraws, the same redraw applied to context, gibberish and null vectors (whiskers on the per-cell slide).",
 "Rule learned twice: the chat template is a context. It moved OLMo's SFT reading by 0.09 and widened Qwen3's gibberish band four-fold. Floors and contexts are compared within one regime only."], 14)
# 4-11 figures
fig_slide("Gemma-2-27B: the coding harness rotates confidence to persona scale; who-is-watching contexts sit near the floors", "s1_spread_by_trait_gemma.png",
 "Measured: rotation per trait (x) and context (colour), layer 22. Diamonds: mean over the ten occupational personas, faint points each persona. Caveat: gibberish band from four traits (100 questions); other traits show the single prompt.")
fig_slide("Under a fixed raw format the gibberish floor is tight on every family; the chat template is what breaks Qwen3's", "s2_spread_by_trait_models.png",
 "Same measure, one panel per model, layer and regime; shared y axis. Occupational means shown where that grid exists. Caveat: template-regime bands for Gemma, Llama L15 and Qwen3-32B cover four traits.")
fig_slide("Every cell with its interval: which contexts rotate a trait beyond what an arbitrary prompt does", "s3_context_effect_ci.png",
 "Dot: context rotation minus the gibberish prompt's rotation; whisker: 95% paired question-bootstrap interval (200 redraws). Positive and clear of zero = the context reorganises the direction beyond gibberish. Reference is the single 24-word gibberish prompt.")
fig_slide("Honesty is the most context-invariant direction on every model; risk-taking and confidence the most sensitive", "s4_trait_invariance_heatmap.png",
 "Cell: spread over the seven evaluation contexts minus the gibberish rotation (band mean where five prompts exist). Zero is the floor; negative cells are the template regime on Qwen3, where gibberish rotates more than any context.")
fig_slide("Trait rotation is orthogonal to displacement along the assistant axis", "s5_axis.png",
 "Gemma layer 22, Lu et al. axis. Left: fraction of each context's activation shift on the axis against its mean rotation. Right: rotation before and after projecting the axis out of both vectors, 144 cells on the diagonal.")
fig_slide("OLMo-2 stages: three quarters of the reported SFT drop was the prompt-format switch; a small monotone decline remains", "s6_olmo_persona_spread.png",
 "Shared variance of the eight trait directions across ten occupational personas, per stage, layer 15. Orange: the original run, which switched from raw text to the chat template at SFT. Blue: raw format on every stage.")
fig_slide("OLMo-2 stages: post-training broadens which contexts reshape trait directions; DPO and RLVR add breadth, not depth", "s7_olmo_context_spread.png",
 "Mean rotation per evaluation context across base, SFT, DPO and Instruct (RLVR), raw format on every stage. Shaded: mean of per-trait 95% paired intervals. The gibberish prompt's own rotation falls from 0.050 to 0.039.")
fig_slide("Qwen3: the chat template widens the gibberish band and swallows the contexts; raw format restores a tight floor", "s8_qwen_template_vs_raw.png",
 "Same model and layer, left through the chat template, right as raw text. Under raw format the who-evaluates contexts rotate most at 8B; at 32B only the harness rotates and no context clears the band.")
# limitations, next, questions
text_slide("Limitations", [
 "Forced-choice answer-token activations only; no free-form behaviour and no link from rotation to misaligned output yet.",
 "Outside the harness on Gemma and the grader on Llama, evaluation-context rotations are 0.02–0.08, a fifth of an occupational persona's.",
 "One template regime per model in the main grids; the raw-format regime exists for Qwen3 and OLMo only.",
 "Gibberish band covers all eight traits only for runs analysed after 15 Sep; four earlier runs have it on four traits.",
 "Pairwise context-by-context dispersion matrices and depth profiles at every layer are computable from stored means but not yet drawn (pod stopped)."])
text_slide("Next steps and resource request", [
 "Reward-hacking trajectory on gpt-oss-120b: run the base dequantised on 2× B200 (or 2× H200), apply the Tinker adapters functionally, pass three gates (step-0 unchanged; graded drift across steps; forced-choice shift on 60 School of Reward Hacks items), then five checkpoints × nine contexts × eight traits. One GPU day.",
 "Upload the thinned activations to the private Hugging Face dataset (any pod, one hour).",
 "CPU: depth profiles at every layer from stored means; stage-versus-stage paired intervals for OLMo; assistant-axis projection on Qwen3-32B and Llama with the released axes.",
 "Writing: paper due 25 Sep; abstract (Rhea) due 18 Sep."])
text_slide("Questions for the group", [
 "Is 'the chat template is a context' a result to foreground, or a method rule in the setup section?",
 "Which regime is canonical for the cross-model table: the model's own template, or raw text on every model?",
 "Do we report the harness effect as the headline for Gemma when it is absent at Qwen3-32B and small on Llama?",
 "Should the OLMo breadth claim wait for stage-versus-stage intervals, or is the per-stage band comparison enough?"])
# appendix: prompts
P = "/Users/jacobdavies/persona-steering/data/personas"
for slug in ["auto_grader","human_evaluator","simulated_env","real_deployment","under_evaluation","unobserved","coding_harness"]:
    d = yaml.safe_load(open(f"{P}/{slug}.yaml")); v = d["system_prompt_variants"]
    s = prs.slides.add_slide(BL); title(s, f"Appendix: context prompt, {slug.replace('_',' ')} (paraphrase 0 of 5)")
    tb(s, 0.6, 1.3, 12.1, 2.0, v[0].strip(), 14)
    tb(s, 0.6, 3.5, 12.1, 3.5, ["Paraphrase %d: %s" % (i+1, x.strip()) for i, x in enumerate(v[1:])], 10, color=INK2)
text_slide("Appendix: extraction details", [
 "CAA: 500 A/B questions per trait (499 for empathy); the correct letter appended as the assistant turn; one forward pass; activation read at that token; vector = mean(pos) − mean(neg).",
 "Persona text prepended to the user message on every model (Gemma 2 has no system role); Qwen3 thinking off; raw format = 'Context: … / Question: … / Answer: X' with no chat template.",
 "Main vector: paraphrase 0 × 500 questions (same schema as the published Gemma grid, which is 1 × 500, not 5 × 100 as its README states). Companion: paraphrases 1–4 × first 100 questions.",
 "Stored per question at analysis layers only (layers.json sidecar) plus all-layer per-cell means; OLMo stages at layers 8/12/15/20.",
 "Layers: Gemma 22/46; Qwen3-32B 31 (depth-matched) and 42; Llama 15 and 20; Qwen3-8B 18; Qwen3-14B 20; Qwen2.5-32B 31/42; OLMo 15/32."], 13)
text_slide("Appendix: code-drift and consistency checks", [
 "Same-run null vs the published April null on Gemma: cells agree to three decimals.",
 "Ignacio's Qwen port vs the main pipeline: cosine 1.0000 and identical norm on therapist/honesty, layer 42.",
 "Ignacio's Gemma rerun through his port reproduces the published vectors to cosine ≥ 0.9999.",
 "Step-0 reward-hacking adapter (untrained) leaves gpt-oss activations unchanged, cosine 1.000: the built-in control for the trajectory.",
 "Data: results branch iclr2027/results (tables, figures, scripts); private HF dataset pending upload; vault ledger holds every number."], 13)
out = os.path.join(H, "iclr2027_results_2026-09-17_v4.pptx"); prs.save(out); print("saved", out, len(prs.slides), "slides")

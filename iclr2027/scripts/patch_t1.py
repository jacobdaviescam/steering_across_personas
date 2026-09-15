import pathlib
p = pathlib.Path("pipeline/t1_trajectory_activations.py"); s = p.read_text()
if "--force-raw-format" in s:
    print("already patched"); raise SystemExit
s = s.replace('''def output_dir_for_stage(spec: CheckpointSpec) -> Path:
    base_short = model_short_name(spec.model.hf_id)
    return OUTPUTS_DIR / base_short / spec.stage_label / "caa_activations"''',
'''def output_dir_for_stage(spec: CheckpointSpec, suffix: str = "") -> Path:
    base_short = model_short_name(spec.model.hf_id)
    return OUTPUTS_DIR / base_short / f"{spec.stage_label}{suffix}" / "caa_activations"''')
s = s.replace('''    parser.add_argument(
        "--dry-run", action="store_true",''','''    parser.add_argument(
        "--force-raw-format", action="store_true",
        help="Use the raw 'Context/Question/Answer' prompt format on every stage, even when "
             "the tokenizer has a chat template (control for the template switch at SFT)",
    )
    parser.add_argument(
        "--dir-suffix", type=str, default="",
        help="Suffix appended to each stage's output directory name (e.g. _rawfmt)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",''')
s = s.replace('''    output_dir = output_dir_for_stage(spec)
''','''    output_dir = output_dir_for_stage(spec, args.dir_suffix)
''')
s = s.replace('''    has_template = _has_chat_template(pm.tokenizer)
    log.info("[%s] Chat template: %s", spec.stage_label, "yes" if has_template else "no (using raw prompts)")''',
'''    has_template = _has_chat_template(pm.tokenizer) and not args.force_raw_format
    log.info("[%s] Chat template: %s", spec.stage_label,
             "yes" if has_template else ("forced raw prompts" if args.force_raw_format else "no (using raw prompts)"))''')
assert s.count("args.dir_suffix") == 1 and "not args.force_raw_format" in s
p.write_text(s); print("t1 patched")

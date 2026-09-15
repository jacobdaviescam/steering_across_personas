import pathlib
p = pathlib.Path("pipeline/2c_caa_activations.py"); s = p.read_text()
s = s.replace('''    parser.add_argument(
        "--dry-run", action="store_true",''','''    parser.add_argument(
        "--paraphrase-variants", type=int, default=0,
        help="Also extract system-prompt paraphrase variants 1..N on the first "
             "--paraphrase-questions questions, saved to <stem>.variants.pt (default: 0 = off)",
    )
    parser.add_argument(
        "--paraphrase-questions", type=int, default=100,
        help="Questions per paraphrase variant for the companion file (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",''')
s = s.replace("import argparse\nimport sys", "import argparse\nimport dataclasses\nimport sys")
old = '''        if activations:
            torch.save(activations, output_path)
            log.info("Saved %d activations to %s", len(activations), output_path.name)
        else:
            log.warning("No activations extracted for %s/%s/%s", persona_slug, trait.value, direction)
'''
new = old + '''
        # Paraphrase floor: variants 1..N on the first K questions -> companion file
        if args.paraphrase_variants > 0 and len(persona.system_prompt_variants) > 1:
            var_path = output_path.with_name(output_path.stem + ".variants.pt")
            if not var_path.exists():
                k = args.paraphrase_questions
                sub = dataclasses.replace(dataset, questions=dataset.questions[:k])
                var_acts = {}
                n_var = min(args.paraphrase_variants, len(persona.system_prompt_variants) - 1)
                for vi in range(1, n_var + 1):
                    acts_v = extract_caa_activations(
                        pm=pm,
                        persona_system_prompt=persona.system_prompt_variants[vi],
                        dataset=sub,
                        direction=direction,
                        batch_size=args.batch_size,
                    )
                    for key, t in acts_v.items():
                        var_acts[f"v{vi}_{key}"] = t
                torch.save(var_acts, var_path)
                log.info("Saved %d paraphrase activations to %s", len(var_acts), var_path.name)
'''
assert old in s, "anchor not found"
s = s.replace(old, new)
p.write_text(s)
print("patched")

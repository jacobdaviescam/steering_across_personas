import pathlib
p = pathlib.Path("pipeline/2c_caa_activations.py"); s = p.read_text()
if "--raw-format" in s: print("already patched"); raise SystemExit
s = s.replace('''    parser.add_argument(
        "--dry-run", action="store_true",''','''    parser.add_argument(
        "--raw-format", action="store_true",
        help="Bypass the chat template: 'Context: <persona>\\\\n\\\\nQuestion: <q>\\\\n\\\\nAnswer: <letter>' as raw text "
             "(for base models, and for base-vs-instruct comparisons with the format held fixed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",''')
s = s.replace('''    persona_placement: str = "auto",
) -> dict[str, torch.Tensor]:''','''    persona_placement: str = "auto",
    raw_format: bool = False,
) -> dict[str, torch.Tensor]:''')
old = '''        answer_pos = find_answer_token_position(
            tokenizer, conv_no_assistant, conv_with_assistant, answer_letter,
        )

        # Tokenize the full conversation for the forward pass
        full_text = _apply_template(tokenizer, conv_with_assistant, False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
'''
new = '''        if raw_format:
            parts = []
            if persona_system_prompt:
                parts.append(f"Context: {persona_system_prompt}\\n")
            parts.append(f"Question: {user_msg}\\n")
            parts.append(f"Answer: {answer_letter}")
            full_text = "\\n".join(parts)
            full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
            answer_pos = len(full_ids) - 1
            if answer_letter not in tokenizer.decode([full_ids[answer_pos]]):
                log.warning("raw-format: last token %r does not contain %s", tokenizer.decode([full_ids[answer_pos]]), answer_letter)
        else:
            answer_pos = find_answer_token_position(
                tokenizer, conv_no_assistant, conv_with_assistant, answer_letter,
            )
            full_text = _apply_template(tokenizer, conv_with_assistant, False)
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
'''
assert old in s, "anchor missing"; s = s.replace(old, new)
s = s.replace('''            persona_placement=args.persona_placement,
        )
''','''            persona_placement=args.persona_placement,
            raw_format=args.raw_format,
        )
''')
s = s.replace('''                        persona_placement=args.persona_placement,
                    )
''','''                        persona_placement=args.persona_placement,
                        raw_format=args.raw_format,
                    )
''')
assert s.count("raw_format=args.raw_format") == 2
p.write_text(s); print("raw-format patched")

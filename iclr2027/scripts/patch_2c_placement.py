import pathlib
p = pathlib.Path("pipeline/2c_caa_activations.py"); s = p.read_text()
if "--persona-placement" in s:
    print("already patched"); raise SystemExit
s = s.replace('''    parser.add_argument(
        "--dry-run", action="store_true",''','''    parser.add_argument(
        "--persona-placement", choices=["auto", "user"], default="auto",
        help="'auto' uses the system field when the chat template supports it; "
             "'user' always prepends the persona to the user message (Gemma-2 style, "
             "use for cross-model comparability)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",''')
# thread placement into extract_caa_activations
s = s.replace('''    direction: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:''','''    direction: str,
    batch_size: int,
    persona_placement: str = "auto",
) -> dict[str, torch.Tensor]:''')
s = s.replace('''    supports_system = pm.supports_system_prompt()
''','''    supports_system = pm.supports_system_prompt() and persona_placement == "auto"
''')
# pass through at both call sites
s = s.replace('''            direction=direction,
            batch_size=args.batch_size,
        )
''','''            direction=direction,
            batch_size=args.batch_size,
            persona_placement=args.persona_placement,
        )
''')
s = s.replace('''                        direction=direction,
                        batch_size=args.batch_size,
                    )
''','''                        direction=direction,
                        batch_size=args.batch_size,
                        persona_placement=args.persona_placement,
                    )
''')
# Qwen3: disable thinking in the chat template where supported
s = s.replace('''def find_answer_token_position(''','''def _apply_template(tokenizer, conversation, add_generation_prompt):
    """apply_chat_template with thinking disabled where the template supports it (Qwen3)."""
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def find_answer_token_position(''')
s = s.replace('''    prefix_text = tokenizer.apply_chat_template(
        conversation_without_assistant,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        conversation_with_assistant,
        tokenize=False,
        add_generation_prompt=False,
    )''','''    prefix_text = _apply_template(tokenizer, conversation_without_assistant, True)
    full_text = _apply_template(tokenizer, conversation_with_assistant, False)''')
s = s.replace('''        full_text = tokenizer.apply_chat_template(
            conv_with_assistant, tokenize=False, add_generation_prompt=False,
        )''','''        full_text = _apply_template(tokenizer, conv_with_assistant, False)''')
assert s.count("persona_placement=args.persona_placement") == 2, "call sites not patched"
assert "_apply_template(tokenizer, conv_with_assistant, False)" in s
p.write_text(s); print("patched placement + thinking-off")

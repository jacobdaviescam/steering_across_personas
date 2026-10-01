"""Add --save-layers to 2c and t1: store per-question activations only at the listed layers, with a layers.json
sidecar in the output dir; also store the per-cell all-layer mean pos/neg in <stem>.means.pt so vectors at
every layer remain computable."""
import pathlib, re
for fn, is_t1 in [("pipeline/2c_caa_activations.py", False), ("pipeline/t1_trajectory_activations.py", True)]:
    p = pathlib.Path(fn); s = p.read_text()
    if "--save-layers" in s: print(fn, "already patched"); continue
    s = s.replace('''    parser.add_argument(
        "--dry-run", action="store_true",''','''    parser.add_argument(
        "--save-layers", type=str, default="",
        help="Comma-separated layer indices to keep per question (default: all). A layers.json sidecar is written; "
             "per-cell all-layer mean pos/neg activations are stored alongside in <stem>.means.pt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",''')
    # helper inserted before main()
    helper = '''
def _save_with_layers(results: dict, output_path, save_layers: list[int] | None) -> None:
    """Save per-question activations (optionally restricted to save_layers) plus all-layer mean and count."""
    import json
    if results:
        stack = torch.stack(list(results.values())).float()          # (n, L, d)
        torch.save({"mean": stack.mean(0).half(), "n": stack.shape[0]}, output_path.with_name(output_path.stem + ".means.pt"))
    if save_layers:
        results = {k: v[save_layers].clone() for k, v in results.items()}
        sidecar = output_path.parent / "layers.json"
        if not sidecar.exists():
            sidecar.write_text(json.dumps({"layers": save_layers}))
    torch.save(results, output_path)


def main() -> None:'''
    s = s.replace("\ndef main() -> None:", helper, 1)
    # parse arg into list right after args parsed
    s = s.replace("    args = parse_args()\n", "    args = parse_args()\n    save_layers = [int(x) for x in args.save_layers.split(',') if x.strip()] or None\n", 1)
    if not is_t1:
        s = s.replace('''        if activations:
            torch.save(activations, output_path)
            log.info("Saved %d activations to %s", len(activations), output_path.name)''','''        if activations:
            _save_with_layers(activations, output_path, save_layers)
            log.info("Saved %d activations to %s", len(activations), output_path.name)''')
        s = s.replace('''                torch.save(var_acts, var_path)''','''                _save_with_layers(var_acts, var_path, save_layers)''')
        assert s.count("_save_with_layers(") == 3
    else:
        # t1: run_stage receives args; save inside it
        s = s.replace('''        if activations:
            torch.save(activations, output_path)''','''        if activations:
            _save_with_layers(activations, output_path, [int(x) for x in args.save_layers.split(',') if x.strip()] or None)''')
        assert s.count("_save_with_layers(") == 2
    p.write_text(s); print(fn, "patched")

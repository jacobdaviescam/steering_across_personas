# Contributing

Welcome to the Persona-Conditional Steering Vectors project. This guide gets new
members productive and keeps `main` clean.

## 1. Get set up

```bash
git clone git@github.com:jacobdaviescam/steering_across_personas.git
cd steering_across_personas
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e ".[dev]"
```

Then **start with the onboarding notebooks** in [`notebooks/`](notebooks/) — they run on
CPU and pull the pre-computed vectors from Hugging Face, so you can reproduce the
headline findings without a GPU. Read [`docs/overview.md`](docs/overview.md) alongside them.

Copy `.env.example` to `.env` and add your keys (`ANTHROPIC_API_KEY` for the LLM judge /
data generation, `HF_TOKEN` for gated models). GPU + model weights are only needed for
the generation/extraction steps (pipeline steps 1–2); analysis works from the HF vectors.

## 2. The workflow — branch → PR → review

`main` is **protected**: no direct pushes. All changes go through a pull request.

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>      # e.g. feat/sycophancy-trait
# ... do your work, commit ...
git push -u origin <branch>
gh pr create --fill                              # or open a PR in the GitHub UI
```

Branch naming: `feat/…`, `fix/…`, `exp/…` (experiments), `docs/…`, `chore/…`.

A PR can merge once:
1. **CI is green** (the `lint` check is required; `smoke-import` runs too).
2. **One approving review.**
3. The PR-template checklist is satisfied.

Keep PRs focused and reasonably small — easier to review, faster to merge.

## 3. Data policy (important)

**Do not commit large data to git.** Activations, steering vectors, responses, and
figures are *outputs*, not source. They belong on the Hugging Face dataset
([`girishgupta/persona-steering-activations`](https://huggingface.co/datasets/girishgupta/persona-steering-activations)),
not in the repo. `outputs/` is gitignored; please keep bulk `data/` artifacts out too.
If you generate new vectors/activations worth sharing, upload them to HF and reference
them — don't `git add` them.

## 4. Code conventions

- Pipeline scripts are numbered (`pipeline/N_*.py`); analysis helpers live in
  `persona_steering/`. Import the package, don't copy-paste helpers into scripts.
- New Python deps go in `pyproject.toml`.
- Run `ruff check .` and `ruff format .` before pushing (config in `pyproject.toml`).
- Results claims must be reproducible from a script or notebook, and must state the
  **model** and **layer** used.

## 5. Picking up work

Browse the [issues](https://github.com/jacobdaviescam/steering_across_personas/issues).
Issues labelled **good first issue** are a gentle ramp; **research** issues are larger
experiment threads. Comment to claim one before starting so we don't duplicate effort.

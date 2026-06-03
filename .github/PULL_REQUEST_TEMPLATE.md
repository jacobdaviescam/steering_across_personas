<!-- Thanks for contributing! Fill in the sections below and tick the checklist. -->

## What & why

<!-- One or two sentences: what does this PR change, and why? Link any issue: "Closes #123". -->

## Type of change

- [ ] New experiment / analysis
- [ ] Bug fix
- [ ] Pipeline / infrastructure
- [ ] Docs / notebooks
- [ ] Refactor (no behaviour change)

## How to review / reproduce

<!-- Exact commands a reviewer can run, or the notebook/figure to look at.
     e.g. `python pipeline/e4_probe_transfer.py --activations-dir ... --layer 22` -->

## Checklist

- [ ] Branched from up-to-date `main`; PR targets `main`
- [ ] Code compiles and CI is green (lint + smoke-import)
- [ ] **No large data committed** — activations/vectors/responses go to the HF dataset, not git (`outputs/` and bulk `data/` stay untracked)
- [ ] No secrets/keys committed; `.env` is untouched
- [ ] Docs updated if behaviour/results changed (`docs/`, `README.md`, or `CLAUDE.md`)
- [ ] New deps added to `pyproject.toml` (not ad-hoc installs)
- [ ] For results changes: numbers backed by a reproducible script or notebook, and the model + layer are stated

## Results / screenshots (if applicable)

<!-- Paste key metrics, a figure, or a table. State model and layer. -->

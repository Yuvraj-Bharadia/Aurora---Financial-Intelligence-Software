# Contributing to Aurora

Contributions are welcome. This document covers reporting problems, proposing changes, and
running the test suite.

## Reporting a bug

Open an issue including:

- a minimal reproducible snippet
- what you expected and what happened instead
- your Python version, operating system, and versions of numpy, pandas, scipy,
  scikit-learn and hmmlearn

## Reporting a suspected correctness problem

Aurora makes claims about causal integrity that are enforced by tests in
`tests/test_non_anticipation.py`. If you believe a result is contaminated by look-ahead, or
that a calibration guarantee is not holding, say so explicitly in the issue title. These take
priority over feature requests, and a failing test case is the most useful thing you can
send.

## Proposing a change

1. Open an issue first for anything beyond a typo, so the approach can be agreed before you
   spend time on it.
2. Fork, branch from `main`, and make your change.
3. Add a test. Changes touching `aurora/data/availability.py`, `aurora/regimes/`,
   `aurora/evaluation/conformal.py` or `aurora/ensembles/` need a test demonstrating the new
   behaviour while leaving the existing assertions passing.
4. Run the suite locally and the linters.
5. Open a pull request describing what changed and why.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: pip install -r requirements-dev.txt
pytest -v
```

Linting matches CI:

```bash
ruff check aurora/ tests/
black --check aurora/ tests/
isort --check-only aurora/ tests/
```

## Conventions

- Format with `black` at the project's configured line length; imports sorted with `isort`.
- Public functions carry type hints and a docstring stating what the function guarantees, not
  only what it computes.
- Where code exists to prevent a specific failure, say so in a comment. Several mechanisms
  here look arbitrary until you know which bug they catch. The net-lag assertion in
  `aurora/data/availability.py` is the clearest example.

## Getting help

Open a GitHub issue or discussion. For questions about the research method rather than the
software, `paper/paper.md` and `docs/methodology.md` are the references.

# AGENTS.md

This file provides guidance to AI coding agents (e.g., GitHub Copilot, Cursor, OpenAI Codex)
working within the Skala repository. Follow these conventions to produce consistent, high-quality
contributions.

## Repository overview

Skala is a neural network-based exchange-correlation (XC) functional for density functional theory
(DFT). The codebase includes:

| Path | Description |
|------|-------------|
| `skala/` | Published ASE, PySCF, and GPU4PySCF runtime plus tests |
| `model/` | Trainable model definition, tests, and LibTorch/FTorch examples |
| `gauxc/` | GauXC exporter, native examples, tests, and documentation |
| `benchmark/` | Benchmark runner, reference data, report tooling, and tests |
| `website/` | Main Sphinx site |
| `.github/workflows/` | CI workflows (test, docs) |

## Development environment

1. **Python version**: 3.11–3.13 (target 3.11 for compatibility tooling).
2. **Environment setup** (Pixi 0.75):
   ```bash
  pixi install --locked -e default
   ```
3. **Pre-commit hooks** (required before committing):
   ```bash
  pixi run -e default pre-commit install
  pixi run -e default pre-commit run --all-files
   ```

## Code style & linting

- **Formatter/linter**: Ruff (`ruff format`, `ruff check --fix --select I`).
- **Type checking**: mypy in strict mode. Ignore missing type information only for explicitly
  named untyped dependencies in `pyproject.toml`; do not use global `--ignore-missing-imports`.
- Line length: 100 characters (Black-compatible).
- Imports sorted via Ruff's isort rules.
- Generated build, coverage, and documentation output is excluded from static analysis.

When editing code:
- Run `ruff format <file>` and `ruff check --fix <file>` before committing.
- Add type hints to new public functions and classes.
- Use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.

## Testing

- Framework: pytest with pytest-cov.
- Run tests:
  ```bash
  OMP_NUM_THREADS=4 pixi run -e default pytest -v --doctest-modules \
    --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
    --durations=50 --durations-min=1.0 skala/src/skala/ skala/tests/
  ```
- Keep tests beside their owning component with a `test_` prefix.
- Use fixtures for expensive setup (molecule construction, model loading).
- Prefer fast unit tests; integration tests that run DFT should be marked or placed separately.

## Documentation

- Engine: Sphinx with myst-nb (executes notebooks during build).
- Build locally:
  ```bash
  pixi run -e docs sphinx-build -b html website website/_build/html
  pixi run -e docs sphinx-build -b html gauxc/docs website/_build/html/gauxc
  touch website/_build/html/.nojekyll
  ```
- Notebooks in `website/` should be executable with a 5-minute timeout.
- Use reStructuredText for standalone pages; Jupyter notebooks for tutorials.

## Pull request guidelines

1. Create a feature branch from `main`.
2. Ensure pre-commit hooks pass.
3. Add or update tests for new functionality.
4. Update documentation if public API changes.
5. Keep commits atomic; write clear commit messages.
6. CI must pass (tests, linting, docs build).

## Architecture notes

- **Runtime functional API** (`skala/src/skala/functional/`): Loads traced checkpoints and defines
  traditional functionals and the runtime interface.
- **Model definition** (`model/src/skala_model/`): Defines trainable layers and the
  enhancement-factor network; it is not part of release artifacts.
- **PySCF integration** (`skala/src/skala/pyscf/`): Custom `numint` module and `SkalaKS` class hook the
  model into PySCF's DFT machinery.
- **ASE calculator** (`skala/src/skala/ase/`): Provides an ASE-compatible calculator for energy/force
  evaluations and geometry optimizations.
- **GauXC integration** (`gauxc/`): Exporter and C/C++/Fortran examples for external GauXC builds.

## Common commands

| Task | Command |
|------|---------|
| Format code | `pixi run -e default ruff format skala/ model/ gauxc/ benchmark/ docs/` |
| Lint code | `pixi run -e default pre-commit run --all-files` |
| Run runtime tests | `OMP_NUM_THREADS=4 pixi run -e default pytest -v --doctest-modules --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html --durations=50 --durations-min=1.0 skala/src/skala/ skala/tests/` |
| Run component tests | `OMP_NUM_THREADS=4 pixi run -e default pytest -v model/tests/test_model.py model/tests/test_utils.py gauxc/tests/ benchmark/tests/` |
| Build docs | `pixi run -e docs sphinx-build -b html website website/_build/html && pixi run -e docs sphinx-build -b html gauxc/docs website/_build/html/gauxc && touch website/_build/html/.nojekyll` |
| Type check | `pixi run -e default mypy skala/src model/src gauxc/src benchmark/src` |

## Contact

- Issues: https://github.com/microsoft/skala/issues
- Security: See `SECURITY.md`
- Code of Conduct: Microsoft Open Source CoC (see `CONTRIBUTING.md`)

# AGENTS.md

This file provides guidance to AI coding agents (e.g., GitHub Copilot, Cursor, OpenAI Codex)
working within the Skala repository. Follow these conventions to produce consistent, high-quality
contributions.

## Repository overview

Skala is a neural network-based exchange-correlation (XC) functional for density functional theory
(DFT). The codebase includes:

| Path | Description |
|------|-------------|
| `src/skala/` | Core Python package (model, PySCF/ASE integrations, utilities) |
| `tests/` | Pytest test suite |
| `docs/` | Sphinx documentation (RST + Jupyter notebooks via myst-nb) |
| `examples/` | Usage examples (Python scripts and C++ integrations) |
| `third_party/gauxc/` | Vendored GauXC library with Skala support |
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
  pixi run -e default lint
   ```

## Code style & linting

- **Formatter/linter**: Ruff (`ruff format`, `ruff check --fix --select I`).
- **Type checking**: mypy in strict mode. Ignore missing type information only for explicitly
  named untyped dependencies in `pyproject.toml`; do not use global `--ignore-missing-imports`.
- Line length: 100 characters (Black-compatible).
- Imports sorted via Ruff's isort rules.
- Exclude `third_party/` and `tests/` from mypy; exclude `third_party/` from Ruff.

When editing code:
- Run `ruff format <file>` and `ruff check --fix <file>` before committing.
- Add type hints to new public functions and classes.
- Use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.

## Testing

- Framework: pytest with pytest-cov.
- Run tests:
  ```bash
  pixi run -e default test
  ```
- Keep test files in `tests/` with `test_` prefix.
- Use fixtures for expensive setup (molecule construction, model loading).
- Prefer fast unit tests; integration tests that run DFT should be marked or placed separately.

## Documentation

- Engine: Sphinx with myst-nb (executes notebooks during build).
- Build locally:
  ```bash
  pixi run -e docs docs-html
  ```
- Notebooks in `docs/` should be executable with a 5-minute timeout.
- Use reStructuredText for standalone pages; Jupyter notebooks for tutorials.

## Pull request guidelines

1. Create a feature branch from `main`.
2. Ensure pre-commit hooks pass.
3. Add or update tests for new functionality.
4. Update documentation if public API changes.
5. Keep commits atomic; write clear commit messages.
6. CI must pass (tests, linting, docs build).

## Architecture notes

- **Functional implementation** (`src/skala/functional/`): Defines the Skala model layers, density
  features, and enhancement-factor network. The pre-trained weights are loaded via Hugging Face Hub.
- **PySCF integration** (`src/skala/pyscf/`): Custom `numint` module and `SkalaKS` class hook the
  model into PySCF's DFT machinery.
- **ASE calculator** (`src/skala/ase/`): Provides an ASE-compatible calculator for energy/force
  evaluations and geometry optimizations.
- **GauXC add-on** (`third_party/gauxc/`): C++ library enabling Skala in compiled DFT codes; uses
  LibTorch for inference.

## Common tasks

| Task | Command |
|------|---------|
| Format code | `pixi run -e default ruff format src/ tests/` |
| Lint code | `pixi run -e default lint` |
| Run tests | `pixi run -e default test` |
| Build docs | `pixi run -e docs docs-html` |
| Type check | `pixi run -e default mypy src/skala` |

## Contact

- Issues: https://github.com/microsoft/skala/issues
- Security: See `SECURITY.md`
- Code of Conduct: Microsoft Open Source CoC (see `CONTRIBUTING.md`)

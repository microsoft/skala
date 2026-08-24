# Contributing

This project welcomes contributions and suggestions.
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution.
For details, visit [https://cla.microsoft.com](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment).
Simply follow the instructions provided by the bot.
You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development setup

Install Pixi 0.75, then create the default locked development environment from the repository root:

```bash
pixi install --locked -e default
pixi run -e default pre-commit install
```

Run the standard checks in their Pixi environments:

```bash
pixi run -e default pytest -v --doctest-modules \
	--cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
	--durations=50 --durations-min=1.0 src/skala/ tests/
pixi run -e default pre-commit run --all-files
pixi run -e docs sphinx-build -b html docs docs/_build/html
```

The generic `pytest` task sets `OMP_NUM_THREADS=4` and forwards all additional arguments.

Named compatibility environments cover Python 3.11 through 3.13, PySCF 2.14,
PyTorch 2.12 and 2.13, GPU4PySCF 1.8.1, and CUDA 12 and 13. Keep `pixi.lock`
synchronized with changes to `pixi.toml` or `pyproject.toml`.


## Model development

The torch model in the `src/skala/functional` folder serves as a representation of what our
model does. The real model is inside the respective `.fun` files on [hugging face](https://huggingface.co/microsoft/skala-1.1), which contains a fully
traced model. So the model folder is not production code, but more an explanation in code, which we
will try to keep in line with our traced models.

PRs that modify the model code consequently do not affect our hugging face model checkpoints.
Due to the extensive testing needed, which far exceeds public GitHub resources,
we will only integrate model changes, if they are very promising in performance, speed or
memory consumption.

Source-model compatibility tests load the published Skala 1.1 weights into the current
implementation, trace it locally, and compare that trace directly with the published trace for
forward accuracy, Vxc feature backpropagation, runtime, and peak memory. Backpropagation covers
density, density-gradient, and kinetic-energy-density inputs. Accuracy also compares the model
derivatives used for nuclear gradients with respect to grid coordinates, grid weights, and coarse
atomic coordinates. Full nuclear forces require additional molecular AO and grid-response terms
outside this synthetic benchmark. The tests use 200,000 deterministic grid points; they require
neither molecular setup nor golden output data. Run the CPU cases with four threads:

```bash
MKL_NUM_THREADS=4 pixi run -e default \
	pytest -v -m model_benchmark -k cpu tests/test_traced_model_comparison.py
```

On a CUDA-capable runner, execute both CPU and GPU cases by omitting the CPU filter:

```bash
MKL_NUM_THREADS=4 pixi run -e gpu-cuda12-torch213 \
	pytest -v -m model_benchmark -k cuda tests/test_traced_model_comparison.py
```

The same comparison can be run as a standalone report. It prints maximum accuracy differences,
local and published runtime medians, and isolated peak allocations for forward and backward work:

```bash
pixi run -e default python tests/test_traced_model_comparison.py --device cpu
pixi run -e gpu-cuda12-torch213 python tests/test_traced_model_comparison.py --device cuda
```

Despite that feel free to open issues and PRs proposing model improvements, we are very
grateful for your input and we will take these into account for new model releases.

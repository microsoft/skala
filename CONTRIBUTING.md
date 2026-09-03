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

Install Pixi 0.78, then create the default locked development environment from the repository root:

```bash
pixi install --locked -e default
pixi run -e default pre-commit install
```

On Linux x86_64 with a CUDA 12-compatible driver, the `dev` environment is the
GPU-enabled development superset. It adds GPU4PySCF, profiling, IPython, a
notebook kernel, and plotting tools while retaining the default test, lint,
model, and benchmark tooling:

```bash
pixi install --locked -e dev
pixi run -e dev ipython
```

Run the standard checks in their Pixi environments:

```bash
OMP_NUM_THREADS=4 pixi run -e default pytest -v --doctest-modules \
	--cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
	--durations=50 --durations-min=1.0 skala/src/skala/ skala/tests/
OMP_NUM_THREADS=4 pixi run -e default pytest -v model/tests/test_model.py \
	model/tests/test_utils.py benchmark/tests/
pixi run -e default pre-commit run --all-files
pixi run -e docs sphinx-build -b html website website/_build/html
touch website/_build/html/.nojekyll
```

Set `OMP_NUM_THREADS=4` when running tests locally to match CI.

Named compatibility environments cover Python 3.11 through 3.13, PySCF 2.14,
PyTorch 2.12 and 2.13, GPU4PySCF 1.8.1, and CUDA 12 and 13. Keep `pixi.lock`
synchronized with changes to `pixi.toml` or any component `pyproject.toml`.

## SkalaXC development

SkalaXC development uses the root Pixi workspace. Its host environment is
locked for Linux x86-64, Linux ARM64, and macOS ARM64. Initialize the GauXC
submodule and install the environment once:

```bash
git submodule update --init SkalaXC/external/GauXC
pixi install --locked -e skalaxc-host
```

Configure and build the native C++ and C APIs, the Fortran binding, tests, and
consumer examples:

```bash
pixi run -e skalaxc-host cmake \
	-S SkalaXC \
	-B SkalaXC/build-contrib \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$PWD/SkalaXC/install-contrib" \
	-DSKALAXC_BUILD_FORTRAN=ON \
	-DSKALAXC_BUILD_TESTS=ON \
	-DSKALAXC_BUILD_EXAMPLES=ON \
	-DSKALAXC_DOWNLOAD_MODELS=ON \
	-DSKALAXC_ENABLE_CUDA=OFF \
	-DSKALAXC_ENABLE_MPI=OFF \
	-DSKALAXC_ENABLE_OPENMP=ON
pixi run -e skalaxc-host cmake --build SkalaXC/build-contrib --parallel 2
```

Run every native CTest registration, including the C and Fortran binding
tests, then install the native package. The fixed seed makes the randomly
generated UKS density matrices reproducible, so numerical failures can be
replayed locally and compared directly with CI:

```bash
OMP_NUM_THREADS=4 SKALAXC_TEST_SEED=20260729 \
	pixi run -e skalaxc-host ctest \
	--test-dir SkalaXC/build-contrib \
	--no-tests=error \
	--output-on-failure
pixi run -e skalaxc-host cmake --install SkalaXC/build-contrib
```

Build the Python binding against that exact native installation and run its
tests. `SkalaXC_DIR` must be absolute so scikit-build cannot select another
SkalaXC package:

```bash
pixi run -e skalaxc-host env \
	SkalaXC_DIR="$PWD/SkalaXC/install-contrib/lib/cmake/SkalaXC" \
	SKALAXC_PYTHON_LAYOUT=WHEEL \
	python -m pip install SkalaXC/python \
		--no-build-isolation --no-deps --force-reinstall
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
	pytest -v SkalaXC/python/tests/
```

The checked-in Pixi tasks remain the shortest way to exercise the standard
host build and the focused static-analysis/documentation checks:

```bash
OMP_NUM_THREADS=4 pixi run -e skalaxc-host skalaxc-test-host
pixi run -e skalaxc-host-clang skalaxc-clang-tidy
pixi run -e skalaxc-tools skalaxc-doxygen
```

CUDA 12 and 13 use the custom platforms `linux-64-cuda12` and
`linux-64-cuda13`; pass the matching platform with `-p` to both `pixi install`
and `pixi run`.

## Model development

The torch model in `model/src/skala_model` serves as a representation of what our
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
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 pixi run -e default \
	pytest -v -m model_benchmark -k cpu model/tests/test_traced_model_comparison.py
```

On a CUDA-capable runner, execute both CPU and GPU cases by omitting the CPU filter:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 pixi run -e gpu-cuda12-torch213 \
	pytest -v -m model_benchmark -k cuda model/tests/test_traced_model_comparison.py
```

The same comparison can be run as a standalone report. It prints maximum accuracy differences,
local and published runtime medians, and isolated peak allocations for forward and backward work:

```bash
pixi run -e default python model/tests/test_traced_model_comparison.py --device cpu
pixi run -e gpu-cuda12-torch213 python model/tests/test_traced_model_comparison.py --device cuda
```

Despite that feel free to open issues and PRs proposing model improvements, we are very
grateful for your input and we will take these into account for new model releases.

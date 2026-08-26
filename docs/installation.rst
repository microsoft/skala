Installation
============

Install from PyPI
-----------------

.. image:: https://img.shields.io/pypi/v/skala?logo=pypi&logoColor=white
   :alt: PyPI
   :target: https://pypi.org/project/skala/

To install *skala*, you can use pip:

.. code-block:: bash

   pip install skala

This will install the *skala* packages and all its dependencies, including

* `torch <https://pytorch.org>`__,
  `e3nn <https://e3nn.org>`__, and
  `opt_einsum_fx <https://opt-einsum-fx.readthedocs.io>`__
  for running the exchange-correlation model
* `pyscf <https://pyscf.org>`__
  for running the self-consistent field (SCF) calculations and evaluating the density features
* `pyscf-dispersion <https://github.com/pyscf/dispersion>`__ on Linux or
  `dftd3 <https://dftd3.readthedocs.io>`__ on macOS on Apple Silicon for computing
  the D3 dispersion correction to the total energy

Skala supports Linux and macOS on Apple Silicon with Python 3.11 through 3.13, PySCF 2.14,
and PyTorch 2.12 or 2.13.

The default Pytorch installation is the GPU version, which the *skala* package in combination with PySCF doesn't leverage. To install only the much smaller CPU version of Pytorch, run the following before installing the *skala* package:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu


Reproducible source environments
--------------------------------

Skala uses `Pixi <https://pixi.sh>`__ for source dependency management. The
repository contains one ``pixi.toml`` and one committed ``pixi.lock`` covering
all supported Python, PySCF, PyTorch, CUDA, documentation, release, and native
build environments. Install Pixi 0.75, then clone the repository and install the
default CPU environment:

.. code-block:: bash

   git clone https://github.com/microsoft/skala
   cd skala
   pixi install --locked -e default

The local Skala package is installed editable. Run commands through Pixi so they
always use the selected environment:

.. code-block:: bash

   pixi run -e default python your_script.py
   pixi run -e default pytest -v --doctest-modules \
    --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
    --durations=50 --durations-min=1.0 src/skala/ tests/
   pixi run -e default pre-commit run --all-files

The generic ``pytest`` task sets ``OMP_NUM_THREADS=4`` and forwards all
additional arguments.

The locked compatibility environments are:

.. list-table::
   :header-rows: 1

   * - Environment
     - Python
     - PySCF
     - PyTorch / CUDA
   * - ``test-py311-pyscf214-torch213``
     - 3.11
     - 2.14
     - 2.13 CPU
   * - ``test-py312-pyscf214-torch212``
     - 3.12
     - 2.14
     - 2.12 CPU
   * - ``test-py312-pyscf214-torch213``
     - 3.12
     - 2.14
     - 2.13 CPU
   * - ``test-py313-pyscf214-torch213``
     - 3.13
     - 2.14
     - 2.13 CPU
   * - ``gpu-cuda12-torch212``
     - 3.12
     - 2.14
     - 2.12 / CUDA 12
   * - ``gpu-cuda12-torch213``
     - 3.12
     - 2.14
     - 2.13 / CUDA 12
   * - ``gpu-cuda13-torch213``
     - 3.12
     - 2.14
     - 2.13 / CUDA 13

For example, install and test the primary CUDA environment with:

.. code-block:: bash

  pixi install --locked -e gpu-cuda12-torch213
  pixi run -e gpu-cuda12-torch213 python tools/verify_gpu.py
  pixi run -e gpu-cuda12-torch213 \
    pytest -v -m 'gpu and not profiling and not model_benchmark' tests/

The CUDA platforms are encoded in the lockfile, so container builds do not need
a CUDA override when no GPU is attached. Runtime GPU checks still require a
compatible NVIDIA driver and device.

For development purposes, please initialize the pre-commit hooks via:

.. code-block:: bash

   pixi run -e default pre-commit install

To test your installation, you can run the tests:

.. code-block:: bash

  pixi run -e default pytest -v --doctest-modules \
    --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
    --durations=50 --durations-min=1.0 src/skala/ tests/


Model checkpoints
-----------------

The pre-trained Skala model checkpoints are hosted `on Hugging Face <https://huggingface.co/microsoft/skala-1.0>`__ and downloaded automatically by the Python package in this repository from there for running calculations.

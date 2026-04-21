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
* `dftd3 <https://dftd3.readthedocs.io>`__
  for computing the D3 dispersion correction to the total energy

The default Pytorch installation is the GPU version, which the *skala* package in combination with PySCF doesn't leverage. To install only the much smaller CPU version of Pytorch, run the following before installing the *skala* package:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu


Install from conda-forge
------------------------

.. image:: https://img.shields.io/conda/vn/conda-forge/skala
   :alt: conda-forge
   :target: https://github.com/conda-forge/skala-feedstock

The *skala* package is available on conda-forge, to install it use

.. code-block:: bash

   mamba install -c conda-forge skala

You can select between GPU and CPU version of pytorch by requesting the ``cuda*`` build or the ``cpu*`` build.
For the CPU version use

.. code-block:: bash

   mamba install -c conda-forge skala "pytorch=*=cpu*"

For the GPU version use (e.g. with Cuda 12)

.. code-block:: bash

   mamba install -c conda-forge skala "pytorch=*=cuda12*"


Installing from source
----------------------

If you prefer to install Skala from the source code, you can clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/microsoft/skala
   cd skala
   mamba env create -n skala -f environment-cpu.yml
   mamba activate skala
   pip install -e .

where ``environment-cpu.yml`` can be replaced with ``environment-gpu.yml`` for
GPU support via `GPU4PySCF <https://github.com/pyscf/gpu4pyscf>`__. The GPU
environment pins ``cuda-toolkit 12``, ``cuda-version 12``, ``cutensor``, and
installs ``gpu4pyscf-cuda12x 1.5`` from PyPI as part of the environment file —
no separate install step is required:

.. code-block:: bash

   mamba env create -n skala -f environment-gpu.yml
   mamba activate skala
   pip install -e .

If you are building inside a container without a GPU attached (for example CI,
or a Docker image built on a CPU-only host), set ``CONDA_OVERRIDE_CUDA`` so the
solver proceeds without a device:

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.0 mamba env create -n skala -f environment-gpu.yml

For CUDA 11 or 13, adjust ``cuda-toolkit``, ``cuda-version``, and the
``gpu4pyscf-cuda{11,13}x`` pin in ``environment-gpu.yml`` accordingly. Check
your driver's maximum supported CUDA version with ``nvidia-smi``.

To install the development dependencies, you can run:

.. code-block:: bash

    pip install -e .[dev]

For development purposes, please initialize the pre-commit hooks via:

.. code-block:: bash

   pre-commit install

To test your installation, you can run the tests:

.. code-block:: bash

   pytest -v tests/


Model checkpoints
-----------------

The pre-trained Skala model checkpoints are hosted `on Hugging Face <https://huggingface.co/microsoft/skala-1.0>`__ and downloaded automatically by the Python package in this repository from there for running calculations.

Using Skala with gpu4pyscf
==========================

The Skala functional can also be used in GPU4PySCF with an appropriate PyTorch CUDA version by creating a new Kohn-Sham calculator based on the `SkalaKS` constructor from the ``skala.gpu4pyscf`` module.

.. code-block:: python

   from pyscf import gto

   from skala.gpu4pyscf import SkalaKS

   mol = gto.M(
       atom="""H 0 0 0; H 0 0 1.4""",
       basis="def2-tzvp",
   )
   ks = SkalaKS(mol, xc="skala-1.1")
   ks.kernel()

   print(ks.dump_scf_summary())


Installation
------------

The recommended way to set up a GPU environment is the provided
``environment-gpu.yml``, which pins ``pytorch-gpu``, ``cuda-toolkit 12``,
``cuda-version 12``, ``cutensor``, and installs ``gpu4pyscf-cuda12x 1.5`` from
PyPI as part of the environment file:

.. code-block:: bash

   mamba env create -n skala -f environment-gpu.yml
   mamba activate skala
   pip install skala

For CUDA 11 or 13, adjust ``cuda-toolkit``, ``cuda-version``, and the
``gpu4pyscf-cuda{11,13}x`` pin in ``environment-gpu.yml`` accordingly.

See the :doc:`installation guide </installation>` for more details, including
how to install from conda-forge or inside a container without a GPU attached.
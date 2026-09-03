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

The repository provides three locked GPU compatibility environments. They
combine conda-forge PyTorch and CUDA libraries with PyPI PySCF and the latest
tested GPU4PySCF release, 1.8.1:

* ``gpu-cuda12-torch212``
* ``gpu-cuda12-torch213``
* ``gpu-cuda13-torch213``

Install and verify one from the repository root, for example:

.. code-block:: bash

   pixi install --locked -e gpu-cuda12-torch213
   pixi run -e gpu-cuda12-torch213 python tools/verify_gpu.py

CUDA 12 and CUDA 13 are supported. Check your driver's maximum supported CUDA
version with ``nvidia-smi`` before selecting an environment.

See the :doc:`installation guide </installation>` for the complete compatibility
matrix and container-build details.

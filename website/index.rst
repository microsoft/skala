Skala: Accurate and scalable exchange-correlation with deep learning
====================================================================

Overview
--------

*Skala* is a deep-learning exchange-correlation functional designed to provide chemical accuracy (less than 1 kcal/mol) for a wide range of chemical systems without using expensive non-local features like exact exchange or hand-crafted density convolutions.
The model is trained on a large dataset of highly accurate total atomization energies, thermochemical properties, like ionization potentials and proton affinities, conformer energies, reaction paths, and non-covalent interactions.


.. admonition:: Learn more
   :class: important

   To learn more about the Skala functional, check out the `blog post <https://aka.ms/skaladft/blog>`__ or the `preprint <https://aka.ms/skaladft/preprint>`__.


What Skala provides
-------------------

*Skala* is first of all the functional itself: a trained neural network model.
Around the model, this project provides several ways to use it, from a ready-to-use Python package to building blocks for integrating Skala into your own electronic structure code.

The Skala model
   The trained exchange-correlation functional.
   Released checkpoints are distributed as TorchScript files (``.fun``) on `Hugging Face <https://huggingface.co/microsoft/skala-1.1>`__ and documented in the :doc:`model cards <model-card/index>`.
   A checkpoint takes density features on an integration grid (density, density gradient, kinetic energy density, grid coordinates and weights, and atomic coordinates) and returns the exchange-correlation energy density on that grid.
   Because the checkpoints are TorchScript files, they can be evaluated from Python as well as from compiled languages through LibTorch.

The ``skala`` Python package
   The published package on `PyPI <https://pypi.org/project/skala/>`__ loads the released checkpoints and adds Skala to `PySCF <https://pyscf.org>`__ as an add-on.
   It provides drop-in Kohn-Sham classes for :doc:`PySCF <pyscf/singlepoint>` on CPU and :doc:`GPU4PySCF <pyscf/gpu4pyscf>` on GPU, as well as an :doc:`ASE calculator <ase>` for energies, forces, and geometry optimizations.
   This is the easiest way to run calculations with Skala.

C++ and Fortran examples
   Minimal programs that load a Skala checkpoint and evaluate the exchange-correlation energy and its derivatives directly from :doc:`C++ with LibTorch <cpp>` and from :doc:`Fortran with FTorch <ftorch>`.
   They start from precomputed input features and are the starting point if you want to call the model from your own code.

GauXC
   `GauXC <https://github.com/wavefunction91/GauXC>`__ is a CPU/GPU library for the numerical integration of exchange-correlation functionals used by electronic structure codes.
   We provide a version of GauXC with Skala support and `bindings for C++, C, and Fortran <https://microsoft.github.io/skala/gauxc/>`__.
   GauXC builds the integration grid and the density features from a density matrix and returns the exchange-correlation energy and potential, which makes it the recommended way to add Skala to an existing DFT code.

Benchmarks
   A reproducible :doc:`timing benchmark <benchmarks>` that compares Skala with traditional functionals in PySCF and GPU4PySCF on molecules of increasing size, together with an interactive report.


Skala in electronic structure packages
--------------------------------------

Skala is already available in the following electronic structure codes.
Refer to their documentation for installation and input options.

`CP2K <https://www.cp2k.org>`__
   See the `GauXC section of the CP2K manual <https://manual.cp2k.org/trunk/methods/dft/gauxc.html>`__.

`FHI-aims <https://fhi-aims.org>`__
   See the `exchange-correlation section of the FHI-aims manual <https://fhi-aims.org/uploads/manual/Ch3/S3.html#skala_model_path>`__.


Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - I want to …
     - Start with
   * - run DFT calculations with Skala from Python
     - :doc:`installation`, then :doc:`pyscf/singlepoint`, :doc:`pyscf/gpu4pyscf`, or :doc:`ase`
   * - use Skala in CP2K or FHI-aims
     - the documentation of these codes, see `Skala in electronic structure packages`_
   * - add Skala to an existing electronic structure code
     - `GauXC <https://microsoft.github.io/skala/gauxc/>`__ with its C++, C, or Fortran bindings
   * - evaluate the Skala model from my own C++ or Fortran code
     - :doc:`cpp` or :doc:`ftorch`
   * - know what performance to expect
     - :doc:`benchmarks`


.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   Installation <installation>

.. toctree::
   :maxdepth: 1
   :caption: Python package
   :hidden:

   PySCF single-point calculations <pyscf/singlepoint>
   GPU4PySCF <pyscf/gpu4pyscf>
   SCF settings <pyscf/scf_settings>
   ASE <ase>

.. toctree::
   :maxdepth: 1
   :caption: Compiled languages
   :hidden:

   C++ (LibTorch) <cpp>
   Fortran (FTorch) <ftorch>
   GauXC (C++, C, Fortran) <https://microsoft.github.io/skala/gauxc/>

.. toctree::
   :maxdepth: 1
   :caption: Skala in other codes
   :hidden:

   CP2K <https://manual.cp2k.org/trunk/methods/dft/gauxc.html>
   FHI-aims <https://fhi-aims.org/uploads/manual/Ch3/S3.html#skala_model_path>

.. toctree::
   :maxdepth: 1
   :caption: Performance
   :hidden:

   Performance & scaling <benchmarks>

.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   model-card/skala-1.1
   model-card/index
   Skala preprint <https://aka.ms/skaladft/preprint>
   Breaking bonds, breaking ground <https://aka.ms/skaladft/blog>

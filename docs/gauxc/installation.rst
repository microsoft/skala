.. _gauxc_install:

Installing GauXC
================

In this section, we will provide instructions on how to install GauXC with Skala support based on the conda-forge ecosystem.
As part of this tutorial we will be

* installing dependencies for building GauXC
* configuring GauXC with different options
* testing the Skala implementation in GauXC
* installing the GauXC library
* reusing GauXC from the CMake build system


Prerequisites
-------------

For this tutorial, we will use the `mamba <https://mamba.readthedocs.io/en/latest/>`__ package manager for setting up the environment and installing dependencies.
If you do not have mamba installed, you can download the `miniforge <https://conda-forge.org/download/>`__ installer.

First, we will create a new environment with all the required dependencies for building GauXC with Skala support.
We provide three different configurations depending on whether you want to build GauXC with OpenMP, MPI, or CUDA support.

.. note::

   A full list of dependencies can be found at :ref:`gauxc-cmake-deps` in the CMake configuration documentation.

For this, create a file named `environment.yml` with the following content:

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-openmp.yml
         :caption: environment.yml
         :language: yaml

   .. tab-item:: MPI

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-mpi.yml
         :caption: environment.yml
         :language: yaml

   .. tab-item:: CUDA

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-cuda.yml
         :caption: environment.yml
         :language: yaml

Create and activate the environment:

.. code-block:: none

   mamba env create -n gauxc-dev -f environment.yml
   mamba activate gauxc-dev

Verify that the toolchain is visible:

.. code-block:: bash

   cmake --version
   python -c "import torch; print(torch.__version__)"


Obtain GauXC with Skala
-----------------------

Download the pre-packaged source bundle from the Skala release page:

.. code-block:: none

   curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r1.tar.gz | tar xzv

.. tip::

   To verify the downloaded tarball you can obtain a checksum

   .. code-block:: none

      curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r1.tar.gz > gauxc-skala-r1.tar.gz
      curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r1.tar.gz.sha256 | sha256sum -c
      tar xzvf gauxc-skala-r1.tar.gz

The archive expands into a ``gauxc`` directory that already contains the Skala patches.
One convenient layout is

.. code-block:: text

   work/
   ├── gauxc/
   └── build/

.. note::

   You can also obtain the latest version of GauXC with Skala support by downloading the `skala branch of GauXC <https://github.com/wavefunction91/gauxc/tree/skala>`__.

   .. code-block:: none

      curl -L https://github.com/wavefunction91/GauXC/archive/refs/heads/skala.tar.gz | tar xzv


Configure and build
-------------------

Create an out-of-tree build directory and pick the configuration that matches your backend.

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=off \
           -DGAUXC_ENABLE_CUDA=off \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

   .. tab-item:: MPI

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=on \
           -DGAUXC_ENABLE_CUDA=off \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

   .. tab-item:: CUDA

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=off \
           -DGAUXC_ENABLE_CUDA=on \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

.. note::

   To enable the C or Fortran bindings, set :cmake:variable:`GAUXC_ENABLE_C` or :cmake:variable:`GAUXC_ENABLE_FORTRAN` in your CMake configuration step.
   For a full list of available CMake options, see :ref:`gauxc-cmake-options` in the CMake configuration documentation.

.. tip::

   If CMake cannot find libtorch, the ``Torch_DIR`` variable can be set to help discover the package.
   For conda-forge installed pytorch this should be set as ``-DTorch_DIR=${CONDA_PREFIX}/share/cmake/Torch``
   and for pip installed pytorch the CMake config file will be in ``${CONDA_PREFIX}/lib/python3.11/site-packages/torch/share/cmake/Torch``
   where the Python version should be adjusted accordingly to the environment.


Quick verification
------------------

After the build finishes, run the bundled regression test to confirm that Skala-enabled functionals
are working correctly. The Skala implementation can run different traditional functionals, like PBE and TPSS,
which can be compared against other libraries.

.. code-block:: bash

   cd gauxc/tests/ref_data
   ../../../build/tests/standalone_driver onedft_input.inp

Expected output includes the total TPSS energy computed using a checkpoint compatible for the Skala implementation
for the reference density matrix.

.. tip::
   
   If the executable cannot locate libtorch or other shared libraries, double-check
   that ``LD_LIBRARY_PATH`` includes ``${CONDA_PREFIX}/lib``
   (activating the environment usually handles this).


Install the library
-------------------

Install into the active conda environment so downstream projects can pick up the CMake config files.

.. code-block:: bash

   cmake --install build

This installs headers, libraries, and CMake config.

.. note::

   For using GauXC in your own CMake project, check out :ref:`gauxc-cmake-integration` in the CMake configuration documentation.
   Alternatively, you can follow the instructions in the :ref:`gauxc-cpp-library` tutorial for a full standalone example.


Troubleshooting
---------------

Torch not found
  ensure ``Torch_DIR`` points to the libtorch CMake package inside the active environment,
  or export ``Torch_DIR`` before running CMake.

CUDA mismatch
  the CUDA toolkit selected by conda must match the version baked into the
  ``pytorch`` build; reinstall ``pytorch`` if necessary (e.g., ``pytorch ==2.3.* cuda118*``).

Linker errors for BLAS/MPI
  verify that the conda environment stayed active during the build and that ``cmake`` picked
  the toolchain from ``${CONDA_PREFIX}`` via ``CMAKE_PREFIX_PATH``.

Standalone driver cannot find densities
  run it from ``gauxc/tests/ref_data`` since paths in density files are specified relative to the
  current directory.

.. note::

   Need help? Open an issue on the `Skala repository <https://github.com/microsoft/skala/issues>`__.
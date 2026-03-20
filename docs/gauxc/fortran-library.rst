GauXC in Fortran
================

This guide demonstrates how to use the GauXC library from Fortran to evaluate exchange-correlation functionals, including the Skala neural network functional.
By the end of this tutorial, you will:

* Set up a CMake project with GauXC as a dependency
* Initialize the GauXC runtime environment
* Read molecule, basis set, and density matrix from an HDF5 input file
* Configure the integration grid, load balancer, and XC integrator
* Evaluate the exchange-correlation energy and potential

.. _gauxc_fortran_api:

Before you start
----------------

This example assumes that you already have access to a GauXC source tree with Skala support, either because you followed :ref:`gauxc_install` first or because you let this example download GauXC automatically through CMake.
The example CMake files are written to accept either workflow.

Before configuring the project, make sure you have the following available:

- a Fortran compiler together with C and C++ compilers
- CMake and Ninja
- HDF5 development libraries
- an installed GauXC with both the C and Fortran APIs enabled, or network access so the example can build GauXC with those APIs enabled for you

If you want to reproduce the sample ``He_def2-svp.h5`` input file from this guide, you also need the Python ``skala`` package installed.

For a minimal OpenMP-capable environment for this guide, use the example environment file:

.. literalinclude:: ../../examples/fortran/gauxc_integration/environment-openmp.yml
   :language: yaml
   :caption: environment-openmp.yml

For MPI builds, switch to ``examples/fortran/gauxc_integration/environment-mpi.yml`` so that both MPI and the matching HDF5 variant are available.

Setting up CMake
----------------

GauXC integrates most conveniently via CMake.
We will set up a minimal CMake project for a command-line driver that uses GauXC.
In addition to GauXC, we will include other dependencies as needed.

The directory structure for the project will be

.. code-block:: text

   ├── CMakeLists.txt
   ├── app
   │   └── main.F90
   └── cmake
       ├── skala-flap.cmake
       ├── skala-dep-versions.cmake
       ├── skala-gauxc.cmake
       └── skala-hdf5.cmake

First, we create the main ``CMakeLists.txt`` to define our project, include dependencies, and declare the executable.

.. literalinclude:: ../../examples/fortran/gauxc_integration/CMakeLists.txt
   :language: cmake
   :caption: CMakeLists.txt

To manage dependencies cleanly, we create a separate file in the ``cmake/`` subdirectory that stores URLs and checksums for all external packages.

.. literalinclude:: ../../examples/fortran/gauxc_integration/cmake/skala-dep-versions.cmake
   :language: cmake
   :caption: cmake/skala-dep-versions.cmake

Each dependency gets its own CMake include file.
First, we define how to include GauXC, our main dependency.
CMake will first attempt to discover an installed GauXC via its config file; if that fails, it will download and build GauXC from source.
The options defined in the main CMake file are passed through to GauXC to ensure the library provides the requested features.
After GauXC is available, we verify that our requirements are satisfied.
For the Fortran driver this includes the Skala implementation as well as the C and Fortran APIs, which are checked via ``GAUXC_HAS_ONEDFT``, ``GAUXC_HAS_C``, and ``GAUXC_HAS_FORTRAN``.

.. literalinclude:: ../../examples/fortran/gauxc_integration/cmake/skala-gauxc.cmake
   :language: cmake
   :caption: cmake/skala-gauxc.cmake

For the command-line interface, we use the `FLAP library <https://github.com/szaghi/FLAP/wiki>`__ (Fortran command Line Arguments Parser).

.. literalinclude:: ../../examples/fortran/gauxc_integration/cmake/skala-flap.cmake
   :language: cmake
   :caption: cmake/skala-flap.cmake

Finally, we will use the HDF5 Fortran interface for reading our input data from an HDF5 file.

.. literalinclude:: ../../examples/fortran/gauxc_integration/cmake/skala-hdf5.cmake
   :language: cmake
   :caption: cmake/skala-hdf5.cmake

The Fortran example links against the HDF5 Fortran interface and the high-level HDF5 libraries.
If HDF5 is missing or only a partial HDF5 installation is visible to CMake, configuration or link-time failures are expected.

This completes the CMake setup required for our command-line driver.

Module imports
--------------

The main driver program imports the relevant GauXC modules.
We also use GauXC's HDF5 I/O module and the FLAP module for command-line parsing.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 1-28

Each GauXC component has a corresponding Fortran module:

`gauxc_status`
  Handles GauXC status codes and error messages.

`gauxc_molecule`
  Manages molecular data structures (atomic numbers and Cartesian coordinates).

`gauxc_basisset`
  Manages basis set data (shell definitions, exponents, contraction coefficients).

`gauxc_molgrid`
  Sets up and manages the numerical integration grid.

`gauxc_runtime_environment`
  Interfaces with MPI and device (GPU) runtime environments.

`gauxc_load_balancer`
  Distributes grid points across available compute resources.

`gauxc_molecular_weights`
  Computes Becke-style partitioning weights for the molecular grid.

`gauxc_xc_functional`
  Handles exchange-correlation functional definitions.

`gauxc_integrator`
  Performs the numerical integration of the exchange-correlation energy and potential.

Next, we declare variables for the GauXC-specific types:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 32-43

We also declare variables for input parameters and intermediate values:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 45-53

When compiled with MPI support, we initialize MPI at program startup.
The ``gauxc/gauxc_config.f`` header provides the ``GAUXC_HAS_MPI`` preprocessor macro for guarding MPI-specific calls.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 55-57

Similarly, at the end of the program we finalize MPI:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 262-264

Command line interface
----------------------

This guide uses an HDF5 input file to provide molecular data.
(GauXC also supports constructing these objects programmatically, which is more convenient when embedding GauXC in a library.)
We also expose parameters for configuring the integration grid and parallelization strategy.
The main command-line arguments are:

``input_file``
  Path to an HDF5 file containing the molecule, basis set, and density matrix.

``model``
  The model checkpoint to evaluate (e.g., ``PBE`` for a traditional functional or a path to a Skala checkpoint).

``grid_spec``
  The grid size specification, which controls the number of angular and radial integration points.

``rad_quad_spec``
  The radial quadrature scheme, which determines the spacing of radial integration points.

``prune_spec``
  The pruning scheme, which controls how atomic grids are combined into a molecular grid.

We use the ``command_line_interface`` type from the FLAP library to define the CLI.
First, we initialize variables with their default values.
These assignments appear immediately above the named ``input`` block in ``main.F90``:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 59-64

Next, we initialize the CLI and define the available arguments.
A named ``input`` block provides convenient error handling.
Add the following inside that block, right after ``input: block`` and before the call to ``cli%parse``:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 66-92,132-140

.. note::

   The molecular grid settings (``grid_spec``, ``rad_quad_spec``, ``prune_spec``) are documented in detail in the :ref:`gauxc_molecular_grid_settings` reference.

After defining the CLI, we parse it within the ``input`` block and retrieve the values:

This is still part of the same ``input`` block.
The block ends after the argument values are read, at ``end block input``.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 94-131

Before adding further implementation, let's build and test the project.
Configure and compile with:

.. code-block:: shell

   cmake -B build -G Ninja -S .
   cmake --build build

Once built, run the driver with ``--help`` to verify it works:

.. code-block:: shell

   ./build/Skala --help

You should see the help page generated by FLAP:

.. code-block:: text

   usage: ./build/Skala  value --model value [--grid value] [--radial-quadrature value] [--pruning-scheme value] [--lb-exec-space value] [--int-exec-space value] [--batch-size value] [--help] [--markdown] [--version]

   Driver for using Skala


   Required switches:
     value
       1-th argument
       Input HDF5 file containing molecule, basis set and density matrix
      --model value
       Model to use for the calculation

   Optional switches:
      --grid value, value in: `fine,ultrafine,superfine,gm3,gm5`
       default value fine
       Molecular grid specification
      --radial-quadrature value, value in: `becke,muraknowles,treutlerahlrichs,murrayhandylaming`
       default value muraknowles
       Radial quadrature to use
      --pruning-scheme value, value in: `unpruned,robust,treutler`
       default value robust
       Pruning scheme to use
      --lb-exec-space value, value in: `host,device`
       default value host
       Execution space for load balancer
      --int-exec-space value, value in: `host,device`
       default value host
       Execution space for integrator
      --batch-size value
       default value 512
       Batch size for grid point processing
      --help, -h
       Print this help message
      --markdown, -md
       Save this help message in a Markdown file
      --version, -v
       Print version

The CLI is now fully functional and allows flexible configuration from the command line.

Initializing GauXC
------------------

We begin by initializing the GauXC runtime environment.
All GauXC-related calls are placed inside the named ``main`` block for streamlined error handling.
This block starts after the command-line parsing section and contains the remainder of the GauXC workflow:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 146-157

At the end of the block, we check the status and clean up the runtime environment:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 245-250

The runtime environment provides the MPI world rank and size (for both MPI and non-MPI builds).
Still inside the ``main`` block, we print the configuration obtained from the command line:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90
   :lines: 159-168

From here on, we use ``world_rank`` to ensure only rank 0 produces output.


Molecule data
-------------

We use GauXC's built-in HDF5 reader to load the molecule data.
Add this immediately after the runtime configuration printout in the ``main`` block.

.. note::

   GauXC stores the information about the atomic numbers and their coordinates as an array of structs, defined in C++ as

   .. code-block:: c++

      struct GauXC::Atom {
        int64_t Z; ///< Atomic number
        double x;  ///< X coordinate (bohr)
        double y;  ///< Y coordinate (bohr)
        double z;  ///< Z coordinate (bohr)
      };
      class GauXC::Molecule : public std::vector<GauXC::Atom> {
        ...
      };

   The HDF5 wrapper maps this struct representation directly to the HDF5 dataset.

The ``gauxc_read_hdf5_record`` subroutine loads the molecule data:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (read molecule)
   :lines: 170-175

For proper memory management, we free the molecule object at program end:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (free molecule)
   :lines: 252


Basis set data
--------------

We load the basis set using the same HDF5 approach as for the molecule.
Place this directly after the molecule-loading code in the ``main`` block.

.. note::

   The basis set is represented as an array of shell objects, each containing primitive exponents,
   contraction coefficients, angular momentum, and shell center:

   .. code-block:: c++

      template <typename F>
      class alignas(256) GauXC::Shell {
        std::array<F, 32> alpha; ///< exponents of primitives
        std::array<F, 32> coeff; ///< contraction coefficients
        std::array<double, 3> O; ///< origin of the shell
        int32_t nprim; ///< number of primitives
        int32_t l; ///< angular moment of the shell
        int32_t pure; ///< pure=1: spherical Gaussians; pure=0: cartesian Gaussians
      };

      template <typename F>
      struct GauXC::BasisSet : public std::vector<GauXC::Shell<F>> {
        ...
      };

   Again, this allows the object's representation to map directly to an HDF5 dataset.

We read the basis set using ``gauxc_read_hdf5_record``:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (read basisset)
   :lines: 177-182

We free the basis set at program end:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (free basisset)
   :lines: 253

Integration grid
----------------

The integration grid defines the spatial points for evaluating the exchange-correlation functional.
Three parameters control grid construction:

- **Grid size**: Density of angular and radial points (e.g., ``fine``, ``ultrafine``)
- **Radial quadrature**: Spacing scheme for radial points (e.g., ``muraknowles``)
- **Pruning scheme**: How atomic grids combine into the molecular grid (e.g., ``robust``)

GauXC uses enumerators for these settings. We define helper functions to convert CLI strings:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (enumerator conversion functions)
   :lines: 266,279-322

We create the molecular grid from our input parameters.
The batch size controls how many grid points are processed together; larger values (up to ~10000)
improve performance, though the default is 512:

These lines follow the molecule and basis-set setup in the ``main`` block.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (grid setup)
   :lines: 184-189

We free the grid at program end:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (free grid)
   :lines: 254

Exchange-correlation integrator
-------------------------------

The load balancer distributes XC functional evaluation across available resources (host or device).
A helper function converts the execution-space CLI string to an enumerator:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (execution space enumerator)
   :lines: 266-277

The load balancer manages access to the molecule, basis, and grid data for all subsequent GauXC operations.
We create it from our input parameters:

Add this in the ``main`` block right after the grid has been constructed.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (load balancer setup)
   :lines: 191-207

Finally, we create the XC integrator.
The functional and load balancer are passed directly to the integrator constructor:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (integrator setup)
   :lines: 209-212

We free the integrator and associated objects at program end:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (free integrator)
   :lines: 255-260

Density matrix
--------------

The density matrix is the final input for GauXC.
Unlike the molecule and basis set, we read the density matrix using our own HDF5 helper subroutine ``read_matrix_from_hdf5_record``.
This subroutine opens the HDF5 file, reads a 2D dataset into an allocatable array, and performs error handling for each HDF5 operation.

The helper subroutine lives in the ``contains`` section at the end of the file, while the call site shown below remains in the ``main`` block after the integrator has been created.

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (HDF5 matrix reader)
   :lines: 324-380

With this helper we can read the density matrices from the input file:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (read density matrix)
   :lines: 214-218

Exchange-correlation evaluation
-------------------------------

With all inputs ready, we perform the XC evaluation:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (exchange-correlation evaluation)
   :lines: 225-229

.. tip::

   To measure evaluation time, define a helper function:

   .. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
      :language: fortran
      :caption: app/main.F90 (time helper function)
      :lines: 382-387

   Use it to wrap the evaluation and print elapsed time:

   .. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
      :language: fortran
      :caption: app/main.F90 (timed exchange-correlation evaluation)
      :lines: 220-243

We output the computed XC energy:

.. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
   :language: fortran
   :caption: app/main.F90 (exchange-correlation output)
   :lines: 237-243

Rebuild the project:

.. code-block:: shell

   cmake --build build

Run the driver from the build directory:

.. code-block:: shell

   ./build/Skala He_def2-svp.h5 --model PBE

.. note::

   Create the ``He_def2-svp.h5`` input file using the ``skala`` package:

   .. literalinclude:: scripts/export-h5.py
      :language: python

The output shows results for the PBE functional:

.. code-block:: text

   Configuration
   -> Input file        : He_def2-svp.h5
   -> Model             : PBE
   -> Grid              : fine
   -> Radial quadrature : muraknowles
   -> Pruning scheme    : robust

   Results
   Exc          = -1.0540318683E+00 Eh
   |VXC(a+b)|_F =  1.4559829661E+00
   |VXC(a-b)|_F =  0.0000000000E+00
   Runtime XC   =  2.5566819200E-01

Download checkpoint from HuggingFace
------------------------------------

To evaluate Skala, download the model checkpoint from HuggingFace using the ``hf`` CLI
from the ``huggingface_hub`` package:

.. code-block:: shell

   hf download microsoft/skala skala-1.0.fun --local-dir .
   ./build/Skala He_def2-svp.h5 --model ./skala-1.0.fun

The output shows results for the Skala functional:

.. code-block:: text

   Configuration
   -> Input file        : He_def2-svp.h5
   -> Model             : ./skala-1.0.fun
   -> Grid              : fine
   -> Radial quadrature : muraknowles
   -> Pruning scheme    : robust

   Results
   Exc          = -1.0712560874E+00 Eh
   |VXC(a+b)|_F =  1.5002997546E+00
   |VXC(a-b)|_F =  0.0000000000E+00
   Runtime XC   =  1.5986489670E+00

Full source code
----------------

.. dropdown:: Full source code of the main driver

   .. literalinclude:: ../../examples/fortran/gauxc_integration/app/main.F90
      :language: fortran
      :caption: app/main.F90

Summary
-------

This guide demonstrated GauXC usage in Fortran applications. We:

1. Created a minimal CMake project with GauXC as a dependency
2. Built a CLI driver that reads molecule, basis set, and density matrix from HDF5
3. Configured the integration grid, load balancer, and XC integrator
4. Evaluated the exchange-correlation energy and potential
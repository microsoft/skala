GauXC in C
==========

This guide shows how to use the GauXC library in C applications.
We will cover

* setting up a CMake project to include GauXC as dependency
* initializing the GauXC runtime environment
* reading molecule, basis set, and density matrix from an HDF5 input file
* setting up the integration grid, load balancer, and exchange-correlation integrator
* performing the exchange-correlation evaluation and outputting the result

Setting up CMake
----------------

GauXC can be most conveniently used via CMake, therefore we will setup a minimal CMake project for a command line driver to use GauXC.
Next to GauXC we will use other dependencies as needed.

The directory structure for the project will be

.. code-block:: text

   ├── CMakeLists.txt
   ├── app
   │   └── main.c
   └── cmake
       ├── skala-argtable3.cmake
       ├── skala-dep-versions.cmake
       └── skala-gauxc.cmake

First we create the main ``CMakeLists.txt`` to define our project, include our dependencies, and declare our executable.

.. literalinclude:: ../../examples/c/gauxc_integration/CMakeLists.txt
   :language: cmake
   :caption: CMakeLists.txt

For handling the dependencies, we create a separate file in the ``cmake/`` subdirectory to include the path and checksums for all our dependencies.

.. literalinclude:: ../../examples/c/gauxc_integration/cmake/skala-dep-versions.cmake
   :language: cmake
   :caption: cmake/skala-dep-versions.cmake

For each dependency we will create a separate CMake include file for finding and making it available.
First, we define how we will include GauXC our main dependency.
For this we can rely in most cases to discover the GauXC config file, however we also provide a fallback to download and build GauXC in case it is not available in the environment.
The options we defined in the main CMake file will be passed through to GauXC to ensure the library provides the requested features.
Furthermore, after having GauXC available, we double check whether our requirements for GauXC are satisfied, this is especially necessary for the Skala implementation, which requires the ``GAUXC_HAS_ONEDFT`` feature flag.

.. literalinclude:: ../../examples/c/gauxc_integration/cmake/skala-gauxc.cmake
   :language: cmake
   :caption: cmake/skala-gauxc.cmake

For our command line driver, we will be using Argtable3 to create the command line interface.

.. literalinclude:: ../../examples/c/gauxc_integration/cmake/skala-argtable3.cmake
   :language: cmake
   :caption: cmake/skala-argtable3.cmake

With this we have the full CMake setup we need for creating our command line driver.


Setting up headers
------------------

For our main driver program we include the relevant headers from GauXC, next to the ones needed for the HDF5 I/O and command line interface.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 1-23
   :caption: app/main.c (header includes)

For each of the GauXC components we will be using, we include the respective header file from GauXC.

`gauxc/status.h`
  For handling GauXC status codes and errors.

`gauxc/molecule.h`
  For handling molecular data structures, like atomic numbers and coordinates.

`gauxc/basisset.h`
  For handling basis set data, like basis function definitions.

`gauxc/molgrid.h`
  For setting up and managing the integration grid.

`gauxc/runtime_environment.h`
  For interacting with MPI and device runtime environments.

`gauxc/load_balancer.h`
  For setting up and managing the load balancer for distributing grid points.

`gauxc/molecular_weights.h`
  For computing molecular weights needed for grid generation.

`gauxc/functional.h`
  For handling exchange-correlation functionals.

`gauxc/xc_integrator.h`
  For setting up and managing the exchange-correlation integrator.

`gauxc/matrix.h`
  For handling matrix data structures, like density matrices and potentials.

We start our main driver program with a guarded call to the MPI initialization.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 115-120
   :caption: app/main.c (MPI initialize)

For the finalization of the MPI environment we also add a guarded call to the MPI finalize function at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 294-297
   :caption: app/main.c (MPI finalize)

GauXC provides a macro ``GAUXC_HAS_MPI`` to inform users whether GauXC was built with MPI support.
We can use this macro to conditionally compile the MPI initialization and finalization code only when GauXC was built with MPI support.


Creating the Command Line Driver
--------------------------------

We will provide our input data mainly via an HDF5 input file.
This file will contain the molecular structure, basis set, and density matrix.
Additionally, we have parameters for defining the integration grid and also the parallelization strategy.
Our main parameters for the input will therefore be

``input_file``
  an HDF5 input file to provide the molecule, basis and density matrix

``model``
  the model checkpoint we want to evaluate

``grid_spec``
  the grid size specification which defines the number of angular and radial integration points

``rad_quad_spec``
  the radial quadrature scheme which defines the spacing of radial integration points

``prune_spec``
  the pruning scheme for combining the atomic grids to a molecular one

For the command line driver we will use Argtable3 to handle the command line arguments concisely.
We define the command line arguments for the input HDF5 file, model type, and other parameters using Argtable3.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 121-146
   :caption: app/main.c (command line arguments)

.. note::

   The settings for the molecular grid (``grid_spec``, ``rad_quad_spec``, ``prune_spec``) are defined in more detail in :ref:`gauxc_molecular_grid_settings` reference.

With this we can parse the command line and handle potential errors.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 148-162
   :caption: app/main.c (parse command line arguments)

Finally, we can extract the values of the command line arguments and store them in variables for later use.
For this purpose we will define two helper functions, one for copying the values from the Argtable3 structs and a normalization function to ensure all inputs are lowercase.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 97-113
   :caption: app/main.c (helper functions for command line arguments)

With this we can extract the command line argument values.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 164-172
   :caption: app/main.c (extract command line argument values)

At this point we can already free the Argtable3 structures as we do not need them anymore.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 174-175
   :caption: app/main.c (free Argtable3 structures)

Also, we want to ensure our input string variables will get freed at the end of our program.
We add the respective ``free()`` calls at the end of our main program, before the MPI finalization.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 270-277
   :caption: app/main.c (free input strings)

Before adding any further implementation, we will create a first build of the project.
To configure the CMake build run

.. code-block:: shell

   cmake -B build -G Ninja -S .
   cmake --build build

After we build our project successfully, we can run our driver directly from the build directory with

.. code-block:: shell

   ./build/Skala --help

As output you should see the help page generated by our command-line interface

.. code-block:: text

   Usage: ./_build/Skala <file> --model=<str> [--grid-spec=<str>] [--radial-quad=<str>] [--prune-scheme=<str>] [--lb-exec-space=<str>] [--int-exec-space=<str>] [--batch-size=<int>] [--basis-tol=<double>] [--help]

   Options:
     <file>                    Input file containing molecular geometry and density matrix
     --model=<str>             OneDFT model to use, can be a path to a checkpoint
     --grid-spec=<str>         Atomic grid size specification (default: Fine)
                               Possible values are: Fine, UltraFine, SuperFine, GM3, GM5
     --radial-quad=<str>       Radial quadrature scheme (default: MuraKnowles)
                               Possible values are: Becke, MuraKnowles, TreutlerAhlrichs, MurrayHandyLaming
     --prune-scheme=<str>      Pruning scheme (default: Robust)
                               Possible values are: Unpruned, Robust, Treutler
     --lb-exec-space=<str>     Load balancer execution space
                               Possible values are: Host, Device
     --int-exec-space=<str>    Integrator execution space
                               Possible values are: Host, Device
     --batch-size=<int>        Batch size for grid point processing (default: 512)
     --basis-tol=<double>      Basis function evaluation tolerance (default: 1e-10)
     --help                    Print this help and exit

With this we are able to change our configuration conveniently from the command line.

.. tip::

   If you encounter any issues when running the driver, like segmentation faults, rerun your binary with a debugger, like gdb.

   .. code-block:: shell

      gdb --args ./build/Skala --help

   This way you can inspect the stack trace and find the source of the error more easily.


As first step for any interaction with GauXC, we need to initialize the GauXC status object and runtime environment.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 177-183
   :caption: app/main.c (GauXC status and runtime environment)

The ``world_size`` and ``world_rank`` variables describe our MPI environment and contain dummy values if we do not use MPI.

At this point we want to show the configuration we are using before proceeding further.
We can use the ``world_rank`` variable to ensure that only the root process outputs the configuration in case of MPI parallel execution.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 185-193
   :caption: app/main.c (configuration summary)

Finally, we also add calls to free the runtime environment at the end of our main program.
To ease freeing memory we will define a type generic macro for calling the respective free functions.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :lines: 25-39
   :caption: app/main.c (free macro)

With this we can free the runtime environment at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free runtime)
   :lines: 281

Molecule data
-------------

For reading the molecule we will use GauXC's built-in functionality to read from an HDF5 dataset.

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

   The HDF5 wrapper directly maps this struct representation to an HDF5 dataset.

We use ``gauxc_molecule_read_hdf5_record`` function which implements the reading of the molecule data.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (read molecule)
   :lines: 195-198

To ensure proper memory management we also add a call to free the molecule object at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free molecule)
   :lines: 278


Basis set data
--------------

For the basis set we will use the same approach as for the molecule and use GauXC's built-in HDF5 reading functionality.

.. note::

   Similar to the molecule the basis set object is built as an array of shell objects.
   The shell object contains the information about the primitives, contraction coefficients, angular momentum, and center of the shell.

   .. code-block:: c++

      template <typename F>
      class alignas(256) GauXC::Shell {
        std::array<F, 32> alpha; ///< exponents of primitives
        std::array<F, 32> coeff; ///< contraction coefficients
        std::array<double, 3> O; ///< origin of the shell
        int32_t nprim; ///< number of primitives
        int32_t l; ///< angular moment of the shell
        int32_t pure; ///< pure=1: spherical Gaussianss; pure=0: cartesian Gaussianss
      };

      template <typename F>
      struct GauXC::BasisSet : public std::vector<GauXC::Shell<F>> {
        ...
      };

   Again, this allows to directly map the object's representation to an HDF5 dataset.

With GauXC's ``gauxc_basisset_read_hdf5_record`` function we can read the basis set data conveniently from the HDF5 file.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (read basisset)
   :lines: 200-203

To avoid memory leaks we also add a call to free the basis set object at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free basisset)
   :lines: 279

Integration grid
----------------

To setup the integration grid, which is the part of the input to GauXC for computing the exchange-correlation functional, we create a molecular grid.
We have three main input parameters here, the grid size which defines the density of angular and radial points, the radial quadrature scheme which defines the spacing of the radial points, and the pruning scheme which defines how atomic grids are combined to a molecular grid.
In GauXC these are defined as enumerators and we add a number of helper functions for turning the input strings from the command-line to the respective enumerator values.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (enumerator conversion functions)
   :lines: 52-95

For the main program we can now create the molecular grid based on our input parameters.
We also have to define the batch size for the grid, the default is 512 points per batch, however larger values up around 10000 are recommended for better performance.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (grid setup)
   :lines: 205-215

We free the molecular grid object at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free grid)
   :lines: 280

Exchange-correlation integrator
-------------------------------

To distribute the work of evaluating the exchange-correlation functional on the grid points, we create a load balancer.
The load balancer will take care of distributing the grid points to the available resources, either host or device, based on the execuation space we provide.
Again we have a helper function to convert the input string to the respective enumerator value.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (execuation space enumerator)
   :lines: 41-50

Note that the load balancer will provide access to the molecule, basis and grid data for all further usage in GauXC.
We can now create the load balancer based on our input parameters.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (load balancer setup)
   :lines: 217-231

Finally, we can create the main GauXC integrator, for this we setup the exchange-correlation integrator factory for producing an instance of the integrator.
To configure the integrator we create an additional settings object which holds the model checkpoint we want to evaluate.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (integrator setup)
   :lines: 233-237

We free the integrator and its associated objects at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free integrator)
   :lines: 282-288

Density matrix
--------------

The final input we need to provide to GauXC is the density matrix.
Similar to the molecule and basis set we will read it from our HDF5 input file.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (read density matrix)
   :lines: 243-247

Also for the density matrix we add a call to free the matrix object at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free density matrix)
   :lines: 289-290

Exchange-correlation evaluation
-------------------------------

With all inputs provided we can now perform the exchange-correlation evaluation.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (exchange-correlation evaluation)
   :lines: 253-258

After the evaluation we can output the computed exchange-correlation energy.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (exchange-correlation output)
   :lines: 264-268

Since we have allocated new matrices for the exchange-correlation potential, we also need to free them at the end of our main program.

.. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
   :language: c
   :caption: app/main.c (free exchange-correlation potential)
   :lines: 291-292

Now we can rebuild our project with

.. code-block:: shell

   cmake --build build

After we build our project successfully, we run the driver again from the build directory

.. code-block:: shell

   ./build/Skala He_def2-svp.h5 --model PBE

.. note::

   The ``He_def2-svp.h5`` input file can be created with the ``skala`` package.

   .. literalinclude:: scripts/export-h5.py
      :language: python

As output we can see the results for the PBE functional

.. code-block:: text

   Configuration
   -> Input file        : He_def2-svp.h5
   -> Model             : PBE
   -> Grid              : fine
   -> Radial quadrature : muraknowles
   -> Pruning scheme    : robust

   Results
   -> EXC : -1.0540318683

Download checkpoint from HuggingFace
------------------------------------

To evaluate Skala we first need to download the model checkpoint from HuggingFace.
For this we can use the ``hf`` command line tool from the ``huggingface_hub`` Python package.
After downloading the model checkpoint we can run our driver again with the new model

.. code-block:: shell

   hf download microsoft/skala skala-1.0.fun --local-dir .
   ./build/Skala He_def2-svp.h5 --model ./skala-1.0.fun

In the output we can see the results for the Skala functional

.. code-block:: text

   Configuration
   -> Input file        : He_def2-svp.h5
   -> Model             : ./skala-1.0.fun
   -> Grid              : fine
   -> Radial quadrature : muraknowles
   -> Pruning scheme    : robust

   Results
   -> EXC : -1.0712560886

Full source code
----------------

.. dropdown:: Full source code of the main driver

   .. literalinclude:: ../../examples/c/gauxc_integration/app/main.c
      :language: c
      :caption: app/main.c

Summary
-------

Within this guide the usage of GauXC in C applications was demonstrated.
A minimal CMake project was created to setup the build environment and include GauXC.
We created a command line driver which reads the molecule, basis set, and density matrix from an HDF5 input file and sets up the integration grid, load balancer, and exchange-correlation integrator.
Finally, we evaluated the exchange-correlation energy and potential and output the result.
GauXC in C++
============

In this guide we will cover how to use the GauXC library in C++.
We will cover

* setting up a CMake project including GauXC as dependency
* initializing the GauXC runtime environment
* reading molecule, basis set, and density matrix from an HDF5 input file
* setting up the integration grid, load balancer, and exchange-correlation integrator
* performing the exchange-correlation evaluation and outputting the results

.. tip::

   For building GauXC and installing in the conda environment checkout :ref:`gauxc_install`.

Setting up CMake
----------------

GauXC can be most conveniently used via CMake, therefore we will setup a minimal CMake project for a command line driver to use GauXC.
Next to GauXC we will use other dependencies as needed.

The directory structure for the project will be

.. code-block:: text

   ├── CMakeLists.txt
   ├── app
   │   └── main.cxx
   └── cmake
       ├── skala-cli11.cmake
       ├── skala-dep-versions.cmake
       ├── skala-eigen3.cmake
       └── skala-gauxc.cmake

First we create the main ``CMakeLists.txt`` to define our project, include our dependencies, and declare our executable.

.. literalinclude:: ../../examples/cpp/gauxc_integration/CMakeLists.txt
   :language: cmake
   :caption: CMakeLists.txt

For handling the dependencies, we create a separate file in the ``cmake/`` subdirectory to include the path and checksums for all our dependencies.

.. literalinclude:: ../../examples/cpp/gauxc_integration/cmake/skala-dep-versions.cmake
   :language: cmake
   :caption: cmake/skala-dep-versions.cmake

For each dependency we will create a separate CMake include file for finding and making it available.
First, we define how we will include GauXC our main dependency.
For this we can rely in most cases to discover the GauXC config file, however we also provide a fallback to download and build GauXC in case it is not available in the environment.
The options we defined in the main CMake file will be passed through to GauXC to ensure the library provides the requested features.
Furthermore, after having GauXC available, we double check whether our requirements for GauXC are satisfied, this is especially necessary for the Skala implementation, which requires the ``GAUXC_HAS_ONEDFT`` feature flag.

.. literalinclude:: ../../examples/cpp/gauxc_integration/cmake/skala-gauxc.cmake
   :language: cmake
   :caption: cmake/skala-gauxc.cmake

While GauXC provides the implementation to evaluate the exchange-correlation functional, it is independent to the library used for storing matrices.
For our example here we will be using Eigen3.
Similar to GauXC we will attempt to find Eigen3 via its config file and fallback to downloading it.
Since Eigen3 is a header-only library, we just need to reexport the include directory of the project.

.. literalinclude:: ../../examples/cpp/gauxc_integration/cmake/skala-eigen3.cmake
   :language: cmake
   :caption: cmake/skala-eigen3.cmake

For our command line driver, we will be using CLI11 to create the command line interface.
Similar to Eigen3, CLI11 is a header-only library and we will use the same approach for including its headers if the dependency can not be found in the environment.

.. literalinclude:: ../../examples/cpp/gauxc_integration/cmake/skala-cli11.cmake
   :language: cmake
   :caption: cmake/skala-cli11.cmake

With this we have the full CMake setup we need for creating our command line driver.

Initializing GauXC
------------------

For the main program we start with including the relevant headers for GauXC.
In our case those come from GauXC for the main functionality of the library, HighFive for access to HDF5 files, Eigen3 for matrix types, and CLI11 for creating the command line interface.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (headers)
   :lines: 1-20

We start defining our main program with a set of default variables.
In this tutorial we will be using

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

Furthermore, we have the variables which define where GauXC is executing (host or device) and ``batch_size`` and ``basis_tol`` for the numerical settings of the evaluation.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (initialization)
   :lines: 118-133

.. note::

   The settings for the molecular grid (``grid_spec``, ``rad_quad_spec``, ``prune_spec``) are defined in more detail in :ref:`gauxc_molecular_grid_settings` reference.

Command line interface
----------------------

Next we will create a command line interface based on the CLI11 library.
Each of the default options we specified will be included there.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (command-line)
   :lines: 134-152

Before adding any further implementation, we can add the finalization to our main and create a first build of the project.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (finalization)
   :lines: 244-248

To configure the CMake build run

.. code-block:: shell

   cmake -B build -G Ninja -S .
   cmake --build build

After we build our project successfully, we can run our driver directly from the build directory with

.. code-block:: shell

   ./build/Skala --help

As output you should see the help page generated by our command-line interface

.. code-block:: text

   Skala GauXC driver 
   
   
   ./build/Skala [OPTIONS] input
   
   
   POSITIONALS:
     input TEXT:FILE REQUIRED    Input file in HDF5 format 
   
   OPTIONS:
     -h,     --help              Print this help message and exit 
             --model TEXT        Model checkpoint to evaluate 
             --grid-size TEXT [fine]  
                                 Grid specification (fine|ultrafine|superfine|gm3|gm5) 
             --radial-quad TEXT [muraknowles]  
                                 Radial quadrature specification 
                                 (becke|muraknowles|treutlerahlrichs|murrayhandylaming) 
             --prune-scheme TEXT [robust]  
                                 Pruning scheme (unpruned|robust|treutler) 
             --lb-exec-space TEXT [host]  
                                 Load balancing execution space 
             --int-exec-space TEXT [host]  
                                 Integration execution space 
             --batch-size INT [512]  
             --basis-tol FLOAT [1e-10]

Now that we are able to change our program variables conveniently from the command-line, we will initialize the GauXC runtime.
For this we are defining a new function to handle different cases, like having access to a device (GPU) or running MPI parallel.
GauXC provides preprocessor guards like ``GAUXC_HAS_DEVICE`` and ``GAUXC_HAS_MPI`` or convenience macros like ``GAUXC_MPI_CODE`` for defining conditional code paths.
When creating the runtime environment, we will provide it with the MPI world communicator if available and if we have a device preallocate memory on the device.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (get_runtime)
   :lines: 22-41

We return the runtime environment in the main program.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (runtime setup)
   :lines: 153-156

The ``world_size`` and ``world_rank`` variables describe our MPI environment and contain dummy values if we do not use MPI.

Before we continue with setting up GauXC, we include an output of our program variables.
Here we can use ``world_rank`` provided by the runtime environment to ensure only the root rank outputs the information.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (inputs)
   :lines: 158-167

Molecule data
-------------

For reading the molecule we will use GauXC's built-in functionality to read from an HDF5 dataset.

.. note::

   The ``GauXC::Molecule`` stores the information about the atomic numbers and their coordinates as an array of structs.

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

   This allows to directly map the object's representation to an HDF5 dataset.

We use ``GauXC::read_hdf5_record`` function which implements the reading of the molecule data.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (read_molecule)
   :lines: 43-50

In the main program we will just use our small wrapper function to obtain the molecule.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (molecule)
   :lines: 169-170

Basis set data
--------------

For the basis set we will use the same approach as for the molecule and use GauXC's built-in HDF5 reading functionality.

.. note::

   Similar to the molecule the ``GauXC::BasisSet`` object is built as an array of ``GauXC::Shell`` objects.
   The ``GauXC::Shell`` object contains the information about the primitives, contraction coefficients, angular momentum, and center of the shell.

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

With GauXC's ``GauXC::read_hdf5_record`` function we can read the basis set data conveniently from the HDF5 file.
Additionally, we are setting the basis set tolerance on the loaded basis set data, which will be taken from our input variables, by default we use a tolerance of 1e-10.
The basis set tolerance will be used for screening small contributions during the evaluation of the density on the grid points.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (read_basis)
   :lines: 52-62

In the main program we can use our wrapper function to load the basis set.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (basis)
   :lines: 172-173

Integration grid
----------------

To setup the integration grid, which is the part of the input to GauXC for computing the exchange-correlation functional, we create a molecular grid.
We have three main input parameters here, the grid size which defines the density of angular and radial points, the radial quadrature scheme which defines the spacing of the radial points, and the pruning scheme which defines how atomic grids are combined to a molecular grid.
In GauXC these are defined as enumerators and we add a number of helper functions for turning the input strings from the command-line to the respective enumerator values.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (grid setting)
   :lines: 64-98

For the main program we can now create the molecular grid based on our input parameters.
We also have to define the batch size for the grid, the default is 512 points per batch, however larger values up around 10000 are recommended for better performance.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (grid)
   :lines: 175-181

Exchange-correlation integrator
-------------------------------

To distribute the work of evaluating the exchange-correlation functional on the grid points, we create a load balancer.
The load balancer will take care of distributing the grid points to the available resources, either host or device, based on the execuation space we provide.
Note that the load balancer will provide access to the molecule, basis and grid data for all further usage in GauXC.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (load balancer)
   :lines: 183-205

Finally, we can create the main GauXC integrator, for this we setup the exchange-correlation integrator factory for producing an instance of the integrator.
To configure the integrator we create an additional settings object which holds the model checkpoint we want to evaluate.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (integrator)
   :lines: 207-214

Density matrix
--------------

The final input we need to provide to GauXC is the density matrix.
Similar to the molecule and basis set we will read it from our HDF5 input file using the HighFive library directly.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (load_density_matrix)
   :lines: 100-116

For the model checkpoint we always use two spin channels and therefore have the scalar density matrix (alpha + beta spin channel) and the polarization density matrix (alpha - beta spin channel).

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (density matrix)
   :lines: 216-218

Exchange-correlation evaluation
-------------------------------

With all inputs provided we can now perform the exchange-correlation evaluation.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (exchange-correlation)
   :lines: 225-228

.. tip::

   For timing the execution we can add an optional timer around the evaluation and synchronize the MPI processes before and after the evaluation to get accurate timings.

   .. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
      :language: c++
      :caption: app/main.cxx (timer)
      :lines: 220-234

After the evaluation we can output the computed exchange-correlation energy and potential.

.. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
   :language: c++
   :caption: app/main.cxx (output)
   :lines: 236-243

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

   EXC          = -1.054031868349e+00 Eh
   |VXC(a+b)|_F = 1.455982966065e+00
   |VXC(a-b)|_F = 0.000000000000e+00
   Runtime XC   = 4.382018760000e-01 s

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
   -> Model             : ./skala-v1.0.fun
   -> Grid              : fine
   -> Radial quadrature : muraknowles
   -> Pruning scheme    : robust

   EXC          = -1.071256087389e+00 Eh
   |VXC(a+b)|_F = 1.500299750739e+00
   |VXC(a-b)|_F = 0.000000000000e+00
   Runtime XC   = 1.792662281000e+00 s

Full source code
----------------

.. dropdown:: Full source code of the main driver

   .. literalinclude:: ../../examples/cpp/gauxc_integration/app/main.cxx
      :language: c++
      :caption: app/main.cxx

Summary
-------

In this guide we covered how to use the GauXC library in C++.
We created a minimal CMake project to setup the build environment and included GauXC.
We created a command line driver which reads the molecule, basis set, and density matrix from an HDF5 input file and sets up the integration grid, load balancer, and exchange-correlation integrator.
Finally, we performed the exchange-correlation evaluation and output the results.

Troubleshooting
---------------

The link interface of target "gauxc::gauxc" contains "gau2grid::gg" but the target was not found
    Explicitly find the gau2grid package together with GauXC by adding the following lines at the end of the GauXC CMake include file

    .. code-block:: cmake
       :caption: cmake/skala-gauxc.cmake (append)

       if(GAUXC_HAS_GAU2GRID AND NOT TARGET gau2grid::gg)
         find_package(gau2grid CONFIG REQUIRED)
       endif()

OpenMP not found with Apple Clang
    On macOS OpenMP support is not provided by default with Apple Clang.
    Either disable OpenMP using the CMake option ``-DSkala_GauXC_ENABLE_OPENMP=OFF`` or install a version of Clang with OpenMP support, for example from conda-forge, and set the ``CXX`` environment variable to point to the new compiler.

libtorch not found
    Ensure that the libtorch library is installed and in the ``CMAKE_PREFIX_PATH``.
    The libtorch library is part of the pytorch package on conda-forge and can be installed with

    .. code-block:: shell

       conda install -c conda-forge pytorch
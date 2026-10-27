Using Skala in Fortran with FTorch
==================================

This page provides an example of how to use Skala in Fortran with the FTorch library.
The example demonstrates how to load a Skala model, prepare input features, compute the exchange-correlation energy and potential, and access the results.

.. note::

   This example evaluates the Skala model on precomputed density features and is meant as a starting point for calling the model from your own code.
   To add Skala to an existing electronic structure code, where the integration grid, the density features, and the exchange-correlation potential have to be computed from a density matrix, see `SkalaXC <https://github.com/microsoft/skala/tree/main/SkalaXC>`__ and its supported backends, which also provide Fortran bindings.
   A :doc:`C++ version <cpp>` of this example is also available.


Setting up development environment
----------------------------------

The repository's locked ``ftorch`` `Pixi <https://pixi.sh>`__ environment provides everything needed for this example:
LibTorch, CMake, Ninja, and the C, C++, and Fortran toolchains, the ``hf`` CLI to download the model checkpoint, and the ``skala`` runtime with PySCF to generate the input features.
From the repository root, activate the environment:

.. code-block:: bash

   pixi shell --locked -e ftorch

All following commands are executed in this shell from the repository root.
Configure, build, and install the example with:

.. code-block:: bash

   cmake -B build_example -S model/examples/fortran/ftorch_integration -G Ninja \
     -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
   cmake --build build_example
   ctest --test-dir build_example --output-on-failure
   cmake --install build_example


Preparing the model and input features
--------------------------------------

To evaluate Skala, we download the model checkpoint from HuggingFace using the ``hf`` CLI from the ``huggingface_hub`` package:

.. code-block:: bash

   hf download microsoft/skala-1.1 skala-1.1-rev1.fun --local-dir .

The example reads the input features from torch ``.pt`` files, one file per feature.
They are generated for the H2 molecule with the def2-QZVP basis set by the ``model/examples/prepare_inputs.py`` script:

.. code-block:: bash

   python model/examples/prepare_inputs.py --output-dir features

.. dropdown:: Feature preparation script

   .. literalinclude:: ../model/examples/prepare_inputs.py
      :language: python
      :caption: model/examples/prepare_inputs.py

.. tip::

   Here we load the features directly from torch ``.pt`` files, in your application you may want to create them directly from Fortran arrays as supported in FTorch.

With the checkpoint and the features in place, run the example:

.. code-block:: bash

   Skala ./skala-1.1-rev1.fun ./features

Model trust and loading errors
------------------------------

The example accepts only the SHA-256-pinned ``skala-1.1-rev1.fun`` checkpoint.
The native loader hashes the file before deserializing it and loads the same
verified bytes. OpenSSL's Crypto library is required and is included in the
locked environment. To use a different model, obtain its digest through a trusted
channel and explicitly update the application; do not trust a digest supplied
alongside an untrusted download.

The Fortran API requires ``expected_sha256`` as its third argument:

.. code-block:: fortran

   call skala_model_load(model, path, expected_sha256, status=status)
   if (status /= 0) then
     ! Handle the failure without using model.
   end if

Loading and metadata errors are logged by the native wrapper and return a null
handle rather than allowing C++ exceptions across the C boundary. With the
optional ``status`` argument, failure returns a nonzero status; without it,
the Fortran wrapper terminates explicitly with ``error stop``.
Only load feature tensor files from trusted sources as well.


Build system setup
------------------

For building the Fortran application, we will use CMake.
We will use the following directory structure and files:

.. code-block:: text

   model/examples/
   ├── prepare_inputs.py
   └── fortran/
       └── ftorch_integration/
           ├── CMakeLists.txt
           ├── app/
           │   └── main.f90
           ├── cmake/
           │   ├── skala-dep-versions.cmake
           │   └── skala-ftorch.cmake
           └── src/
               ├── skala_ftorch.cxx
               └── skala_ftorch.f90

For the main ``CMakeLists.txt`` file, we will set up the project and include the necessary CMake modules for Skala and FTorch:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/CMakeLists.txt
   :language: cmake
   :caption: CMakeLists.txt

To ensure that we have the correct versions of our dependencies, we will include a CMake module that specifies the versions of Skala and FTorch:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/cmake/skala-dep-versions.cmake
   :language: cmake
   :caption: cmake/skala-dep-versions.cmake

Next, we will include the CMake modules for Skala and FTorch, which will handle finding the libraries and setting up the necessary include directories and link targets:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/cmake/skala-ftorch.cmake
   :language: cmake
   :caption: cmake/skala-ftorch.cmake


FTorch addons
-------------

We create a number of extensions to the FTorch library in the ``src/`` directory, which provide the necessary Fortran bindings to interact with Skala.
However, we will not go into the details of these files here, as they are primarily focused on the Fortran-C++ interoperability.

.. dropdown:: Fortran bindings for Skala

   The files ``skala_ftorch.cxx`` and ``skala_ftorch.f90`` contain the C++ and Fortran code, respectively, that define the bindings between Skala and FTorch.
   These files include functions for loading Skala models, preparing input features, and computing exchange-correlation energies and potentials.

   .. literalinclude:: ../model/examples/fortran/ftorch_integration/src/skala_ftorch.cxx
      :language: c++
      :caption: src/skala_ftorch.cxx

   .. literalinclude:: ../model/examples/fortran/ftorch_integration/src/skala_ftorch.f90
      :language: fortran
      :caption: src/skala_ftorch.f90


Fortran application
-------------------

Finally, we have the Fortran application itself, which demonstrates how to use the Skala bindings to compute exchange-correlation energies and potentials.
We start the main program with the necessary module imports and variable declarations:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: declarations
   :end-before: ! docs-end: declarations

First, we obtain the command line arguments for the model path and feature directory, and check that they are provided:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: command-line
   :end-before: ! docs-end: command-line

We define a small contained helper procedure to read the command line arguments:

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: get-argument
   :end-before: ! docs-end: get-argument

The main types provided by the Skala bindings are the ``skala_model`` type, which extends the ``torch_model`` provided by FTorch and has a custom ``skala_model_load`` procedure for loading the Skala model and its meta data.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: load-model
   :end-before: ! docs-end: load-model

The input to the Skala model is passed via a dictionary of tensors, which we prepare by loading the necessary features from disk and converting them to the appropriate format.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: load-features
   :end-before: ! docs-end: load-features

To place the features in the correct format for Skala, we add them to the input dictionary with the appropriate keys.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: prepare-input
   :end-before: ! docs-end: prepare-input

.. note::

   The input dictionary accepts both single tensors and lists of tensors.
   If your program operates on multiple grids, you can pass those together to the input dictionary and Skala will handle them correctly.

   .. code-block:: fortran

      type(skala_dict) :: input
      type(torch_tensor) :: grid_coords(3), grid_weights(3)
      ! Load the grid features into the arrays
      call input%insert(skala_feature%grid_coords, grid_coords)
      call input%insert(skala_feature%grid_weights, grid_weights)

With this we can now compute the exchange-correlation energy and potential by calling the Skala model with the prepared inputs.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: evaluate-model
   :end-before: ! docs-end: evaluate-model

To get the exchange-correlation energy, we need to weight the exc values by the grid weights and sum over the grid points, which we can do using the provided tensor operations.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: integrate-energy
   :end-before: ! docs-end: integrate-energy

.. note::

   The used FTorch version 1.0.0 does not yet support the ``torch_tensor_sum`` operation, therefore we provide an implementation ourselves.
   Future versions of FTorch might cover more features which are provided by the FTorch extensions in the ``src/`` directory, so be sure to check the documentation for the latest updates.

Now we can build and install our application:

.. code-block:: bash

   cmake --build build_example
   cmake --install build_example

And run it, passing the path to the Skala model and the feature directory as command line arguments:

.. code-block:: bash

   Skala ./skala-1.1-rev1.fun ./features

The output for the H2 molecule with the def2-QZVP basis set should look like this:

.. code-block:: text

   [1] Loading model from ./skala-1.1-rev1.fun
   [2] Loading features from ./features
    -> Loading atomic_grid_size_bound_shape
    -> Loading coarse_0_atomic_coords
    -> Loading grad
    -> Loading grid_coords
    -> Loading kin
    -> Loading atomic_grid_sizes
    -> Loading atomic_grid_weights
    -> Loading grid_weights
    -> Loading density
   [3] Preparing input dictionary
   [4] Running model inference
   [5] Computing XC energy = sum(exc * grid_weights)
    -> E_xc =  -6.24491437792573E-01

The ``get_exc_vxc`` procedure computes the exchange-correlation energy and potential, which we can then access from the returned dictionary.
The potential terms are stored under the same keys as the input features and can be extracted as tensors.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: extract-gradients
   :end-before: ! docs-end: extract-gradients

We can use those tensors for further processing in our application, for example to compute the norm of the potential

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: gradient-means
   :end-before: ! docs-end: gradient-means

Or by converting them to Fortran arrays and using the built-in array operations.

.. literalinclude:: ../model/examples/fortran/ftorch_integration/app/main.f90
   :language: fortran
   :caption: app/main.f90
   :start-after: ! docs-start: array-access
   :end-before: ! docs-end: array-access

We rebuild and reinstall the application to include the latest changes:

.. code-block:: bash

   cmake --build build_example
   cmake --install build_example

Finally, we can run the application again to see the updated output with the computed potential values.

.. code-block:: bash

   Skala ./skala-1.1-rev1.fun ./features

In the output we can see the computed exchange-correlation energy as well as the mean values of the potential components, and the raw tensor data for each component.

.. code-block:: text

   [1] Loading model from ./skala-1.1-rev1.fun
   [2] Loading features from ./features
    -> Loading atomic_grid_size_bound_shape
    -> Loading coarse_0_atomic_coords
    -> Loading grad
    -> Loading grid_coords
    -> Loading kin
    -> Loading atomic_grid_sizes
    -> Loading atomic_grid_weights
    -> Loading grid_weights
    -> Loading density
   [3] Preparing input dictionary
   [4] Running model inference
   [5] Computing XC energy = sum(exc * grid_weights)
    -> E_xc =  -6.24491437792573E-01
   [6] Extracting vxc components
   [7] Gradient means (dexc/dx)
    -> mean(dexc/d_density)                =  -4.22419335262521E-03
    -> mean(dexc/d_grad)                   =   7.34686947724400E-14
    -> mean(dexc/d_kin)                    =  -9.29253735658196E-04
    -> mean(dexc/d_grid_coords)            =  -6.87739518316912E-15
    -> mean(dexc/d_grid_weights)           =  -3.47625888953444E-02
    -> mean(dexc/d_coarse_0_atomic_coords) =   6.74534920784954E-11
   [8] Accessing tensor data as Fortran arrays
    -> exc: shape = (19616)
         [ -2.27818608240367E-01 -2.27817104148283E-01 -2.27798286191804E-01 ...]
    -> dexc/d_density: shape = (19616, 2)
        [[ -7.42783701343921E-15 -2.88117252256267E-12 -9.41193381837591E-11 ...]
         [ -7.42783701343921E-15 -2.88117252256267E-12 -9.41193381837591E-11 ...]]
    -> dexc/d_grad: shape = (19616, 3, 2)
       [[[ -2.26087793682497E-19 -8.04381692417593E-16 -9.40376765067130E-14 ...]
         [  7.74835723433842E-31  2.99945981309767E-28  9.57635302305716E-27 ...]]
        [[  ...                    ]]]
    -> dexc/d_kin: shape = (19616, 2)
        [[  7.35445499050557E-16  2.85382228628491E-13  9.36659695851134E-12 ...]
         [  7.35445499050557E-16  2.85382228628491E-13  9.36659695851134E-12 ...]]
    -> dexc/d_grid_coords: shape = (3, 19616)
        [[ -1.18742738203596E-20 -4.99830311741504E-29  1.59096762782194E-21 ...]
         [ -4.23435244490882E-17 -1.78174745984425E-25  5.66422957219616E-18 ...]]
    -> dexc/d_grid_weights: shape = (19616)
         [ -2.27818608240367E-01 -2.27817104148283E-01 -2.27798286191804E-01 ...]
    -> dexc/d_coarse_0_atomic_coords: shape = (3, 2)
        [[ -1.68408204319782E-11 -9.23455002109453E-12  7.69511908655255E-04]
         [  ...]]

.. note::

   You are expected to add the D3(BJ) dispersion correction (using B3LYP damping parameters) to the final energy of Skala.
   You can use e.g. `simple-dftd3 <https://github.com/loriab/simple-dftd3>`_ for that.

Summary
-------

In this example, we have demonstrated how to use Skala in Fortran with the FTorch library.
We have shown how to set up the development environment, build the application using CMake, and run the application to compute exchange-correlation energies and potentials using a Skala model.
This example serves as a starting point for integrating Skala into your Fortran applications using FTorch, and can be extended to include more complex features and functionalities as needed.

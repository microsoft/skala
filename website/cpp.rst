Using Skala in C++ with LibTorch
================================

This page provides an example of how to use Skala in C++ with the LibTorch library.
The example demonstrates how to load a Skala model, prepare input features, compute the exchange-correlation energy and its derivatives with respect to the input features, and access the results.

.. note::

   This example evaluates the Skala model on precomputed density features and is meant as a starting point for calling the model from your own code.
   To add Skala to an existing electronic structure code, where the integration grid, the density features, and the exchange-correlation potential have to be computed from a density matrix, use `GauXC <https://microsoft.github.io/skala/gauxc/>`__ instead.
   A :doc:`Fortran version <ftorch>` of this example is also available.


Setting up development environment
----------------------------------

The repository's locked ``cpp-integration`` `Pixi <https://pixi.sh>`__ environment provides everything needed for this example:
LibTorch, nlohmann_json, CMake, Ninja, and the C and C++ toolchains, the ``hf`` CLI to download the model checkpoint, and the ``skala`` runtime with PySCF to generate the input features.
From the repository root, activate the environment:

.. code-block:: bash

   pixi shell --locked -e cpp-integration

All following commands are executed in this shell from the repository root.
Configure and build the example with:

.. code-block:: bash

   cmake -B build_example -S model/examples/cpp/cpp_integration -G Ninja
   cmake --build build_example


Preparing the model and input features
--------------------------------------

To evaluate Skala, we download the model checkpoint from HuggingFace using the ``hf`` CLI from the ``huggingface_hub`` package:

.. code-block:: bash

   hf download microsoft/skala-1.1 skala-1.1-rev1.fun --local-dir .

The example reads the input features from torch ``.pt`` files, one file per feature.
They are generated for the H2 molecule with the def2-QZVP basis set by the ``model/examples/prepare_inputs.py`` script, which is shared by the C++ and :doc:`Fortran <ftorch>` examples:

.. code-block:: bash

   python model/examples/prepare_inputs.py --output-dir features

.. dropdown:: Feature preparation script

   .. literalinclude:: ../model/examples/prepare_inputs.py
      :language: python
      :caption: model/examples/prepare_inputs.py

.. tip::

   Here we load the features directly from torch ``.pt`` files, in your application you may want to create them directly from your own arrays, for example with ``torch::from_blob``.

With the checkpoint and the features in place, run the example:

.. code-block:: bash

   ./build_example/skala_cpp_integration ./skala-1.1-rev1.fun ./features


Build system setup
------------------

For building the C++ application, we will use CMake.
We will use the following directory structure and files:

.. code-block:: text

   model/examples/
   ├── prepare_inputs.py
   └── cpp/
       └── cpp_integration/
           ├── CMakeLists.txt
           └── main.cpp

The ``CMakeLists.txt`` file sets up the project, finds LibTorch and nlohmann_json, and declares the executable:

.. literalinclude:: ../model/examples/cpp/cpp_integration/CMakeLists.txt
   :language: cmake
   :caption: CMakeLists.txt


C++ application
---------------

We start the program with the necessary headers and some type aliases for the TorchScript calling convention:

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: imports
   :end-before: // docs-end: imports

The Skala checkpoint is a TorchScript module.
Next to the module, the checkpoint stores the list of required input features and the protocol version as extra files, which we read while loading the model.
The ``get_exc_density`` method of the module evaluates the exchange-correlation energy density on the grid.

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: load-model
   :end-before: // docs-end: load-model

The input to the Skala model is passed via a dictionary of tensors.
We load each required feature from its ``.pt`` file and insert it into the dictionary under its feature name:

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: load-features
   :end-before: // docs-end: load-features

With the model and the features available, we evaluate the exchange-correlation energy density by calling the model method with the feature dictionary as ``mol`` keyword argument:

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: evaluate-energy
   :end-before: // docs-end: evaluate-energy

To obtain the derivatives of the exchange-correlation energy with respect to the input features, which are needed to build the exchange-correlation potential, we enable gradients on the relevant feature tensors and use LibTorch's autograd:

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: evaluate-gradients
   :end-before: // docs-end: evaluate-gradients

Finally, the main program reads the command line arguments, loads model and features, and computes the exchange-correlation energy by weighting the energy density with the grid weights and summing over the grid points:

.. literalinclude:: ../model/examples/cpp/cpp_integration/main.cpp
   :language: c++
   :caption: main.cpp
   :start-after: // docs-start: main
   :end-before: // docs-end: main

Running the example
-------------------

Rebuild the application after changing the sources:

.. code-block:: bash

   cmake --build build_example

And run it, passing the path to the Skala model and the feature directory as command line arguments:

.. code-block:: bash

   ./build_example/skala_cpp_integration ./skala-1.1-rev1.fun ./features

The output for the H2 molecule with the def2-QZVP basis set should look like this:

.. code-block:: text

   Compute Exc...
   Exc = -0.624491
   Compute Exc and dExc/dfeat...
   Exc = -0.624491
   |dExc/d(density)| = 1.39944
   |dExc/d(kin)| = 0.846093
   |dExc/d(grad)| = 0.0237195
   |dExc/d(grid_coords)| = 0.000438993
   |dExc/d(grid_weights)| = 9.72895
   |dExc/d(coarse_0_atomic_coords)| = 0.00108825

.. note::

   You are expected to add the D3(BJ) dispersion correction (using B3LYP damping parameters) to the final energy of Skala.
   You can use e.g. `simple-dftd3 <https://github.com/loriab/simple-dftd3>`_ for that.

.. tip::

   The `performance tuning guide <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`__ from Intel provides useful tips on how to tune the performance of PyTorch models on CPU.


Summary
-------

In this example, we have demonstrated how to use Skala in C++ with the LibTorch library.
We have shown how to set up the development environment, build the application using CMake, and run the application to compute exchange-correlation energies and their derivatives using a Skala model.
This example serves as a starting point for integrating Skala into your C++ applications using LibTorch, and can be extended to include more complex features and functionalities as needed.
For integrating Skala into an existing electronic structure code, see also the `GauXC integration <https://microsoft.github.io/skala/gauxc/>`__.

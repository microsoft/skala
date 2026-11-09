C++ tutorial: evaluate a density matrix
==========================================

After following :doc:`installation`, run the checked-in C++ example:

.. code-block:: bash

   pixi run -e gauxc-openmp SkalaXC/build/examples/skalaxc_eval_cpp \
     SkalaXC/tests/ref_data/skala_he_def2qzvp_pbe_uks.hdf5 PBE

The fixture supplies ``/MOLECULE``, ``/BASIS``, ``/DENSITY_SCALAR``, and
``/DENSITY_Z``. The program prints the model, basis size, exchange-correlation
energy, and scalar-potential norm. In an SCF application, supply your own
density matrices instead of loading the fixture.

The example constructs the runtime, grid, load balancer, and molecular weights
before creating the integrator. Weight partitioning must happen before
evaluation. Eigen and HighFive belong to this example, not the public
SkalaXC dependency surface.

.. literalinclude:: ../../SkalaXC/examples/cpp/skalaxc_eval.cpp
   :language: cpp

To obtain nuclear gradients with the same integrator:

.. code-block:: cpp

   auto gradient = integrator.eval_exc_grad(Ps, Pz);

The result contains exactly three components per atom in xyz order.
Keep the runtime and any MPI communicator alive until dependent objects have
been destroyed. The supplied example initializes/finalizes MPI conditionally.

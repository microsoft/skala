Fortran tutorial: evaluate a density matrix
============================================================

Follow :doc:`installation` with the Fortran-enabled environment. The example
requires the HDF5 Fortran component to read its own density input:

.. code-block:: bash

   pixi run -e gauxc-openmp-fortran SkalaXC/build/examples/skalaxc_eval_fortran \
     SkalaXC/tests/ref_data/skala_he_def2qzvp_tpss_uks.hdf5 TPSS

The program creates the runtime, molecule, basis, grid, load balancer,
weights, functional, and integrator in a scoped block. It reports the model,
basis size, exchange-correlation energy and scalar-potential norm.

.. literalinclude:: ../../SkalaXC/examples/fortran/skalaxc_eval.F90
   :language: fortran

Replace the fixture-reading section with the scalar and z-spin density
arrays supplied by your SCF program. The example assumes a trusted bundled
fixture; validate dataset dimensions and HDF5 return codes when accepting
external data. Keep all API status checks and the scoped lifetime.

For MPI, use ``gauxc-mpi-fortran``, configure ``SKALAXC_ENABLE_MPI=ON``, and
launch the example with your MPI launcher. All ranks in the runtime
communicator must participate. See :doc:`fortran-api` for move-only ownership
and gradient buffer requirements.

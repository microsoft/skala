Python tutorial: evaluate a density matrix
============================================================

After :doc:`python-api`, run the checked-in tutorial with the bundled PBE
fixture:

.. code-block:: bash

   OMP_NUM_THREADS=4 pixi run -e skalaxc-python-host \
     python SkalaXC/python/examples/evaluate.py \
     SkalaXC/tests/ref_data/skala_he_def2qzvp_pbe_uks.hdf5 PBE

.. literalinclude:: ../../SkalaXC/python/examples/evaluate.py
   :language: python

The example prints energy, potential norm, and gradient shape. In an SCF
program, replace HDF5 loading with your scalar and z-spin density matrices.
The arrays returned by the integrator are independent NumPy outputs.

The same program supports an MPI-enabled installation: use
``skalaxc-python-mpi``, enable native ``SKALAXC_ENABLE_MPI=ON``, and launch
the program on all ranks. It passes ``MPI.COMM_WORLD`` explicitly at the
application boundary; the library uses that supplied communicator.

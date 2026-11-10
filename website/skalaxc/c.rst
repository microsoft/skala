C tutorial: evaluate a density matrix
============================================================

Follow :doc:`installation`, including the install step so named models are
available. With examples enabled, run:

.. code-block:: bash

   pixi run -e gauxc-openmp SkalaXC/build/examples/skalaxc_eval_c \
     SkalaXC/tests/ref_data/skala_he_def2qzvp_lda_uks.hdf5 LDA

The program prints the model, basis size, exchange-correlation energy and
scalar-potential norm. HDF5 is used only to load this application's density
input, not as a dependency exposed by the SkalaXC API.

.. literalinclude:: ../../SkalaXC/examples/c/skalaxc_eval.c
   :language: c

The companion reader checks multiplication overflow and requires an exact
two-dimensional ``nbf`` by ``nbf`` dataset before reading into the allocation:

.. literalinclude:: ../../SkalaXC/examples/c/density_io.h
   :language: c

For a real SCF solver, replace file loading with your scalar and z-spin
density arrays. Retain the checked allocation, status handling and reverse
cleanup order. See :doc:`c-api` for the evaluation and gradient contracts.

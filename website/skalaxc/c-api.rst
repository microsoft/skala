C API
=====

Include ``<skalaxc/skalaxc.h>``. The C ABI uses opaque handles and plain data;
no C++, GauXC, or LibTorch types cross it. A C-only consumer can use:

.. code-block:: cmake

   project(my_solver LANGUAGES C)
   find_package(SkalaXC REQUIRED CONFIG COMPONENTS C)
   add_executable(my_solver main.c)
   target_link_libraries(my_solver PRIVATE SkalaXC::skalaxc)

The pipeline is runtime, molecule/basis, grid, load balancer, molecular
weights, functional, and integrator. Check each status against
``SKALAXC_SUCCESS`` and report ``skalaxc_last_error_message()`` before cleanup.
Initialize handles to ``NULL`` and destroy each owned handle exactly once,
in reverse construction order. The :doc:`c` demonstrates cleanup on every
failure path.

Evaluation accepts caller-owned ``double`` arrays in column-major order,
with ``nbf * nbf`` elements per density or potential channel. Nuclear
gradients need ``3 * natoms`` elements in atom-major xyz order. Inputs are
the scalar and z-spin density channels, not separate alpha/beta matrices.
The supported calculation is UKS; RKS is not implemented.

MPI builds take the caller's native communicator and require collective
evaluation over that communicator. Keep it alive until dependent handles
are destroyed, and destroy those handles before finalizing MPI.

Status and error reporting
------------------------------------------------------------

.. literalinclude:: ../../SkalaXC/include/skalaxc/c/status.h
   :language: c

Evaluation and diagnostics
------------------------------------------------------------

.. literalinclude:: ../../SkalaXC/include/skalaxc/c/integrator.h
   :language: c

.. literalinclude:: ../../SkalaXC/include/skalaxc/c/diagnostics.h
   :language: c

The umbrella header lists the remaining construction and configuration APIs:

.. literalinclude:: ../../SkalaXC/include/skalaxc/skalaxc.h
   :language: c

Fortran API
============================================================

The ``skalaxc`` module wraps only the public C ABI through ``iso_c_binding``.
Use the installed module and its CMake target:

.. code-block:: cmake

   project(my_solver LANGUAGES Fortran)
   find_package(SkalaXC REQUIRED CONFIG COMPONENTS Fortran)
   add_executable(my_solver main.f90)
   target_link_libraries(my_solver PRIVATE SkalaXC::skalaxc_fortran)

The module is compiler-specific: build the library and consumer with
compatible Fortran compilers. No GauXC or LibTorch include paths are needed.
``SKALAXC_BUILD_FORTRAN=OFF`` omits the module and its CMake component.

Ownership and error handling
------------------------------------------------------------

Each derived handle type owns its native resource. Handles are move-only:
ordinary assignment terminates with an explicit diagnostic rather than
creating aliased owners. This applies to scalars and arrays of every rank.
Transfer ownership explicitly:

.. code-block:: fortran

   call destination%move_from(source)

This destroys any previous destination resource and empties the source.
For arrays, move individual elements. Scoped handles finalize automatically;
explicit ``%destroy()`` is safe before scope exit. Use a ``block`` or procedure
scope, as demonstrated in :doc:`fortran`, so resources are finalized before
``MPI_Finalize``. Do not copy enclosing structures containing owned handles.

Status-returning operations must be checked against ``SKALAXC_SUCCESS``.
Report ``skalaxc_last_error()`` on failure. Array storage is
``real(c_double)``, column-major, with ``nbf * nbf`` values for each scalar
or z-spin density/potential channel; gradients contain ``3 * natoms``
atom-major xyz values. The evaluation methods accept flat contiguous arrays.

Public handle operations
------------------------------------------------------------

.. literalinclude:: ../../SkalaXC/src/fortran/skalaxc.F90
   :language: fortran
   :start-at:    type :: skalaxc_runtime_environment_t
   :end-at:    end type skalaxc_xc_integrator_t

Settings and diagnostics mirror the C API. In MPI builds, create the runtime
with the caller's Fortran communicator and keep that communicator alive while
any dependent handle exists. Evaluations are collective over that communicator.

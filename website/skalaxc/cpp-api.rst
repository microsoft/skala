C++ API
========

Include ``<skalaxc/skalaxc.hpp>`` and link the public target:

.. code-block:: cmake

   find_package(SkalaXC CONFIG REQUIRED COMPONENTS CXX)
   target_link_libraries(my_application PRIVATE SkalaXC::skalaxc)

Only standard C++ types and SkalaXC-owned values appear at the library boundary.
The deliberate MPI exception is a native ``MPI_Comm`` in MPI-enabled builds.
GauXC, LibTorch, ATen, and Eigen types are not exported by the shared library.
The templated matrix adapter accepts a caller-owned compatible matrix type
without requiring Eigen in the public library ABI.

Pipeline
--------

* ``RuntimeEnvironment`` owns execution resources and selects the communicator.
* ``Molecule`` and ``BasisSet<double>`` describe the system.
* ``MolGridFactory`` constructs the integration grid.
* ``LoadBalancerFactory`` assigns complete atomic domains.
* ``MolecularWeightsFactory`` partitions the quadrature weights.
* ``functional_type`` selects the owned model.
* ``XCIntegratorFactory<Matrix>`` constructs an integrator.
* ``eval_exc_vxc`` returns energy and scalar/z potentials.
* ``eval_exc_grad`` returns nuclear gradients.

Use allocating results or caller-owned output overloads as appropriate.
Invalid inputs raise ``SkalaXC::Exception``. Do not evaluate the same
integrator concurrently; use independent instances or serialize calls.
``diagnostics()`` reports rank-local counters and optional timing data.

The complete public declarations are maintained with the implementation:

.. literalinclude:: ../../SkalaXC/include/skalaxc/skalaxc.hpp
   :language: cpp

See :doc:`cpp` for a compiled end-to-end example.

Runtime environment
===================

This section provides a reference for the runtime environment in GauXC, including C++ class definitions, C bindings, and Fortran bindings for creating and managing runtime environments used in GauXC computations.
The runtime environment encapsulates information about the MPI communicator and device configuration for GauXC computations.


C++ definitions
---------------

.. cpp:class:: GauXC::RuntimeEnvironment

   A class representing the runtime environment for GauXC computations, including information about the MPI communicator and device configuration.

   .. cpp:function:: RuntimeEnvironment(MPI_Comm comm)

      Construct a RuntimeEnvironment object with the specified MPI communicator.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.

   .. cpp:function:: int comm_rank() const

      Get the rank of the current process in the MPI communicator associated with this runtime environment.

      :returns: The rank of the current process in the MPI communicator.

   .. cpp:function:: int comm_size() const

      Get the size of the MPI communicator associated with this runtime environment.

      :returns: The size of the MPI communicator.

   .. cpp:function:: DeviceRuntimeEnvironment as_device_runtime() const

      Convert the current runtime environment to a device runtime environment.


.. cpp:class:: GauXC::DeviceRuntimeEnvironment : public RuntimeEnvironment

   A class representing a device runtime environment for GauXC computations, including information about the device configuration and memory usage.

   .. cpp:function:: DeviceRuntimeEnvironment(MPI_Comm comm, double fill_fraction)

      Construct a DeviceRuntimeEnvironment object with the specified MPI communicator and device memory fill fraction.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.
      :param fill_fraction: The fraction of the device memory to be used for GauXC computations (between 0 and 1).

   .. cpp:function:: DeviceRuntimeEnvironment(MPI_Comm comm, void* mem, size_t mem_size)

      Construct a DeviceRuntimeEnvironment object with the specified MPI communicator and user-provided memory.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.
      :param mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param mem_size: The size of the user-provided memory in bytes.

   .. cpp:function:: void* device_memory() const

      Get a pointer to the device memory associated with this runtime environment.

      :returns: A pointer to the device memory associated with this runtime environment.

   .. cpp:function:: size_t device_memory_size() const

      Get the size of the device memory associated with this runtime environment in bytes.

      :returns: The size of the device memory associated with this runtime environment in bytes.

   .. cpp:function:: bool owns_memory() const

      Check if this runtime environment owns the device memory.

      :returns: True if this runtime environment owns the device memory, false otherwise.

   .. cpp:function:: void release_buffer()

      Release the device memory buffer associated with this runtime environment, if it is owned by this runtime environment.

   .. cpp:function:: void set_buffer(void* mem, size_t mem_size)

      Set the device memory buffer for this runtime environment to the specified user-provided memory.

      :param mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param mem_size: The size of the user-provided memory in bytes.


C bindings
----------

.. c:struct:: GauXCRuntimeEnvironment

   Opaque struct representing the runtime environment in the GauXC C API.

   .. c:function:: GauXCRuntimeEnvironment gauxc_runtime_environment_new(GauXCStatus* status, MPI_Comm comm)

      Create a new GauXCRuntimeEnvironment object.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.
      :returns: A new GauXCRuntimeEnvironment object.

   .. c:function:: GauXCRuntimeEnvironment gauxc_device_runtime_environment_new(GauXCStatus* status, MPI_Comm comm, double fill_fraction)

      Create a new GauXCRuntimeEnvironment object for a specific device.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.
      :param fill_fraction: The fraction of the device memory to be used for GauXC computations (between 0 and 1).
      :returns: A new GauXCRuntimeEnvironment object for the specified device.

   .. c:function:: GauXCRuntimeEnvironment gauxc_runtime_environment_new_mem(GauXCStatus* status, MPI_Comm comm, void* mem, size_t mem_size)

      Create a new GauXCRuntimeEnvironment object using user-provided memory.

      .. important:: Signature changes if :c:macro:`GAUXC_HAS_MPI` is defined and :cmake:variable:`GAUXC_ENABLE_MPI` is enabled.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param comm: The MPI communicator to be used for GauXC computations. Only possible to pass if MPI support is enabled in GauXC.
      :param mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param mem_size: The size of the user-provided memory in bytes.
      :returns: A new GauXCRuntimeEnvironment object using the provided memory.

   .. c:function:: int gauxc_runtime_environment_comm_rank(GauXCStatus* status, GauXCRuntimeEnvironment env)

      Get the rank of the current process in the MPI communicator associated with the GauXCRuntimeEnvironment.
      Returns 0 if MPI support is not enabled in GauXC.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param env: The GauXCRuntimeEnvironment object for which to get the MPI rank.
      :returns: The rank of the current process in the MPI communicator.

   .. c:function:: int gauxc_runtime_environment_comm_size(GauXCStatus* status, GauXCRuntimeEnvironment env)

      Get the size of the MPI communicator associated with the GauXCRuntimeEnvironment.
      Returns 1 if MPI support is not enabled in GauXC.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param env: The GauXCRuntimeEnvironment object for which to get the MPI size.
      :returns: The size of the MPI communicator.

   .. c:function:: void gauxc_runtime_environment_delete(GauXCStatus* status, GauXCRuntimeEnvironment* env)

      Delete a GauXCRuntimeEnvironment object.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param env: Pointer to the GauXCRuntimeEnvironment object to be deleted.


Fortran bindings
----------------

.. f:module:: gauxc_runtime_environment
   :synopsis: Fortran bindings for GauXC runtime environment objects.

.. f:currentmodule:: gauxc_runtime_environment

.. f:type:: gauxc_runtime_environment_type

   Opaque type representing the runtime environment in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_runtime_environment`.

   .. f:function:: gauxc_runtime_environment_new(status)

      Create a new GauXCRuntimeEnvironment object.
      If MPI support is enabled in GauXC, this function creates a runtime environment with the default MPI communicator (``MPI_COMM_WORLD``).
      Part of the :f:func:`gauxc_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object.

   .. f:function:: gauxc_runtime_environment_new_mpi(status, comm)

      Create a new GauXCRuntimeEnvironment object.
      This function is only available if MPI support is enabled in GauXC and allows specifying the MPI communicator to be used for GauXC computations.
      Part of the :f:func:`gauxc_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param integer comm: The MPI communicator to be used for GauXC computations.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object.

   .. f:function:: gauxc_runtime_environment_new_mpi_f08(status, comm)

      Create a new GauXCRuntimeEnvironment object.
      This function is only available if MPI support is enabled in GauXC and allows specifying the MPI communicator to be used for GauXC computations.
      Part of the :f:func:`gauxc_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param type(MPI_Comm) comm: The MPI communicator to be used for GauXC computations.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object.

   .. f:function:: gauxc_device_runtime_environment_new(status, fill_fraction)

      Create a new GauXCRuntimeEnvironment object for a specific device.
      If MPI support is enabled in GauXC, this function creates a runtime environment with the default MPI communicator (``MPI_COMM_WORLD``).
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the fraction of the device memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param integer comm: The MPI communicator to be used for GauXC computations.
      :param real(c_double) fill_fraction: The fraction of the device memory to be used for GauXC computations (between 0 and 1).
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object for the specified device.

   .. f:function:: gauxc_device_runtime_environment_new_mpi(status, comm, fill_fraction)

      Create a new GauXCRuntimeEnvironment object for a specific device.
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the MPI communicator and the fraction of the device memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param integer comm: The MPI communicator to be used for GauXC computations.
      :param real(c_double) fill_fraction: The fraction of the device memory to be used for GauXC computations (between 0 and 1).
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object for the specified device.

   .. f:function:: gauxc_device_runtime_environment_new_mpi_f08(status, comm, fill_fraction)

      Create a new GauXCRuntimeEnvironment object for a specific device.
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the MPI communicator and the fraction of the device memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param type(MPI_Comm) comm: The MPI communicator to be used for GauXC computations.
      :param real(c_double) fill_fraction: The fraction of the device memory to be used for GauXC computations (between 0 and 1).
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object for the specified device.

   .. f:function:: gauxc_device_runtime_environment_new_mem(status, mem, mem_size)

      Create a new GauXCRuntimeEnvironment object using user-provided memory.
      If MPI support is enabled in GauXC, this function creates a runtime environment with the default MPI communicator (``MPI_COMM_WORLD``).
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the user-provided memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param type(c_ptr) mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param integer(c_size_t) mem_size: The size of the user-provided memory in bytes.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object using the provided memory.

   .. f:function:: gauxc_device_runtime_environment_new_mem_mpi(status, comm, mem, mem_size)

      Create a new GauXCRuntimeEnvironment object using user-provided memory.
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the MPI communicator and the user-provided memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param integer comm: The MPI communicator to be used for GauXC computations.
      :param type(c_ptr) mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param integer(c_size_t) mem_size: The size of the user-provided memory in bytes.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object using the provided memory.

   .. f:function:: gauxc_device_runtime_environment_new_mem_mpi_f08(status, comm, mem, mem_size)

      Create a new GauXCRuntimeEnvironment object using user-provided memory.
      This function is only available if device support (CUDA or HIP) is enabled in GauXC and allows specifying the MPI communicator and the user-provided memory to be used for GauXC computations.
      Part of the :f:func:`gauxc_device_runtime_environment_new` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param type(MPI_Comm) comm: The MPI communicator to be used for GauXC computations.
      :param type(c_ptr) mem: Pointer to the user-provided memory to be used for GauXC computations.
      :param integer(c_size_t) mem_size: The size of the user-provided memory in bytes.
      :returns type(gauxc_runtime_environment_type): A new GauXCRuntimeEnvironment object using the provided memory.

   .. f:function:: gauxc_runtime_environment_delete(status, env)

      Delete a GauXCRuntimeEnvironment object.
      Part of the :f:func:`gauxc_delete` interface.

      :param type(gauxc_status_type) status: Variable to store the status of the operation.
      :param type(gauxc_runtime_environment_type) env: The GauXCRuntimeEnvironment object to be deleted.
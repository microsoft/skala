Load balancer API
=================

This section provides a reference for the load balancer in GauXC, including C++ class definitions, C bindings, and Fortran bindings for creating and managing load balancers used in GauXC computations.


C++ definitions
---------------

.. cpp:class:: GauXC::LoadBalancer

   A class representing a load balancer for distributing computational work across multiple processes in GauXC.


.. cpp:class:: GauXC::LoadBalancerFactory

   A factory class for creating LoadBalancer objects based on the specified load balancing strategy.

   .. cpp:function:: LoadBalancerFactory(ExecutionSpace ex, std::string kernel_name)

      Construct a LoadBalancerFactory object with the specified execution space and kernel name.

      Currently accepted values for Host execution space:

      :"DEFAULT":
        Read as "REPLICATED-PETITE"
      :"REPLICATED":
        Read as "REPLICATED-PETITE"
      :"REPLICATED-PETITE":
        Replicate the load balancer function, only keep non negligible basis functions
      :"REPLICATED-FILLIN":
        Same as "REPLICATED-PETITE" except if two non-adjacent basis functions are kept, the gaps are filled in.
        This gurantees contiguous memory access but leads to significantly more work.
        Not advised for general usage.
     
      Currently accepted values for Device execution space:

      :"DEFAULT":
        Read as "REPLICATED"
      :"REPLICATED":
        Same as Host::REPLICATED-PETITE

      :param ex: The execution space for which to create load balancers.
      :param kernel_name: The name of the kernel for which to create load balancers.

   .. cpp:function:: LoadBalancer get_instance(const RuntimeEnvironment& rt, const Molecule& mol, const MolGrid& mg, const BasisSet<double>& bs)
      
      Get a LoadBalancer instance for the specified runtime environment, molecule, molecular grid, and basis set.

      :param rt: The runtime environment for which to get the load balancer instance.
      :param mol: The molecule for which to get the load balancer instance.
      :param mg: The molecular grid for which to get the load balancer instance.
      :param bs: The basis set for which to get the load balancer instance.

   .. cpp:function:: std::shared_ptr<LoadBalancer> get_shared_instance(RuntimeEnvironment& rt, const Molecule& mol, const MolGrid& mg, const BasisSet<double>& bs)

      Get a shared pointer to a LoadBalancer instance for the specified runtime environment, molecule, molecular grid, and basis set.

      :param rt: The runtime environment for which to get the load balancer instance.
      :param mol: The molecule for which to get the load balancer instance.
      :param mg: The molecular grid for which to get the load balancer instance.
      :param bs: The basis set for which to get the load balancer instance.


.. cpp:enum-class:: GauXC::ExecutionSpace

   Enumeration of execution spaces for which load balancers can be created in GauXC.

   The following execution spaces are available:

   .. cpp:enumerator:: Host

      Load balancers for execution on the host CPU.

   .. cpp:enumerator:: Device

      Load balancers for execution on a device (e.g., GPU).


C bindings
----------

.. c:struct:: GauXCLoadBalancer

   Opaque struct representing a load balancer in the GauXC C API.

   .. c:function:: void gauxc_load_balancer_delete(GauXCStatus* status, GauXCLoadBalancer* lb)

      Delete a load balancer instance.
      Part of the :f:func:`gauxc_delete` interface.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param lb: Pointer to the GauXCLoadBalancer instance to be deleted.

.. c:struct:: GauXCLoadBalancerFactory

   Opaque struct representing a load balancer factory in the GauXC C API.

   .. c:function:: GauXCLoadBalancerFactory* gauxc_load_balancer_factory_new(GauXCStatus* status, enum GauXC_ExecutionSpace ex, const char* kernel_name)

      Create a new load balancer factory with the specified execution space and kernel name.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param ex: The execution space for which to create load balancers.
      :param kernel_name: The name of the kernel for which to create load balancers.

   .. c:function:: GauXCLoadBalancer gauxc_load_balancer_factory_get_instance(GauXCStatus* status, GauXCLoadBalancerFactory lbf, const GauXCRuntimeEnvironment rt, const GauXCMolecule mol, const GauXCMolGrid mg, const GauXCBasisSet bs)

      Get a load balancer instance for the specified runtime environment, molecule, molecular grid, and basis set.
      Part of the :f:func:`gauxc_get_instance` interface.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param lbf: GauXCLoadBalancerFactory struct from which to get the load balancer instance.
      :param rt: Pointer to the runtime environment for which to get the load balancer instance.
      :param mol: Pointer to the molecule for which to get the load balancer instance.
      :param mg: Pointer to the molecular grid for which to get the load balancer instance.
      :param bs: Pointer to the basis set for which to get the load balancer instance.

   .. c:function:: void gauxc_load_balancer_factory_delete(GauXCStatus* status, GauXCLoadBalancerFactory* lbf)

      Delete a load balancer factory instance.
      Part of the :f:func:`gauxc_delete` interface.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param lbf: Pointer to the GauXCLoadBalancerFactory instance to be deleted.


.. c:enum:: GauXC_ExecutionSpace

   Enumeration of execution spaces for which load balancers can be created in the GauXC C API.

   The following execution spaces are available:

   .. c:enumerator:: GauXC_ExecutionSpace_Host

      Load balancers for execution on the host CPU.

   .. c:enumerator:: GauXC_ExecutionSpace_Device

      Load balancers for execution on a device (e.g., GPU).


Fortran bindings
----------------

.. f:module:: gauxc_load_balancer
   :synopsis: Fortran bindings for GauXC LoadBalancer and LoadBalancerFactory

.. f:currentmodule:: gauxc_load_balancer

.. f:type:: gauxc_load_balancer_type

   Opaque type representing a load balancer in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_load_balancer`.

   .. f:function:: gauxc_load_balancer_delete(status, lb)

      Delete a load balancer instance.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_load_balancer_type) lb: The load balancer instance to be deleted.

.. f:type:: gauxc_load_balancer_factory_type

   Opaque type representing a load balancer factory in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_load_balancer`.

   .. f:function:: gauxc_load_balancer_factory_new(status, ex, kernel_name)

      Create a new load balancer factory with the specified execution space and kernel name.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param integer(c_int) ex: The execution space for which to create load balancers.
      :param character(len=*) kernel_name: The name of the kernel for which to create load balancers.

   .. f:function:: gauxc_load_balancer_factory_get_instance(status, lbf, rt, mol, mg, bs)

      Get a load balancer instance for the specified runtime environment, molecule, molecular grid, and basis set.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_load_balancer_factory_type) lbf: The load balancer factory from which to get the load balancer instance.
      :param type(gauxc_runtime_environment_type) rt: The runtime environment for which to get the load balancer instance.
      :param type(gauxc_molecule_type) mol: The molecule for which to get the load balancer instance.
      :param type(gauxc_mol_grid_type) mg: The molecular grid for which to get the load balancer instance.
      :param type(gauxc_basis_set_type) bs: The basis set for which to get the load balancer instance.

   .. f:subroutine:: gauxc_load_balancer_factory_delete(status, lbf)

      Delete a load balancer factory instance.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_load_balancer_factory_type) lbf: The load balancer factory instance to be deleted.

.. f:currentmodule:: gauxc_enums

.. f:type:: gauxc_execution_space

   Parameter instance of a derived type with members corresponding to the execution spaces for which load balancers can be created in the GauXC Fortran API.

   :f integer(c_int) host: Load balancers for execution on the host CPU.

   :f integer(c_int) device: Load balancers for execution on a device (e.g., GPU).
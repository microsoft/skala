Molecular grid weights API
==========================

This section provides a reference for the GauXC API related to molecular grid weights, which can be applied to LoadBalancer tasks to modify their weights based on the molecular grid. The API includes C++ class definitions

C++ definitions
---------------

.. cpp:class:: GauXC::MolecularWeights

   A class representing molecular grid weights in GauXC, which can be applied to a LoadBalancer's tasks.

   .. cpp:function:: void modify_weights(LoadBalancer& lb)

      Apply molecular weights to a LoadBalancer's tasks

      :param lb: LoadBalancer instance to which the molecular grid weights will be applied.

.. cpp:class:: GauXC::MolecularWeightsFactory

   A factory class for creating MolecularWeights instances in GauXC.

   .. cpp:function:: MolecularWeightsFactory(ExecutionSpace ex, std::string local_work_kernel_name, MolecularWeightsSettings settings)

      Construct a MolecularWeightsFactory instance with the specified settings.

      :param ex: The execution space for which to create molecular grid weights.
      :param local_work_kernel_name: Name of the local work kernel to be used.
      :param settings: A MolecularWeightsSettings struct containing settings for computing molecular grid weights.
      :returns: A MolecularWeightsFactory instance initialized with the specified settings.

   .. cpp:function:: MolecularWeights get_instance()

      Get a MolecularWeights instance created by the factory.

      :returns: A MolecularWeights instance created by the factory.

   .. cpp:function:: std::shared_ptr<MolecularWeights> get_shared_instance()

      Get a shared pointer to a MolecularWeights instance created by the factory.

      :returns: A shared pointer to a MolecularWeights instance created by the factory.

.. cpp:struct:: GauXC::MolecularWeightsSettings

   Struct representing settings for computing molecular grid weights in GauXC.

   .. cpp:member:: XCWeightAlg weight_alg = XCWeightAlg::Becke

      The algorithm to be used for computing molecular grid weights. Possible values are defined in the XCWeightAlg enum.

   .. cpp:member:: bool becke_size_adjustment

      Whether to apply Becke size adjustment to the molecular grid weights.
      Default should be true.

.. cpp:enum:: GauXC::XCWeightAlg

   The following options are available:

   .. cpp:enumerator:: NOTPARTITIONED

      No partitioning.

   .. cpp:enumerator:: Becke

      Becke's original algorithm for computing molecular grid weights.

   .. cpp:enumerator:: SSF

      Stratmann-Scuseria-Frisch algorithm for computing molecular grid weights.

   .. cpp:enumerator:: LKO

      Lauqua-Kuessman-Ochsenfeld algorithm for computing molecular grid weights.


C bindings
----------

.. c:struct:: GauXCMolecularWeights

   Opaque struct representing molecular grid weights in the GauXC C API.

   .. c:function:: void gauxc_molecular_weights_modify_weights(GauXCStatus* status, GauXCMolecularWeights mw, LoadBalancer lb)

      Apply molecular weights to a LoadBalancer's tasks

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param mw: GauXCMolecularWeights struct from which to get the molecular grid weight.
      :param lb: LoadBalancer struct to which the molecular grid weights will be applied.

   .. c:function:: void gauxc_molecular_weights_delete(GauXCStatus* status, GauXCMolecularWeights* mw)

      Delete a molecular grid weights instance.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param mw: Pointer to the GauXCMolecularWeights instance to be deleted.

.. c:struct:: GauXCMolecularWeightsFactory

   Opaque struct representing a factory for creating molecular grid weights in the GauXC C API.

   .. c:function:: GauXCMolecularWeightsFactory gauxc_molecular_weights_factory_new(GauXCStatus* status, GauXCMolecularWeightsSettings settings)

      Create a new GauXCMolecularWeightsFactory instance with the specified settings.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param settings: GauXCMolecularWeightsSettings struct containing settings for computing molecular grid weights.
      :returns: A new GauXCMolecularWeightsFactory instance initialized with the specified settings.

   .. c:function:: GauXCMolecularWeights gauxc_molecular_weights_factory_get_instance(GauXCStatus* status, GauXCMolecularWeightsFactory mwf, enum GauXC_ExecutionSpace ex, char* local_work_kernel_name, GauXCMolecularWeightsSettings settings)

      Get a GauXCMolecularWeights instance created by the factory.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param mwf: GauXCMolecularWeightsFactory struct from which to get the molecular grid weights instance.
      :param local_work_kernel_name: Name of the local work kernel to be used.
      :param settings: GauXCMolecularWeightsSettings struct containing settings for computing molecular grid weights.
      :returns: A GauXCMolecularWeights instance created by the factory.

   .. c:function:: void gauxc_molecular_weights_factory_delete(GauXCStatus* status, GauXCMolecularWeightsFactory* mwf)

      Delete a GauXCMolecularWeightsFactory instance.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param mwf: Pointer to the GauXCMolecularWeightsFactory instance to be deleted.

.. c:struct:: GauXCMolecularWeightsSettings

   Representation of settings for computing molecular grid weights in the GauXC C API.

   .. c:member:: enum GauXC_XCWeightAlg weight_alg

      The algorithm to be used for computing molecular grid weights. Possible values are defined in the GauXC_XCWeightAlg enum.

   .. c:member:: bool becke_size_adjustment

      Whether to apply Becke size adjustment to the molecular grid weights.
      Default should be true.

.. c:enum:: GauXC_XCWeightAlg

   Enumeration of algorithms for computing molecular grid weights in the GauXC C API.

   The following algorithms are available:

   .. c:enumerator:: GauXC_XCWeightAlg_NOTPARTITIONED

      No partitioning.

   .. c:enumerator:: GauXC_XCWeightAlg_Becke

      Becke's original algorithm for computing molecular grid weights.

   .. c:enumerator:: GauXC_XCWeightAlg_SSF

      Stratmann-Scuseria-Frisch algorithm for computing molecular grid weights.

   .. c:enumerator:: GauXC_XCWeightAlg_LKO

      Lauqua-Kuessman-Ochsenfeld algorithm for computing molecular grid weights.


Fortran bindings
----------------

.. f:module:: gauxc_molecular_weights
   :synopsis: Fortran bindings for GauXC molecular grid weights

.. f:currentmodule:: gauxc_molecular_weights

.. f:type:: gauxc_molecular_weights_type

   Opaque type representing molecular grid weights in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_molecular_weights`.

   .. f:function:: gauxc_molecular_weights_modify_weights(status, mw, lb)

      Apply molecular weights to a LoadBalancer's tasks

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_molecular_weights_type) mw: The molecular grid weights instance from which to get the molecular grid weight.
      :param type(gauxc_load_balancer_type) lb: The LoadBalancer instance to which the molecular grid weights will be applied.

   .. f:function:: gauxc_molecular_weights_delete(status, mw)

      Delete a molecular grid weights instance.
      Part of the :f:func:`gauxc_delete` interface.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_molecular_weights_type) mw: The molecular grid weights instance to be deleted.


.. f:type:: gauxc_molecular_weights_factory_type

   Opaque type representing a factory for creating molecular grid weights in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_molecular_weights`.

   .. f:function:: gauxc_molecular_weights_factory_new(status, ex, local_work_kernel_name, settings)

      Create a new GauXCMolecularWeightsFactory instance with the specified settings.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param integer(c_int) ex: The execution space for which to get the molecular grid weights instance. Possible values are defined in the gauxc_xc_weight_alg type.
      :param character(len=*) local_work_kernel_name: Name of the local work kernel to be used.
      :param type(gauxc_molecular_weights_settings_type) settings: The settings for computing molecular grid weights.
      :returns type(gauxc_molecular_weights_factory_type): A new GauXCMolecularWeightsFactory instance initialized with the specified settings.

   .. f:function:: gauxc_molecular_weights_factory_get_instance(status, mwf)

      Get a GauXCMolecularWeights instance created by the factory.
      Part of the :f:func:`gauxc_get_instance` interface.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_molecular_weights_factory_type) mwf: The GauXCMolecularWeightsFactory instance from which to get the molecular grid weights instance.
      :returns type(gauxc_molecular_weights_type): A GauXCMolecularWeights instance created by the factory.

   .. f:function:: gauxc_molecular_weights_factory_delete(status, mwf)

      Delete a GauXCMolecularWeightsFactory instance.
      Part of the :f:func:`gauxc_delete` interface.

      :param type(gauxc_status_type) status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param type(gauxc_molecular_weights_factory_type) mwf: The GauXCMolecularWeightsFactory instance to be deleted.

   .. code-block:: fortran
      :caption: Example

      use gauxc_enums, only : gauxc_executionspace
      use gauxc_load_balancer, only : gauxc_load_balancer_type
      use gauxc_molecular_weights, only : gauxc_molecular_weights_factory_type, gauxc_molecular_weights_factory_new, &
        & gauxc_get_instance, gauxc_molecular_weights_type, gauxc_molecular_weights_modify_weight, gauxc_delete

      type(gauxc_load_balancer_type) :: lb
      type(gauxc_status_type) :: status
      type(gauxc_molecular_weights_factory_type) :: mwf
      type(gauxc_molecular_weights_type) :: mw

      ! setup load balancer here

      main: block
      call gauxc_molecular_weights_factory_new(status, gauxc_executionspace%host, &
        & "Default", gauxc_molecular_weights_settings())
      if (status%code /= 0) exit main
      mw = gauxc_get_instance(status, mwf)
      if (status%code /= 0) exit main
      call gauxc_molecular_weights_modify_weight(status, mw, lb)
      if (status%code /= 0) exit main
      end block main
      if (status%code /= 0) then
         ! handle error
      end if

      call gauxc_delete(status, mw)
      call gauxc_delete(status, mwf)


.. f:type:: gauxc_molecular_weights_settings

   Parameter instance of a derived type representing settings for computing molecular grid weights in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_molecular_weights`.

   :f integer(c_int) weight_alg: The algorithm to be used for computing molecular grid weights. Possible values are defined in the gauxc_xc_weight_alg type.
   :f logical(c_bool) becke_size_adjustment: Whether to apply Becke size adjustment to the molecular grid weights. Default should be true.

   .. code-block:: fortran
      :caption: Example

      use gauxc_enums, only : gauxc_xcweightalg
      use gauxc_molecular_weights, only : gauxc_molecular_weights_settings
      type(gauxc_molecular_weights_settings) :: settings
      settings%weight_alg = gauxc_xcweightalg%becke
      settings%becke_size_adjustment = .true.

.. f:currentmodule:: gauxc_enums

.. f:type:: gauxc_xcweightalg

   Parameter instance of a derived type representing the algorithm to be used for computing molecular grid weights in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_enums`.

   The following algorithms are available:

   :f integer(c_int) notpartitioned: No partitioning.
   :f integer(c_int) becke: Becke's original algorithm for computing molecular grid weights.
   :f integer(c_int) ssf: Stratmann-Scuseria-Frisch algorithm for computing molecular grid weights.
   :f integer(c_int) lko: Lauqua-Kuessman-Ochsenfeld algorithm for computing molecular grid weights.
Exchange-correlation integrator
===============================

C++ definitions
---------------

.. cpp:class:: template <typename MatrixType> GauXC::XCIntegrator

   .. cpp:type:: matrix_type = MatrixType

      The type of matrix data expected by the integrator kernels.

   .. cpp:type:: value_type = matrix_type::value_type

      The scalar value type corresponding to the matrix type.

   .. cpp:function:: value_type integrate_den(const MatrixType& density_matrix) const

      Compute the total density to get the number of electrons.

      :param density_matrix: The input density matrix data.
      :returns: The computed total density.

   .. cpp:function:: value_type eval_exc(const MatrixType& density_matrix) const

      Compute the exchange-correlation energy for a given density matrix.

      :param density_matrix: The input density matrix data.
      :returns: The computed exchange-correlation energy.

   .. cpp:function:: value_type eval_exc(const MatrixType& density_matrix_s, const MatrixType& density_matrix_z) const

      Compute the exchange-correlation energy for a given density matrix.

      :param density_matrix_s: The input density matrix data for the spin-up component.
      :param density_matrix_z: The input density matrix data for the spin-down component.
      :returns: The computed exchange-correlation energy.

   .. cpp:function:: value_type eval_exc(const MatrixType& density_matrix_s, const MatrixType& density_matrix_z, const MatrixType& density_matrix_y, const MatrixType& density_matrix_x) const

      Compute the exchange-correlation energy for a given density matrix.

      :param density_matrix_s: The input density matrix data for the spin-up component.
      :param density_matrix_z: The input density matrix data for the spin-down component.
      :param density_matrix_y: The input density matrix data for the spin-y component.
      :param density_matrix_x: The input density matrix data for the spin-x component.
      :returns: The computed exchange-correlation energy.

   .. cpp:function:: std::tuple<value_type, matrix_type> eval_exc_vxc(const MatrixType& density_matrix)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param density_matrix: The input density matrix data.
      :returns: The computed exchange-correlation energy and the computed exchange-correlation potential matrix.

   .. cpp:function:: std::tuple<value_type, matrix_type, matrix_type> eval_exc_vxc(const MatrixType& density_matrix_s, const MatrixType& density_matrix_z)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param density_matrix_s: The input density matrix data for the spin-up component.
      :param density_matrix_z: The input density matrix data for the spin-down component.
      :returns: The computed exchange-correlation energy and the computed exchange-correlation potential matrices for the total and spin-polarization components.

   .. cpp:function:: std::tuple<value_type, matrix_type, matrix_type> eval_exc_vxc_onedft(const MatrixType& density_matrix_s, const MatrixType& density_matrix_z, const OneDFTSettings settings)

       Compute the exchange-correlation energy and potential for a given density matrix.

       .. important::

          This function is available if :c:macro:`GAUXC_HAS_ONEDFT` is defined or the CMake option :cmake:variable:`GAUXC_ENABLE_ONEDFT` is enabled.
          It requires a compatible checkpoint for the Skala implementation of the functional, which can be specified with the ``settings`` parameter.

       :param density_matrix_s: The input density matrix data for the spin-up component.
       :param density_matrix_z: The input density matrix data for the spin-down component.
       :param settings: The OneDFTSettings struct containing parameters for the Skala evaluation.
       :returns: The computed exchange-correlation energy and the computed exchange-correlation potential matrices for the total and spin-polarization components.

   .. cpp:function:: std::tuple<value_type, matrix_type, matrix_type, matrix_type, matrix_type> eval_exc_vxc(const MatrixType& density_matrix_s, const MatrixType& density_matrix_z, const MatrixType& density_matrix_y, const MatrixType& density_matrix_x)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param density_matrix_s: The input density matrix data for the spin-up component.
      :param density_matrix_z: The input density matrix data for the spin-down component.
      :param density_matrix_y: The input density matrix data for the spin-y component.
      :param density_matrix_x: The input density matrix data for the spin-x component.
      :returns: The computed exchange-correlation energy and the computed exchange-correlation potential matrices for the total and spin-polarization components.

.. cpp:class:: GauXC::XCIntegratorFactory

   A factory class for creating instances of the XCIntegrator class based on specified parameters.

   .. cpp:function:: XCIntegratorFactory(ExecutionSpace ex, std::string integrator_input_type, std::string integrator_kernel_name, std::string local_work_kernel_name, std::string reduction_kernel_name)

      Create a new XCIntegratorFactory instance with the specified execution space and kernel names.

      :param ex: The execution space for which to create integrators.
      :param integrator_input_type: The type of input data expected by the integrator kernels.
      :param integrator_kernel_name: The name of the kernel to use for the main integration step.
      :param local_work_kernel_name: The name of the kernel to use for computing local work sizes.
      :param reduction_kernel_name: The name of the kernel to use for reduction operations.

   .. cpp:function:: XCIntegrator get_instance(const ExchCXX::Functional& func, const LoadBalancer& lb)

      Get an instance of the XCIntegrator class for the specified functional and load balancer.

      :param func: The exchange-correlation functional for which to get the integrator instance.
      :param lb: The load balancer to use for the integrator instance.
      :returns: An instance of the XCIntegrator class initialized with the specified parameters.

   .. cpp:function:: std::shared_ptr<XCIntegrator> get_shared_instance(const ExchCXX::Functional& func, const LoadBalancer& lb)

      Get a shared pointer to an instance of the XCIntegrator class for the specified functional and load balancer.

      :param func: The exchange-correlation functional for which to get the integrator instance.
      :param lb: The load balancer to use for the integrator instance.
      :returns: A shared pointer to an instance of the XCIntegrator class initialized with the specified parameters.

.. cpp:struct:: GauXC::OneDFTSettings

   A struct containing settings for the Skala implementation of the exchange-correlation functional.

   .. cpp:member:: model

      The model checkpoint to use for evaluating the exchange-correlation energy and potential with the Skala implementation.

.. cpp:class:: ExchCXX::Functional

   A class representing an exchange-correlation functional in the ExchCXX library, which can be used with GauXC integrators.


C bindings
----------

.. c:struct:: GauXCIntegrator

   Opaque type representing an exchange-correlation integrator instance.

   .. c:function:: GauXCIntegrator gauxc_integrator_new(GauXCStatus* status, const GauXCFunctional functional, const GauXCLoadBalancer lb, enum GauXC_ExecutionSpace ex, const char* integrator_input_type, const char* integrator_kernel_name, const char* local_work_kernel_name, const char* reduction_kernel_name)

      Create a new exchange-correlation integrator instance with the specified functional, load balancer, execution space, and kernel names.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param functional: The exchange-correlation functional for which to create the integrator.
      :param lb: The load balancer to use for the integrator.
      :param ex: The execution space for which to create the integrator.
      :param integrator_input_type: The type of input data expected by the integrator kernels.
      :param integrator_kernel_name: The name of the kernel to use for the main integration step.
      :param local_work_kernel_name: The name of the kernel to use for computing local work sizes.
      :param reduction_kernel_name: The name of the kernel to use for reduction operations.
      :returns: A new GauXCIntegrator instance initialized with the specified parameters.

   .. c:function:: void gauxc_integrator_integrate_den(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix, const int64_t ldp, double* den)

      Compute the total density to get the number of electrons

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix: Pointer to the input density matrix data.
      :param ldp: The leading dimension of the density matrix.
      :param den: Pointer to the output variable where the computed total density will be stored.

   .. c:function:: void gauxc_integrator_eval_exc_rks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix, const int64_t ldp, double* exc)

      Compute the exchange-correlation energy for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix: Pointer to the input density matrix data.
      :param ldp: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.

   .. c:function:: void gauxc_integrator_eval_exc_uks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix_s, const int64_t ldp_s, const double* density_matrix_z, const int64_t ldp_z, double* exc)

      Compute the exchange-correlation energy for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix_s: Pointer to the input density matrix data.
      :param ldp_s: The leading dimension of the density matrix.
      :param density_matrix_z: Pointer to the input density matrix data.
      :param ldp_z: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.

   .. c:function:: void gauxc_integrator_eval_exc_gks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix_s, const int64_t ldp_s, const double* density_matrix_z, const int64_t ldp_z, const double* density_matrix_y, const int64_t ldp_y, const double* density_matrix_x, const int64_t ldp_x, double* exc)

      Compute the exchange-correlation energy for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix_s: Pointer to the input density matrix data.
      :param ldp_s: The leading dimension of the density matrix.
      :param density_matrix_z: Pointer to the input density matrix data.
      :param ldp_z: The leading dimension of the density matrix.
      :param density_matrix_y: Pointer to the input density matrix data.
      :param ldp_y: The leading dimension of the density matrix.
      :param density_matrix_x: Pointer to the input density matrix data.
      :param ldp_x: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.

   .. c:function:: void gauxc_integrator_eval_exc_vxc_rks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix, const int64_t ldp, double* exc, double* vxc_matrix, const int64_t ldp_vxc)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix: Pointer to the input density matrix data.
      :param ldp: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.
      :param vxc_matrix: Pointer to the output array where the computed exchange-correlation potential matrix will be stored.
      :param ldp_vxc: The leading dimension of the vxc_matrix array.

   .. c:function:: void gauxc_integrator_eval_exc_vxc_uks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix_s, const int64_t ldp_s, const double* density_matrix_z, const int64_t ldp_z, double* exc, double* vxc_matrix_s, const int64_t ldp_vxc_s, double* vxc_matrix_z, const int64_t ldp_vxc_z)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix_s: Pointer to the input density matrix data.
      :param ldp_s: The leading dimension of the density matrix.
      :param density_matrix_z: Pointer to the input density matrix data.
      :param ldp_z: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.
      :param vxc_matrix_s: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param ldp_vxc_s: The leading dimension of the vxc_matrix_s array.
      :param vxc_matrix_z: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.
      :param ldp_vxc_z: The leading dimension of the vxc_matrix_z array.


   .. c:function:: void gauxc_integrator_eval_exc_vxc_onedft_uks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix_s, const int64_t ldp_s, const double* density_matrix_z, const int64_t ldp_z, const char* model, double* exc, double* vxc_matrix_s, const int64_t ldp_vxc_s, double* vxc_matrix_z, const int64_t ldp_vxc_z)

      Compute the exchange-correlation energy and potential for a given density matrix.

      .. important::
         
         This function is available if :c:macro:`GAUXC_HAS_ONEDFT` is defined or the CMake option :cmake:variable:`GAUXC_ENABLE_ONEDFT` is enabled.
         It requires a compatible checkpoint for the Skala implementation of the functional, which can be specified with the ``model`` parameter.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix_s: Pointer to the input density matrix data.
      :param ldp_s: The leading dimension of the density matrix.
      :param density_matrix_z: Pointer to the input density matrix data.
      :param ldp_z: The leading dimension of the density matrix.
      :param model: The model checkpoint to use for evaluating the exchange-correlation energy and potential.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.
      :param vxc_matrix_s: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param ldp_vxc_s: The leading dimension of the vxc_matrix_s array.
      :param vxc_matrix_z: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.
      :param ldp_vxc_z: The leading dimension of the vxc_matrix_z array.

   .. c:function:: void gauxc_integrator_eval_exc_vxc_gks(GauXCStatus* status, GauXCIntegrator integrator, const int64_t m, const int64_t n, const double* density_matrix_s, const int64_t ldp_s, const double* density_matrix_z, const int64_t ldp_z, const double* density_matrix_y, const int64_t ldp_y, const double* density_matrix_x, const int64_t ldp_x, double* exc, double* vxc_matrix_s, const int64_t ldp_vxc_s, double* vxc_matrix_z, const int64_t ldp_vxc_z, double* vxc_matrix_y, const int64_t ldp_vxc_y, double* vxc_matrix_x, const int64_t ldp_vxc_x)

      Compute the exchange-correlation energy and potential for a given density matrix.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: The GauXCIntegrator instance to use for the integration.
      :param m: The number of rows in the density matrix.
      :param n: The number of columns in the density matrix.
      :param density_matrix_s: Pointer to the input density matrix data.
      :param ldp_s: The leading dimension of the density matrix.
      :param density_matrix_z: Pointer to the input density matrix data.
      :param ldp_z: The leading dimension of the density matrix.
      :param density_matrix_y: Pointer to the input density matrix data.
      :param ldp_y: The leading dimension of the density matrix.
      :param density_matrix_x: Pointer to the input density matrix data.
      :param ldp_x: The leading dimension of the density matrix.
      :param exc: Pointer to the output variable where the computed exchange-correlation energy will be stored.
      :param vxc_matrix_s: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param ldp_vxc_s: The leading dimension of the vxc_matrix_s array.
      :param vxc_matrix_z: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.
      :param ldp_vxc_z: The leading dimension of the vxc_matrix_z array.
      :param vxc_matrix_y: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-y component will be stored.
      :param ldp_vxc_y: The leading dimension of the vxc_matrix_y array.
      :param vxc_matrix_x: Pointer to the output array where the computed exchange-correlation potential matrix for the spin-x component will be stored.
      :param ldp_vxc_x: The leading dimension of the vxc_matrix_x array.

   .. c:function:: void gauxc_integrator_delete(GauXCStatus* status, GauXCIntegrator* integrator)

      Delete an exchange-correlation integrator instance.

      :param status: Pointer to a GauXCStatus struct where the status of the operation will be stored.
      :param integrator: Pointer to the GauXCIntegrator instance to be deleted.

.. c:struct:: GauXCFunctional

   Opaque type representing an exchange-correlation functional in the GauXC C API, which can be used with GauXC integrators.


Fortran bindings
----------------

.. f:module:: gauxc_integrator
   :synopsis: Fortran bindings for GauXC exchange-correlation integrator objects.

.. f:currentmodule:: gauxc_integrator

.. f:type:: gauxc_integrator_type

   Opaque type representing an exchange-correlation integrator instance in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_integrator`.

   .. f:function:: gauxc_integrator_new(status, functional, lb, ex[, integrator_input_type, integrator_kernel_name, local_work_kernel_name, reduction_kernel_name])

      Create a new exchange-correlation integrator instance with the specified functional, load balancer, execution space, and kernel names.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_functional_type) functional: The exchange-correlation functional for which to create the integrator.
      :param type(gauxc_load_balancer_type) lb: The load balancer to use for the integrator.
      :param integer(c_int) ex: The execution space for which to create the integrator.
      :optional character(len=*) integrator_input_type: The type of input data expected by the integrator kernels. Default: "Replicated"
      :optional character(len=*) integrator_kernel_name: The name of the kernel to use for the main integration step. Default: "Default"
      :optional character(len=*) local_work_kernel_name: The name of the kernel to use for computing local work sizes. Default: "Default"
      :optional character(len=*) reduction_kernel_name: The name of the kernel to use for reduction operations. Default: "Default"
      :returns type(gauxc_integrator_type): A new gauxc_integrator_type object initialized with the specified parameters.

   .. f:subroutine:: gauxc_integrator_integrate_den(status, integrator, density_matrix, den)

      Compute the total density to get the number of electrons.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix [dimension(:,:)]: The input density matrix data.
      :param real(c_double) den: Output variable where the computed total density will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_rks(status, integrator, density_matrix, exc)

      Compute the exchange-correlation energy for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix [dimension(:,:)]: The input density matrix data.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_uks(status, integrator, density_matrix_s, density_matrix_z, exc)

      Compute the exchange-correlation energy for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix_s [dimension(:,:)]: The input density matrix data for the spin-up component.
      :param real(c_double) density_matrix_z [dimension(:,:)]: The input density matrix data for the spin-down component.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_gks(status, integrator, density_matrix_s, density_matrix_z, density_matrix_y, density_matrix_x, exc)

      Compute the exchange-correlation energy for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix_s [dimension(:,:)]: The input density matrix data for the spin-up component.
      :param real(c_double) density_matrix_z [dimension(:,:)]: The input density matrix data for the spin-down component.
      :param real(c_double) density_matrix_y [dimension(:,:)]: The input density matrix data for the spin-y component.
      :param real(c_double) density_matrix_x [dimension(:,:)]: The input density matrix data for the spin-x component.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_vxc_rks(status, integrator, density_matrix, exc, vxc_matrix, ldp_vxc)

      Compute the exchange-correlation energy and potential for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc_vxc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix [dimension(:,:)]: The input density matrix data.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.
      :param real(c_double) vxc_matrix [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix will be stored.
      :param integer(c_int64_t) ldp_vxc: The leading dimension of the vxc_matrix array.

   .. f:subroutine:: gauxc_integrator_eval_exc_vxc_uks(status, integrator, density_matrix_s, density_matrix_z, exc, vxc_matrix_s, vxc_matrix_z)
      
      Compute the exchange-correlation energy and potential for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc_vxc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix_s [dimension(:,:)]: The input density matrix data for the spin-up component.
      :param real(c_double) density_matrix_z [dimension(:,:)]: The input density matrix data for the spin-down component.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.
      :param real(c_double) vxc_matrix_s [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param real(c_double) vxc_matrix_z [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_vxc_onedft_uks(status, integrator, density_matrix_s, density_matrix_z, model, exc, vxc_matrix_s, vxc_matrix_z)

      Compute the exchange-correlation energy and potential for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc_vxc` interface.

      .. important::
         
         This function is available if :c:macro:`GAUXC_HAS_ONEDFT` is defined or the CMake option :cmake:variable:`GAUXC_ENABLE_ONEDFT` is enabled.
         It requires a compatible checkpoint for the Skala implementation of the functional, which can be specified with the ``model`` parameter.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix_s [dimension(:,:)]: The input density matrix data for the spin-up component.
      :param real(c_double) density_matrix_z [dimension(:,:)]: The input density matrix data for the spin-down component.
      :param character(len=*) model: The model checkpoint to use for evaluating the exchange-correlation energy and potential.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.
      :param real(c_double) vxc_matrix_s [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param real(c_double) vxc_matrix_z [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.

   .. f:subroutine:: gauxc_integrator_eval_exc_vxc_gks(status, integrator, density_matrix_s, density_matrix_z, density_matrix_y, density_matrix_x, exc, vxc_matrix_s, vxc_matrix_z, vxc_matrix_y, vxc_matrix_x)

      Compute the exchange-correlation energy and potential for a given density matrix.
      Part of the :f:func:`gauxc_eval_exc_vxc` interface.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_integrator_type) integrator: The GauXCIntegrator instance to use for the integration.
      :param real(c_double) density_matrix_s [dimension(:,:)]: The input density matrix data for the spin-up component.
      :param real(c_double) density_matrix_z [dimension(:,:)]: The input density matrix data for the spin-down component.
      :param real(c_double) density_matrix_y [dimension(:,:)]: The input density matrix data for the spin-y component.
      :param real(c_double) density_matrix_x [dimension(:,:)]: The input density matrix data for the spin-x component.
      :param real(c_double) exc: Output variable where the computed exchange-correlation energy will be stored.
      :param real(c_double) vxc_matrix_s [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-up component will be stored.
      :param real(c_double) vxc_matrix_z [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-down component will be stored.
      :param real(c_double) vxc_matrix_y [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-y component will be stored.
      :param real(c_double) vxc_matrix_x [dimension(:,:)]: Output array where the computed exchange-correlation potential matrix for the spin-x component will be stored.

.. f:currentmodule:: gauxc_xc_functional

.. f:type:: gauxc_functional_type

   Opaque type representing an exchange-correlation functional in the GauXC Fortran API, which can be used with GauXC integrators.
/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "cuda_aos_scheme1.hpp"
#include "device/cuda/cuda_backend.hpp"
#include "cuda_aos_scheme1_weights.hpp"
#include "device/common/device_blas.hpp"

namespace GauXC {

template <typename Base>
std::unique_ptr<XCDeviceData> CudaAoSScheme1<Base>::create_device_data(const DeviceRuntimeEnvironment& rt) {
  return std::make_unique<Data>(rt);
}

template <typename Base> 
void CudaAoSScheme1<Base>::partition_weights( XCDeviceData* _data ) {
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  auto device_backend = dynamic_cast<CUDABackend*>(data->device_backend_);
  if( !device_backend ) GAUXC_BAD_BACKEND_CAST();


  // Compute distances from grid to atomic centers
  const auto ldatoms = data->get_ldatoms();
  auto base_stack    = data->base_stack;
  auto static_stack  = data->static_stack;
  auto scheme1_stack = data->scheme1_stack;
  cuda_aos_scheme1_weights_wrapper( data->total_npts_task_batch, data->global_dims.natoms,
    base_stack.points_x_device, base_stack.points_y_device, base_stack.points_z_device,
    static_stack.rab_device, ldatoms, static_stack.coords_device, 
    scheme1_stack.dist_scratch_device, ldatoms, scheme1_stack.iparent_device, 
    scheme1_stack.dist_nearest_device, base_stack.weights_device, *device_backend->master_stream );
}

template <typename Base> 
void CudaAoSScheme1<Base>::eval_weight_1st_deriv_contracted( XCDeviceData* _data, XCWeightAlg alg ) {
  if( alg != XCWeightAlg::SSF ) {
    GAUXC_GENERIC_EXCEPTION("Weight Algorithm NYI for CUDA AoS Scheme1");
  }
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  auto device_backend = dynamic_cast<CUDABackend*>(data->device_backend_);
  if( !device_backend ) GAUXC_BAD_BACKEND_CAST();

  // make w times f vector
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;
  auto base_stack    = data->base_stack;
  if( is_pol )
    increment( data->device_backend_->master_blas_handle(), base_stack.den_z_eval_device, 
    base_stack.den_s_eval_device, data->total_npts_task_batch ); 

  hadamard_product(data->device_backend_->master_blas_handle(), data->total_npts_task_batch, 1, base_stack.den_s_eval_device, 1, 
    base_stack.eps_eval_device, 1); 


  // Compute distances from grid to atomic centers
  const auto ldatoms = data->get_ldatoms();
  auto static_stack  = data->static_stack;
  auto scheme1_stack = data->scheme1_stack;
  cuda_aos_scheme1_weight_1st_deriv_wrapper( data->total_npts_task_batch, data->global_dims.natoms,
    base_stack.points_x_device, base_stack.points_y_device, base_stack.points_z_device,
    static_stack.rab_device, ldatoms, static_stack.coords_device, 
    scheme1_stack.dist_scratch_device, ldatoms, scheme1_stack.iparent_device, 
    scheme1_stack.dist_nearest_device, base_stack.eps_eval_device, static_stack.exc_grad_device, *device_backend->master_stream );
}


template struct CudaAoSScheme1<AoSScheme1Base>;
#ifdef GAUXC_HAS_MAGMA
template struct CudaAoSScheme1<AoSScheme1MAGMABase>;
#endif
#ifdef GAUXC_HAS_CUTLASS
template struct CudaAoSScheme1<AoSScheme1CUTLASSBase>;
#endif



}

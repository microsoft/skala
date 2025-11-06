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
#include "hip_aos_scheme1.hpp"
#include "device/hip/hip_backend.hpp"
#include "kernels/grid_to_center.hpp"
#include "kernels/hip_ssf_1d.hpp"
#include "kernels/hip_ssh_2d.hpp"

namespace GauXC {

template <typename Base>
std::unique_ptr<XCDeviceData> HipAoSScheme1<Base>::create_device_data(const DeviceRuntimeEnvironment& rt) {
  return std::make_unique<Data>(rt);
}


template <typename Base>
void HipAoSScheme1<Base>::partition_weights( XCDeviceData* _data ) {
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  auto device_backend = dynamic_cast<HIPBackend*>(data->device_backend_);
  if( !device_backend ) GAUXC_BAD_BACKEND_CAST();

  const auto ldatoms = data->get_ldatoms();
  auto base_stack    = data->base_stack;
  auto static_stack  = data->static_stack;
  auto scheme1_stack = data->scheme1_stack;

  // Compute distances from grid to atomic centers
  compute_grid_to_center_dist( data->total_npts_task_batch, data->global_dims.natoms,
    static_stack.coords_device, base_stack.points_x_device, 
    base_stack.points_y_device, base_stack.points_z_device,
    scheme1_stack.dist_scratch_device, ldatoms, *device_backend->master_stream );

#if 1
  partition_weights_ssf_2d( data->total_npts_task_batch, data->global_dims.natoms,
    static_stack.rab_device, ldatoms, static_stack.coords_device,
    scheme1_stack.dist_scratch_device, ldatoms, scheme1_stack.iparent_device,
    scheme1_stack.dist_nearest_device, base_stack.weights_device,
    *device_backend->master_stream );

#else
  // Modify weights
  partition_weights_ssf_1d( data->total_npts_task_batch, data->global_dims.natoms,
    static_stack.rab_device, ldatoms, static_stack.coords_device, 
    scheme1_stack.dist_scratch_device, ldatoms, scheme1_stack.iparent_device, 
    scheme1_stack.dist_nearest_device, base_stack.weights_device,
    *device_backend->master_stream );
#endif

}


template struct HipAoSScheme1<AoSScheme1Base>;
#ifdef GAUXC_HAS_MAGMA
template struct HipAoSScheme1<AoSScheme1MAGMABase>;
#endif


}

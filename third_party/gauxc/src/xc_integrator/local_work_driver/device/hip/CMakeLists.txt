#
# GauXC Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy).
#
# (c) 2024-2025, Microsoft Corporation
#
# All rights reserved.
#
# See LICENSE.txt for details
#
target_sources(gauxc PRIVATE
  hip_aos_scheme1_data.cxx
  hip_aos_scheme1.cxx

  xc_functional_eval_wrapper.cxx

  kernels/collocation_device.hip
  kernels/grid_to_center.hip
  kernels/hip_ssf_1d.hip
  kernels/hip_ssh_2d.hip
  #cuda_aos_scheme1_weights.cu # cuda_ssf_2d w/ CudaAoSScheme1 constants
  kernels/pack_submat.hip
  kernels/hipblas_extensions.hip
  kernels/uvvars.hip
  kernels/zmat_vxc.hip
  kernels/hip_inc_potential.hip
  kernels/symmetrize_mat.hip

)

if(GAUXC_HAS_MAGMA)
  # MAGMA requires linkage to hipSPARSE for some strange reason....
  find_package( hipsparse REQUIRED )
  target_link_libraries( gauxc PUBLIC roc::hipsparse )
endif()

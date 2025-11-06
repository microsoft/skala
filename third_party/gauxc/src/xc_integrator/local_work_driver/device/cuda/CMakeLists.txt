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
  cuda_aos_scheme1_data.cxx
  cuda_aos_scheme1.cxx

  xc_functional_eval_wrapper.cxx

  kernels/collocation_device.cu
  #kernels/cuda_weights.cu
  kernels/grid_to_center.cu
  kernels/cuda_ssf_1d.cu
  cuda_aos_scheme1_weights.cu # cuda_ssf_2d w/ CudaAoSScheme1 constants
  kernels/pack_submat.cu
  kernels/cublas_extensions.cu
  kernels/uvvars.cu
  kernels/zmat_vxc.cu
  kernels/zmat_fxc.cu
  kernels/cuda_inc_potential.cu
  kernels/symmetrize_mat.cu
  kernels/increment_exc_grad.cu
  kernels/exx_ek_screening_bfn_stats.cu
  kernels/onedft.cu
)

# Check if CMAKE_CUDA_ARCHITECTURES is set
if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
  message( FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES Must Be Set" )
endif()

# Check that only CUDA CC 6.0+ is enabled
foreach( cuda_arch ${CMAKE_CUDA_ARCHITECTURES} )
  if( cuda_arch LESS 60 )
    message(FATAL_ERROR "GauXC Requires CUDA CC 6.0+ For FP64 Atomics")
  endif()
endforeach()

add_subdirectory(obara_saika)
if(GAUXC_ENABLE_CUTLASS)
    include( gauxc-cutlass )
    target_link_libraries( gauxc PRIVATE $<BUILD_INTERFACE:gauxc_cutlass> )
    target_sources( gauxc PRIVATE
      scheme1_cutlass_base.cxx 
      scheme1_cutlass_data_base.cxx   

      kernels/cutlass_wrapper.cu
    )
endif()

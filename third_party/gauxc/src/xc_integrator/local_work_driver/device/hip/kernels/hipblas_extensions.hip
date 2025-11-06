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
#include "hip/hip_runtime.h"
#include <gauxc/util/div_ceil.hpp>
#include "exceptions/hipblas_exception.hpp"

#include "device_specific/hip_device_constants.hpp"
#include "device_specific/hipblas_util.hpp"
#include "device/common/device_blas.hpp"

namespace GauXC {

hipblasOperation_t device_op_to_hipblas( DeviceBlasOp op ) {
  switch( op ) {
    case DeviceBlasOp::NoTrans: return HIPBLAS_OP_N;
    case DeviceBlasOp::Trans:   return HIPBLAS_OP_T;
    default:
      GAUXC_GENERIC_EXCEPTION("Unsupported DeviceBlasOp");
      return HIPBLAS_OP_N;
  }
}

hipblasFillMode_t device_uplo_to_hipblas( DeviceBlasUplo uplo ) {
  switch(uplo) {
    case DeviceBlasUplo::Upper: return HIPBLAS_FILL_MODE_UPPER;
    case DeviceBlasUplo::Lower: return HIPBLAS_FILL_MODE_LOWER;
    default:
      GAUXC_GENERIC_EXCEPTION("Unsupported DeviceBlasUplo");
      return HIPBLAS_FILL_MODE_LOWER;
  }
}

template <typename T>
__global__ void increment_kernel( const T* X, T* Y ) {
  const auto tid = blockIdx.x;
  if( tid < 1 ) (*Y) += (*X);
}

template <typename T>
void increment( const T* X, T* Y, hipStream_t stream ) {
  hipLaunchKernelGGL(increment_kernel, dim3(1), dim3(1), 0, stream, X,Y);
}

template <>
void dot( device_blas_handle generic_handle,
          int            N,
          const double*  X,
          int            INCX,
          const double*  Y,
          int            INCY,
          double*        RES ) {

  hipblasHandle_t handle = generic_handle.blas_handle_as<util::hipblas_handle>();

  auto stat = hipblasDdot( handle, N, X, INCX, Y, INCY, RES );
  GAUXC_HIPBLAS_ERROR("HIPBLAS DDOT FAILED", stat );

}

template <typename T>
void gdot( device_blas_handle generic_handle,
           int       N,
           const T*  X,
           int       INCX,
           const T*  Y,
           int       INCY,
           T*        SCR,
           T*        RES ) {


  dot( generic_handle, N, X, INCX, Y, INCY, SCR );
  hipblasHandle_t handle = generic_handle.blas_handle_as<util::hipblas_handle>();
  auto stream = util::get_stream(handle);
  increment( SCR, RES, stream );

}

template 
void gdot( device_blas_handle generic_handle,
           int            N,
           const double*  X,
           int            INCX,
           const double*  Y,
           int            INCY,
           double*        SCR,
           double*        RES );










template <typename T>
void __global__ hadamard_product_kernel( int      M,
                                         int      N,
                                         const T* A,
                                         int      LDA,
                                         T*       B,
                                         int      LDB ) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < M and tid_y < N ) {
    B[ tid_x + tid_y*LDB ] *= A[ tid_x + tid_y*LDA ];
  }

}



template <typename T>
void hadamard_product( device_blas_handle generic_handle,
                       int            M,
                       int            N,
                       const T*       A,
                       int            LDA,
                       T*             B,
                       int            LDB ) {


  hipblasHandle_t handle = generic_handle.blas_handle_as<util::hipblas_handle>();
  auto stream = util::get_stream(handle);
  dim3 threads(hip::warp_size, hip::max_warps_per_thread_block);
  dim3 blocks( util::div_ceil( M, threads.x ),
               util::div_ceil( N, threads.y ) );

  hipLaunchKernelGGL(hadamard_product_kernel, dim3(blocks), dim3(threads), 0, stream ,  M, N, A, LDA, B, LDB );

}
 
template 
void hadamard_product( device_blas_handle generic_handle,
                       int            M,
                       int            N,
                       const double*  A,
                       int            LDA,
                       double*        B,
                       int            LDB ); 




template <>
void gemm( device_blas_handle generic_handle, 
           DeviceBlasOp TA, DeviceBlasOp TB,
           int M, int N, int K, double ALPHA, 
           const double* A, int LDA, const double* B, int LDB,
           double BETA, double* C, int LDC ) {


  hipblasHandle_t handle = generic_handle.blas_handle_as<util::hipblas_handle>();
  auto stat = hipblasDgemm( handle, device_op_to_hipblas(TA), 
    device_op_to_hipblas(TB), M, N, K, &ALPHA, A, LDA,
    B, LDB, &BETA, C, LDC );
  GAUXC_HIPBLAS_ERROR("HIPBLAS DGEMM FAILED", stat);

}


template <>
void syr2k( device_blas_handle generic_handle, 
            DeviceBlasUplo UPLO, DeviceBlasOp Trans,
            int M, int K, double ALPHA, 
            const double* A, int LDA, const double* B, int LDB,
            double BETA, double* C, int LDC ) {

  hipblasHandle_t handle = generic_handle.blas_handle_as<util::hipblas_handle>();
  auto stat = hipblasDsyr2k( handle, device_uplo_to_hipblas(UPLO), 
    device_op_to_hipblas(Trans), M, K, &ALPHA, A, LDA, B, LDB,
    &BETA, C, LDC );
  GAUXC_HIPBLAS_ERROR("HIPBLAS DSYR2K FAILED", stat);

}

}


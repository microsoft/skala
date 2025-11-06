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
#pragma once
#include "device/device_blas_handle.hpp"

namespace GauXC {

enum class DeviceBlasOp : unsigned char {
  NoTrans,
  Trans
};

enum class DeviceBlasUplo : unsigned char {
  Upper,
  Lower
};

template <typename T>
void increment( device_blas_handle generic_handle, const T* X, T* Y, int N );


template <typename T>
void dot( device_blas_handle handle,
          int            N,
          const T*       X,
          int            INCX,
          const T*       Y,
          int            INCY,
          T*             RES );

template <typename T>
void gdot( device_blas_handle handle,
          int            N,
           const T*       X,
           int            INCX,
           const T*       Y,
           int            INCY,
           T*             SCR,
           T*             RES );


template <typename T>
void hadamard_product( device_blas_handle handle,
                       int            M,
                       int            N,
                       const T*       A,
                       int            LDA,
                       T*             B,
                       int            LDB );
                       

template <typename T>
void gemm( device_blas_handle handle, 
           DeviceBlasOp TA, DeviceBlasOp TB,
           int M, int N, int K, T ALPHA, 
           const T* A, int LDA, const T* B, int LDB,
           T BETA, T* C, int LDC );

template <typename T>
void syr2k( device_blas_handle handle, 
            DeviceBlasUplo UPLO, DeviceBlasOp Trans,
            int M, int K, T ALPHA, 
            const T* A, int LDA, const T* B, int LDB,
            T BETA, T* C, int LDC );

}


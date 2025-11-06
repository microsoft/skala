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

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>

#include "device/device_queue.hpp"

namespace GauXC {


/**
 * @brief Runs a vbatch GEMM operation with CUTLASS
 *
 * Executes a set of GEMM operations where each operation has a different problem size. MAGMA 
 * calls this a vbatch operation, and CUTLASS refers to it as a grouped operation. 
 *
 * @param[in] problem_sizes_device Device buffer containing problem sizes (m, n, k) of gemms
 * @param[in] problem_sizes_host   Host buffer containing problem sizes (m, n, k) of gemms
 * @param[in] problem_count        Number of problems
 * @param[in] ptr_A                Device buffer containing pointers to A matrices
 * @param[in] ptr_B                Device buffer containing pointers to B matrices
 * @param[in] ptr_C                Device buffer containing pointers to C matrices
 * @param[in] ptr_D                Device buffer containing pointers to D matrices
 * @param[in] lda                  Device buffer containing leading dimension of A matrices
 * @param[in] ldb                  Device buffer containing leading dimension of B matrices
 * @param[in] ldc                  Device buffer containing leading dimension of C matrices
 * @param[in] ldd                  Device buffer containing leading dimension of D matrices
 * @param[in] alpha                Alpha parameter for gemm operations
 * @param[in] beta                 Beta parameter for gemm operations
 * @param[in] queue                device queue to which operations will be submitted
 *
 */
void cutlass_gemm(
  cutlass::gemm::GemmCoord* problem_sizes_device,
  cutlass::gemm::GemmCoord* problem_sizes_host,
  const int problem_count,
  double ** ptr_A,
  double ** ptr_B,
  double ** ptr_C,
  double ** ptr_D,
  int64_t* lda,
  int64_t* ldb,
  int64_t* ldc,
  int64_t* ldd,
  const double alpha,
  const double beta,
  device_queue queue
);

/**
 * @brief Runs a vbatch Syr2k operation with CUTLASS
 *
 * Executes a set of SYR2K operations where each operation has a different problem size. MAGMA 
 * calls this a vbatch operation, and CUTLASS refers to it as a grouped operation. 
 *
 * @param[in] problem_sizes_device Device buffer containing problem sizes (m, n, k) of syr2k
 * @param[in] problem_sizes_host   Host buffer containing problem sizes (m, n, k) of syr2k
 * @param[in] problem_count        Number of problems
 * @param[in] ptr_A                Device buffer containing pointers to A matrices
 * @param[in] ptr_B                Device buffer containing pointers to B matrices
 * @param[in] ptr_C                Device buffer containing pointers to C matrices
 * @param[in] ptr_D                Device buffer containing pointers to D matrices
 * @param[in] lda                  Device buffer containing leading dimension of A matrices
 * @param[in] ldb                  Device buffer containing leading dimension of B matrices
 * @param[in] ldc                  Device buffer containing leading dimension of C matrices
 * @param[in] ldd                  Device buffer containing leading dimension of D matrices
 * @param[in] alpha                Alpha parameter for syr2k operations
 * @param[in] beta                 Beta parameter for syr2k operations
 * @param[in] queue                device queue to which operations will be submitted
 *
 */
void cutlass_syr2k(
  cutlass::gemm::GemmCoord* problem_sizes_device,
  cutlass::gemm::GemmCoord* problem_sizes_host,
  const int problem_count,
  double ** ptr_A,
  double ** ptr_B,
  double ** ptr_C,
  double ** ptr_D,
  int64_t* lda,
  int64_t* ldb,
  int64_t* ldc,
  int64_t* ldd,
  const double alpha,
  const double beta,
  device_queue queue
);

}

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
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal.h>

#include <cutlass/gemm/kernel/rank_2k_grouped.h>
#include <cutlass/gemm/kernel/default_rank_2k_grouped.h>
#include <cutlass/gemm/device/rank_2k_grouped.h>
#include <cutlass/gemm/device/rank_2k.h>

#include <cutlass/util/device_memory.h>

#include "device_specific/cuda_device_constants.hpp"
#include "device_specific/cuda_util.hpp"
#include "device/device_queue.hpp"
#include "exceptions/cutlass_exception.hpp"

namespace GauXC {

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
) {
  // Template parameters defining data types and layouts
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementA = double; 
  using ElementB = double; 
  using ElementC = double; 

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  constexpr int kAlignmentA = 1;
  constexpr int kAlignmentB = 1;
  
  constexpr cutlass::ComplexTransform kTransformA = cutlass::ComplexTransform::kNone;
  constexpr cutlass::ComplexTransform kTransformB = cutlass::ComplexTransform::kNone;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,  1,
    ElementAccumulator,
    ElementAccumulator>; 

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

  // Tunable and arch specific Parameters
  // Perform scheduling on device
  constexpr GroupScheduleMode kGroupScheduleMode = GroupScheduleMode::kDeviceOnly;  
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;  // Size of  Gemm each thread block will perform
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;  // Size of Gemm each warp will perform
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;  // Size of DMMA Tensor Core in Ampere
  constexpr int kStages = 4;  // Number of shared memory stages

  // Define CUTLASS GEMM Type
  using GemmGroupKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, kTransformA, kAlignmentA,
    ElementB, LayoutB, kTransformB, kAlignmentB,
    ElementOutput, LayoutC, 
    ElementAccumulator, 
    OperatorClass,
    ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kGroupScheduleMode>::GemmKernel;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmGroupKernel>; 

  const int threadblock_count = GemmGrouped::sufficient(problem_sizes_host, problem_count);

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  cutlass::Status status = cutlass::Status::kSuccess;
  typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(alpha, beta);

  // Configure GEMM arguments
  typename GemmGrouped::Arguments args(
    problem_sizes_device,
    problem_count,
    threadblock_count,
    epilogue_op,
    ptr_A,
    ptr_B,
    ptr_C,
    ptr_D,
    lda,
    ldb,
    ldc,
    ldd,
    problem_sizes_host
  );

  // Initialize the GEMM object
  GemmGrouped gemm;

  size_t workspace_size = gemm.get_workspace_size(args);
  if (workspace_size) {
    GAUXC_GENERIC_EXCEPTION("CUTLASS Workspace Size Must Be Zero");
  }

  status = gemm.initialize(args, nullptr);
  GAUXC_CUTLASS_ERROR("CUTLASS Group Gemm Initialization Failed", status);

  status = gemm.run(stream);
  GAUXC_CUTLASS_ERROR("CUTLASS Group Gemm Run Failed", status);
}


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
) {
  // Template parameters defining data types and layouts
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementA = double; 
  using ElementB = double; 
  using ElementC = double; 

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  constexpr int kAlignmentA = 1;
  constexpr int kAlignmentB = 1;

  constexpr cutlass::ComplexTransform kTransformA = cutlass::ComplexTransform::kNone;
  constexpr cutlass::ComplexTransform kTransformB = cutlass::ComplexTransform::kNone;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,  1,
    ElementAccumulator,
    ElementAccumulator>; 

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  
  // Tunable and arch specific Parameters
  // Perform scheduling on device
  constexpr GroupScheduleMode kGroupScheduleMode = GroupScheduleMode::kDeviceOnly;
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;  // Size of  Gemm each thread block will perform
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;  // Size of Gemm each warp will perform
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;  // Size of DMMA Tensor Core in Ampere
  constexpr int kStages = 4;  // Number of shared memory stages

  // Syr2k specific
  constexpr cutlass::FillMode kFillModeC = cutlass::FillMode::kLower;
  using Operator = cutlass::arch::OpMultiplyAdd;
  constexpr cutlass::BlasMode kBlasMode = cutlass::BlasMode::kSymmetric;

  // Define CUTLASS SYR2k Type
  using SYR2KGroupkernel = typename cutlass::gemm::kernel::DefaultRank2KGrouped<
    ElementA, LayoutA, kTransformA, kAlignmentA,
    ElementB, LayoutB, kTransformB, kAlignmentB,
    ElementOutput, LayoutC, kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    Operator, kBlasMode,
    kGroupScheduleMode>::Rank2Kkernel;

  using Syr2kGrouped = cutlass::gemm::device::Rank2KGrouped<SYR2KGroupkernel>;

  const int threadblock_count = Syr2kGrouped::sufficient(problem_sizes_host, problem_count);

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  cutlass::Status status = cutlass::Status::kSuccess;
  typename Syr2kGrouped::EpilogueOutputOp::Params epilogue_op(alpha, beta);

   typename Syr2kGrouped::Arguments args(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_sizes_device,
    problem_count,
    threadblock_count,
    epilogue_op,
    ptr_A,
    ptr_B,
    ptr_C,
    ptr_D,
    lda,
    ldb,
    ldc,
    ldd,
    problem_sizes_host
  ); 

  Syr2kGrouped gemm;
  size_t workspace_size = gemm.get_workspace_size(args);
  if (workspace_size) {
    GAUXC_GENERIC_EXCEPTION("CUTLASS Workspace Size Must Be Zero");
  }

  status = gemm.initialize(args, nullptr);
  GAUXC_CUTLASS_ERROR("CUTLASS Group Syr2k Initialization Failed", status);
  status = gemm.run(stream);
  GAUXC_CUTLASS_ERROR("CUTLASS Group Syr2k Run Failed", status);
}

}

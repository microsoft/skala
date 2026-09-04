/**
 * @file
 * @brief SkalaXC C API: lightweight rank-local diagnostics.
 */
#pragma once

#include <stdint.h>

#include <skalaxc/c/enums.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Number of entries in a diagnostics timing array. */
#define SKALAXC_TIMING_METRIC_COUNT 11

/** @brief Stable identifiers for XC-integrator timing phases. */
enum SkalaXC_TimingMetric {
  SkalaXC_TimingMetric_ModelLoad,            ///< TorchScript model loading
  SkalaXC_TimingMetric_FeatureConstruction,  ///< ML feature construction
  SkalaXC_TimingMetric_ModelBatchPacking,    ///< Complete-domain batch packing
  SkalaXC_TimingMetric_ModelForward,         ///< ML model forward evaluation
  SkalaXC_TimingMetric_ModelBackward,        ///< ML model backward evaluation
  SkalaXC_TimingMetric_PotentialMapping,     ///< Potential mapping to grid data
  SkalaXC_TimingMetric_AOAssembly,           ///< AO matrix assembly
  SkalaXC_TimingMetric_GradientAssembly,     ///< Nuclear-gradient assembly
  SkalaXC_TimingMetric_MPIReduction,         ///< MPI result reduction
  SkalaXC_TimingMetric_TotalEXCVXC,          ///< Complete EXC/VXC evaluation
  SkalaXC_TimingMetric_TotalEXCGradient      ///< Complete gradient evaluation
};

/** @brief Availability of a timing value in a diagnostics snapshot. */
enum SkalaXC_TimingStatus {
  SkalaXC_TimingStatus_Unavailable,  ///< Timing is unavailable for this backend
  SkalaXC_TimingStatus_Pending,  ///< Asynchronous timing is not yet complete
  SkalaXC_TimingStatus_Complete  ///< Timing contains a complete value
};

/** @brief Rank-local timing and debug-logging configuration. */
typedef struct skalaxc_timing_settings {
  int32_t verbose;       /**< Nonzero requests complete CUDA timings on read. */
  int32_t debug_logging; /**< Nonzero emits rank-local diagnostics to stderr. */
} skalaxc_timing_settings_t;

/** @brief Last and cumulative values for one timing phase. */
typedef struct skalaxc_timing_value {
  int64_t last_nanoseconds;   ///< Most recent completed duration
  int64_t total_nanoseconds;  ///< Sum of all completed durations
  int64_t call_count;         ///< Number of recorded phase invocations
  int32_t status;             ///< One of enum SkalaXC_TimingStatus
} skalaxc_timing_value_t;

/** @brief Rank-local, non-collective snapshot of integrator diagnostics. */
typedef struct skalaxc_diagnostics_snapshot {
  int32_t backend;               /**< Host or device execution space. */
  int32_t rank;                  /**< Rank in the runtime communicator. */
  int32_t communicator_size;     /**< Size of the runtime communicator. */
  int32_t device_id;             /**< CUDA ordinal, or -1 for host execution. */
  int32_t openmp_threads;        /**< Maximum OpenMP threads at construction. */
  double device_memory_fraction; /**< GauXC CUDA arena fraction, or zero. */
  int32_t domain_batch_mode;     /**< Configured SkalaXC_DomainBatchMode. */
  skalaxc_timing_value_t timings[SKALAXC_TIMING_METRIC_COUNT];
  /**< Timing values indexed by enum SkalaXC_TimingMetric. */
  int64_t exc_vxc_calls;            ///< Number of EXC/VXC evaluations
  int64_t exc_gradient_calls;       ///< Number of gradient evaluations
  int64_t model_batches;            ///< Model batches evaluated since reset
  int64_t domains;                  ///< Atomic domains evaluated since reset
  int64_t tasks;                    ///< Local quadrature tasks processed
  int64_t points;                   ///< Local quadrature points processed
  int64_t local_atoms;              ///< Atomic domains owned by this rank
  int64_t configured_model_batches; /**< Model batches planned at construction.
                                     */
  int64_t task_points_min;          ///< Minimum points in a processed task
  int64_t task_points_max;          ///< Maximum points in a processed task
  int64_t task_basis_min;           ///< Minimum basis functions in a task
  int64_t task_basis_max;           ///< Maximum basis functions in a task
  int64_t model_batch_points_min;   ///< Minimum points in a model batch
  int64_t model_batch_points_max;   ///< Maximum points in a model batch
  int64_t max_domains_per_model_batch; /**< Maximum domains per model batch. */
} skalaxc_diagnostics_snapshot_t;

/**
 * @brief Initialize timing settings with synchronization and logging disabled.
 * @param settings Output settings; NULL is tolerated.
 */
void skalaxc_timing_settings_default(skalaxc_timing_settings_t* settings)
    SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif

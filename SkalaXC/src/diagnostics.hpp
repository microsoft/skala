#pragma once

#include "index_types.hpp"

#include <skalaxc/skalaxc.hpp>

#include <chrono>
#include <cstdint>

namespace SkalaXC::detail {

/** @return Maximum OpenMP thread count available to the caller. */
int maximum_openmp_threads() noexcept;

/** @brief Rank-local task and grid-point counts. */
struct LocalWorkload {
  types::TaskCount tasks{};        ///< Number of local tasks.
  types::GridPointCount points{};  ///< Number of local grid points.
};

/** @brief Rank-local model batching geometry. */
struct ModelWorkload {
  types::AtomCount local_atoms{};  ///< Complete atomic domains owned locally.
  types::ModelBatchCount configured_batches{};  ///< Configured local batches.
  types::CountRange<types::GridPointCount> task_points{};  ///< Points per task.
  types::CountRange<types::BasisFunctionCount>
      task_basis_functions{};  ///< Basis functions per task.
  types::CountRange<types::GridPointCount> batch_points{};  ///< Points per
                                                            ///< batch.
  types::DomainCount max_domains_per_batch{};  ///< Largest batch domain count.
};

/** @brief Parallel and backend setup recorded for one integrator. */
struct ParallelSetup {
  types::CommunicatorSize communicator_size{};  ///< Runtime communicator size.
  types::DeviceId device_id{};                  ///< Selected CUDA device.
  types::OpenMPThreadCount openmp_threads{};    ///< Available OpenMP threads.
  double device_memory_fraction = 0.0;  ///< GauXC device arena fraction.
  DomainBatchMode domain_batch_mode =
      DomainBatchMode::Conservative;  ///< Batching policy.
};

/** @brief Own and update one integrator's rank-local diagnostics snapshot. */
class DiagnosticsRegistry {
 public:
  /**
   * @brief Initialize one rank-local registry.
   * @param settings Timing and logging settings.
   * @param backend Evaluation backend.
   * @param rank Runtime-communicator rank.
   */
  DiagnosticsRegistry(TimingSettings settings, ExecutionSpace backend,
                      types::CommunicatorRank rank) noexcept;

  /**
   * @brief Record one completed phase.
   * @param metric Phase identifier.
   * @param duration Elapsed wall time.
   */
  void record(TimingMetric metric, std::chrono::nanoseconds duration) noexcept;
  /** @brief Increment the EXC/VXC call count. */
  void increment_exc_vxc_calls() noexcept;
  /** @brief Increment the gradient call count. */
  void increment_exc_gradient_calls() noexcept;
  /** @brief Replace recorded parallel setup. @param setup Setup values. */
  void set_parallel_setup(ParallelSetup setup) noexcept;
  /** @brief Replace recorded local workload. @param workload Workload values.
   */
  void set_local_workload(LocalWorkload workload) noexcept;
  /** @brief Replace recorded model workload. @param workload Workload values.
   */
  void set_model_workload(ModelWorkload workload) noexcept;
  /** @brief Record one processed model batch. @param domains Domains in the
   * batch. */
  void record_model_batch(types::DomainCount domains) noexcept;
  /** @return Current rank-local snapshot. */
  DiagnosticsSnapshot snapshot() const noexcept;
  /** @brief Clear evaluation counters and timings while retaining setup. */
  void reset_evaluation() noexcept;

 private:
  TimingSettings settings_;
  DiagnosticsSnapshot snapshot_;
};

/** @brief Record elapsed host wall time for one diagnostics phase. */
class HostTimingScope {
 public:
  /**
   * @brief Start timing a host phase.
   * @param registry Destination registry.
   * @param metric Phase identifier.
   */
  HostTimingScope(DiagnosticsRegistry& registry, TimingMetric metric) noexcept;
  /** @brief Record elapsed time unless already finished. */
  ~HostTimingScope() noexcept;

  /** @brief Record elapsed time immediately; subsequent calls are no-ops. */
  void finish() noexcept;

  HostTimingScope(const HostTimingScope&) = delete;
  HostTimingScope& operator=(const HostTimingScope&) = delete;

 private:
  DiagnosticsRegistry* registry_ = nullptr;
  TimingMetric metric_ = TimingMetric::Count;
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace SkalaXC::detail

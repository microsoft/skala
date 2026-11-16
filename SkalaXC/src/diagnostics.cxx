#include "diagnostics.hpp"
#include "saturating_math.hpp"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace SkalaXC::detail {

namespace {

std::uint64_t as_nanoseconds(std::chrono::nanoseconds duration) noexcept {
  if (duration.count() <= 0) return 0;
  return static_cast<std::uint64_t>(duration.count());
}

}  // namespace

int maximum_openmp_threads() noexcept {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

DiagnosticsRegistry::DiagnosticsRegistry(TimingSettings settings,
                                         ExecutionSpace backend,
                                         types::CommunicatorRank rank) noexcept
    : settings_(settings) {
  snapshot_.backend = backend;
  snapshot_.rank = rank.raw();
}

void DiagnosticsRegistry::record(TimingMetric metric,
                                 std::chrono::nanoseconds duration) noexcept {
  if (metric == TimingMetric::Count) return;
  auto& value = snapshot_.timings[static_cast<std::size_t>(metric)];
  value.last_nanoseconds = as_nanoseconds(duration);
  value.total_nanoseconds =
      saturating_add(value.total_nanoseconds, value.last_nanoseconds);
  value.call_count = saturating_add(value.call_count, 1);
  value.status = TimingStatus::Complete;
}

void DiagnosticsRegistry::increment_exc_vxc_calls() noexcept {
  snapshot_.exc_vxc_calls = saturating_add(snapshot_.exc_vxc_calls, 1);
}

void DiagnosticsRegistry::increment_exc_gradient_calls() noexcept {
  snapshot_.exc_gradient_calls =
      saturating_add(snapshot_.exc_gradient_calls, 1);
}

void DiagnosticsRegistry::set_parallel_setup(ParallelSetup setup) noexcept {
  snapshot_.communicator_size = setup.communicator_size.raw();
  snapshot_.device_id = setup.device_id.raw();
  snapshot_.openmp_threads = setup.openmp_threads.raw();
  snapshot_.device_memory_fraction = setup.device_memory_fraction;
  snapshot_.domain_batch_mode = setup.domain_batch_mode;
}

void DiagnosticsRegistry::set_local_workload(LocalWorkload workload) noexcept {
  snapshot_.tasks = workload.tasks.raw();
  snapshot_.points = static_cast<std::uint64_t>(workload.points.raw());
}

void DiagnosticsRegistry::set_model_workload(ModelWorkload workload) noexcept {
  snapshot_.local_atoms = workload.local_atoms.raw();
  snapshot_.configured_model_batches = workload.configured_batches.raw();
  snapshot_.task_points_min =
      static_cast<std::uint64_t>(workload.task_points.minimum.raw());
  snapshot_.task_points_max =
      static_cast<std::uint64_t>(workload.task_points.maximum.raw());
  snapshot_.task_basis_min =
      static_cast<std::uint64_t>(workload.task_basis_functions.minimum.raw());
  snapshot_.task_basis_max =
      static_cast<std::uint64_t>(workload.task_basis_functions.maximum.raw());
  snapshot_.model_batch_points_min =
      static_cast<std::uint64_t>(workload.batch_points.minimum.raw());
  snapshot_.model_batch_points_max =
      static_cast<std::uint64_t>(workload.batch_points.maximum.raw());
  snapshot_.max_domains_per_model_batch = workload.max_domains_per_batch.raw();
}

void DiagnosticsRegistry::record_model_batch(
    types::DomainCount domains) noexcept {
  snapshot_.model_batches = saturating_add(snapshot_.model_batches, 1);
  snapshot_.domains = saturating_add(snapshot_.domains, domains.raw());
}

DiagnosticsSnapshot DiagnosticsRegistry::snapshot() const noexcept {
  return snapshot_;
}

void DiagnosticsRegistry::reset_evaluation() noexcept {
  for (std::size_t index =
           static_cast<std::size_t>(TimingMetric::ModelLoad) + 1;
       index < timing_metric_count; ++index)
    snapshot_.timings[index] = {};
  snapshot_.exc_vxc_calls = 0;
  snapshot_.exc_gradient_calls = 0;
  snapshot_.model_batches = 0;
  snapshot_.domains = 0;
}

HostTimingScope::HostTimingScope(DiagnosticsRegistry& registry,
                                 TimingMetric metric) noexcept
    : registry_(&registry),
      metric_(metric),
      start_(std::chrono::steady_clock::now()) {}

HostTimingScope::~HostTimingScope() noexcept { finish(); }

void HostTimingScope::finish() noexcept {
  if (!registry_) return;
  registry_->record(metric_,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start_));
  registry_ = nullptr;
}

}  // namespace SkalaXC::detail
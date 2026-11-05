#include "skala_driver.hpp"

#include "model_grid_layout.hpp"
#include "saturating_math.hpp"

#include <gauxc/xc_task.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>

namespace SkalaXC {
namespace {

struct MatrixSummary {
  double trace = 0.0;
  double norm = 0.0;
  double max_abs = 0.0;
};

template <typename Derived>
MatrixSummary summarize_matrix(const Eigen::MatrixBase<Derived>& matrix) {
  return {matrix.trace(), matrix.norm(), matrix.cwiseAbs().maxCoeff()};
}

const char* metric_name(TimingMetric metric) {
  switch (metric) {
    case TimingMetric::ModelLoad:
      return "model_load";
    case TimingMetric::FeatureConstruction:
      return "feature_construction";
    case TimingMetric::ModelBatchPacking:
      return "model_batch_packing";
    case TimingMetric::ModelForward:
      return "model_forward";
    case TimingMetric::ModelBackward:
      return "model_backward";
    case TimingMetric::PotentialMapping:
      return "potential_mapping";
    case TimingMetric::AOAssembly:
      return "ao_assembly";
    case TimingMetric::GradientAssembly:
      return "gradient_assembly";
    case TimingMetric::MPIReduction:
      return "mpi_reduction";
    case TimingMetric::TotalEXCVXC:
      return "total_exc_vxc";
    case TimingMetric::TotalEXCGradient:
      return "total_exc_gradient";
    case TimingMetric::Count:
      return "count";
  }
  return "unknown";
}

void write_summary(std::ostream& output, std::string_view name,
                   const MatrixSummary& summary) {
  output << std::scientific << std::setprecision(8) << name
         << "_trace=" << summary.trace << ' ' << name
         << "_norm=" << summary.norm << ' ' << name
         << "_max_abs=" << summary.max_abs;
}

}  // namespace

void SkalaDriver::set_setup_diagnostics(
    types::CommunicatorSize communicator_size, types::DeviceId device_id,
    double device_memory_fraction, DomainBatchMode batch_mode,
    const std::vector<GauXC::XCTask>& tasks,
    const std::vector<ModelDomainBatch>& batches) noexcept {
  std::int64_t local_points = 0;
  types::CountRange<types::GridPointCount> task_points;
  types::CountRange<types::BasisFunctionCount> task_basis_functions;
  if (!tasks.empty()) {
    task_points.minimum =
        types::GridPointCount{std::numeric_limits<std::int64_t>::max()};
    task_basis_functions.minimum =
        types::BasisFunctionCount{std::numeric_limits<std::int64_t>::max()};
    for (const auto& task : tasks) {
      const types::GridPointCount points{
          static_cast<std::int64_t>(task.points.size())};
      const types::BasisFunctionCount basis_functions{
          static_cast<std::int64_t>(task.bfn_screening.nbe)};
      local_points += points.raw();
      task_points.minimum = types::GridPointCount{
          std::min(task_points.minimum.raw(), points.raw())};
      task_points.maximum = types::GridPointCount{
          std::max(task_points.maximum.raw(), points.raw())};
      task_basis_functions.minimum = types::BasisFunctionCount{
          std::min(task_basis_functions.minimum.raw(), basis_functions.raw())};
      task_basis_functions.maximum = types::BasisFunctionCount{
          std::max(task_basis_functions.maximum.raw(), basis_functions.raw())};
    }
  }

  std::uint64_t local_atoms = 0;
  types::CountRange<types::GridPointCount> batch_points;
  std::uint64_t max_domains_per_batch = 0;
  if (!batches.empty()) {
    batch_points.minimum =
        types::GridPointCount{std::numeric_limits<std::int64_t>::max()};
    for (const auto& batch : batches) {
      const auto domains = static_cast<std::uint64_t>(batch.atoms.size());
      const auto points = batch.point_count;
      local_atoms = detail::saturating_add(local_atoms, domains);
      batch_points.minimum = types::GridPointCount{
          std::min(batch_points.minimum.raw(), points.raw())};
      batch_points.maximum = types::GridPointCount{
          std::max(batch_points.maximum.raw(), points.raw())};
      max_domains_per_batch = std::max(max_domains_per_batch, domains);
    }
  }

  diagnostics_.set_parallel_setup(
      {communicator_size, device_id,
       types::OpenMPThreadCount{detail::maximum_openmp_threads()},
       device_memory_fraction, batch_mode});
  diagnostics_.set_local_workload(
      {types::TaskCount{tasks.size()}, types::GridPointCount{local_points}});
  diagnostics_.set_model_workload(
      {types::AtomCount{local_atoms}, types::ModelBatchCount{batches.size()},
       task_points, task_basis_functions, batch_points,
       types::DomainCount{max_domains_per_batch}});
}

void SkalaDriver::log_setup(
    const std::string& model, const std::vector<std::string>& feature_keys,
    bool is_gga, bool is_mgga,
    const std::vector<ModelDomainBatch>& batches) const noexcept {
  if (!debug_log_.enabled()) return;
  const auto snapshot = diagnostics_.snapshot();
  debug_log_.log("setup", [&](std::ostream& output) {
    output << "communicator_size=" << snapshot.communicator_size
           << " openmp_threads=" << snapshot.openmp_threads;
    if (snapshot.backend == ExecutionSpace::Device)
      output << " device_id=" << snapshot.device_id
             << " memory_fraction=" << snapshot.device_memory_fraction;
  });
  debug_log_.log("model", [&](std::ostream& output) {
    output << "selector=" << model
           << " approximation=" << (is_mgga ? "MGGA" : (is_gga ? "GGA" : "LDA"))
           << " features=";
    for (std::size_t index = 0; index < feature_keys.size(); ++index) {
      if (index != 0) output << ',';
      output << feature_keys[index];
    }
  });
  debug_log_.log("workload", [&](std::ostream& output) {
    output << "tasks=" << snapshot.tasks << " points=" << snapshot.points
           << " local_atoms=" << snapshot.local_atoms
           << " task_points_min=" << snapshot.task_points_min
           << " task_points_max=" << snapshot.task_points_max
           << " task_basis_min=" << snapshot.task_basis_min
           << " task_basis_max=" << snapshot.task_basis_max;
  });
  debug_log_.log("batching", [&](std::ostream& output) {
    output << "mode="
           << (snapshot.domain_batch_mode == DomainBatchMode::Aggressive
                   ? "aggressive"
                   : "conservative")
           << " batches=" << snapshot.configured_model_batches
           << " batch_points_min=" << snapshot.model_batch_points_min
           << " batch_points_max=" << snapshot.model_batch_points_max
           << " max_domains_per_batch=" << snapshot.max_domains_per_model_batch;
  });
  for (std::size_t index = 0; index < batches.size(); ++index) {
    const auto& batch = batches[index];
    debug_log_.log("batch", [&](std::ostream& output) {
      output << "index=" << index << " atoms=";
      for (std::size_t atom_index = 0; atom_index < batch.atoms.size();
           ++atom_index) {
        if (atom_index != 0) output << ',';
        output << batch.atoms[atom_index].raw();
      }
      output << " task_blocks=" << batch.task_blocks.size()
             << " grid_size=" << batch.grid_size.raw()
             << " points=" << batch.point_count.raw();
    });
  }
}

void SkalaDriver::log_model_load_timing() const noexcept {
  if (!debug_log_.enabled()) return;
  const auto snapshot = diagnostics_.snapshot();
  const auto& timing = snapshot.timing(TimingMetric::ModelLoad);
  if (timing.status != TimingStatus::Complete) return;
  debug_log_.log("timing", [&](std::ostream& output) {
    output << "metric=" << metric_name(TimingMetric::ModelLoad)
           << " milliseconds=" << std::fixed << std::setprecision(3)
           << static_cast<double>(timing.last_nanoseconds) / 1.0e6;
  });
}

DiagnosticsSnapshot SkalaDriver::log_evaluation_start(
    std::string_view evaluation, const ConstColMajorMatrixMap& scalar_density,
    const ConstColMajorMatrixMap& spin_density) const noexcept {
  const auto before = diagnostics_.snapshot();
  if (!debug_log_.enabled()) return before;
  const auto scalar = summarize_matrix(scalar_density);
  const auto spin = summarize_matrix(spin_density);
  debug_log_.log("evaluation", [&](std::ostream& output) {
    output << "kind=" << evaluation
           << " event=start basis_size=" << scalar_density.rows() << ' ';
    write_summary(output, "scalar_density", scalar);
    output << ' ';
    write_summary(output, "spin_density", spin);
  });
  return before;
}

void SkalaDriver::log_exc_vxc_result(
    std::string_view evaluation, double exc,
    const ColMajorMatrixMap& scalar_potential,
    const ColMajorMatrixMap& spin_potential) const noexcept {
  if (!debug_log_.enabled()) return;
  const auto scalar = summarize_matrix(scalar_potential);
  const auto spin = summarize_matrix(spin_potential);
  debug_log_.log("evaluation", [&](std::ostream& output) {
    output << std::scientific << std::setprecision(8) << "kind=" << evaluation
           << " event=end exc=" << exc << ' ';
    write_summary(output, "scalar_potential", scalar);
    output << ' ';
    write_summary(output, "spin_potential", spin);
  });
}

void SkalaDriver::log_gradient_result(
    std::string_view evaluation,
    const RowMajorMatrixMap& gradient) const noexcept {
  if (!debug_log_.enabled()) return;
  const auto translation = gradient.colwise().sum();
  debug_log_.log("evaluation", [&](std::ostream& output) {
    output << std::scientific << std::setprecision(8) << "kind=" << evaluation
           << " event=end gradient_norm=" << gradient.norm()
           << " gradient_max_abs=" << gradient.cwiseAbs().maxCoeff()
           << " translation_residual=" << translation.norm();
  });
}

void SkalaDriver::log_host_timing_delta(
    std::string_view evaluation,
    const DiagnosticsSnapshot& before) const noexcept {
  if (!debug_log_.enabled()) return;
  const auto after = diagnostics_.snapshot();
  for (std::size_t index =
           static_cast<std::size_t>(TimingMetric::FeatureConstruction);
       index < timing_metric_count; ++index) {
    const auto& old_value = before.timings[index];
    const auto& new_value = after.timings[index];
    if (new_value.call_count <= old_value.call_count) continue;
    const auto elapsed =
        new_value.total_nanoseconds - old_value.total_nanoseconds;
    debug_log_.log("timing", [&](std::ostream& output) {
      output << "kind=" << evaluation
             << " metric=" << metric_name(static_cast<TimingMetric>(index))
             << " milliseconds=" << std::fixed << std::setprecision(3)
             << static_cast<double>(elapsed) / 1.0e6
             << " calls=" << new_value.call_count - old_value.call_count;
    });
  }
}

void SkalaDriver::log_device_timing_unavailable(
    std::string_view evaluation) const noexcept {
  debug_log_.log("timing", [&](std::ostream& output) {
    output
        << "kind=" << evaluation
        << " status=unavailable reason=cuda_event_collection_not_implemented";
  });
}

}  // namespace SkalaXC
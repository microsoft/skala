#pragma once

#include "debug_log.hpp"
#include "diagnostics.hpp"
#include "host/eigen_types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace GauXC {
struct XCTask;
}

namespace SkalaXC {

struct ModelDomainBatch;

/** @brief Backend-independent interface for UKS evaluation and diagnostics. */
class SkalaDriver {
 public:
  /**
   * @brief Initialize shared diagnostics.
   * @param timing_settings Timing settings.
   * @param backend Evaluation backend.
   * @param rank Runtime rank.
   * @param size Runtime size.
   */
  SkalaDriver(TimingSettings timing_settings, ExecutionSpace backend,
              types::CommunicatorRank rank, types::CommunicatorSize size)
      : diagnostics_(timing_settings, backend, rank),
        debug_log_(timing_settings, backend, rank, size) {}
  virtual ~SkalaDriver() noexcept = default;

  /** @return Current rank-local diagnostics snapshot. */
  DiagnosticsSnapshot diagnostics() const noexcept {
    return diagnostics_.snapshot();
  }

  /** @brief Clear evaluation diagnostics while retaining setup. */
  void reset_diagnostics() noexcept { diagnostics_.reset_evaluation(); }

  /** @brief Emit the completed model-load timing when logging is enabled. */
  void log_model_load_timing() const noexcept;

  /**
   * @brief Evaluate UKS XC energy and potentials.
   * @param scalar_density Scalar density matrix.
   * @param spin_density Spin-z density matrix.
   * @param scalar_potential Output scalar potential.
   * @param spin_potential Output spin-z potential.
   * @return XC energy.
   */
  virtual double eval_exc_vxc_uks(ConstColMajorMatrixMap scalar_density,
                                  ConstColMajorMatrixMap spin_density,
                                  ColMajorMatrixMap scalar_potential,
                                  ColMajorMatrixMap spin_potential) = 0;

  /**
   * @brief Evaluate a UKS XC nuclear gradient.
   * @param scalar_density Scalar density matrix.
   * @param spin_density Spin-z density matrix.
   * @param gradient Output atom-major Cartesian gradient.
   */
  virtual void eval_exc_grad_uks(ConstColMajorMatrixMap scalar_density,
                                 ConstColMajorMatrixMap spin_density,
                                 RowMajorMatrixMap gradient) = 0;

 protected:
  /**
   * @brief Log model and batch setup.
   * @param model Model selector.
   * @param feature_keys Model features.
   * @param is_gga Whether gradients are used.
   * @param is_mgga Whether kinetic density is used.
   * @param batches Configured batches.
   */
  void log_setup(const std::string& model,
                 const std::vector<std::string>& feature_keys, bool is_gga,
                 bool is_mgga,
                 const std::vector<ModelDomainBatch>& batches) const noexcept;
  /**
   * @brief Record parallel and workload setup.
   * @param communicator_size Runtime size.
   * @param device_id Device identifier.
   * @param device_memory_fraction Device arena fraction.
   * @param batch_mode Batching policy.
   * @param tasks Local tasks.
   * @param batches Configured batches.
   */
  void set_setup_diagnostics(
      types::CommunicatorSize communicator_size, types::DeviceId device_id,
      double device_memory_fraction, DomainBatchMode batch_mode,
      const std::vector<GauXC::XCTask>& tasks,
      const std::vector<ModelDomainBatch>& batches) noexcept;
  /**
   * @brief Log evaluation inputs.
   * @param evaluation Operation name.
   * @param scalar_density Scalar density.
   * @param spin_density Spin density.
   * @return Snapshot before evaluation.
   */
  DiagnosticsSnapshot log_evaluation_start(
      std::string_view evaluation, const ConstColMajorMatrixMap& scalar_density,
      const ConstColMajorMatrixMap& spin_density) const noexcept;
  /**
   * @brief Log energy and potentials.
   * @param evaluation Operation name.
   * @param exc XC energy.
   * @param scalar_potential Scalar potential.
   * @param spin_potential Spin potential.
   */
  void log_exc_vxc_result(
      std::string_view evaluation, double exc,
      const ColMajorMatrixMap& scalar_potential,
      const ColMajorMatrixMap& spin_potential) const noexcept;
  /**
   * @brief Log a nuclear gradient.
   * @param evaluation Operation name.
   * @param gradient Atom-major gradient.
   */
  void log_gradient_result(std::string_view evaluation,
                           const RowMajorMatrixMap& gradient) const noexcept;
  /**
   * @brief Log host timing changes.
   * @param evaluation Operation name.
   * @param before Snapshot before evaluation.
   */
  void log_host_timing_delta(std::string_view evaluation,
                             const DiagnosticsSnapshot& before) const noexcept;
  /** @brief Log unavailable device timing. @param evaluation Operation name. */
  void log_device_timing_unavailable(
      std::string_view evaluation) const noexcept;

  detail::DiagnosticsRegistry diagnostics_;  ///< Rank-local diagnostics state.
  detail::DebugLogger debug_log_;            ///< Best-effort debug writer.
};

}  // namespace SkalaXC

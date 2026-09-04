#pragma once
/**
 * @file
 * @brief CUDA orchestration backend for SkalaXC ML functionals.
 */

#include "skala_driver.hpp"

#include <gauxc/load_balancer.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>

#include <memory>
#include <string>
#include <vector>

namespace SkalaXC {

class DeviceModelGridExchange;
class SkalaModel;

/** @brief Evaluate SkalaXC ML functionals with GauXC's CUDA local work API. */
class SkalaDeviceDriver final : public SkalaDriver {
 public:
  /**
   * @brief Construct a driver from weight-partitioned integration tasks.
   * @param weighted_lb Load balancer containing sorted, partitioned tasks.
   * @param raw_weights Per-task quadrature weights before partitioning.
   * @param model Bundled model selector or TorchScript `.fun` path.
   * @param device_id CUDA device ordinal used for evaluation.
   * @param device_memory_fraction GauXC CUDA arena memory fraction.
   * @param timing_settings Rank-local diagnostics configuration.
   * @param batch_mode Exact-size complete-domain batching policy.
   */
  SkalaDeviceDriver(const GauXC::LoadBalancer& weighted_lb,
                    const std::vector<std::vector<double>>& raw_weights,
                    const std::string& model, types::DeviceId device_id,
                    double device_memory_fraction,
                    TimingSettings timing_settings = {},
                    DomainBatchMode batch_mode = DomainBatchMode::Conservative);
  /** @brief Destroy the owned CUDA model and work state. */
  ~SkalaDeviceDriver() noexcept override;

  /** @brief Device drivers cannot be copied. */
  SkalaDeviceDriver(const SkalaDeviceDriver&) = delete;
  /** @brief Device drivers cannot be copy-assigned. */
  SkalaDeviceDriver& operator=(const SkalaDeviceDriver&) = delete;

  /**
   * @brief Evaluate UKS ML exchange-correlation energy and potential.
   * @param scalar_density Scalar-spin column-major density matrix.
   * @param spin_density Z-spin column-major density matrix.
   * @param scalar_potential Output scalar XC potential matrix.
   * @param spin_potential Output z-spin XC potential matrix.
   * @return Global exchange-correlation energy.
   */
  double eval_exc_vxc_uks(ConstColMajorMatrixMap scalar_density,
                          ConstColMajorMatrixMap spin_density,
                          ColMajorMatrixMap scalar_potential,
                          ColMajorMatrixMap spin_potential) override;

  /**
   * @brief Evaluate the UKS ML exchange-correlation nuclear gradient.
   * @param scalar_density Scalar-spin column-major density matrix.
   * @param spin_density Z-spin column-major density matrix.
   * @param gradient Output atom-major matrix with xyz columns.
   */
  void eval_exc_grad_uks(ConstColMajorMatrixMap scalar_density,
                         ConstColMajorMatrixMap spin_density,
                         RowMajorMatrixMap gradient) override;

 private:
  /** @brief CUDA device ordinal selected for every operation. */
  types::DeviceId device_id_;
  /** @brief Weight-partitioned GauXC task owner. */
  GauXC::LoadBalancer lb_;
  /** @brief CUDA local work driver. */
  std::unique_ptr<GauXC::LocalWorkDriver> lwd_;
  /** @brief Model resident on the selected CUDA device. */
  std::unique_ptr<SkalaModel> model_;
  /** @brief Exchange between GauXC task buffers and model tensors. */
  std::unique_ptr<DeviceModelGridExchange> model_grid_exchange_;
};

}  // namespace SkalaXC
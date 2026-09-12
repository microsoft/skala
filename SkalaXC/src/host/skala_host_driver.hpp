#pragma once
/**
 * @file
 * @brief SkalaXC host ML backend.
 *
 * Reimplementation of GauXC/skala
 * reference_replicated_xc_host_integrator_onedft.hpp as a standalone driver.
 *
 * Instead of living as methods on GauXC's ReferenceReplicatedXCHostIntegrator
 * and entering GauXC's XCIntegrator dispatch, SkalaHostDriver owns the
 * reusable GauXC components (LoadBalancer + LocalWorkDriver) and drives their
 * collocation / xmat / uvvar primitives directly.
 *
 * Per-task ML data lives in parallel input and output vectors (GauXC master's
 * XCTask has no `feat` member and is never modified).
 *
 * Internal header: uses GauXC types freely and must never be included by any
 * SkalaXC public header (ABI isolation). LibTorch stays confined to the .cxx.
 */
#include <gauxc/basisset.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/molgrid.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <skalaxc/skalaxc.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include "eigen_types.hpp"
#include "skala_driver.hpp"
#include "task_data.hpp"

namespace SkalaXC {

class ModelGridExchange;
class SkalaModel;

/**
 * @brief Standalone host driver for SkalaXC ML exchange-correlation
 * functionals.
 */
class SkalaHostDriver final : public SkalaDriver {
 public:
  /**
   * @brief Construct a host driver from a weight-partitioned load balancer.
   * @param weighted_lb LoadBalancer whose tasks are already sorted and
   *        weight-partitioned (see MolecularWeights::modify_weights).
   * @param raw_weights Per-task pre-partition quadrature weights, aligned to
   *        the balancer's (sorted) task order.
   * @param model ML model selector ("LDA"/"PBE"/"TPSS") or a .fun path.
   * @param timing_settings Diagnostics timing and logging settings.
   * @param domain_batch_mode Exact-size complete-domain batching policy.
   *
   * The runtime environment, molecule, and basis are taken from
   * @p weighted_lb; the driver builds only its host LocalWorkDriver and ML
   * state. It does not partition weights (that is done up front by
   * MolecularWeights::modify_weights).
   */
  SkalaHostDriver(
      const GauXC::LoadBalancer& weighted_lb,
      const std::vector<std::vector<double>>& raw_weights,
      const std::string& model, TimingSettings timing_settings = {},
      DomainBatchMode domain_batch_mode = DomainBatchMode::Conservative);
  ~SkalaHostDriver() noexcept override;

  SkalaHostDriver(const SkalaHostDriver&) = delete;
  SkalaHostDriver& operator=(const SkalaHostDriver&) = delete;

  /**
   * @brief Build a host driver directly from a system description.
   * @param rt Runtime environment.
   * @param mol Molecule.
   * @param mg Molecular integration grid.
   * @param basis AO basis set.
   * @param model ML model selector ("LDA"/"PBE"/"TPSS") or a .fun path.
   * @param domain_batch_mode Exact-size complete-domain batching policy.
   * @param timing_settings Diagnostics timing and logging settings.
   * @return A ready-to-use, weight-partitioned host driver.
   *
   * Convenience for white-box tests: constructs the GauXC load balancer, sorts
   * and SSF-partitions its tasks while snapshotting the pre-partition ("raw")
   * quadrature weights, and returns the driver. This mirrors the public SkalaXC
   * pipeline (LoadBalancerFactory then MolecularWeights::modify_weights) in one
   * call so tests need no GauXC internal headers.
   */
  static SkalaHostDriver from_system(
      const GauXC::RuntimeEnvironment& rt, const GauXC::Molecule& mol,
      const GauXC::MolGrid& mg, const GauXC::BasisSet<double>& basis,
      const std::string& model,
      DomainBatchMode domain_batch_mode = DomainBatchMode::Conservative,
      TimingSettings timing_settings = {});

  /**
   * @brief Evaluate UKS ML exchange-correlation energy and potential.
   * @param scalar_density Scalar-spin density matrix.
   * @param spin_density Z-spin density matrix.
   * @param scalar_potential Output scalar XC potential matrix.
   * @param spin_potential Output z XC potential matrix.
   * @return Exchange-correlation energy EXC.
   */
  double eval_exc_vxc_uks(ConstColMajorMatrixMap scalar_density,
                          ConstColMajorMatrixMap spin_density,
                          ColMajorMatrixMap scalar_potential,
                          ColMajorMatrixMap spin_potential) override;

  /**
   * @brief Evaluate the UKS ML exchange-correlation nuclear gradient.
   * @param scalar_density Scalar-spin density matrix.
   * @param spin_density Z-spin density matrix.
   * @param gradient Output atom-major Cartesian gradient.
   */
  void eval_exc_grad_uks(ConstColMajorMatrixMap scalar_density,
                         ConstColMajorMatrixMap spin_density,
                         RowMajorMatrixMap gradient) override;

 private:
  std::unique_ptr<SkalaModel> model_;
  GauXC::LoadBalancer lb_;
  std::unique_ptr<GauXC::LocalWorkDriver> lwd_;
  std::vector<TaskFeatureData> task_features_;
  std::vector<TaskPotentialData> task_potentials_;
  std::vector<std::vector<double>> raw_weights_;  ///< Pre-partition quadrature
                                                  ///< weights per task.
  std::unique_ptr<ModelGridExchange> model_grid_exchange_;

  /** @brief Prepare per-task local data before ML evaluation. */
  void pre_skala_local_work_(const GauXC::BasisSet<double>& basis,
                             ConstColMajorMatrixMap scalar_density,
                             ConstColMajorMatrixMap spin_density,
                             double& electron_count, bool is_gga, bool is_mgga,
                             bool needs_laplacian);

  /** @brief Accumulate ML potentials into AO matrices after ML evaluation. */
  void post_skala_local_work_(const GauXC::BasisSet<double>& basis,
                              ColMajorMatrixMap scalar_potential,
                              ColMajorMatrixMap spin_potential, bool is_gga,
                              bool is_mgga, bool needs_laplacian);

  /** @brief Accumulate local Pulay and grid-weight derivative terms. */
  void exc_grad_local_work_(ConstColMajorMatrixMap scalar_density,
                            ConstColMajorMatrixMap spin_density,
                            RowMajorMatrixMap gradient, bool is_gga,
                            bool is_mgga);
};

}  // namespace SkalaXC

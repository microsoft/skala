/**
 * SkalaXC host ML backend implementation.
 *
 * Ported from GauXC/skala reference_replicated_xc_host_integrator_onedft.hpp
 * (onedft -> skala) and restructured from integrator methods into a standalone
 * driver that OWNS the reusable GauXC components and drives their primitives
 * directly. Per-task ML data lives in parallel input and output vectors;
 * GauXC master's XCTask is never modified, and the LoadBalancer's task list
 * is never reordered (atom ordering is applied through an index permutation).
 */
#include "skala_host_driver.hpp"
#include "component_matrix_map.hpp"
#include "exceptions.hpp"
#include "model_grid_exchange.hpp"
#include "mpi_wrapper.hpp"
#include "skala_model.hpp"
#include "skala_util.hpp"

// GauXC reusable internals (reachable via the in-tree `gauxc` target's
// PUBLIC BUILD_INTERFACE include of ${GauXC}/src).
#include "xc_integrator/integrator_util/integrator_common.hpp"
#include "xc_integrator/local_work_driver/host/local_host_work_driver.hpp"
#include "xc_integrator/replicated/host/xc_host_data.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <gauxc/basisset_map.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/util/mpi.hpp>
#include <stdexcept>
#include <string>

namespace SkalaXC {

namespace {

class GauXCTaskAdapter {
 public:
  GauXCTaskAdapter(GauXC::LocalHostWorkDriver& driver,
                   const GauXC::XCTask& task,
                   const GauXC::BasisSet<double>& basis)
      : driver_(driver),
        task_(task),
        basis_(basis),
        npts_(task.points.size()),
        nbe_(task.bfn_screening.nbe),
        nshells_(task.bfn_screening.shell_list.size()) {}

  void eval_collocation(ComponentMatrixMap& basis_components,
                        bool needs_gradient) const {
    if (needs_gradient) {
      driver_.eval_collocation_gradient(npts_, nshells_, nbe_,
                                        task_.points.data()->data(), basis_,
                                        task_.bfn_screening.shell_list.data(),
                                        basis_components.component_data(0),
                                        basis_components.component_data(1),
                                        basis_components.component_data(2),
                                        basis_components.component_data(3));
    } else {
      driver_.eval_collocation(npts_, nshells_, nbe_,
                               task_.points.data()->data(), basis_,
                               task_.bfn_screening.shell_list.data(),
                               basis_components.component_data(0));
    }
  }

  void eval_collocation_hessian(ComponentMatrixMap& basis_components) const {
    driver_.eval_collocation_hessian(
        npts_, nshells_, nbe_, task_.points.data()->data(), basis_,
        task_.bfn_screening.shell_list.data(),
        basis_components.component_data(0), basis_components.component_data(1),
        basis_components.component_data(2), basis_components.component_data(3),
        basis_components.component_data(4), basis_components.component_data(5),
        basis_components.component_data(6), basis_components.component_data(7),
        basis_components.component_data(8), basis_components.component_data(9));
  }

  void eval_xmat(std::size_t point_components, std::size_t nbf,
                 const GauXC::LocalHostWorkDriver::submat_map_t& submat_map,
                 ConstColMajorMatrixMap density,
                 const ComponentMatrixMap& basis_components,
                 ComponentMatrixMap& x_components, Eigen::Index x_component,
                 std::vector<double>& scratch) const {
    driver_.eval_xmat(
        point_components * npts_, nbf, nbe_, submat_map, 1.0, density.data(),
        density.outerStride(), basis_components.component_data(0), nbe_,
        x_components.component_data(x_component), nbe_, scratch.data());
  }

  void eval_lda_model_features_uks(const ComponentMatrixMap& basis_components,
                                   const ComponentMatrixMap& x_components,
                                   Eigen::Index spin_component,
                                   AlphaBetaMatrix& alpha_beta_density) const {
    driver_.eval_uvvar_lda_uks(npts_, nbe_, basis_components.component_data(0),
                               x_components.component_data(0), nbe_,
                               x_components.component_data(spin_component),
                               nbe_, alpha_beta_density.data());
  }

  void eval_gga_model_features_uks(const ComponentMatrixMap& basis_components,
                                   const ComponentMatrixMap& x_components,
                                   Eigen::Index spin_component,
                                   AlphaBetaMatrix& alpha_beta_density,
                                   SpinGradient& alpha_beta_density_gradient,
                                   std::vector<double>& gamma,
                                   ScalarZGradient& scalar_z_scratch) const {
    scalar_z_scratch.resize(npts_);
    driver_.eval_uvvar_gga_uks(
        npts_, nbe_, basis_components.component_data(0),
        basis_components.component_data(1), basis_components.component_data(2),
        basis_components.component_data(3), x_components.component_data(0),
        nbe_, x_components.component_data(spin_component), nbe_,
        alpha_beta_density.data(), scalar_z_scratch.direction_data(X),
        scalar_z_scratch.direction_data(Y), scalar_z_scratch.direction_data(Z),
        gamma.data());
    convert_scalar_z_to_alpha_beta(scalar_z_scratch,
                                   alpha_beta_density_gradient);
  }

  void eval_mgga_model_features_uks(const ComponentMatrixMap& basis_components,
                                    const ComponentMatrixMap& x_components,
                                    Eigen::Index spin_component,
                                    AlphaBetaMatrix& alpha_beta_density,
                                    SpinGradient& alpha_beta_density_gradient,
                                    std::vector<double>& gamma,
                                    AlphaBetaMatrix& alpha_beta_kinetic,
                                    std::vector<double>& laplacian,
                                    ScalarZGradient& scalar_z_scratch) const {
    scalar_z_scratch.resize(npts_);
    driver_.eval_uvvar_mgga_uks(
        npts_, nbe_, basis_components.component_data(0),
        basis_components.component_data(1), basis_components.component_data(2),
        basis_components.component_data(3), nullptr,
        x_components.component_data(0), nbe_,
        x_components.component_data(spin_component), nbe_,
        x_components.component_data(1), x_components.component_data(2),
        x_components.component_data(3), nbe_,
        x_components.component_data(spin_component + 1),
        x_components.component_data(spin_component + 2),
        x_components.component_data(spin_component + 3), nbe_,
        alpha_beta_density.data(), scalar_z_scratch.direction_data(X),
        scalar_z_scratch.direction_data(Y), scalar_z_scratch.direction_data(Z),
        gamma.data(), alpha_beta_kinetic.data(), laplacian.data());
    convert_scalar_z_to_alpha_beta(scalar_z_scratch,
                                   alpha_beta_density_gradient);
  }

  void eval_zmat_lda_vxc_uks(const AlphaBetaMatrix& density_potential,
                             const ComponentMatrixMap& basis_components,
                             ComponentMatrixMap& zmat_components,
                             Eigen::Index spin_component) const {
    driver_.eval_zmat_lda_vxc_uks(
        npts_, nbe_, density_potential.data(),
        basis_components.component_data(0), zmat_components.component_data(0),
        nbe_, zmat_components.component_data(spin_component), nbe_);
  }

  void eval_mmat_mgga_vxc_uks(const AlphaBetaMatrix& kinetic_potential,
                              const ComponentMatrixMap& basis_components,
                              ComponentMatrixMap& zmat_components,
                              Eigen::Index spin_component) const {
    driver_.eval_mmat_mgga_vxc_uks(
        npts_, nbe_, kinetic_potential.data(), nullptr,
        basis_components.component_data(1), basis_components.component_data(2),
        basis_components.component_data(3), zmat_components.component_data(1),
        zmat_components.component_data(2), zmat_components.component_data(3),
        nbe_, zmat_components.component_data(spin_component + 1),
        zmat_components.component_data(spin_component + 2),
        zmat_components.component_data(spin_component + 3), nbe_);
  }

  void inc_vxc(std::size_t point_components, std::size_t nbf,
               const ComponentMatrixMap& basis_components,
               const GauXC::LocalHostWorkDriver::submat_map_t& submat_map,
               const ComponentMatrixMap& zmat_components,
               Eigen::Index zmat_component, ColMajorMatrixMap potential,
               std::vector<double>& scratch) const {
    driver_.inc_vxc(point_components * npts_, nbf, nbe_,
                    basis_components.component_data(0), submat_map,
                    zmat_components.component_data(zmat_component), nbe_,
                    potential.data(), potential.outerStride(), scratch.data());
  }

  void eval_weight_1st_deriv_contracted(
      GauXC::XCWeightAlg weight_alg, const GauXC::Molecule& molecule,
      const GauXC::MolMeta& molecule_metadata,
      const std::vector<double>& weighted_dE_dw,
      RowMajorMatrixMap gradient) const {
    driver_.eval_weight_1st_deriv_contracted(
        weight_alg, molecule, molecule_metadata, task_, weighted_dE_dw.data(),
        gradient.data());
  }

 private:
  GauXC::LocalHostWorkDriver& driver_;
  const GauXC::XCTask& task_;
  const GauXC::BasisSet<double>& basis_;
  std::size_t npts_;
  std::size_t nbe_;
  std::size_t nshells_;
};

// ---------------------------------------------------------------------------
// File-local helpers (ported from the branch free functions; adapted to read
// per-task ML data from parallel storage rather than a mutated XCTask list).
// ---------------------------------------------------------------------------

void validate_zmat_inputs(const AlphaBetaMatrix& density_potential,
                          const SpinGradient& gradient_potential,
                          const ComponentMatrixMap& basis_components,
                          const ComponentMatrixMap& zmat_components,
                          Eigen::Index spin_component) {
  if (basis_components.components() < 4 || spin_component < 0 ||
      spin_component >= zmat_components.components() ||
      zmat_components.rows() != basis_components.rows() ||
      zmat_components.points() != basis_components.points())
    SKALAXC_EXCEPTION("Invalid Z-matrix dimensions");

  const auto point_count = basis_components.points();
  if (density_potential.rows() != point_count ||
      density_potential.cols() != spin_dimension ||
      gradient_potential.points() != point_count)
    SKALAXC_EXCEPTION("Invalid Z-matrix potential dimensions");
}

void eval_zmat_gga_vxc_uks(const AlphaBetaMatrix& density_potential,
                           const SpinGradient& gradient_potential,
                           const ComponentMatrixMap& basis_components,
                           ComponentMatrixMap& zmat_components,
                           Eigen::Index spin_component) {
  validate_zmat_inputs(density_potential, gradient_potential, basis_components,
                       zmat_components, spin_component);

  const auto density_scalar_z = alpha_beta_to_scalar_z(density_potential);
  const auto basis_value = basis_components.component(0);
  auto zmat_scalar = zmat_components.component(0);
  auto zmat_spin = zmat_components.component(spin_component);
  zmat_scalar.array() =
      basis_value.array().rowwise() *
      (0.5 * density_scalar_z.col(PauliChannel::Scalar)).transpose().array();
  zmat_spin.array() =
      basis_value.array().rowwise() *
      (0.5 * density_scalar_z.col(PauliChannel::SpinZ)).transpose().array();

  for (Eigen::Index direction = 0; direction < direction_dimension;
       ++direction) {
    const auto potential =
        gradient_potential.direction(static_cast<Direction>(direction));
    const auto gradient_scalar_z = alpha_beta_to_scalar_z(potential);
    const auto basis_derivative = basis_components.component(direction + 1);
    zmat_scalar.array() +=
        basis_derivative.array().rowwise() *
        gradient_scalar_z.col(PauliChannel::Scalar).transpose().array();
    zmat_spin.array() +=
        basis_derivative.array().rowwise() *
        gradient_scalar_z.col(PauliChannel::SpinZ).transpose().array();
  }
}

}  // anonymous namespace

// ===========================================================================
// SkalaHostDriver
// ===========================================================================

SkalaHostDriver::SkalaHostDriver(
    const GauXC::LoadBalancer& weighted_lb,
    const std::vector<std::vector<double>>& raw_weights,
    const std::string& model, TimingSettings timing_settings,
    DomainBatchMode domain_batch_mode)
    : SkalaDriver(timing_settings, ExecutionSpace::Host,
                  types::CommunicatorRank{weighted_lb.runtime().comm_rank()},
                  types::CommunicatorSize{weighted_lb.runtime().comm_size()}),
      lb_(weighted_lb),
      lwd_(GauXC::LocalWorkDriverFactory::make_local_work_driver(
          GauXC::ExecutionSpace::Host, "Default")),
      raw_weights_(raw_weights) {

  {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelLoad);
    model_ = std::make_unique<SkalaModel>(model, weighted_lb.runtime());
  }

  // The balancer's tasks are already sorted and weight-partitioned (by
  // MolecularWeights::modify_weights); raw_weights_ holds the pre-partition
  // quadrature weights aligned to that sorted task order. The driver only
  // builds its host LocalWorkDriver, ML model, and per-task feature storage.
  if (not lb_.state().modified_weights_are_stored)
    SKALAXC_EXCEPTION(
        "SkalaHostDriver requires weight-partitioned tasks; call "
        "MolecularWeights::modify_weights first");

  auto& tasks = lb_.get_tasks();
  const auto& rt = lb_.runtime();
  model_grid_exchange_ = std::make_unique<ModelGridExchange>(
      tasks,
      types::AtomCount{static_cast<std::uint64_t>(lb_.molecule().natoms())}, rt,
      domain_batch_mode);
  const auto& local_batches = model_grid_exchange_->local_batches();
  set_setup_diagnostics(types::CommunicatorSize{rt.comm_size()},
                        types::DeviceId{-1}, 0.0, domain_batch_mode, tasks,
                        local_batches);

  const bool needs_gradient = model_->is_gga() || model_->is_mgga();
  task_features_.resize(tasks.size());
  task_potentials_.resize(tasks.size());
  for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
    const Eigen::Index point_count = tasks[task_index].points.size();
    auto& features = task_features_[task_index];
    features.density.resize(point_count, spin_dimension);
    features.density_gradient.resize(needs_gradient ? point_count : 0);
    features.kinetic.resize(model_->is_mgga() ? point_count : 0,
                            spin_dimension);

    auto& potentials = task_potentials_[task_index];
    potentials.density.resize(point_count, spin_dimension);
    potentials.density_gradient.resize(needs_gradient ? point_count : 0);
    potentials.kinetic.resize(model_->is_mgga() ? point_count : 0,
                              spin_dimension);
    potentials.dE_dw.resize(point_count);
  }
  log_setup(model, model_->feature_keys(), model_->is_gga(), model_->is_mgga(),
            local_batches);
}

SkalaHostDriver::~SkalaHostDriver() noexcept = default;

SkalaHostDriver SkalaHostDriver::from_system(
    const GauXC::RuntimeEnvironment& rt, const GauXC::Molecule& mol,
    const GauXC::MolGrid& mg, const GauXC::BasisSet<double>& basis,
    const std::string& model, DomainBatchMode domain_batch_mode,
    TimingSettings timing_settings) {
  // Mirror the public SkalaXC pipeline (LoadBalancerFactory ->
  // MolecularWeights::modify_weights): build the load balancer, then sort and
  // weight-partition its tasks while snapshotting the pre-partition ("raw")
  // quadrature weights that Skala ML models consume.
  GauXC::LoadBalancerFactory lb_factory(GauXC::ExecutionSpace::Host, "Default");
  GauXC::LoadBalancer lb = lb_factory.get_instance(rt, mol, mg, basis);

  auto lwd = GauXC::LocalWorkDriverFactory::make_local_work_driver(
      GauXC::ExecutionSpace::Host, "Default");
  auto* host_lwd = dynamic_cast<GauXC::LocalHostWorkDriver*>(lwd.get());
  if (!host_lwd) SKALAXC_EXCEPTION("Expected a LocalHostWorkDriver");

  auto& tasks = lb.get_tasks();
  std::stable_sort(tasks.begin(), tasks.end(),
                   [](const GauXC::XCTask& a, const GauXC::XCTask& b) {
                     return (a.points.size() * a.bfn_screening.nbe) >
                            (b.points.size() * b.bfn_screening.nbe);
                   });

  std::vector<std::vector<double>> raw_weights(tasks.size());
  for (std::size_t i = 0; i < tasks.size(); ++i)
    raw_weights[i] = tasks[i].weights;

  const GauXC::XCWeightAlg weight_alg = GauXC::XCWeightAlg::SSF;
  host_lwd->partition_weights(weight_alg, lb.molecule(), lb.molmeta(),
                              tasks.begin(), tasks.end());
  lb.state().modified_weights_are_stored = true;
  lb.state().weight_alg = weight_alg;

  return SkalaHostDriver(lb, raw_weights, model, timing_settings,
                         domain_batch_mode);
}

double SkalaHostDriver::eval_exc_vxc_uks(ConstColMajorMatrixMap scalar_density,
                                         ConstColMajorMatrixMap spin_density,
                                         ColMajorMatrixMap scalar_potential,
                                         ColMajorMatrixMap spin_potential) {
  const auto& basis = lb_.basis();
  const Eigen::Index basis_size = basis.nbf();
  const auto valid_ao_matrix = [basis_size](const auto& matrix) {
    return matrix.rows() == basis_size && matrix.cols() == basis_size &&
           matrix.innerStride() == 1 && matrix.outerStride() == basis_size;
  };
  if (!valid_ao_matrix(scalar_density) || !valid_ao_matrix(spin_density) ||
      !valid_ao_matrix(scalar_potential) || !valid_ao_matrix(spin_potential))
    SKALAXC_EXCEPTION(
        "UKS density and potential matrices must be dense nbf x nbf "
        "column-major views");

  const auto diagnostics_before =
      log_evaluation_start("exc_vxc", scalar_density, spin_density);
  detail::HostTimingScope total_timer(diagnostics_, TimingMetric::TotalEXCVXC);
  diagnostics_.increment_exc_vxc_calls();

  auto& tasks = lb_.get_tasks();
  auto rt = lb_.runtime();
  double N_EL = 0.0;

  const auto& feature_keys = model_->feature_keys();
  const bool is_gga = model_->is_gga();
  const bool is_mgga = model_->is_mgga();

  // Local features: collocation -> xmat -> uvvar.
  {
    detail::HostTimingScope timer(diagnostics_,
                                  TimingMetric::FeatureConstruction);
    pre_skala_local_work_(basis, scalar_density, spin_density, N_EL, is_gga,
                          is_mgga, false);
  }

  double EXC = 0.0;
  for (const auto& batch : model_grid_exchange_->local_batches()) {
    FeatureDict features_dict;
    {
      detail::HostTimingScope timer(diagnostics_,
                                    TimingMetric::ModelBatchPacking);
      features_dict = model_grid_exchange_->prepare_local_features(
          batch, tasks, task_features_, raw_weights_, lb_.molecule(),
          feature_keys);
    }
    diagnostics_.record_model_batch(types::DomainCount{batch.atoms.size()});
    at::Tensor exc;
    {
      detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelForward);
      exc = evaluate_model_energy(*model_, features_dict,
                                  c10::Device(c10::DeviceType::CPU));
      validate_model_tensor_finite(exc, "host model energy");
    }
    {
      detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelBackward);
      exc.backward();
    }
    EXC += exc.item().to<double>();
    {
      detail::HostTimingScope timer(diagnostics_,
                                    TimingMetric::PotentialMapping);
      model_grid_exchange_->distribute_local_potentials(
          batch, is_gga || is_mgga, is_mgga, features_dict, task_potentials_);
    }
  }

  {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::AOAssembly);
    post_skala_local_work_(basis, scalar_potential, spin_potential, is_gga,
                           is_mgga, false);
  }

#ifdef GAUXC_HAS_MPI
  if (rt.comm_size() > 1) {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::MPIReduction);
    SkalaXC::mpi::allreduce_sum(scalar_potential, rt);
    SkalaXC::mpi::allreduce_sum(spin_potential, rt);
    MPI_Allreduce(MPI_IN_PLACE, &EXC, 1, MPI_DOUBLE, MPI_SUM, rt.comm());
    MPI_Allreduce(MPI_IN_PLACE, &N_EL, 1, MPI_DOUBLE, MPI_SUM, rt.comm());
  }
#endif
  (void)N_EL;
  total_timer.finish();
  log_exc_vxc_result("exc_vxc", EXC, scalar_potential, spin_potential);
  log_host_timing_delta("exc_vxc", diagnostics_before);
  return EXC;
}

void SkalaHostDriver::eval_exc_grad_uks(ConstColMajorMatrixMap scalar_density,
                                        ConstColMajorMatrixMap spin_density,
                                        RowMajorMatrixMap gradient) {
  const auto& basis = lb_.basis();
  auto& tasks = lb_.get_tasks();
  const auto rt = lb_.runtime();
  const int natoms = lb_.molecule().natoms();
  const Eigen::Index basis_size = basis.nbf();
  const Eigen::Index atom_count = natoms;
  const auto valid_density = [basis_size](const auto& density) {
    return density.rows() == basis_size && density.cols() == basis_size &&
           density.innerStride() == 1 && density.outerStride() == basis_size;
  };
  if (!valid_density(scalar_density) || !valid_density(spin_density) ||
      gradient.rows() != atom_count || gradient.cols() != 3 ||
      gradient.innerStride() != 1 || gradient.outerStride() != 3)
    SKALAXC_EXCEPTION("Invalid density matrix or atom-major gradient view");

  const auto diagnostics_before =
      log_evaluation_start("exc_gradient", scalar_density, spin_density);
  detail::HostTimingScope total_timer(diagnostics_,
                                      TimingMetric::TotalEXCGradient);
  diagnostics_.increment_exc_gradient_calls();

  const auto& feature_keys = model_->feature_keys();
  const bool is_gga = model_->is_gga();
  const bool is_mgga = model_->is_mgga();

  double N_EL = 0.0;
  {
    detail::HostTimingScope timer(diagnostics_,
                                  TimingMetric::FeatureConstruction);
    pre_skala_local_work_(basis, scalar_density, spin_density, N_EL, is_gga,
                          is_mgga, false);
  }

  gradient.setZero();
  for (const auto& batch : model_grid_exchange_->local_batches()) {
    FeatureDict features_dict;
    {
      detail::HostTimingScope timer(diagnostics_,
                                    TimingMetric::ModelBatchPacking);
      features_dict = model_grid_exchange_->prepare_local_features(
          batch, tasks, task_features_, raw_weights_, lb_.molecule(),
          feature_keys);
    }
    diagnostics_.record_model_batch(types::DomainCount{batch.atoms.size()});
    const auto points_key = feat_map().at(SKALA_FEATURE::POINTS);
    const auto coords_key = feat_map().at(SKALA_FEATURE::COORDS);
    const auto weights_key = feat_map().at(SKALA_FEATURE::WEIGHTS);
    if (features_dict.find(points_key) != features_dict.end())
      features_dict.at(points_key).requires_grad_(true);
    if (features_dict.find(coords_key) != features_dict.end())
      features_dict.at(coords_key).requires_grad_(true);
    features_dict.at(weights_key).requires_grad_(true);

    at::Tensor exc;
    {
      detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelForward);
      exc = evaluate_model_energy(*model_, features_dict,
                                  c10::Device(c10::DeviceType::CPU));
      validate_model_tensor_finite(exc, "host model energy");
    }
    {
      detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelBackward);
      exc.backward();
    }

    auto dE_dw_cpu = validated_model_gradient(features_dict.at(weights_key),
                                              "host model dE/dw");
    validate_model_tensor_finite(dE_dw_cpu, "host model dE/dw");
    std::vector<double> dE_dw_values(
        static_cast<std::size_t>(batch.point_count.raw()));
    std::memcpy(dE_dw_values.data(), dE_dw_cpu.data_ptr<double>(),
                dE_dw_values.size() * sizeof(double));

    {
      detail::HostTimingScope timer(diagnostics_,
                                    TimingMetric::GradientAssembly);
      if (features_dict.find(points_key) != features_dict.end()) {
        auto point_grad = features_dict.at(points_key).grad();
        model_grid_exchange_->accumulate_local_point_gradient(batch, point_grad,
                                                              gradient);
      }

      if (features_dict.find(coords_key) != features_dict.end()) {
        auto coords_grad = features_dict.at(coords_key).grad();
        model_grid_exchange_->accumulate_local_coordinate_gradient(
            batch, coords_grad, gradient);
      }
    }
    {
      detail::HostTimingScope timer(diagnostics_,
                                    TimingMetric::PotentialMapping);
      model_grid_exchange_->distribute_local_potentials(
          batch, is_gga || is_mgga, is_mgga, features_dict, task_potentials_);
      model_grid_exchange_->distribute_local_dE_dw(
          batch, std::move(dE_dw_values), task_potentials_);
    }
  }

  {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::GradientAssembly);
    exc_grad_local_work_(scalar_density, spin_density, gradient, is_gga,
                         is_mgga);
  }

#ifdef GAUXC_HAS_MPI
  if (rt.comm_size() > 1) {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::MPIReduction);
    SkalaXC::mpi::allreduce_sum(gradient, rt);
  }
#endif
  (void)N_EL;
  total_timer.finish();
  log_gradient_result("exc_gradient", gradient);
  log_host_timing_delta("exc_gradient", diagnostics_before);
}

void SkalaHostDriver::exc_grad_local_work_(
    ConstColMajorMatrixMap scalar_density, ConstColMajorMatrixMap spin_density,
    RowMajorMatrixMap gradient, bool is_gga, bool is_mgga) {
  auto* lwd = dynamic_cast<GauXC::LocalHostWorkDriver*>(lwd_.get());
  if (!lwd) SKALAXC_EXCEPTION("Expected a LocalHostWorkDriver");

  const auto& basis = lb_.basis();
  const auto& mol = lb_.molecule();
  const auto& molmeta = lb_.molmeta();
  auto& lb_state = lb_.state();
  if (!lb_state.modified_weights_are_stored)
    SKALAXC_EXCEPTION("Weights Have Not Been Modified");
  const GauXC::XCWeightAlg weight_alg = lb_state.weight_alg;

  GauXC::BasisSetMap basis_map(basis, mol);
  const int32_t nbf = basis.nbf();
  const auto& tasks = lb_.get_tasks();
  const size_t ntasks = tasks.size();
  constexpr std::array<std::array<Eigen::Index, 3>, 3> hessian_components{
      {{{4, 5, 6}}, {{5, 7, 8}}, {{6, 8, 9}}}};

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    GauXC::XCHostData<double> host_data;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t iT = 0; iT < ntasks; ++iT) {
      const auto& task = tasks[iT];
      const auto& potentials = task_potentials_[iT];
      const GauXCTaskAdapter task_work(*lwd, task, basis);
      const int32_t npts = task.points.size();
      const int32_t nbe = task.bfn_screening.nbe;
      const int32_t nshells = task.bfn_screening.shell_list.size();

      host_data.basis_eval.resize((is_gga || is_mgga ? 10 : 4) * npts * nbe);
      host_data.zmat.resize((is_gga || is_mgga ? 8 : 2) * npts * nbe);
      host_data.nbe_scr.resize(nbe * nbe);
      host_data.eps.resize(npts);

      const Eigen::Index basis_component_count = (is_gga || is_mgga) ? 10 : 4;
      ComponentMatrixMap basis_components(host_data.basis_eval.data(),
                                          basis_component_count, nbe, npts);

      const Eigen::Index zmat_component_count = (is_gga || is_mgga) ? 8 : 2;
      ComponentMatrixMap zmat_components(host_data.zmat.data(),
                                         zmat_component_count, nbe, npts);
      const Eigen::Index spin_zmat_component = (is_gga || is_mgga) ? 4 : 1;
      std::vector<std::array<int32_t, 3>> submat_map;
      std::tie(submat_map, std::ignore) = GauXC::gen_compressed_submat_map(
          basis_map, task.bfn_screening.shell_list, nbf, nbf);
      if (is_gga || is_mgga) {
        task_work.eval_collocation_hessian(basis_components);
      } else {
        task_work.eval_collocation(basis_components, true);
      }

      const int xmat_len = (is_gga || is_mgga) ? 4 : 1;
      task_work.eval_xmat(xmat_len, nbf, submat_map, scalar_density,
                          basis_components, zmat_components, 0,
                          host_data.nbe_scr);
      task_work.eval_xmat(xmat_len, nbf, submat_map, spin_density,
                          basis_components, zmat_components,
                          spin_zmat_component, host_data.nbe_scr);

      // GauXC's contracted partition derivative expects w_i * f_i. The model
      // boundary cotangent is f_i = dE/dw_i.
      VectorMap(host_data.eps.data(), npts) =
          potentials.dE_dw.array() *
          ConstVectorMap(task.weights.data(), npts).array();
      task_work.eval_weight_1st_deriv_contracted(weight_alg, mol, molmeta,
                                                 host_data.eps, gradient);

      const auto density_scalar_z = alpha_beta_to_scalar_z(potentials.density);

      CartesianMatrix gradient_scalar;
      CartesianMatrix gradient_spin;
      if (is_gga || is_mgga) {
        gradient_scalar.resize(npts, direction_dimension);
        gradient_spin.resize(npts, direction_dimension);
        for (Eigen::Index direction = 0; direction < direction_dimension;
             ++direction) {
          const auto potential = potentials.density_gradient.direction(
              static_cast<Direction>(direction));
          const auto gradient_scalar_z = alpha_beta_to_scalar_z(potential);
          gradient_scalar.col(direction) =
              gradient_scalar_z.col(PauliChannel::Scalar);
          gradient_spin.col(direction) =
              gradient_scalar_z.col(PauliChannel::SpinZ);
        }
      }

      ScalarZChannels kinetic_scalar_z;
      if (is_mgga)
        kinetic_scalar_z = alpha_beta_to_scalar_z(potentials.kinetic);

      const auto contract = [](const auto& left, const auto& right,
                               const auto& point_potential) {
        return ((left.array().rowwise() * point_potential.transpose().array()) *
                right.array())
            .sum();
      };

      const auto xN_component = zmat_components.component(0);
      const auto xZ_component = zmat_components.component(spin_zmat_component);
      Eigen::Index basis_offset = 0;
      for (int32_t ish = 0; ish < nshells; ++ish) {
        const int sh_idx = task.bfn_screening.shell_list[ish];
        const Eigen::Index shell_size = basis[sh_idx].size();
        const int iAt = basis_map.shell_to_center(sh_idx);
        if (iAt == task.iParent) {
          basis_offset += shell_size;
          continue;
        }

        const auto xN = xN_component.middleRows(basis_offset, shell_size);
        const auto xZ = xZ_component.middleRows(basis_offset, shell_size);
        Eigen::Vector3d shell_gradient = Eigen::Vector3d::Zero();
        for (Eigen::Index force = 0; force < direction_dimension; ++force) {
          const auto basis_derivative =
              basis_components.component(force + 1).middleRows(basis_offset,
                                                               shell_size);
          shell_gradient(force) =
              contract(xN, basis_derivative,
                       density_scalar_z.col(PauliChannel::Scalar)) +
              contract(xZ, basis_derivative,
                       density_scalar_z.col(PauliChannel::SpinZ));

          if (is_gga || is_mgga) {
            for (Eigen::Index response = 0; response < direction_dimension;
                 ++response) {
              const auto basis_hessian =
                  basis_components
                      .component(hessian_components[force][response])
                      .middleRows(basis_offset, shell_size);
              const auto xN_derivative =
                  zmat_components.component(response + 1)
                      .middleRows(basis_offset, shell_size);
              const auto xZ_derivative =
                  zmat_components.component(response + 5)
                      .middleRows(basis_offset, shell_size);
              shell_gradient(force) +=
                  contract(basis_hessian, xN, gradient_scalar.col(response)) +
                  contract(basis_derivative, xN_derivative,
                           gradient_scalar.col(response)) +
                  contract(basis_hessian, xZ, gradient_spin.col(response)) +
                  contract(basis_derivative, xZ_derivative,
                           gradient_spin.col(response));

              if (is_mgga)
                shell_gradient(force) +=
                    0.5 *
                    (contract(basis_hessian, xN_derivative,
                              kinetic_scalar_z.col(PauliChannel::Scalar)) +
                     contract(basis_hessian, xZ_derivative,
                              kinetic_scalar_z.col(PauliChannel::SpinZ)));
            }
          }
        }

        for (Eigen::Index direction = 0; direction < direction_dimension;
             ++direction) {
          const double contribution = -2.0 * shell_gradient(direction);
#ifdef _OPENMP
#pragma omp atomic
#endif
          gradient(iAt, direction) += contribution;
#ifdef _OPENMP
#pragma omp atomic
#endif
          gradient(task.iParent, direction) -= contribution;
        }
        basis_offset += shell_size;
      }
    }
  }
}

void SkalaHostDriver::pre_skala_local_work_(
    const GauXC::BasisSet<double>& basis, ConstColMajorMatrixMap scalar_density,
    ConstColMajorMatrixMap spin_density, double& electron_count, bool is_gga,
    bool is_mgga, bool /*needs_laplacian*/) {

  const bool needs_gradient = is_gga || is_mgga;
  auto* lwd = dynamic_cast<GauXC::LocalHostWorkDriver*>(lwd_.get());
  if (!lwd) SKALAXC_EXCEPTION("Expected a LocalHostWorkDriver");
  const auto& mol = lb_.molecule();
  GauXC::BasisSetMap basis_map(basis, mol);
  const int32_t nbf = basis.nbf();

  auto& tasks = lb_.get_tasks();
  const size_t ntasks = tasks.size();

  auto& lb_state = lb_.state();
  if (not lb_state.modified_weights_are_stored)
    SKALAXC_EXCEPTION("Weights Have Not Been Modified");

  double NEL_WORK = 0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+ : NEL_WORK)
#endif
  {
    GauXC::XCHostData<double> host_data;
    ScalarZGradient scalar_z_gradient;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t iT = 0; iT < ntasks; ++iT) {
      auto& task = tasks[iT];
      auto& features = task_features_[iT];
      const GauXCTaskAdapter task_work(*lwd, task, basis);

      const int32_t npts = task.points.size();
      const int32_t nbe = task.bfn_screening.nbe;
      const int32_t mgga_component_count = is_mgga ? 4 : 1;
      const int32_t basis_component_count = needs_gradient ? 4 : 1;

      host_data.nbe_scr.resize(nbe * nbe);
      host_data.zmat.resize(npts * nbe * spin_dimension * mgga_component_count);
      host_data.basis_eval.resize(basis_component_count * npts * nbe);
      if (needs_gradient) host_data.gamma.resize(direction_dimension * npts);

      ComponentMatrixMap basis_components(host_data.basis_eval.data(),
                                          basis_component_count, nbe, npts);
      ComponentMatrixMap zmat_components(host_data.zmat.data(),
                                         spin_dimension * mgga_component_count,
                                         nbe, npts);
      std::vector<std::array<int32_t, 3>> submat_map;
      std::tie(submat_map, std::ignore) = GauXC::gen_compressed_submat_map(
          basis_map, task.bfn_screening.shell_list, nbf, nbf);

      task_work.eval_collocation(basis_components, needs_gradient);
      task_work.eval_xmat(mgga_component_count, nbf, submat_map, scalar_density,
                          basis_components, zmat_components, 0,
                          host_data.nbe_scr);
      task_work.eval_xmat(mgga_component_count, nbf, submat_map, spin_density,
                          basis_components, zmat_components,
                          mgga_component_count, host_data.nbe_scr);

      if (is_mgga) {
        task_work.eval_mgga_model_features_uks(
            basis_components, zmat_components, mgga_component_count,
            features.density, features.density_gradient, host_data.gamma,
            features.kinetic, host_data.lapl, scalar_z_gradient);
      } else if (is_gga) {
        task_work.eval_gga_model_features_uks(
            basis_components, zmat_components, mgga_component_count,
            features.density, features.density_gradient, host_data.gamma,
            scalar_z_gradient);
      } else {
        task_work.eval_lda_model_features_uks(basis_components, zmat_components,
                                              mgga_component_count,
                                              features.density);
      }

      NEL_WORK += ConstVectorMap(task.weights.data(), npts)
                      .dot(features.density.rowwise().sum());
    }
  }  // omp parallel
  electron_count = NEL_WORK;
}

void SkalaHostDriver::post_skala_local_work_(
    const GauXC::BasisSet<double>& basis, ColMajorMatrixMap scalar_potential,
    ColMajorMatrixMap spin_potential, bool is_gga, bool is_mgga,
    bool /*needs_laplacian*/) {

  const bool needs_gradient = is_gga || is_mgga;
  auto* lwd = dynamic_cast<GauXC::LocalHostWorkDriver*>(lwd_.get());
  if (!lwd) SKALAXC_EXCEPTION("Expected a LocalHostWorkDriver");
  const auto& mol = lb_.molecule();
  GauXC::BasisSetMap basis_map(basis, mol);
  const int32_t nbf = basis.nbf();
  scalar_potential.setZero();
  spin_potential.setZero();

  auto& tasks = lb_.get_tasks();
  const size_t ntasks = tasks.size();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    GauXC::XCHostData<double> host_data;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t iT = 0; iT < ntasks; ++iT) {
      const auto& task = tasks[iT];
      const auto& potentials = task_potentials_[iT];
      const GauXCTaskAdapter task_work(*lwd, task, basis);

      const int32_t npts = task.points.size();
      const int32_t nbe = task.bfn_screening.nbe;
      const int32_t mgga_component_count = is_mgga ? 4 : 1;
      const int32_t basis_component_count = needs_gradient ? 4 : 1;

      host_data.nbe_scr.resize(nbe * nbe);
      host_data.zmat.resize(npts * nbe * spin_dimension * mgga_component_count);
      host_data.basis_eval.resize(basis_component_count * npts * nbe);

      ComponentMatrixMap basis_components(host_data.basis_eval.data(),
                                          basis_component_count, nbe, npts);
      ComponentMatrixMap zmat_components(host_data.zmat.data(),
                                         spin_dimension * mgga_component_count,
                                         nbe, npts);

      std::vector<std::array<int32_t, 3>> submat_map;
      std::tie(submat_map, std::ignore) = GauXC::gen_compressed_submat_map(
          basis_map, task.bfn_screening.shell_list, nbf, nbf);

      task_work.eval_collocation(basis_components, needs_gradient);

      if (needs_gradient) {
        eval_zmat_gga_vxc_uks(potentials.density, potentials.density_gradient,
                              basis_components, zmat_components,
                              mgga_component_count);
        if (is_mgga)
          task_work.eval_mmat_mgga_vxc_uks(potentials.kinetic, basis_components,
                                           zmat_components,
                                           mgga_component_count);
      } else {
        task_work.eval_zmat_lda_vxc_uks(potentials.density, basis_components,
                                        zmat_components, mgga_component_count);
      }

      task_work.inc_vxc(mgga_component_count, nbf, basis_components, submat_map,
                        zmat_components, 0, scalar_potential,
                        host_data.nbe_scr);
      task_work.inc_vxc(mgga_component_count, nbf, basis_components, submat_map,
                        zmat_components, mgga_component_count, spin_potential,
                        host_data.nbe_scr);
    }
  }  // omp parallel

  scalar_potential.template triangularView<Eigen::StrictlyUpper>() =
      scalar_potential.transpose();
  spin_potential.template triangularView<Eigen::StrictlyUpper>() =
      spin_potential.transpose();
}

}  // namespace SkalaXC

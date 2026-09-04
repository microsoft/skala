// SkalaXC black-box public-API test.
//
// This translation unit is a *consumer*: it and its test utility include only
// the public SkalaXC header and have NO access to GauXC or LibTorch
// headers/libraries. Eigen is an explicit consumer-side dependency used to own
// matrices. Successful compilation and linking prove the SkalaXC API remains
// ABI-isolated.
//
// It drives the public pipeline exactly as documented (RuntimeEnvironment ->
// Molecule/BasisSet -> MolGrid -> LoadBalancer -> MolecularWeights ->
// functional_type -> XCIntegratorFactory -> XCIntegrator), mirroring GauXC.
//
// HighFive is used solely to load the test's own density input from the HDF5
// fixture (the consumer's responsibility); it is unrelated to GauXC.

#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>
#include <highfive/H5File.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Integrator = SkalaXC::XCIntegrator<Matrix>;

static_assert(!std::is_copy_constructible_v<Integrator>);
static_assert(!std::is_copy_assignable_v<Integrator>);
static_assert(std::is_nothrow_move_constructible_v<Integrator>);
static_assert(std::is_nothrow_move_assignable_v<Integrator>);

// A fully-built integrator plus the sizes a consumer tracks alongside it.
struct BuiltIntegrator {
  Integrator integrator;
  std::int64_t nbf;
  std::int64_t natoms;
};

struct PipelineSettings {
  SkalaXC::PruningScheme pruning = SkalaXC::PruningScheme::Unpruned;
  std::int64_t batch_size = 512;
  SkalaXC::RadialQuad radial = SkalaXC::RadialQuad::MuraKnowles;
  SkalaXC::AtomicGridSizeDefault atomic_grid =
      SkalaXC::AtomicGridSizeDefault::UltraFineGrid;
  SkalaXC::XCWeightAlg weight_algorithm = SkalaXC::XCWeightAlg::SSF;
};

template <typename Function>
bool throws_skala_exception(Function function) {
  try {
    function();
  } catch (const SkalaXC::Exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

struct StateContractResults {
  bool uninitialized_integrator;
  bool uninitialized_caller_owned_integrator;
  bool unmodified_weights;
  bool empty_functional;
  bool mismatched_execution_space;
  bool duplicate_weight_modification;
  bool reusable_load_balancer;
};

// Run the public pipeline for one fixture + model. After the integrator is
// built it is self-contained, so the intermediate stages may go out of scope.
BuiltIntegrator build(const std::string& fixture, const std::string& model,
                      const PipelineSettings& settings = {},
                      SkalaXC::TimingSettings timing_settings = {},
                      SkalaXC::DomainBatchMode domain_batch_mode =
                          SkalaXC::DomainBatchMode::Conservative) {
#ifdef SKALAXC_HAS_MPI
  SkalaXC::RuntimeEnvironment rt(MPI_COMM_WORLD);
#else
  SkalaXC::RuntimeEnvironment rt;
#endif
  const auto system = SkalaXC::test::load_molecular_system(fixture);

  auto mg = SkalaXC::test::make_molgrid(system.molecule, settings.atomic_grid,
                                        settings.batch_size, settings.pruning,
                                        settings.radial);

  SkalaXC::LoadBalancerFactory lb_factory(SkalaXC::ExecutionSpace::Host);
  auto lb = lb_factory.get_instance(rt, system.molecule, mg, system.basis);

  SkalaXC::MolecularWeightsSettings weight_settings;
  weight_settings.weight_alg = settings.weight_algorithm;
  SkalaXC::MolecularWeightsFactory mw_factory(SkalaXC::ExecutionSpace::Host,
                                              "Default", weight_settings);
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  SkalaXC::functional_type func(model);
  SkalaXC::XCIntegratorFactory<Matrix> xc_factory(
      SkalaXC::ExecutionSpace::Host, timing_settings, domain_batch_mode);
  return BuiltIntegrator{xc_factory.get_instance(func, lb),
                         static_cast<std::int64_t>(system.basis.nbf()),
                         static_cast<std::int64_t>(system.molecule.natoms())};
}

StateContractResults check_state_contracts(const std::string& fixture) {
  StateContractResults results{};
  results.uninitialized_integrator = throws_skala_exception([] {
    SkalaXC::XCIntegrator<Matrix> integrator;
    const Matrix density;
    (void)integrator.eval_exc_vxc(density, density);
  });
  results.uninitialized_caller_owned_integrator = throws_skala_exception([] {
    SkalaXC::XCIntegrator<Matrix> integrator;
    Matrix density;
    Matrix scalar_potential;
    Matrix spin_potential;
    (void)integrator.eval_exc_vxc(density, density, scalar_potential,
                                  spin_potential);
  });

#ifdef SKALAXC_HAS_MPI
  SkalaXC::RuntimeEnvironment rt(MPI_COMM_WORLD);
#else
  SkalaXC::RuntimeEnvironment rt;
#endif
  const auto system = SkalaXC::test::load_molecular_system(fixture);
  auto mg = SkalaXC::test::make_molgrid(
      system.molecule, SkalaXC::AtomicGridSizeDefault::UltraFineGrid);
  SkalaXC::LoadBalancerFactory lb_factory(SkalaXC::ExecutionSpace::Host);
  auto lb = lb_factory.get_instance(rt, system.molecule, mg, system.basis);
  SkalaXC::XCIntegratorFactory<Matrix> host_factory(
      SkalaXC::ExecutionSpace::Host);

  results.unmodified_weights = throws_skala_exception([&] {
    (void)host_factory.get_instance(SkalaXC::functional_type("PBE"), lb);
  });

  SkalaXC::MolecularWeightsFactory mw_factory(
      SkalaXC::ExecutionSpace::Host, "Default",
      SkalaXC::MolecularWeightsSettings{});
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  auto first = host_factory.get_instance(SkalaXC::functional_type("PBE"), lb);
  auto second = host_factory.get_instance(SkalaXC::functional_type("PBE"), lb);
  const auto density =
      SkalaXC::test::load_uks_density(fixture, "/DENSITY_SCALAR", "/DENSITY_Z");
  (void)first.eval_exc_vxc(density.scalar, density.spin);
  const auto first_diagnostics = first.diagnostics();
  const auto second_before = second.diagnostics();
  (void)second.eval_exc_vxc(density.scalar, density.spin);
  const auto second_after = second.diagnostics();
  results.reusable_load_balancer =
      first_diagnostics.exc_vxc_calls == 1 &&
      second_before.exc_vxc_calls == 0 && second_after.exc_vxc_calls == 1 &&
      first_diagnostics.timing(SkalaXC::TimingMetric::ModelLoad).call_count ==
          1 &&
      second_before.timing(SkalaXC::TimingMetric::ModelLoad).call_count == 1;

  results.empty_functional = throws_skala_exception(
      [&] { (void)host_factory.get_instance(SkalaXC::functional_type{}, lb); });
  results.mismatched_execution_space = throws_skala_exception([&] {
    SkalaXC::XCIntegratorFactory<Matrix> device_factory(
        SkalaXC::ExecutionSpace::Device);
    (void)device_factory.get_instance(SkalaXC::functional_type("PBE"), lb);
  });
  results.duplicate_weight_modification =
      throws_skala_exception([&] { mw.modify_weights(lb); });
  return results;
}

struct CaseResult {
  double exc, exc_ref, exc_rel_err;
  std::int64_t nbf;
  bool vxc_symmetric;
  bool vxc_finite;
};

CaseResult run_case(const std::string& fixture, const std::string& model,
                    const PipelineSettings& settings = {}) {
  BuiltIntegrator built = build(fixture, model, settings);
  const std::int64_t nbf = built.nbf;

  HighFive::File file(fixture, HighFive::File::ReadOnly);
  const auto density =
      SkalaXC::test::load_uks_density(fixture, "/DENSITY_SCALAR", "/DENSITY_Z");
  double EXC_ref = 0.0;
  file.getDataSet("/EXC").read(&EXC_ref);

  auto [EXC, VXCs, VXCz] =
      built.integrator.eval_exc_vxc(density.scalar, density.spin);

  // VXC must be symmetric (nbf x nbf).
  const bool sym = (VXCs - VXCs.transpose()).cwiseAbs().maxCoeff() <= 1e-10 &&
                   (VXCz - VXCz.transpose()).cwiseAbs().maxCoeff() <= 1e-10;

  CaseResult r;
  r.nbf = nbf;
  r.exc = EXC;
  r.exc_ref = EXC_ref;
  r.exc_rel_err = std::abs(EXC - EXC_ref) / std::max(1.0, std::abs(EXC_ref));
  r.vxc_symmetric = sym;
  r.vxc_finite = VXCs.allFinite() && VXCz.allFinite();
  return r;
}

// A density whose extent does not match the basis must be rejected.
bool rejects_invalid_input_dimensions(const std::string& fixture) {
  BuiltIntegrator built = build(fixture, "PBE");
  const std::int64_t nbf = built.nbf;
  Matrix Ps(nbf, nbf > 1 ? nbf - 1 : nbf + 1);
  Matrix Pz(nbf, nbf);
  try {
    (void)built.integrator.eval_exc_vxc(Ps, Pz);
  } catch (const SkalaXC::Exception&) {
    return true;
  }
  return false;
}

bool caller_owned_potential_api_works(const std::string& fixture) {
  BuiltIntegrator built = build(fixture, "PBE");
  const auto density =
      SkalaXC::test::load_uks_density(fixture, "/DENSITY_SCALAR", "/DENSITY_Z");
  const auto [allocated_exc, allocated_scalar, allocated_spin] =
      built.integrator.eval_exc_vxc(density.scalar, density.spin);

  Matrix scalar(built.nbf, built.nbf);
  Matrix spin(built.nbf, built.nbf);
  scalar.setConstant(-1.0);
  spin.setConstant(-2.0);
  const double caller_owned_exc =
      built.integrator.eval_exc_vxc(density.scalar, density.spin, scalar, spin);
  if (std::abs(caller_owned_exc - allocated_exc) > 1e-12 ||
      (scalar - allocated_scalar).cwiseAbs().maxCoeff() > 1e-12 ||
      (spin - allocated_spin).cwiseAbs().maxCoeff() > 1e-12)
    return false;

  Matrix invalid_scalar(built.nbf, built.nbf > 1 ? built.nbf - 1 : 2);
  Matrix untouched_spin = Matrix::Constant(built.nbf, built.nbf, 17.0);
  if (!throws_skala_exception([&] {
        (void)built.integrator.eval_exc_vxc(density.scalar, density.spin,
                                            invalid_scalar, untouched_spin);
      }))
    return false;
  return (untouched_spin.array() == 17.0).all();
}

bool gradient_api_works(const std::string& fixture) {
  SkalaXC::TimingSettings timing_settings;
  BuiltIntegrator built = build(fixture, "TPSS", {}, timing_settings);
  BuiltIntegrator aggressive = build(fixture, "TPSS", {}, timing_settings,
                                     SkalaXC::DomainBatchMode::Aggressive);
  const std::int64_t natoms = built.natoms;

  const auto density = SkalaXC::test::load_uks_density(fixture, "/DENSITY", "");

  const auto gradient =
      built.integrator.eval_exc_grad(density.scalar, density.spin);
  std::vector<double> caller_owned_gradient(
      static_cast<std::size_t>(3 * natoms), -1.0);
  built.integrator.eval_exc_grad(density.scalar, density.spin,
                                 caller_owned_gradient);
  const auto aggressive_gradient =
      aggressive.integrator.eval_exc_grad(density.scalar, density.spin);
  if (gradient.size() != static_cast<std::size_t>(3 * natoms)) return false;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    if (std::abs(caller_owned_gradient[i] - gradient[i]) > 1e-12) return false;
  if (aggressive_gradient.size() != gradient.size()) return false;

  std::vector<double> invalid_gradient(gradient.size() + 1, 19.0);
  if (!throws_skala_exception([&] {
        built.integrator.eval_exc_grad(density.scalar, density.spin,
                                       invalid_gradient);
      }) ||
      !std::all_of(invalid_gradient.begin(), invalid_gradient.end(),
                   [](double value) { return value == 19.0; }))
    return false;

  double squared_norm = 0.0;
  double translation[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < gradient.size(); ++i) {
    if (!std::isfinite(gradient[i])) return false;
    if (!std::isfinite(aggressive_gradient[i]) ||
        std::abs(aggressive_gradient[i] - gradient[i]) > 1e-10)
      return false;
    squared_norm += gradient[i] * gradient[i];
    translation[i % 3] += gradient[i];
  }

  // SkalaXC supports only weight-derivative-inclusive gradients; requesting
  // otherwise must be rejected.
  bool rejects_no_weight_derivatives = false;
  SkalaXC::IntegratorSettingsEXC_GRAD no_weight_derivatives;
  no_weight_derivatives.include_weight_derivatives = false;
  try {
    (void)built.integrator.eval_exc_grad(density.scalar, density.spin,
                                         no_weight_derivatives);
  } catch (const SkalaXC::Exception&) {
    rejects_no_weight_derivatives = true;
  }

  const auto diagnostics = built.integrator.diagnostics();
  const bool diagnostics_valid =
      diagnostics.exc_gradient_calls == 2 &&
      diagnostics.timing(SkalaXC::TimingMetric::TotalEXCGradient).status ==
          SkalaXC::TimingStatus::Complete &&
      diagnostics.timing(SkalaXC::TimingMetric::GradientAssembly).status ==
          SkalaXC::TimingStatus::Complete;

  return diagnostics_valid && rejects_no_weight_derivatives &&
         std::sqrt(squared_norm) > 1e-3 && std::abs(translation[0]) < 1e-10 &&
         std::abs(translation[1]) < 1e-10 && std::abs(translation[2]) < 1e-10;
}

bool diagnostics_api_works(const std::string& fixture) {
  SkalaXC::TimingSettings timing_settings;
  BuiltIntegrator built = build(fixture, "PBE", {}, timing_settings);

  const auto density =
      SkalaXC::test::load_uks_density(fixture, "/DENSITY_SCALAR", "/DENSITY_Z");

  (void)built.integrator.eval_exc_vxc(density.scalar, density.spin);
  auto snapshot = built.integrator.diagnostics();
  const auto& model_load = snapshot.timing(SkalaXC::TimingMetric::ModelLoad);
  const auto& model_forward =
      snapshot.timing(SkalaXC::TimingMetric::ModelForward);
  const auto& total = snapshot.timing(SkalaXC::TimingMetric::TotalEXCVXC);
  const bool active_rank =
      snapshot.tasks > 0 && snapshot.points > 0 && snapshot.local_atoms == 1 &&
      snapshot.configured_model_batches == 1 && snapshot.task_points_min > 0 &&
      snapshot.task_points_max > 0 && snapshot.model_batches == 1 &&
      snapshot.domains == 1 &&
      model_forward.status == SkalaXC::TimingStatus::Complete &&
      model_forward.call_count == 1;
  const bool idle_rank =
      snapshot.tasks == 0 && snapshot.points == 0 &&
      snapshot.local_atoms == 0 && snapshot.configured_model_batches == 0 &&
      snapshot.task_points_min == 0 && snapshot.task_points_max == 0 &&
      snapshot.model_batches == 0 && snapshot.domains == 0 &&
      model_forward.status == SkalaXC::TimingStatus::Unavailable &&
      model_forward.call_count == 0;
  if (snapshot.backend != SkalaXC::ExecutionSpace::Host ||
      snapshot.communicator_size < 1 || snapshot.device_id != -1 ||
      snapshot.openmp_threads < 1 || snapshot.exc_vxc_calls != 1 ||
      (!active_rank && !idle_rank) ||
      model_load.status != SkalaXC::TimingStatus::Complete ||
      total.status != SkalaXC::TimingStatus::Complete ||
      model_load.call_count != 1 || total.call_count != 1)
    return false;

  built.integrator.reset_diagnostics();
  snapshot = built.integrator.diagnostics();
  return snapshot.timing(SkalaXC::TimingMetric::ModelLoad).call_count == 1 &&
         snapshot.configured_model_batches == (snapshot.tasks > 0 ? 1 : 0) &&
         snapshot.timing(SkalaXC::TimingMetric::ModelForward).status ==
             SkalaXC::TimingStatus::Unavailable &&
         snapshot.exc_vxc_calls == 0 && snapshot.model_batches == 0 &&
         snapshot.domains == 0 &&
         ((snapshot.tasks > 0 && snapshot.points > 0) ||
          (snapshot.tasks == 0 && snapshot.points == 0));
}

}  // namespace

int main() {
  const auto version = SkalaXC::version();
  if (version != SKALAXC_EXPECTED_VERSION) {
    std::printf("[FAIL] SkalaXC version: expected %s, got %.*s\n",
                SKALAXC_EXPECTED_VERSION, static_cast<int>(version.size()),
                version.data());
    return 1;
  }
  std::printf("[PASS] SkalaXC version %.*s\n", static_cast<int>(version.size()),
              version.data());

#ifdef SKALAXC_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) MPI_Init(nullptr, nullptr);
#endif

  const SkalaXC::DeviceRuntimeSettings device_settings;
  if (device_settings.device_id != 0 ||
      std::abs(device_settings.memory_fraction - 0.75) > 1e-15) {
    std::printf("[FAIL] device runtime defaults\n");
    return 1;
  }

  const std::string ref_dir = std::string(SKALAXC_TEST_REF_DATA_PATH);

  struct Fixture {
    const char* name;
    const char* file;
    const char* model;
  };
  const std::vector<Fixture> fixtures = {
      {"HE / def2-qzvp / lda", "/skala_he_def2qzvp_lda_uks.hdf5", "LDA"},
      {"HE / def2-qzvp / pbe", "/skala_he_def2qzvp_pbe_uks.hdf5", "PBE"},
      {"HE / def2-qzvp / tpss", "/skala_he_def2qzvp_tpss_uks.hdf5", "TPSS"},
  };

  int failures = 0;
  for (const auto& fx : fixtures) {
    try {
      const CaseResult r = run_case(ref_dir + fx.file, fx.model);
      const bool ok = (r.exc_rel_err < 1e-5) && r.vxc_symmetric &&
                      r.vxc_finite && std::isfinite(r.exc);
      std::printf(
          "[%s] %s : nbf=%lld EXC=%.10f (ref %.10f, rel %.2e) vxc_sym=%d\n",
          ok ? "PASS" : "FAIL", fx.name, (long long)r.nbf, r.exc, r.exc_ref,
          r.exc_rel_err, (int)r.vxc_symmetric);
      if (!ok) ++failures;
    } catch (const SkalaXC::Exception& e) {
      std::printf("[FAIL] %s : SkalaXC::Exception: %s\n", fx.name, e.what());
      ++failures;
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s : std::exception: %s\n", fx.name, e.what());
      ++failures;
    }
  }

  struct SettingsCase {
    const char* name;
    PipelineSettings settings;
    bool expect_success;
  };
  const std::vector<SettingsCase> settings_cases = {
      {"robust/becke/superfine/becke-weights",
       {SkalaXC::PruningScheme::Robust, 128, SkalaXC::RadialQuad::Becke,
        SkalaXC::AtomicGridSizeDefault::SuperFineGrid,
        SkalaXC::XCWeightAlg::Becke},
       true},
      {"treutler/mhl/gm3/lko-weights",
       {SkalaXC::PruningScheme::Treutler, 256,
        SkalaXC::RadialQuad::MurrayHandyLaming,
        SkalaXC::AtomicGridSizeDefault::GM3, SkalaXC::XCWeightAlg::LKO},
       true},
      {"unpruned/treutler-ahlrichs/gm5/ssf",
       {SkalaXC::PruningScheme::Unpruned, 64,
        SkalaXC::RadialQuad::TreutlerAhlrichs,
        SkalaXC::AtomicGridSizeDefault::GM5, SkalaXC::XCWeightAlg::SSF},
       true},
      {"unpartitioned weights rejected",
       {SkalaXC::PruningScheme::Unpruned, 512, SkalaXC::RadialQuad::MuraKnowles,
        SkalaXC::AtomicGridSizeDefault::FineGrid,
        SkalaXC::XCWeightAlg::NOTPARTITIONED},
       false},
  };
  const std::string settings_fixture = ref_dir + fixtures[1].file;
  for (const auto& settings_case : settings_cases) {
    try {
      const CaseResult result =
          run_case(settings_fixture, "PBE", settings_case.settings);
      const bool passed = settings_case.expect_success &&
                          std::isfinite(result.exc) && result.vxc_finite &&
                          result.vxc_symmetric;
      std::printf("[%s] settings: %s\n", passed ? "PASS" : "FAIL",
                  settings_case.name);
      if (!passed) ++failures;
    } catch (const std::exception& error) {
      const bool passed = !settings_case.expect_success;
      std::printf("[%s] settings: %s%s%s\n", passed ? "PASS" : "FAIL",
                  settings_case.name, passed ? "" : " : ",
                  passed ? "" : error.what());
      if (!passed) ++failures;
    }
  }

  if (!rejects_invalid_input_dimensions(ref_dir + fixtures.front().file)) {
    std::printf("[FAIL] invalid input dimensions were accepted\n");
    ++failures;
  } else {
    std::printf("[PASS] invalid input dimensions rejected\n");
  }

  const StateContractResults state_contracts =
      check_state_contracts(ref_dir + fixtures.front().file);
  const auto check_contract = [&failures](const char* name, bool passed) {
    std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
    if (!passed) ++failures;
  };
  check_contract("uninitialized integrator rejected",
                 state_contracts.uninitialized_integrator);
  check_contract("uninitialized caller-owned integrator rejected",
                 state_contracts.uninitialized_caller_owned_integrator);
  check_contract("integrator rejected unmodified weights",
                 state_contracts.unmodified_weights);
  check_contract("empty functional rejected", state_contracts.empty_functional);
  check_contract("mismatched execution spaces rejected",
                 state_contracts.mismatched_execution_space);
  check_contract("duplicate weight modification rejected",
                 state_contracts.duplicate_weight_modification);
  check_contract("load balancer reused with independent diagnostics",
                 state_contracts.reusable_load_balancer);

  if (!caller_owned_potential_api_works(ref_dir + fixtures[1].file)) {
    std::printf("[FAIL] caller-owned potential API validation failed\n");
    ++failures;
  } else {
    std::printf("[PASS] caller-owned potential API\n");
  }

  const std::string gradient_fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";
  if (!gradient_api_works(gradient_fixture)) {
    std::printf("[FAIL] public gradient API validation failed\n");
    ++failures;
  } else {
    std::printf("[PASS] public gradient API\n");
  }

  if (!diagnostics_api_works(ref_dir + fixtures[1].file)) {
    std::printf("[FAIL] public diagnostics API validation failed\n");
    ++failures;
  } else {
    std::printf("[PASS] public diagnostics API\n");
  }

  std::printf("\npublic-API failures: %d\n", failures);

#ifdef SKALAXC_HAS_MPI
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) MPI_Finalize();
#endif

  return failures == 0 ? 0 : 1;
}

#include <catch2/catch_test_macros.hpp>

#include <skalaxc/skalaxc.hpp>
#include <skalaxc/skalaxc_config.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef SKALAXC_HAS_OPENMP
#include <omp.h>

namespace {

class OpenMPSettingsGuard {
 public:
  OpenMPSettingsGuard()
      : dynamic_(omp_get_dynamic()), max_threads_(omp_get_max_threads()) {
    omp_set_dynamic(0);
  }

  ~OpenMPSettingsGuard() {
    omp_set_num_threads(max_threads_);
    omp_set_dynamic(dynamic_);
  }

  OpenMPSettingsGuard(const OpenMPSettingsGuard&) = delete;
  OpenMPSettingsGuard& operator=(const OpenMPSettingsGuard&) = delete;

 private:
  int dynamic_;
  int max_threads_;
};

using Matrix = Eigen::MatrixXd;

struct HostEvaluation {
  std::tuple<double, Matrix, Matrix> exc_vxc;
  std::vector<double> gradient;
  SkalaXC::DiagnosticsSnapshot diagnostics;
};

HostEvaluation evaluate(const SkalaXC::test::MolecularSystem& system,
                        const SkalaXC::test::UksDensity& density,
                        int thread_count) {
  omp_set_num_threads(thread_count);
  auto grid = SkalaXC::test::make_molgrid(
      system.molecule, SkalaXC::AtomicGridSizeDefault::FineGrid);
  SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD)};
  SkalaXC::LoadBalancerFactory load_balancer_factory(
      SkalaXC::ExecutionSpace::Host);
  auto load_balancer = load_balancer_factory.get_instance(
      runtime, system.molecule, grid, system.basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      SkalaXC::ExecutionSpace::Host, "Default",
      SkalaXC::MolecularWeightsSettings{});
  weights_factory.get_instance().modify_weights(load_balancer);
  SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
      SkalaXC::ExecutionSpace::Host);
  auto integrator = integrator_factory.get_instance(
      SkalaXC::functional_type("TPSS"), load_balancer);

  auto exc_vxc = integrator.eval_exc_vxc(density.scalar, density.spin);
  auto gradient = integrator.eval_exc_grad(density.scalar, density.spin);
  return {std::move(exc_vxc), std::move(gradient), integrator.diagnostics()};
}

}  // namespace
#endif

TEST_CASE("OpenMP thread counts preserve host EXC, VXC, and gradients",
          "[skala][openmp]") {
#ifdef SKALAXC_HAS_OPENMP
  OpenMPSettingsGuard restore_openmp_settings;
  const std::string fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";
  const auto system = SkalaXC::test::load_molecular_system(fixture);
  const auto density = SkalaXC::test::load_uks_density(fixture, "/DENSITY", "");
  const auto single_thread = evaluate(system, density, 1);
  const auto two_threads = evaluate(system, density, 2);

  REQUIRE(single_thread.diagnostics.openmp_threads == 1);
  REQUIRE(two_threads.diagnostics.openmp_threads == 2);

  const double exc_error =
      std::abs(std::get<0>(single_thread.exc_vxc) -
               std::get<0>(two_threads.exc_vxc)) /
      std::max(1.0, std::abs(std::get<0>(single_thread.exc_vxc)));
  const double scalar_error = SkalaXC::test::matrix_error_per_basis(
      std::get<1>(single_thread.exc_vxc), std::get<1>(two_threads.exc_vxc));
  const double spin_error = SkalaXC::test::matrix_error_per_basis(
      std::get<2>(single_thread.exc_vxc), std::get<2>(two_threads.exc_vxc));
  REQUIRE(single_thread.gradient.size() == two_threads.gradient.size());
  double gradient_error = 0.0;
  for (std::size_t i = 0; i < single_thread.gradient.size(); ++i) {
    const double difference =
        single_thread.gradient[i] - two_threads.gradient[i];
    gradient_error += difference * difference;
  }
  gradient_error = std::sqrt(gradient_error) /
                   static_cast<double>(single_thread.gradient.size());
  CHECK(exc_error <= 1e-12);
  CHECK(scalar_error <= 1e-12);
  CHECK(spin_error <= 1e-12);
  CHECK(gradient_error <= 1e-10);
#else
  SKIP("OpenMP disabled");
#endif
}

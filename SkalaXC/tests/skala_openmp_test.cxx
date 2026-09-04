#include <catch2/catch.hpp>

#include <skalaxc/skalaxc.hpp>
#include <skalaxc/skalaxc_config.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
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

}  // namespace
#endif

TEST_CASE("OpenMP thread counts preserve host EXC, VXC, and gradients",
          "[skala][openmp]") {
#ifdef SKALAXC_HAS_OPENMP
  using Matrix = Eigen::MatrixXd;
  const std::string fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";
  const auto system = SkalaXC::test::load_molecular_system(fixture);
  const auto density = SkalaXC::test::load_uks_density(fixture, "/DENSITY", "");
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

  OpenMPSettingsGuard restore_openmp_settings;
  omp_set_num_threads(1);
  const auto single_thread =
      integrator.eval_exc_vxc(density.scalar, density.spin);
  const auto single_thread_gradient =
      integrator.eval_exc_grad(density.scalar, density.spin);
  omp_set_num_threads(2);
  const auto two_threads =
      integrator.eval_exc_vxc(density.scalar, density.spin);
  const auto two_thread_gradient =
      integrator.eval_exc_grad(density.scalar, density.spin);

  const double exc_error =
      std::abs(std::get<0>(single_thread) - std::get<0>(two_threads)) /
      std::max(1.0, std::abs(std::get<0>(single_thread)));
  const double scalar_error = SkalaXC::test::matrix_error_per_basis(
      std::get<1>(single_thread), std::get<1>(two_threads));
  const double spin_error = SkalaXC::test::matrix_error_per_basis(
      std::get<2>(single_thread), std::get<2>(two_threads));
  REQUIRE(single_thread_gradient.size() == two_thread_gradient.size());
  double gradient_error = 0.0;
  for (std::size_t i = 0; i < single_thread_gradient.size(); ++i) {
    const double difference =
        single_thread_gradient[i] - two_thread_gradient[i];
    gradient_error += difference * difference;
  }
  gradient_error = std::sqrt(gradient_error) /
                   static_cast<double>(single_thread_gradient.size());
  CHECK(exc_error <= 1e-12);
  CHECK(scalar_error <= 1e-12);
  CHECK(spin_error <= 1e-12);
  CHECK(gradient_error <= 1e-10);
#else
  SUCCEED("OpenMP disabled");
#endif
}

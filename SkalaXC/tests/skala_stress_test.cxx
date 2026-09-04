#include <catch2/catch.hpp>

#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <tuple>

#ifdef __linux__
#include <unistd.h>
#endif

namespace {

using Matrix = Eigen::MatrixXd;

std::size_t resident_bytes() {
#ifdef __linux__
  std::ifstream statm("/proc/self/statm");
  std::size_t total_pages = 0;
  std::size_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) return 0;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return 0;
  return resident_pages * static_cast<std::size_t>(page_size);
#else
  return 0;
#endif
}

}  // namespace

TEST_CASE("Repeated host evaluations remain stable and bounded",
          "[skala][stress]") {
  SkalaXC::Molecule molecule{{SkalaXC::AtomicNumber(1), -0.7, 0.0, 0.0},
                             {SkalaXC::AtomicNumber(1), 0.7, 0.0, 0.0}};
  auto basis = SkalaXC::test::make_sto3g_hydrogen_basis(molecule);
  auto grid = SkalaXC::test::make_molgrid(
      molecule, SkalaXC::AtomicGridSizeDefault::FineGrid, 128);
  SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD)};
  SkalaXC::LoadBalancerFactory load_balancer_factory(
      SkalaXC::ExecutionSpace::Host);
  auto load_balancer =
      load_balancer_factory.get_instance(runtime, molecule, grid, basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      SkalaXC::ExecutionSpace::Host, "Default");
  weights_factory.get_instance().modify_weights(load_balancer);
  SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
      SkalaXC::ExecutionSpace::Host);
  auto integrator = integrator_factory.get_instance(
      SkalaXC::functional_type("LDA"), load_balancer);

  Matrix scalar_density = Matrix::Constant(2, 2, 0.5);
  Matrix spin_density = Matrix::Zero(2, 2);
  const auto reference = integrator.eval_exc_vxc(scalar_density, spin_density);
  for (int iteration = 0; iteration < 10; ++iteration)
    (void)integrator.eval_exc_vxc(scalar_density, spin_density);
  const std::size_t warmed_resident_bytes = resident_bytes();

  for (int iteration = 0; iteration < 100; ++iteration) {
    const auto result = integrator.eval_exc_vxc(scalar_density, spin_density);
    REQUIRE(std::get<0>(result) ==
            Approx(std::get<0>(reference)).margin(1e-13));
    REQUIRE((std::get<1>(result) - std::get<1>(reference)).norm() <= 1e-13);
    REQUIRE((std::get<2>(result) - std::get<2>(reference)).norm() <= 1e-13);
  }

  const std::size_t final_resident_bytes = resident_bytes();
  if (warmed_resident_bytes != 0 && final_resident_bytes != 0) {
    constexpr std::size_t max_growth_bytes = 64ULL * 1024ULL * 1024ULL;
    REQUIRE(final_resident_bytes <= warmed_resident_bytes + max_growth_bytes);
  }
}

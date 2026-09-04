#include <catch2/catch_test_macros.hpp>

#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#ifdef SKALAXC_HAS_MPI
#include <mpi.h>
#endif

namespace {

using Matrix = Eigen::MatrixXd;

struct HostEvaluation {
  double exc;
  Matrix scalar_potential;
  Matrix spin_potential;
  std::vector<double> gradient;
};

HostEvaluation evaluate_host(const SkalaXC::RuntimeEnvironment& runtime,
                             const SkalaXC::Molecule& molecule,
                             const SkalaXC::BasisSet<double>& basis,
                             const Matrix& scalar_density,
                             const Matrix& spin_density) {
  auto grid = SkalaXC::test::make_molgrid(
      molecule, SkalaXC::AtomicGridSizeDefault::FineGrid);
  SkalaXC::LoadBalancerFactory load_balancer_factory(
      SkalaXC::ExecutionSpace::Host);
  auto load_balancer =
      load_balancer_factory.get_instance(runtime, molecule, grid, basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      SkalaXC::ExecutionSpace::Host, "Default",
      SkalaXC::MolecularWeightsSettings{});
  weights_factory.get_instance().modify_weights(load_balancer);
  SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
      SkalaXC::ExecutionSpace::Host);
  auto integrator = integrator_factory.get_instance(
      SkalaXC::functional_type("TPSS"), load_balancer);
  auto [exc, scalar_potential, spin_potential] =
      integrator.eval_exc_vxc(scalar_density, spin_density);
  auto gradient = integrator.eval_exc_grad(scalar_density, spin_density);
  return HostEvaluation{exc, std::move(scalar_potential),
                        std::move(spin_potential), std::move(gradient)};
}

}  // namespace

TEST_CASE("Skala host evaluation uses the runtime MPI subcommunicator",
          "[skala][mpi][host-subcomm][mpi-only]") {
#ifdef SKALAXC_HAS_MPI
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < 4) {
    SKIP("Requires at least four MPI ranks");
  }

  const int color = world_rank % 2;
  MPI_Comm subcomm = MPI_COMM_NULL;
  REQUIRE(MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm) ==
          MPI_SUCCESS);

  const double displacement = 0.04 * color;
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system(displacement);
  Matrix scalar_density(2, 2);
  scalar_density << 0.5 + 0.02 * color, 0.5, 0.5, 0.5 - 0.02 * color;
  const Matrix spin_density = Matrix::Zero(2, 2);

  const SkalaXC::RuntimeEnvironment subcomm_runtime{subcomm};
  const SkalaXC::RuntimeEnvironment self_runtime{MPI_COMM_SELF};
  const auto subcomm_result =
      evaluate_host(subcomm_runtime, system.molecule, system.basis,
                    scalar_density, spin_density);
  const auto self_result =
      evaluate_host(self_runtime, system.molecule, system.basis, scalar_density,
                    spin_density);

  const double exc_error = std::abs(subcomm_result.exc - self_result.exc) /
                           std::max(1.0, std::abs(self_result.exc));
  const double scalar_error = SkalaXC::test::matrix_error_per_basis(
      subcomm_result.scalar_potential, self_result.scalar_potential);
  const double spin_error = SkalaXC::test::matrix_error_per_basis(
      subcomm_result.spin_potential, self_result.spin_potential);
  REQUIRE(subcomm_result.gradient.size() == self_result.gradient.size());
  double gradient_error = 0.0;
  for (std::size_t index = 0; index < self_result.gradient.size(); ++index)
    gradient_error = std::max(
        gradient_error,
        std::abs(subcomm_result.gradient[index] - self_result.gradient[index]));

  INFO("subcommunicator color=" << color);
  CHECK(exc_error <= 1e-10);
  CHECK(scalar_error <= 1e-7);
  CHECK(spin_error <= 1e-10);
  CHECK(gradient_error <= 1e-8);
  MPI_Comm_free(&subcomm);
#else
  SKIP("MPI disabled");
#endif
}
#include <catch2/catch_test_macros.hpp>

#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
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

TEST_CASE("Host evaluation errors reach idle ranks",
          "[skala][mpi][host-evaluation-errors]") {
#ifdef SKALAXC_HAS_MPI
  MPI_Comm communicator = MPI_COMM_WORLD;
  SECTION("world communicator") {}
  SECTION("split communicator") {
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, -world_rank, &communicator);
  }
#endif
  {
    SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(communicator)};
    auto system = SkalaXC::test::make_rotated_h2_sto3g_system(0.0);
    system.molecule.resize(1);
    system.basis.resize(1);
    auto grid = SkalaXC::test::make_molgrid(
        system.molecule, SkalaXC::AtomicGridSizeDefault::FineGrid);
    auto load_balancer =
        SkalaXC::LoadBalancerFactory(SkalaXC::ExecutionSpace::Host)
            .get_instance(runtime, system.molecule, grid, system.basis);
    SkalaXC::MolecularWeightsFactory(SkalaXC::ExecutionSpace::Host, "Default",
                                     {})
        .get_instance()
        .modify_weights(load_balancer);
    auto integrator =
        SkalaXC::XCIntegratorFactory<Matrix>(SkalaXC::ExecutionSpace::Host)
            .get_instance(SkalaXC::functional_type("LDA"), load_balancer);
    const Matrix invalid =
        Matrix::Constant(1, 1, std::numeric_limits<double>::quiet_NaN());
    const Matrix spin = Matrix::Zero(1, 1);
    for (const bool gradient : {false, true}) {
      std::string error;
      try {
        if (gradient)
          (void)integrator.eval_exc_grad(invalid, spin);
        else
          (void)integrator.eval_exc_vxc(invalid, spin);
      } catch (const SkalaXC::Exception& exception) {
        error = exception.what();
      }
      CHECK(error.find("NaN") != std::string::npos);
#ifdef SKALAXC_HAS_MPI
      int failures = error.empty() ? 0 : 1;
      MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, communicator);
      CHECK(failures == runtime.comm_size());
#endif
    }
    const Matrix valid = Matrix::Constant(1, 1, 0.5);
    CHECK(std::isfinite(std::get<0>(integrator.eval_exc_vxc(valid, spin))));
    const auto gradient = integrator.eval_exc_grad(valid, spin);
    const bool finite_gradient =
        std::all_of(gradient.begin(), gradient.end(),
                    [](double value) { return std::isfinite(value); });
    CHECK(finite_gradient);
  }
#ifdef SKALAXC_HAS_MPI
  if (communicator != MPI_COMM_WORLD) MPI_Comm_free(&communicator);
#endif
}

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
  REQUIRE(subcomm_result.gradient.size() == 3 * system.molecule.natoms());
  REQUIRE(self_result.gradient.size() == 3 * system.molecule.natoms());
  double gradient_error = 0.0;
  for (std::size_t index = 0; index < self_result.gradient.size(); ++index) {
    INFO("gradient component=" << index);
    REQUIRE(std::isfinite(subcomm_result.gradient[index]));
    REQUIRE(std::isfinite(self_result.gradient[index]));
    gradient_error = std::max(
        gradient_error,
        std::abs(subcomm_result.gradient[index] - self_result.gradient[index]));
  }

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
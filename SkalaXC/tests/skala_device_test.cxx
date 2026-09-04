#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#ifdef SKALAXC_HAS_MPI
#include <mpi.h>
#endif

namespace {

using Matrix = Eigen::MatrixXd;
using Result = std::tuple<double, Matrix, Matrix>;

std::string cuda_skala_model() {
  return std::string(SKALAXC_MODEL_PATH) + "/skala-1.1-cuda.fun";
}

Result evaluate(const SkalaXC::RuntimeEnvironment& runtime,
                SkalaXC::ExecutionSpace execution_space,
                const SkalaXC::Molecule& molecule,
                const SkalaXC::BasisSet<double>& basis,
                const std::string& model, const Matrix& scalar_density,
                const Matrix& spin_density, bool repeat,
                SkalaXC::DomainBatchMode batch_mode =
                    SkalaXC::DomainBatchMode::Conservative,
                SkalaXC::TimingSettings timing_settings = {},
                SkalaXC::DiagnosticsSnapshot* diagnostics = nullptr) {
  auto grid = SkalaXC::test::make_molgrid(
      molecule, SkalaXC::AtomicGridSizeDefault::UltraFineGrid);
  SkalaXC::LoadBalancerFactory load_balancer_factory(execution_space,
                                                     "Default");
  auto load_balancer =
      load_balancer_factory.get_instance(runtime, molecule, grid, basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      execution_space, "Default", SkalaXC::MolecularWeightsSettings{});
  weights_factory.get_instance().modify_weights(load_balancer);

  SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
      execution_space, timing_settings, batch_mode);
  auto integrator = integrator_factory.get_instance(
      SkalaXC::functional_type(model), load_balancer);
  Result result = integrator.eval_exc_vxc(scalar_density, spin_density);
  if (repeat) result = integrator.eval_exc_vxc(scalar_density, spin_density);
  if (diagnostics) *diagnostics = integrator.diagnostics();
  return result;
}

std::vector<double> evaluate_gradient(
    const SkalaXC::RuntimeEnvironment& runtime,
    SkalaXC::ExecutionSpace execution_space, const SkalaXC::Molecule& molecule,
    const SkalaXC::BasisSet<double>& basis, const std::string& model,
    const Matrix& scalar_density, const Matrix& spin_density, bool repeat,
    SkalaXC::DomainBatchMode batch_mode =
        SkalaXC::DomainBatchMode::Conservative) {
  auto grid = SkalaXC::test::make_molgrid(
      molecule, SkalaXC::AtomicGridSizeDefault::FineGrid);
  SkalaXC::LoadBalancerFactory load_balancer_factory(execution_space,
                                                     "Default");
  auto load_balancer =
      load_balancer_factory.get_instance(runtime, molecule, grid, basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      execution_space, "Default", SkalaXC::MolecularWeightsSettings{});
  weights_factory.get_instance().modify_weights(load_balancer);
  SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
      execution_space, SkalaXC::TimingSettings{}, batch_mode);
  auto integrator = integrator_factory.get_instance(
      SkalaXC::functional_type(model), load_balancer);
  auto result = integrator.eval_exc_grad(scalar_density, spin_density);
  if (repeat) result = integrator.eval_exc_grad(scalar_density, spin_density);
  return result;
}

}  // namespace

TEST_CASE("Skala CUDA reproduces host EXC and VXC",
          "[skala][cuda][device-reference-integration]") {
  const std::string fixture = std::string(SKALAXC_TEST_REF_DATA_PATH) +
                              "/skala_he_def2qzvp_lda_uks.hdf5";
  const auto system = SkalaXC::test::load_molecular_system(fixture);
  const auto density =
      SkalaXC::test::load_uks_density(fixture, "/DENSITY_SCALAR", "/DENSITY_Z");

  SkalaXC::RuntimeEnvironment host_runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD)};
  SkalaXC::DeviceRuntimeSettings device_settings;
  device_settings.device_id = 0;
  SkalaXC::RuntimeEnvironment device_runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD, )
                                                 device_settings};

  for (const std::string model : {"LDA", "PBE", "TPSS"}) {
    const Result host =
        evaluate(host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
                 system.basis, model, density.scalar, density.spin, false);
    const Result device = evaluate(
        device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
        system.basis, model, density.scalar, density.spin, true);

    const double exc_error = std::abs(std::get<0>(device) - std::get<0>(host)) /
                             std::max(1.0, std::abs(std::get<0>(host)));
    const double scalar_error = SkalaXC::test::matrix_error_per_basis(
        std::get<1>(device), std::get<1>(host));
    const double spin_error = SkalaXC::test::matrix_error_per_basis(
        std::get<2>(device), std::get<2>(host));

    INFO("model=" << model);
    INFO("device EXC=" << std::get<0>(device)
                       << " host EXC=" << std::get<0>(host));
    INFO("EXC relative error=" << exc_error);
    INFO("scalar VXC norm error / nbf=" << scalar_error);
    INFO("z VXC norm error / nbf=" << spin_error);
    CHECK(exc_error <= 1e-10);
    CHECK(scalar_error <= 1e-7);
    CHECK(spin_error <= 1e-10);
  }
}

TEST_CASE("Skala CUDA reproduces host semilocal nuclear gradients",
          "[skala][cuda][device-gradient]") {
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system();
  Matrix scalar_density(2, 2);
  scalar_density << 0.5, 0.5, 0.5, 0.5;
  const Matrix spin_density = Matrix::Zero(2, 2);

  SkalaXC::RuntimeEnvironment host_runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD)};
  SkalaXC::RuntimeEnvironment device_runtime{
      SKALAXC_MPI_CODE(MPI_COMM_WORLD, ) SkalaXC::DeviceRuntimeSettings{}};
  // The current TPSS trace can exceed sm_120 launch resources in its
  // TensorExpr backward kernel. Keep TPSS EXC/VXC coverage above and exercise
  // the primary neural model in the isolated test below until TPSS is retraced.
  for (const std::string model : {"LDA", "PBE"}) {
    INFO("model=" << model);
    const auto host = evaluate_gradient(
        host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
        system.basis, model, scalar_density, spin_density, false);
    const auto device = evaluate_gradient(
        device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
        system.basis, model, scalar_density, spin_density, true);
    REQUIRE(device.size() == host.size());
    double max_error = 0.0;
    for (std::size_t index = 0; index < host.size(); ++index)
      max_error = std::max(max_error, std::abs(device[index] - host[index]));
    INFO("maximum gradient component error=" << max_error);
    CHECK(max_error <= 1e-6);
  }
}

TEST_CASE("Skala CUDA evaluates neural nuclear gradients",
          "[skala][cuda][device-neural-gradient]") {
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system();
  Matrix scalar_density(2, 2);
  scalar_density << 0.5, 0.5, 0.5, 0.5;
  const Matrix spin_density = Matrix::Zero(2, 2);
  SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD, )
                                          SkalaXC::DeviceRuntimeSettings{}};

  const auto gradient = evaluate_gradient(
      runtime, SkalaXC::ExecutionSpace::Device, system.molecule, system.basis,
      cuda_skala_model(), scalar_density, spin_density, false);
  REQUIRE(gradient.size() == 3 * system.molecule.size());
  double squared_norm = 0.0;
  std::array<double, 3> translation{};
  for (std::size_t index = 0; index < gradient.size(); ++index) {
    REQUIRE(std::isfinite(gradient[index]));
    squared_norm += gradient[index] * gradient[index];
    translation[index % translation.size()] += gradient[index];
  }
  CHECK(squared_norm > 1e-6);
  for (const double component : translation)
    CHECK(std::abs(component) <= 1e-10);
}

TEST_CASE("Skala CUDA exact-size batching modes agree",
          "[skala][cuda][device-batching]") {
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system();
  Matrix scalar_density(2, 2);
  scalar_density << 0.5, 0.5, 0.5, 0.5;
  const Matrix spin_density = Matrix::Zero(2, 2);
  const std::string model = "PBE";
  SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD, )
                                          SkalaXC::DeviceRuntimeSettings{}};

  const auto conservative =
      evaluate(runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
               system.basis, model, scalar_density, spin_density, false,
               SkalaXC::DomainBatchMode::Conservative);
  SkalaXC::DiagnosticsSnapshot aggressive_diagnostics;
  SkalaXC::TimingSettings debug_settings;
  debug_settings.debug_logging = true;
  const auto aggressive =
      evaluate(runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
               system.basis, model, scalar_density, spin_density, false,
               SkalaXC::DomainBatchMode::Aggressive, debug_settings,
               &aggressive_diagnostics);
  CHECK(std::abs(std::get<0>(aggressive) - std::get<0>(conservative)) <= 1e-10);
  CHECK(SkalaXC::test::matrix_error_per_basis(
            std::get<1>(aggressive), std::get<1>(conservative)) <= 1e-7);
  CHECK(SkalaXC::test::matrix_error_per_basis(
            std::get<2>(aggressive), std::get<2>(conservative)) <= 1e-10);
  CHECK(aggressive_diagnostics.device_id == 0);
  CHECK(aggressive_diagnostics.device_memory_fraction == Catch::Approx(0.75));
  CHECK(aggressive_diagnostics.domain_batch_mode ==
        SkalaXC::DomainBatchMode::Aggressive);
  CHECK(aggressive_diagnostics.local_atoms == system.molecule.size());
  CHECK(aggressive_diagnostics.configured_model_batches == 1);
  CHECK(aggressive_diagnostics.max_domains_per_model_batch ==
        system.molecule.size());
  CHECK(
      aggressive_diagnostics.timing(SkalaXC::TimingMetric::ModelLoad).status ==
      SkalaXC::TimingStatus::Complete);

  const auto conservative_gradient = evaluate_gradient(
      runtime, SkalaXC::ExecutionSpace::Device, system.molecule, system.basis,
      model, scalar_density, spin_density, false,
      SkalaXC::DomainBatchMode::Conservative);
  const auto aggressive_gradient = evaluate_gradient(
      runtime, SkalaXC::ExecutionSpace::Device, system.molecule, system.basis,
      model, scalar_density, spin_density, false,
      SkalaXC::DomainBatchMode::Aggressive);
  REQUIRE(aggressive_gradient.size() == conservative_gradient.size());
  for (std::size_t index = 0; index < aggressive_gradient.size(); ++index)
    CHECK(std::abs(aggressive_gradient[index] - conservative_gradient[index]) <=
          1e-6);
}

TEST_CASE("Skala CUDA restores a non-default caller Torch stream",
          "[skala][cuda][stream]") {
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system();
  Matrix scalar_density(2, 2);
  scalar_density << 0.5, 0.5, 0.5, 0.5;
  const Matrix spin_density = Matrix::Zero(2, 2);

  SkalaXC::DeviceRuntimeSettings device_settings;
  device_settings.device_id = 0;
  SkalaXC::RuntimeEnvironment runtime{SKALAXC_MPI_CODE(MPI_COMM_WORLD, )
                                          device_settings};
  auto grid = SkalaXC::test::make_molgrid(
      system.molecule, SkalaXC::AtomicGridSizeDefault::FineGrid);
  SkalaXC::LoadBalancerFactory load_balancer_factory(
      SkalaXC::ExecutionSpace::Device, "Default");
  auto load_balancer = load_balancer_factory.get_instance(
      runtime, system.molecule, grid, system.basis);
  SkalaXC::MolecularWeightsFactory weights_factory(
      SkalaXC::ExecutionSpace::Device, "Default",
      SkalaXC::MolecularWeightsSettings{});
  weights_factory.get_instance().modify_weights(load_balancer);

  const auto original_stream =
      c10::cuda::getCurrentCUDAStream(device_settings.device_id);
  const auto caller_stream =
      c10::cuda::getStreamFromPool(false, device_settings.device_id);
  REQUIRE(caller_stream !=
          c10::cuda::getDefaultCUDAStream(device_settings.device_id));

  {
    const c10::cuda::CUDAStreamGuard caller_guard(caller_stream);
    REQUIRE(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
            caller_stream);

    SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
        SkalaXC::ExecutionSpace::Device, SkalaXC::TimingSettings{},
        SkalaXC::DomainBatchMode::Aggressive);
    auto integrator = integrator_factory.get_instance(
        SkalaXC::functional_type("PBE"), load_balancer);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);

    const auto first = integrator.eval_exc_vxc(scalar_density, spin_density);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
    const auto second = integrator.eval_exc_vxc(scalar_density, spin_density);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
    CHECK(std::abs(std::get<0>(second) - std::get<0>(first)) <= 1e-10);
    CHECK(SkalaXC::test::matrix_error_per_basis(std::get<1>(second),
                                                std::get<1>(first)) <= 1e-7);
    CHECK(SkalaXC::test::matrix_error_per_basis(std::get<2>(second),
                                                std::get<2>(first)) <= 1e-10);

    const auto first_gradient =
        integrator.eval_exc_grad(scalar_density, spin_density);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
    const auto second_gradient =
        integrator.eval_exc_grad(scalar_density, spin_density);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
    REQUIRE(second_gradient.size() == first_gradient.size());
    for (std::size_t index = 0; index < first_gradient.size(); ++index)
      CHECK(std::abs(second_gradient[index] - first_gradient[index]) <= 1e-6);

    Matrix invalid_density = scalar_density;
    invalid_density(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(integrator.eval_exc_vxc(invalid_density, spin_density),
                    SkalaXC::Exception);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
    CHECK_THROWS_AS(integrator.eval_exc_grad(invalid_density, spin_density),
                    SkalaXC::Exception);
    CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
          caller_stream);
  }

  CHECK(c10::cuda::getCurrentCUDAStream(device_settings.device_id) ==
        original_stream);
}

TEST_CASE("Skala CUDA supports an MPI rank with no atomic domains",
          "[skala][cuda][mpi][device-idle-rank][mpi-only]") {
#ifdef SKALAXC_HAS_MPI
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < 3) {
    SKIP("Requires at least three MPI ranks");
  }

  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank < 3 ? 0 : MPI_UNDEFINED, world_rank,
                 &subcomm);
  if (subcomm == MPI_COMM_NULL) {
    SKIP("Only the first three ranks participate");
  }

  MPI_Comm local_comm = MPI_COMM_NULL;
  REQUIRE(MPI_Comm_split_type(subcomm, MPI_COMM_TYPE_SHARED, world_rank,
                              MPI_INFO_NULL, &local_comm) == MPI_SUCCESS);
  int local_rank = 0;
  int local_size = 1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  REQUIRE(device_count > 0);
  const int ranks_per_device = (local_size + device_count - 1) / device_count;
  SkalaXC::DeviceRuntimeSettings device_settings;
  device_settings.device_id = local_rank % device_count;
  device_settings.memory_fraction = 0.8 / ranks_per_device;
  MPI_Comm_free(&local_comm);

  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system();
  Matrix scalar_density(2, 2);
  scalar_density << 0.5, 0.5, 0.5, 0.5;
  const Matrix spin_density = Matrix::Zero(2, 2);

  SkalaXC::RuntimeEnvironment host_runtime{subcomm};
  SkalaXC::RuntimeEnvironment device_runtime{subcomm, device_settings};
  const auto host =
      evaluate(host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
               system.basis, "PBE", scalar_density, spin_density, false);
  const auto device =
      evaluate(device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
               system.basis, "PBE", scalar_density, spin_density, false);
  const auto host_gradient = evaluate_gradient(
      host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
      system.basis, "PBE", scalar_density, spin_density, false);
  const auto device_gradient = evaluate_gradient(
      device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
      system.basis, "PBE", scalar_density, spin_density, false);

  CHECK(std::abs(std::get<0>(device) - std::get<0>(host)) <= 1e-10);
  CHECK(SkalaXC::test::matrix_error_per_basis(std::get<1>(device),
                                              std::get<1>(host)) <= 1e-7);
  CHECK(SkalaXC::test::matrix_error_per_basis(std::get<2>(device),
                                              std::get<2>(host)) <= 1e-10);
  REQUIRE(device_gradient.size() == host_gradient.size());
  for (std::size_t index = 0; index < host_gradient.size(); ++index)
    CHECK(std::abs(device_gradient[index] - host_gradient[index]) <= 1e-6);
  MPI_Comm_free(&subcomm);
#else
  SKIP("MPI disabled");
#endif
}

TEST_CASE("Skala CUDA uses the runtime MPI subcommunicator",
          "[skala][cuda][mpi][subcomm][device-subcomm][mpi-only]") {
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
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm);

  MPI_Comm local_comm = MPI_COMM_NULL;
  REQUIRE(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                              MPI_INFO_NULL, &local_comm) == MPI_SUCCESS);
  int local_rank = 0;
  int local_size = 1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  REQUIRE(device_count > 0);
  const int ranks_per_device = (local_size + device_count - 1) / device_count;
  SkalaXC::DeviceRuntimeSettings device_settings;
  device_settings.device_id = local_rank % device_count;
  device_settings.memory_fraction = 0.8 / ranks_per_device;
  MPI_Comm_free(&local_comm);

  const double displacement = 0.04 * color;
  const auto system = SkalaXC::test::make_rotated_h2_sto3g_system(displacement);
  Matrix scalar_density(2, 2);
  scalar_density << 0.5 + 0.02 * color, 0.5, 0.5, 0.5 - 0.02 * color;
  const Matrix spin_density = Matrix::Zero(2, 2);

  SkalaXC::RuntimeEnvironment host_runtime{subcomm};
  SkalaXC::RuntimeEnvironment device_runtime{subcomm, device_settings};
  const auto host =
      evaluate(host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
               system.basis, "PBE", scalar_density, spin_density, false);
  const auto device =
      evaluate(device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
               system.basis, "PBE", scalar_density, spin_density, true);
  const auto host_gradient = evaluate_gradient(
      host_runtime, SkalaXC::ExecutionSpace::Host, system.molecule,
      system.basis, "PBE", scalar_density, spin_density, false);
  const auto device_gradient = evaluate_gradient(
      device_runtime, SkalaXC::ExecutionSpace::Device, system.molecule,
      system.basis, "PBE", scalar_density, spin_density, true);

  const double exc_error = std::abs(std::get<0>(device) - std::get<0>(host)) /
                           std::max(1.0, std::abs(std::get<0>(host)));
  const double scalar_error = SkalaXC::test::matrix_error_per_basis(
      std::get<1>(device), std::get<1>(host));
  const double spin_error = SkalaXC::test::matrix_error_per_basis(
      std::get<2>(device), std::get<2>(host));
  REQUIRE(device_gradient.size() == host_gradient.size());
  double gradient_error = 0.0;
  for (std::size_t index = 0; index < host_gradient.size(); ++index)
    gradient_error = std::max(gradient_error, std::abs(device_gradient[index] -
                                                       host_gradient[index]));

  INFO("subcommunicator color=" << color);
  INFO("CUDA device=" << device_settings.device_id);
  CHECK(exc_error <= 1e-10);
  CHECK(scalar_error <= 1e-7);
  CHECK(spin_error <= 1e-10);
  CHECK(gradient_error <= 1e-6);
  MPI_Comm_free(&subcomm);
#else
  SKIP("MPI disabled");
#endif
}

/**
 * @file
 * @brief Implementation of CUDA SkalaXC energy, potential, and gradient work.
 */
#include "device/skala_device_driver.hpp"

#include "device/common/skala.hpp"
#include "device/common/skala_exc_grad.hpp"
#include "device/model_grid_exchange.hpp"
#include "device/xc_device_aos_data.hpp"
#include "exceptions.hpp"
#include "host/mpi_wrapper.hpp"
#include "host/skala_util.hpp"
#include "skala_model.hpp"

#include "device/local_device_work_driver.hpp"
#include "runtime_environment/device/device_backend.hpp"
#include "runtime_environment/device_specific/cuda_util.hpp"

#include <gauxc/basisset_map.hpp>
#include <gauxc/runtime_environment.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

namespace SkalaXC {

namespace {

/**
 * @brief Select a CUDA device for the calling host thread.
 * @param device_id CUDA device ordinal to activate.
 */
void activate_cuda_device(types::DeviceId device_id) {
  const auto status = cudaSetDevice(device_id.raw());
  if (status != cudaSuccess)
    SKALAXC_EXCEPTION(std::string("Failed to select CUDA device: ") +
                      cudaGetErrorString(status));
}

/** @brief Wrap GauXC's master stream as a non-owning LibTorch stream. */
c10::cuda::CUDAStream torch_stream_for_runtime(
    const GauXC::DeviceRuntimeEnvironment& runtime, types::DeviceId device_id) {
  auto* backend = runtime.device_backend();
  if (!backend) SKALAXC_EXCEPTION("Missing GauXC CUDA device backend");
  auto queue = backend->queue();
  auto* stream = queue.queue_as_ptr<GauXC::util::cuda_stream>();
  if (!stream) SKALAXC_EXCEPTION("GauXC master queue is not a CUDA stream");
  return c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(*stream),
                                          device_id.raw());
}

/**
 * @brief Map model features to GauXC's XC approximation level.
 * @param model Loaded Skala model to inspect.
 * @return LDA, GGA, or kinetic-density meta-GGA approximation.
 */
GauXC::integrator_xc_approx model_approximation(const SkalaModel& model) {
  if (model.is_mgga()) return GauXC::MGGA_TAU;
  if (model.is_gga()) return GauXC::GGA;
  return GauXC::LDA;
}

/**
 * @brief Check that an Eigen view is a dense column-major AO matrix.
 * @tparam Matrix Eigen matrix or map type.
 * @param matrix Matrix view to validate.
 * @param basis_size Required row and column count.
 * @return `true` when dimensions and strides satisfy the device contract.
 */
template <typename Matrix>
bool valid_ao_matrix(const Matrix& matrix, Eigen::Index basis_size) {
  return matrix.rows() == basis_size && matrix.cols() == basis_size &&
         matrix.innerStride() == 1 && matrix.outerStride() == basis_size;
}

/** @brief Deferred device validity check and its diagnostic label. */
struct DeviceFiniteCheck {
  std::string label;
  at::Tensor result;
};

/** @brief Combine deferred validity checks without reading device scalars. */
at::Tensor aggregate_finite_checks(
    const std::vector<DeviceFiniteCheck>& checks) {
  if (checks.empty()) return {};
  std::vector<at::Tensor> results;
  results.reserve(checks.size());
  for (const auto& check : checks) results.push_back(check.result);
  return torch::stack(results).all();
}

/** @brief Report a failed validity check after the device stream is fenced. */
void validate_finite_checks(const std::vector<DeviceFiniteCheck>& checks,
                            const at::Tensor& aggregate) {
  if (!aggregate.defined() || aggregate.item<bool>()) return;
  for (const auto& check : checks)
    if (!check.result.item<bool>())
      SKALAXC_EXCEPTION("Non-finite CUDA " + check.label);
  SKALAXC_EXCEPTION("Non-finite CUDA model value");
}

/** @brief Evaluate and validate one CUDA model batch's integrated energy. */
at::Tensor evaluate_cuda_model_energy(const SkalaModel& model,
                                      const FeatureDict& features,
                                      types::DeviceId device_id,
                                      std::size_t batch_index) {
  const std::string batch =
      "model output in batch " + std::to_string(batch_index);
  return SkalaXC::evaluate_model_energy(
      model, features, c10::Device(c10::DeviceType::CUDA, device_id.raw()));
}

/** @brief Return one batch's validated contiguous `dE/dw` values. */
at::Tensor model_dE_dw(const FeatureDict& features, types::DeviceId device_id,
                       std::int64_t point_count, std::size_t batch_index) {
  const auto& weights = features.at(feat_map().at(SKALA_FEATURE::WEIGHTS));
  auto dE_dw = validated_model_gradient(
      weights, "CUDA model dE/dw in batch " + std::to_string(batch_index));
  validate_model_tensor(
      dE_dw, "CUDA model dE/dw in batch " + std::to_string(batch_index),
      c10::Device(c10::DeviceType::CUDA, device_id.raw()), torch::kFloat64,
      {point_count}, true);
  return dE_dw;
}

}  // namespace

SkalaDeviceDriver::SkalaDeviceDriver(
    const GauXC::LoadBalancer& weighted_lb,
    const std::vector<std::vector<double>>& raw_weights,
    const std::string& model, types::DeviceId device_id,
    double device_memory_fraction, TimingSettings timing_settings,
    DomainBatchMode batch_mode)
    : SkalaDriver(timing_settings, ExecutionSpace::Device,
                  types::CommunicatorRank{weighted_lb.runtime().comm_rank()},
                  types::CommunicatorSize{weighted_lb.runtime().comm_size()}),
      device_id_(device_id),
      lb_(weighted_lb),
      lwd_(GauXC::LocalWorkDriverFactory::make_local_work_driver(
          GauXC::ExecutionSpace::Device, "Default")) {
  activate_cuda_device(device_id_);
  const auto runtime = GauXC::detail::as_device_runtime(lb_.runtime());
  const c10::cuda::CUDAStreamGuard stream_guard(
      torch_stream_for_runtime(runtime, device_id_));
  if (!lb_.state().modified_weights_are_stored)
    SKALAXC_EXCEPTION("SkalaDeviceDriver requires weight-partitioned tasks");

  const auto model_device =
      c10::Device(c10::DeviceType::CUDA, device_id_.raw());
  {
    detail::HostTimingScope timer(diagnostics_, TimingMetric::ModelLoad);
    model_ = std::make_unique<SkalaModel>(model, lb_.runtime(), model_device);
  }
  auto& tasks = lb_.get_tasks();
  model_grid_exchange_ = std::make_unique<DeviceModelGridExchange>(
      tasks,
      types::AtomCount{static_cast<std::uint64_t>(lb_.molecule().natoms())},
      lb_.runtime(), raw_weights, device_id_, batch_mode);
  const auto& local_batches = model_grid_exchange_->local_batches();
  set_setup_diagnostics(types::CommunicatorSize{lb_.runtime().comm_size()},
                        device_id_, device_memory_fraction, batch_mode, tasks,
                        local_batches);
  log_setup(model, model_->feature_keys(), model_->is_gga(), model_->is_mgga(),
            local_batches);
}

SkalaDeviceDriver::~SkalaDeviceDriver() noexcept = default;

double SkalaDeviceDriver::eval_exc_vxc_uks(
    ConstColMajorMatrixMap scalar_density, ConstColMajorMatrixMap spin_density,
    ColMajorMatrixMap scalar_potential, ColMajorMatrixMap spin_potential) {
  activate_cuda_device(device_id_);
  auto runtime = GauXC::detail::as_device_runtime(lb_.runtime());
  const c10::cuda::CUDAStreamGuard stream_guard(
      torch_stream_for_runtime(runtime, device_id_));
  diagnostics_.increment_exc_vxc_calls();
  const auto& basis = lb_.basis();
  const Eigen::Index basis_size = basis.nbf();
  if (!valid_ao_matrix(scalar_density, basis_size) ||
      !valid_ao_matrix(spin_density, basis_size) ||
      !valid_ao_matrix(scalar_potential, basis_size) ||
      !valid_ao_matrix(spin_potential, basis_size))
    SKALAXC_EXCEPTION(
        "UKS density and potential matrices must be dense nbf x nbf "
        "column-major views");
  log_evaluation_start("exc_vxc", scalar_density, spin_density);

  auto* device_lwd = dynamic_cast<GauXC::LocalDeviceWorkDriver*>(lwd_.get());
  if (!device_lwd) SKALAXC_EXCEPTION("Expected a LocalDeviceWorkDriver");
  auto device_data = device_lwd->create_device_data(runtime);
  auto* aos_data = dynamic_cast<GauXC::XCDeviceAoSData*>(device_data.get());
  if (!aos_data) SKALAXC_EXCEPTION("Expected Scheme1 CUDA device data");

  GauXC::integrator_term_tracker terms;
  terms.exc_vxc = true;
  terms.ks_scheme = GauXC::UKS;
  terms.xc_approx = model_approximation(*model_);
  const bool is_mgga = model_->is_mgga();
  const bool needs_gradient = model_->is_gga() || is_mgga;

  auto& tasks = lb_.get_tasks();
  GauXC::BasisSetMap basis_map(basis, lb_.molecule());
  device_data->populate_submat_maps(basis.nbf(), tasks.begin(), tasks.end(),
                                    basis_map);
  device_data->reset_allocations();
  device_data->allocate_static_data_exc_vxc(basis.nbf(), basis.nshells(), terms,
                                            true);
  device_data->send_static_data_density_basis(
      scalar_density.data(), basis.nbf(), spin_density.data(), basis.nbf(),
      nullptr, 0, nullptr, 0, basis);
  device_data->zero_exc_vxc_integrands(terms);

  const auto model_device =
      c10::Device(c10::DeviceType::CUDA, device_id_.raw());
  FeatureDict local_features = model_grid_exchange_->prepare_features(
      model_->feature_keys(), model_device);

  auto task_it = tasks.begin();
  while (task_it != tasks.end()) {
    const auto batch_begin = task_it;
    task_it =
        device_data->generate_buffers(terms, basis_map, task_it, tasks.end());
    if (needs_gradient)
      device_lwd->eval_collocation_gradient(device_data.get());
    else
      device_lwd->eval_collocation(device_data.get());

    const bool need_xmat_gradient = is_mgga;
    for (const auto density : {GauXC::DEN_S, GauXC::DEN_Z}) {
      device_lwd->eval_xmat(1.0, device_data.get(), need_xmat_gradient,
                            density);
      if (is_mgga)
        device_lwd->eval_vvars_mgga(device_data.get(), density, false);
      else if (model_->is_gga())
        device_lwd->eval_vvars_gga(device_data.get(), density);
      else
        device_lwd->eval_vvars_lda(device_data.get(), density);
    }
    if (is_mgga)
      device_lwd->eval_uvars_mgga(device_data.get(), GauXC::UKS, false);
    else if (model_->is_gga())
      device_lwd->eval_uvars_gga(device_data.get(), GauXC::UKS);
    else
      device_lwd->eval_uvars_lda(device_data.get(), GauXC::UKS);

    model_grid_exchange_->pack_post_uvars_features(
        *aos_data,
        types::TaskIndex{static_cast<std::size_t>(
            std::distance(tasks.begin(), batch_begin))},
        local_features);
  }

  FeatureDict local_potentials = model_grid_exchange_->prepare_local_potentials(
      needs_gradient, is_mgga, model_device);
  at::Tensor exc_value_device = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).device(model_device));
  std::vector<DeviceFiniteCheck> finite_checks;
  std::size_t batch_index = 0;
  for (const auto& batch : model_grid_exchange_->local_batches()) {
    FeatureDict features = model_grid_exchange_->prepare_local_batch_features(
        batch, local_features, lb_.molecule(), model_->feature_keys(),
        model_device, false);
    for (const auto& item : features) {
      const auto& tensor = item.value();
      if (tensor.is_floating_point())
        finite_checks.push_back({"model feature '" + item.key() +
                                     "' in batch " +
                                     std::to_string(batch_index),
                                 model_tensor_finite_check(tensor)});
    }
    auto exc =
        evaluate_cuda_model_energy(*model_, features, device_id_, batch_index);
    finite_checks.push_back(
        {"model energy in batch " + std::to_string(batch_index),
         model_tensor_finite_check(exc)});
    exc.backward();
    for (const auto& item : features) {
      const auto gradient = item.value().grad();
      if (gradient.defined())
        finite_checks.push_back({"model gradient for feature '" + item.key() +
                                     "' in batch " +
                                     std::to_string(batch_index),
                                 model_tensor_finite_check(gradient)});
    }
    {
      at::NoGradGuard no_grad;
      exc_value_device.add_(exc.detach());
    }
    model_grid_exchange_->store_local_batch_potentials(
        batch, needs_gradient, is_mgga, features, local_potentials);
    diagnostics_.record_model_batch(types::DomainCount{batch.atoms.size()});
    ++batch_index;
  }
  const at::Tensor all_model_values_finite =
      aggregate_finite_checks(finite_checks);

  task_it = tasks.begin();
  while (task_it != tasks.end()) {
    const auto batch_begin = task_it;
    task_it =
        device_data->generate_buffers(terms, basis_map, task_it, tasks.end());
    model_grid_exchange_->unpack_potentials(
        *aos_data,
        types::TaskIndex{static_cast<std::size_t>(
            std::distance(tasks.begin(), batch_begin))},
        local_potentials);
    if (needs_gradient)
      device_lwd->eval_collocation_gradient(device_data.get());
    else
      device_lwd->eval_collocation(device_data.get());

    std::int32_t max_points = 0;
    std::int32_t max_basis = 0;
    for (const auto& task : aos_data->host_device_tasks) {
      max_points = std::max(max_points, static_cast<std::int32_t>(task.npts));
      max_basis = std::max(max_basis,
                           static_cast<std::int32_t>(task.bfn_screening.nbe));
    }
    for (const auto density : {GauXC::DEN_S, GauXC::DEN_Z}) {
      zmat_skala_vxc(aos_data->host_device_tasks.size(), max_basis, max_points,
                     aos_data->aos_stack.device_tasks, terms.xc_approx, density,
                     device_data->queue());
      if (is_mgga)
        device_lwd->eval_mmat_mgga_vxc(device_data.get(), GauXC::UKS, false,
                                       density);
      device_lwd->inc_vxc(device_data.get(), density, is_mgga);
    }
  }

  device_lwd->symmetrize_vxc(device_data.get(), GauXC::DEN_S);
  device_lwd->symmetrize_vxc(device_data.get(), GauXC::DEN_Z);

  double device_exc = 0.0;
  double electron_count = 0.0;
  device_data->retrieve_exc_vxc_integrands(
      &device_exc, &electron_count, scalar_potential.data(), basis.nbf(),
      spin_potential.data(), basis.nbf(), nullptr, 0, nullptr, 0);
  runtime.device_backend()->master_queue_synchronize();
  validate_finite_checks(finite_checks, all_model_values_finite);
  double exc_value = exc_value_device.item<double>();
#ifdef GAUXC_HAS_MPI
  if (lb_.runtime().comm_size() > 1) {
    mpi::allreduce_sum(scalar_potential, lb_.runtime());
    mpi::allreduce_sum(spin_potential, lb_.runtime());
    std::vector<double> exc_values{exc_value};
    mpi::allreduce_sum(exc_values, lb_.runtime());
    exc_value = exc_values.front();
  }
#endif
  log_exc_vxc_result("exc_vxc", exc_value, scalar_potential, spin_potential);
  log_device_timing_unavailable("exc_vxc");
  return exc_value;
}

void SkalaDeviceDriver::eval_exc_grad_uks(ConstColMajorMatrixMap scalar_density,
                                          ConstColMajorMatrixMap spin_density,
                                          RowMajorMatrixMap gradient) {
  activate_cuda_device(device_id_);
  auto runtime = GauXC::detail::as_device_runtime(lb_.runtime());
  const c10::cuda::CUDAStreamGuard stream_guard(
      torch_stream_for_runtime(runtime, device_id_));
  diagnostics_.increment_exc_gradient_calls();
  const auto& basis = lb_.basis();
  const Eigen::Index basis_size = basis.nbf();
  if (!valid_ao_matrix(scalar_density, basis_size) ||
      !valid_ao_matrix(spin_density, basis_size) ||
      gradient.rows() != static_cast<Eigen::Index>(lb_.molecule().size()) ||
      gradient.cols() != direction_dimension || gradient.innerStride() != 1 ||
      gradient.outerStride() != direction_dimension)
    SKALAXC_EXCEPTION("Invalid density matrix or atom-major gradient view");
  log_evaluation_start("exc_gradient", scalar_density, spin_density);

  auto* device_lwd = dynamic_cast<GauXC::LocalDeviceWorkDriver*>(lwd_.get());
  if (!device_lwd) SKALAXC_EXCEPTION("Expected a LocalDeviceWorkDriver");
  auto device_data = device_lwd->create_device_data(runtime);
  auto* aos_data = dynamic_cast<GauXC::XCDeviceAoSData*>(device_data.get());
  if (!aos_data) SKALAXC_EXCEPTION("Expected Scheme1 CUDA device data");

  GauXC::integrator_term_tracker terms;
  terms.exc_grad = true;
  terms.weights = true;
  terms.ks_scheme = GauXC::UKS;
  terms.xc_approx = model_approximation(*model_);
  const bool is_mgga = model_->is_mgga();
  const bool needs_gradient = model_->is_gga() || is_mgga;

  auto& tasks = lb_.get_tasks();
  GauXC::BasisSetMap basis_map(basis, lb_.molecule());
  device_data->populate_submat_maps(basis.nbf(), tasks.begin(), tasks.end(),
                                    basis_map);
  device_data->reset_allocations();
  device_data->allocate_static_data_exc_grad(basis.nbf(), basis.nshells(),
                                             lb_.molecule().size(), terms);
  device_data->send_static_data_density_basis(
      scalar_density.data(), basis.nbf(), spin_density.data(), basis.nbf(),
      nullptr, 0, nullptr, 0, basis);
  device_data->allocate_static_data_weights(lb_.molecule().size());
  device_data->send_static_data_weights(lb_.molecule(), lb_.molmeta());
  device_data->zero_exc_grad_integrands();

  const auto model_device =
      c10::Device(c10::DeviceType::CUDA, device_id_.raw());
  FeatureDict local_features = model_grid_exchange_->prepare_features(
      model_->feature_keys(), model_device);
  auto task_it = tasks.begin();
  while (task_it != tasks.end()) {
    const auto batch_begin = task_it;
    task_it =
        device_data->generate_buffers(terms, basis_map, task_it, tasks.end());
    if (needs_gradient)
      device_lwd->eval_collocation_hessian(device_data.get());
    else
      device_lwd->eval_collocation_gradient(device_data.get());

    const bool need_xmat_gradient = needs_gradient;
    for (const auto density : {GauXC::DEN_S, GauXC::DEN_Z}) {
      device_lwd->eval_xmat(1.0, device_data.get(), need_xmat_gradient,
                            density);
      if (is_mgga)
        device_lwd->eval_vvars_mgga(device_data.get(), density, false);
      else if (model_->is_gga())
        device_lwd->eval_vvars_gga(device_data.get(), density);
      else
        device_lwd->eval_vvars_lda(device_data.get(), density);
    }
    if (is_mgga)
      device_lwd->eval_uvars_mgga(device_data.get(), GauXC::UKS, false);
    else if (model_->is_gga())
      device_lwd->eval_uvars_gga(device_data.get(), GauXC::UKS);
    else
      device_lwd->eval_uvars_lda(device_data.get(), GauXC::UKS);
    model_grid_exchange_->pack_post_uvars_features(
        *aos_data,
        types::TaskIndex{static_cast<std::size_t>(
            std::distance(tasks.begin(), batch_begin))},
        local_features);
  }

  FeatureDict local_potentials = model_grid_exchange_->prepare_local_potentials(
      needs_gradient, is_mgga, model_device);
  at::Tensor local_dE_dw =
      model_grid_exchange_->prepare_local_dE_dw(model_device);
  std::vector<DeviceFiniteCheck> finite_checks;
  const auto& local_batches = model_grid_exchange_->local_batches();
  for (std::size_t batch_index = 0; batch_index < local_batches.size();
       ++batch_index) {
    const auto& batch = local_batches[batch_index];
    FeatureDict features = model_grid_exchange_->prepare_local_batch_features(
        batch, local_features, lb_.molecule(), model_->feature_keys(),
        model_device, true);
    auto energy =
        evaluate_cuda_model_energy(*model_, features, device_id_, batch_index);
    finite_checks.push_back(
        {"model energy in batch " + std::to_string(batch_index),
         model_tensor_finite_check(energy)});
    energy.backward();
    for (const auto& item : features) {
      const auto& tensor = item.value();
      if (tensor.is_floating_point())
        finite_checks.push_back({"model feature '" + item.key() +
                                     "' in batch " +
                                     std::to_string(batch_index),
                                 model_tensor_finite_check(tensor)});
      const auto gradient = tensor.grad();
      if (gradient.defined())
        finite_checks.push_back({"model gradient for feature '" + item.key() +
                                     "' in batch " +
                                     std::to_string(batch_index),
                                 model_tensor_finite_check(gradient)});
    }
    auto dE_dw =
        model_dE_dw(features, device_id_, batch.point_count.raw(), batch_index);
    finite_checks.push_back(
        {"model dE/dw in batch " + std::to_string(batch_index),
         model_tensor_finite_check(dE_dw)});
    model_grid_exchange_->store_local_batch_potentials(
        batch, needs_gradient, is_mgga, features, local_potentials);
    model_grid_exchange_->store_local_batch_dE_dw(batch, dE_dw, local_dE_dw);
    model_grid_exchange_->accumulate_local_geometry_gradients(
        batch_index, features, *aos_data);
    diagnostics_.record_model_batch(types::DomainCount{batch.atoms.size()});
  }
  const at::Tensor all_model_values_finite =
      aggregate_finite_checks(finite_checks);

  task_it = tasks.begin();
  while (task_it != tasks.end()) {
    const auto batch_begin = task_it;
    task_it =
        device_data->generate_buffers(terms, basis_map, task_it, tasks.end());
    const types::TaskIndex first_task{
        static_cast<std::size_t>(std::distance(tasks.begin(), batch_begin))};
    model_grid_exchange_->unpack_potentials(*aos_data, first_task,
                                            local_potentials);
    if (needs_gradient)
      device_lwd->eval_collocation_hessian(device_data.get());
    else
      device_lwd->eval_collocation_gradient(device_data.get());

    for (const auto density : {GauXC::DEN_S, GauXC::DEN_Z}) {
      device_lwd->eval_xmat(1.0, device_data.get(), needs_gradient, density);
      device_lwd->save_xmat(device_data.get(), needs_gradient, density);
    }

    std::int32_t max_points = 0;
    for (const auto& task : aos_data->host_device_tasks)
      max_points = std::max(max_points, static_cast<std::int32_t>(task.npts));
    if (needs_gradient)
      transform_skala_vxc_for_grad(aos_data->host_device_tasks.size(),
                                   max_points, aos_data->aos_stack.device_tasks,
                                   device_data->queue());

    if (is_mgga)
      device_lwd->inc_exc_grad_mgga(device_data.get(), GauXC::UKS, false, true);
    else if (model_->is_gga())
      device_lwd->inc_exc_grad_gga(device_data.get(), GauXC::UKS, true);
    else
      device_lwd->inc_exc_grad_lda(device_data.get(), GauXC::UKS, true);

    model_grid_exchange_->prepare_weight_derivatives(*aos_data, first_task,
                                                     local_dE_dw);
    device_lwd->eval_weight_1st_deriv_contracted(device_data.get(),
                                                 lb_.state().weight_alg);
  }

  gradient.setZero();
  double electron_count = 0.0;
  device_data->retrieve_exc_grad_integrands(gradient.data(), &electron_count);
  runtime.device_backend()->master_queue_synchronize();
  validate_finite_checks(finite_checks, all_model_values_finite);
#ifdef GAUXC_HAS_MPI
  if (lb_.runtime().comm_size() > 1)
    mpi::allreduce_sum(gradient, lb_.runtime());
#endif
  log_gradient_result("exc_gradient", gradient);
  log_device_timing_unavailable("exc_gradient");
}

}  // namespace SkalaXC

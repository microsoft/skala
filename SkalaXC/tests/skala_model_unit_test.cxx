#include <catch2/catch.hpp>

#include "host/skala_util.hpp"
#include "skala_model.hpp"

#include <gauxc/runtime_environment.hpp>
#include <skalaxc/skalaxc_config.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

class TempModelDirectory {
 public:
  TempModelDirectory() {
    std::mt19937_64 random(std::random_device{}());
    do {
      path_ = std::filesystem::temp_directory_path() /
              ("skalaxc-model-test-" + std::to_string(random()));
    } while (!std::filesystem::create_directory(path_));
  }

  ~TempModelDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void save_with_metadata(const torch::jit::script::Module& module,
                        const std::filesystem::path& path,
                        const std::string& protocol_version,
                        const std::string& features) {
  const torch::jit::ExtraFilesMap extra_files{
      {"protocol_version", protocol_version}, {"features", features}};
  module.save(path.string(), extra_files);
}

void create_file(const std::filesystem::path& path) {
  std::ofstream output(path);
  output << "test";
}

FeatureDict make_model_features(const std::vector<std::string>& feature_keys,
                                const c10::Device& device) {
  constexpr std::int64_t atom_count = 2;
  constexpr std::int64_t grid_size = 4;
  constexpr std::int64_t point_count = atom_count * grid_size;
  const auto double_options =
      torch::TensorOptions().dtype(torch::kFloat64).device(device);
  const auto integer_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  FeatureDict features;

  for (const auto& key : feature_keys) {
    at::Tensor tensor;
    switch (SkalaXC::reverse_feat_map().at(key)) {
      case SkalaXC::SKALA_FEATURE::DEN:
        tensor = torch::full({2, point_count}, 0.4, double_options);
        break;
      case SkalaXC::SKALA_FEATURE::DDEN:
        tensor = torch::full({2, 3, point_count}, 0.05, double_options);
        break;
      case SkalaXC::SKALA_FEATURE::TAU:
        tensor = torch::full({2, point_count}, 0.2, double_options);
        break;
      case SkalaXC::SKALA_FEATURE::POINTS:
        tensor = torch::arange(point_count * 3, double_options)
                     .reshape({point_count, 3})
                     .mul(0.01)
                     .add(0.2);
        break;
      case SkalaXC::SKALA_FEATURE::WEIGHTS:
        tensor = torch::linspace(0.1, 0.8, point_count, double_options);
        tensor.requires_grad_(true);
        break;
      case SkalaXC::SKALA_FEATURE::COORDS:
        tensor = torch::zeros({atom_count, 3}, double_options);
        break;
      case SkalaXC::SKALA_FEATURE::ATOMIC_GRID_WEIGHTS:
        tensor = torch::full({point_count}, 0.25, double_options);
        break;
      case SkalaXC::SKALA_FEATURE::ATOMIC_GRID_SIZES:
        tensor = torch::full({atom_count}, grid_size, integer_options);
        break;
      case SkalaXC::SKALA_FEATURE::ATOMIC_GRID_SIZE_BOUND_SHAPE:
        tensor = torch::zeros({grid_size, 0}, integer_options);
        break;
      default:
        FAIL("Unexpected bundled model feature " << key);
    }
    features.insert(key, std::move(tensor));
  }
  return features;
}

void check_integrated_energy(const std::filesystem::path& path,
                             const c10::Device& device) {
  SkalaXC::SkalaModel model(path.string(), device);
  auto features = make_model_features(model.feature_keys(), device);
  const auto weights =
      features.at(SkalaXC::feat_map().at(SkalaXC::SKALA_FEATURE::WEIGHTS));
  auto energy = SkalaXC::evaluate_model_energy(model, features, device);

  REQUIRE(energy.defined());
  REQUIRE(energy.numel() == 1);
  REQUIRE(energy.scalar_type() == torch::kFloat64);
  CHECK(std::isfinite(energy.item<double>()));

  energy.backward();
  const auto dE_dw = weights.grad();
  REQUIRE(dE_dw.defined());
  REQUIRE(dE_dw.sizes() == weights.sizes());
  CHECK(dE_dw.isfinite().all().item<bool>());
}

}  // namespace

TEST_CASE("Model tensor validation rejects malformed boundary values",
          "[skala][model-validation]") {
  const c10::Device cpu(c10::DeviceType::CPU);
  const auto doubles = torch::TensorOptions().dtype(torch::kFloat64);

  CHECK_THROWS_WITH(SkalaXC::validate_model_tensor({}, "test tensor", cpu,
                                                   torch::kFloat64, {2}),
                    Catch::Contains("Undefined test tensor"));

  const auto wrong_type = torch::zeros({2}, torch::kFloat32);
  CHECK_THROWS_WITH(SkalaXC::validate_model_tensor(wrong_type, "test tensor",
                                                   cpu, torch::kFloat64, {2}),
                    Catch::Contains("wrong dtype"));

  const auto wrong_shape = torch::zeros({3}, doubles);
  CHECK_THROWS_WITH(SkalaXC::validate_model_tensor(wrong_shape, "test tensor",
                                                   cpu, torch::kFloat64, {2}),
                    Catch::Contains("invalid dimensions"));

  const auto nonscalar_energy = torch::zeros({1}, doubles).requires_grad_(true);
  CHECK_THROWS_WITH(SkalaXC::validate_model_tensor(
                        nonscalar_energy, "integrated model energy", cpu,
                        torch::kFloat64, {}, false, true),
                    Catch::Contains("invalid dimensions"));

  const auto noncontiguous = torch::zeros({2, 3}, doubles).transpose(0, 1);
  CHECK_THROWS_WITH(SkalaXC::validate_model_tensor(noncontiguous, "test tensor",
                                                   cpu, torch::kFloat64,
                                                   noncontiguous.sizes(), true),
                    Catch::Contains("must be contiguous"));

  const auto detached = torch::zeros({}, doubles);
  CHECK_THROWS_WITH(
      SkalaXC::validate_model_tensor(detached, "integrated model energy", cpu,
                                     torch::kFloat64, {}, false, true),
      Catch::Contains("not connected to autograd"));

  CHECK_THROWS_WITH(
      SkalaXC::validate_model_tensor_finite(
          torch::full({1}, std::numeric_limits<double>::quiet_NaN(), doubles),
          "test tensor"),
      Catch::Contains("Non-finite test tensor"));
  CHECK_THROWS_WITH(
      SkalaXC::validate_model_tensor_finite(
          torch::full({1}, std::numeric_limits<double>::infinity(), doubles),
          "test tensor"),
      Catch::Contains("Non-finite test tensor"));

#ifdef SKALAXC_HAS_CUDA
  if (torch::cuda::is_available()) {
    const c10::Device cuda(c10::DeviceType::CUDA, 0);
    const auto device_tensor = torch::ones({2}, doubles.device(cuda));
    CHECK_THROWS_WITH(
        SkalaXC::validate_model_tensor(device_tensor, "test tensor", cpu,
                                       torch::kFloat64, {2}),
        Catch::Contains("wrong device"));

    const auto deferred = SkalaXC::model_tensor_finite_check(device_tensor);
    CHECK(deferred.device() == cuda);
    CHECK(deferred.scalar_type() == torch::kBool);
    CHECK(deferred.numel() == 1);
  }
#endif
}

TEST_CASE("Model gradient validation preserves the feature contract",
          "[skala][model-validation]") {
  auto feature = torch::ones({2, 3}, torch::kFloat64).requires_grad_(true);
  CHECK_THROWS_WITH(SkalaXC::validated_model_gradient(feature, "test gradient"),
                    Catch::Contains("Undefined test gradient"));

  feature.square().sum().backward();
  const auto gradient =
      SkalaXC::validated_model_gradient(feature, "test gradient");
  CHECK(gradient.sizes() == feature.sizes());
  CHECK(gradient.scalar_type() == feature.scalar_type());
  CHECK(gradient.device() == feature.device());
  CHECK(gradient.is_contiguous());
  CHECK(SkalaXC::model_tensor_finite_check(gradient).item<bool>());
}

TEST_CASE("Model return values must be tensors", "[skala][model-validation]") {
  torch::jit::script::Module module("NonTensorEnergy");
  module.define(R"JIT(
def forward(self, mol: Dict[str, Tensor]) -> int:
  return 1
)JIT");
  FeatureDict features;
  CHECK_THROWS_WITH(SkalaXC::get_exc(module.get_method("forward"), features),
                    Catch::Contains("must be a tensor"));
}

TEST_CASE("Model resolution uses explicit and configured paths",
          "[skala][model-path]") {
  TempModelDirectory temporary;
  const auto installed = temporary.path() / "installed";
  std::filesystem::create_directory(installed);

  const auto explicit_path = temporary.path() / "explicit.fun";
  create_file(explicit_path);
  REQUIRE(SkalaXC::detail::resolve_model_path(
              explicit_path.string(), installed) == explicit_path.string());

  create_file(installed / "pbe.fun");
  REQUIRE(SkalaXC::detail::resolve_model_path("PBE", installed) ==
          (installed / "pbe.fun").string());

  create_file(installed / "custom.fun");
  REQUIRE(SkalaXC::detail::resolve_model_path("custom.fun", installed) ==
          (installed / "custom.fun").string());

  try {
    (void)SkalaXC::detail::resolve_model_path("missing.fun", installed);
    FAIL("missing model resolution succeeded");
  } catch (const std::exception& error) {
    const std::string message = error.what();
    REQUIRE(message.find((installed / "missing.fun").string()) !=
            std::string::npos);
  }

  CHECK_THROWS_WITH(SkalaXC::detail::resolve_model_path("SKALA", installed),
                    Catch::Contains("specify a local checkpoint path"));
}

TEST_CASE("Skala model loading validates paths and metadata",
          "[skala][model-loading]") {
  const std::filesystem::path valid_path =
      std::filesystem::path(SKALAXC_MODEL_PATH) / "pbe.fun";
  SkalaXC::SkalaModel valid_model(valid_path.string());
  CHECK(valid_model.is_gga());
  CHECK_FALSE(valid_model.is_mgga());
  CHECK_FALSE(valid_model.feature_keys().empty());
  CHECK_NOTHROW(valid_model.energy_function());

  const std::filesystem::path skala_path =
      std::filesystem::path(SKALAXC_MODEL_PATH) / "skala-1.1.fun";
  SkalaXC::SkalaModel skala_model(skala_path.string());
  CHECK_FALSE(skala_model.is_gga());
  CHECK(skala_model.is_mgga());
  CHECK_FALSE(skala_model.feature_keys().empty());
  CHECK_NOTHROW(skala_model.energy_function());

  TempModelDirectory temporary;
  CHECK_THROWS(
      SkalaXC::SkalaModel((temporary.path() / "missing.fun").string()));

  const auto corrupt_path = temporary.path() / "corrupt.fun";
  {
    std::ofstream corrupt(corrupt_path);
    corrupt << "not a TorchScript archive";
  }
  CHECK_THROWS(SkalaXC::SkalaModel(corrupt_path.string()));

  torch::jit::ExtraFilesMap source_metadata{{"protocol_version", ""},
                                            {"features", ""}};
  const auto module =
      torch::jit::load(valid_path.string(), torch::kCPU, source_metadata);

  const auto unsupported_protocol = temporary.path() / "protocol.fun";
  save_with_metadata(module, unsupported_protocol, "1",
                     source_metadata.at("features"));
  CHECK_THROWS(SkalaXC::SkalaModel(unsupported_protocol.string()));

  const auto malformed_features = temporary.path() / "malformed-features.fun";
  save_with_metadata(module, malformed_features, "2", "{}");
  CHECK_THROWS(SkalaXC::SkalaModel(malformed_features.string()));

  const auto unsupported_features =
      temporary.path() / "unsupported-features.fun";
  save_with_metadata(module, unsupported_features, "2",
                     R"(["unsupported_feature"])");
  CHECK_THROWS(SkalaXC::SkalaModel(unsupported_features.string()));
}

TEST_CASE("Bundled models expose integrated energy and dE/dw",
          "[skala][model-integrated-energy]") {
  const auto model_directory = std::filesystem::path(SKALAXC_MODEL_PATH);
  std::vector<std::string> filenames{"ldax.fun", "pbe.fun", "tpss.fun",
                                     "skala-1.1.fun"};
#ifdef SKALAXC_HAS_CUDA
  filenames.emplace_back("skala-1.1-cuda.fun");
#endif
  for (const auto& filename : filenames) {
    DYNAMIC_SECTION(filename) {
      const c10::Device device = filename == "skala-1.1-cuda.fun"
                                     ? c10::Device(c10::DeviceType::CUDA, 0)
                                     : c10::Device(c10::DeviceType::CPU);
      check_integrated_energy(model_directory / filename, device);
    }
  }
}

TEST_CASE("Protocol-v2 models require integrated energy",
          "[skala][model-integrated-energy]") {
  TempModelDirectory temporary;
  torch::jit::script::Module module("MissingIntegratedEnergyFunctional");
  module.define(R"JIT(
def forward(self, mol: Dict[str, Tensor]) -> Tensor:
    return mol["density"].sum(0)
)JIT");
  const auto path = temporary.path() / "missing-integrated-energy.fun";
  save_with_metadata(module, path, "2", R"(["density", "grid_weights"])");

  CHECK_THROWS(SkalaXC::SkalaModel(path.string()));
}

TEST_CASE("Runtime rank zero broadcasts the model archive",
          "[skala][mpi][model-broadcast][mpi-only]") {
#ifdef GAUXC_HAS_MPI
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm runtime_communicator = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank,
                 &runtime_communicator);
  GauXC::RuntimeEnvironment runtime(runtime_communicator);
  const std::filesystem::path valid_path =
      std::filesystem::path(SKALAXC_MODEL_PATH) / "pbe.fun";

  SECTION("non-root selectors are not resolved") {
    const std::string selector = runtime.comm_rank() == 0
                                     ? valid_path.string()
                                     : "/non-root-must-not-read.fun";
    const SkalaXC::SkalaModel model(selector, runtime);
    CHECK(model.is_gga());
    CHECK_FALSE(model.feature_keys().empty());
  }

  SECTION("rank-zero read errors reach every rank") {
    const std::string selector = runtime.comm_rank() == 0
                                     ? "/rank-zero-missing-model.fun"
                                     : valid_path.string();
    CHECK_THROWS(SkalaXC::SkalaModel(selector, runtime));
  }

  MPI_Comm_free(&runtime_communicator);
#else
  SUCCEED("MPI disabled");
#endif
}

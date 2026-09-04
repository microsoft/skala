#include "skala_model.hpp"

#include "exceptions.hpp"
#include "host/mpi_wrapper.hpp"
#include "host/skala_util.hpp"

#include <caffe2/serialize/in_memory_adapter.h>
#include <gauxc/runtime_environment.hpp>
#include <nlohmann/json.hpp>
#include <skalaxc/skalaxc_config.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>

namespace SkalaXC {

namespace detail {

std::string resolve_model_path(const std::string& model,
                               const std::filesystem::path& model_directory) {
  if (std::filesystem::exists(model)) return model;

  if (model == "SKALA")
    SKALAXC_EXCEPTION(
        "To use the Skala functional, specify a local checkpoint path.");

  std::filesystem::path filename = model;
  if (model == "PBE") filename = "pbe.fun";
  if (model == "TPSS") filename = "tpss.fun";
  if (model == "LDA") filename = "ldax.fun";

  const auto candidate = model_directory / filename;
  if (std::filesystem::exists(candidate)) return candidate.string();

  SKALAXC_EXCEPTION("Model " + model + " not found at " + candidate.string());
}

}  // namespace detail

namespace {

std::string read_model_archive(const std::string& model) {
  const char* model_path_override = std::getenv("SKALAXC_MODEL_PATH");
  const std::filesystem::path model_directory =
      model_path_override != nullptr && model_path_override[0] != '\0'
          ? model_path_override
          : SKALAXC_MODEL_PATH_INSTALL;
  const auto path = detail::resolve_model_path(model, model_directory);
  std::ifstream input(path, std::ios::binary);
  if (!input) SKALAXC_EXCEPTION("Unable to open model archive " + path);

  std::string archive((std::istreambuf_iterator<char>(input)),
                      std::istreambuf_iterator<char>());
  if (input.bad()) SKALAXC_EXCEPTION("Unable to read model archive " + path);
  return archive;
}

std::string broadcast_model_archive(const std::string& model,
                                    const GauXC::RuntimeEnvironment& runtime) {
  std::string archive;
  std::string read_error;
  if (runtime.comm_rank() == 0) {
    try {
      archive = read_model_archive(model);
    } catch (const std::exception& error) {
      read_error = error.what();
    } catch (...) {
      read_error = "unknown model archive read error";
    }
  }

  mpi::broadcast_string(read_error, runtime);
  if (!read_error.empty())
    SKALAXC_EXCEPTION("Runtime rank 0 could not read the Skala model: " +
                      read_error);
  mpi::broadcast_string(archive, runtime);
  return archive;
}

}  // namespace

/** @brief Private TorchScript module, metadata, and callable model method. */
struct SkalaModel::Impl {
  Impl(const std::string& archive, const c10::Device& device) {
    torch::jit::ExtraFilesMap extra_files{{"features", ""},
                                          {"protocol_version", ""}};
    try {
      if (archive.size() >
          static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
        SKALAXC_EXCEPTION("Skala model archive exceeds LibTorch limits");
      auto adapter = std::make_shared<caffe2::serialize::MemoryReadAdapter>(
          archive.data(), static_cast<off_t>(archive.size()));
      module = torch::jit::load(adapter, device, extra_files);
    } catch (const c10::Error& error) {
      SKALAXC_EXCEPTION("error loading skala model: " +
                        std::string(error.what()));
    }

    const auto version =
        nlohmann::json::parse(extra_files.at("protocol_version")).get<int>();
    if (version != 2)
      SKALAXC_EXCEPTION("Unsupported protocol version " +
                        std::to_string(version));

    const auto features = nlohmann::json::parse(extra_files.at("features"));
    if (!features.is_array()) SKALAXC_EXCEPTION("features is not an array");

    for (const auto& feature : features) {
      if (!feature.is_string()) SKALAXC_EXCEPTION("feature is not a string");
      feature_keys.push_back(feature.get<std::string>());
    }
    if (feature_keys.empty())
      SKALAXC_EXCEPTION("No feature keys found in model");

    for (const auto& key : feature_keys) {
      if (!valueExists(key))
        SKALAXC_EXCEPTION("Feature Key Required Not Implemented: " + key);
      if (key == feat_map().at(SKALA_FEATURE::TAU)) is_mgga = true;
      if (key == feat_map().at(SKALA_FEATURE::DDEN)) is_gga = true;
    }
    if (is_mgga) is_gga = false;

    module.eval();
    for (auto parameter : module.parameters())
      parameter.set_requires_grad(false);
    energy_func = module.find_method("get_exc");
    if (!energy_func)
      SKALAXC_EXCEPTION(
          "Model archive does not define the required get_exc method");
  }

  torch::jit::script::Module module;
  std::optional<torch::jit::Method> energy_func;
  std::vector<std::string> feature_keys;
  bool is_gga = false;
  bool is_mgga = false;
};

SkalaModel::SkalaModel(const std::string& model, c10::Device device)
    : pimpl_(std::make_unique<Impl>(read_model_archive(model), device)) {}

SkalaModel::SkalaModel(const std::string& model,
                       const GauXC::RuntimeEnvironment& runtime,
                       c10::Device device)
    : pimpl_(std::make_unique<Impl>(broadcast_model_archive(model, runtime),
                                    device)) {}

SkalaModel::~SkalaModel() noexcept = default;

const torch::jit::Method& SkalaModel::energy_function() const {
  if (!pimpl_->energy_func)
    SKALAXC_EXCEPTION("Model integrated-energy function is not initialized");
  return *pimpl_->energy_func;
}

const std::vector<std::string>& SkalaModel::feature_keys() const noexcept {
  return pimpl_->feature_keys;
}

bool SkalaModel::is_gga() const noexcept { return pimpl_->is_gga; }

bool SkalaModel::is_mgga() const noexcept { return pimpl_->is_mgga; }

}  // namespace SkalaXC
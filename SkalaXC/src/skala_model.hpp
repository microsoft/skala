#pragma once

#include <c10/core/Device.h>
#include <torch/script.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace GauXC {
class RuntimeEnvironment;
}

namespace SkalaXC {

namespace detail {

std::string resolve_model_path(const std::string& model,
                               const std::filesystem::path& model_directory);

}  // namespace detail

/** @brief Own a validated TorchScript XC model on one rank and device. */
class SkalaModel {
 public:
  /**
   * @brief Load a model directly on one device.
   * @param model Alias, relative model name, or archive path.
   * @param device Device on which the module executes.
   */
  explicit SkalaModel(const std::string& model,
                      c10::Device device = c10::Device(c10::DeviceType::CPU));
  /**
   * @brief Load a model collectively through a runtime communicator.
   * @param model Alias, relative model name, or archive path.
   * @param runtime Runtime environment defining the communicator.
   * @param device Device on which the rank-local module executes.
   */
  SkalaModel(const std::string& model, const GauXC::RuntimeEnvironment& runtime,
             c10::Device device = c10::Device(c10::DeviceType::CPU));
  ~SkalaModel() noexcept;

  SkalaModel(const SkalaModel&) = delete;
  SkalaModel& operator=(const SkalaModel&) = delete;

  /** @return Mandatory integrated XC energy method. */
  const torch::jit::Method& energy_function() const;

  /** @return Ordered model input feature keys. */
  const std::vector<std::string>& feature_keys() const noexcept;
  /** @return Whether the model requires density gradients. */
  bool is_gga() const noexcept;
  /** @return Whether the model requires kinetic density. */
  bool is_mgga() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace SkalaXC

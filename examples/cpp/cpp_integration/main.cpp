#include <torch/script.h>
#include <torch/csrc/autograd/autograd.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>

using json = nlohmann::json;
using IValueList = std::vector<c10::IValue>;
using IValueMap = std::unordered_map<std::string, c10::IValue>;
using FeatureDict = c10::Dict<std::string, at::Tensor>;

at::Tensor
load_feature(const std::string &filename, torch::DeviceType device)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open())
  {
    throw std::runtime_error("Failed to open feature file: " + filename);
  }
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  input.close();
  return torch::jit::pickle_load(bytes).toTensor().to(device);
}

FeatureDict
load_features(const std::string &prefix, const std::vector<std::string> &keys, torch::DeviceType device)
{
  FeatureDict featmap;
  for (const auto &key : keys)
  {
    featmap.insert(key, load_feature(prefix + "/" + key + ".pt", device));
  }
  return featmap;
}

std::tuple<torch::jit::Method, std::vector<std::string>>
load_model(const std::string &filename, torch::DeviceType device)
{
  torch::jit::script::Module mod;
  torch::jit::ExtraFilesMap extra_files{{"features", ""}, {"protocol_version", ""}};
  std::vector<std::string> keys;

  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(filename, device, extra_files);
  }
  catch (const c10::Error &e)
  {
    throw std::runtime_error("Error loading the model from " + filename + ": " + e.what());
  }

  auto version = json::parse(extra_files.at("protocol_version")).get<int>();
  if (version != 2)
  {
    throw std::runtime_error("Unsupported protocol version " + std::to_string(version));
  }

  auto features = json::parse(extra_files.at("features"));
  // check if features is array
  if (!features.is_array())
  {
    throw std::runtime_error("features is not an array");
  }
  for (const auto &feature : features)
  {
    if (!feature.is_string())
    {
      throw std::runtime_error("feature is not a string");
    }
    keys.push_back(feature.get<std::string>());
  }

  return std::make_tuple(mod.get_method("get_exc_density"), keys);
}

at::Tensor
get_exc(const torch::jit::Method &exc_func, const FeatureDict &features)
{
  IValueList args;
  IValueMap kwargs;
  kwargs["mol"] = features;
  return exc_func(args, kwargs).toTensor();
}

std::tuple<at::Tensor, c10::Dict<std::string, at::Tensor>>
get_exc_and_grad(const torch::jit::Method &exc_func, const FeatureDict &features)
{
  // Create a mutable copy only for the tensors that need gradients
  FeatureDict features_with_grad;
  std::vector<at::Tensor> input_tensors;
  std::vector<std::string> tensor_keys;

  for (const auto &kv : features)
  {
    auto tensor_with_grad = kv.value().clone().requires_grad_(true);
    features_with_grad.insert(kv.key(), tensor_with_grad);
    input_tensors.push_back(tensor_with_grad);
    tensor_keys.push_back(kv.key());
  }

  IValueList args;
  IValueMap kwargs;
  kwargs["mol"] = features_with_grad;

  auto exc_on_grid = exc_func(args, kwargs).toTensor();
  auto exc = (exc_on_grid * features_with_grad.at("grid_weights")).sum();

  auto gradients = torch::autograd::grad(
      {exc},                  // outputs
      input_tensors,          // inputs
      /*grad_outputs=*/{},    // grad_outputs (defaults to ones)
      /*retain_graph=*/false, // retain_graph, necessary for higher-order grads
      /*create_graph=*/false, // create_graph, necessary for higher-order grads
      /*allow_unused=*/true   // allow_unused
  );

  c10::Dict<std::string, at::Tensor> grad;
  for (size_t i = 0; i < tensor_keys.size(); ++i)
  {
    grad.insert(tensor_keys[i], gradients[i]);
  }

  return std::make_tuple(exc_on_grid, grad);
}

int main(int argc, const char *argv[])
{
  if (argc != 3)
  {
    std::cerr << "usage: skala_cpp_integration <path-to-fun-file> <feature-file-directory>\n";
    return -1;
  }

  const torch::DeviceType device = torch::kCPU;

  const auto [exc_func, feature_keys] = load_model(std::string(argv[1]), device);
  const auto features = load_features(std::string(argv[2]), feature_keys, device);

  std::cout << "Compute Exc..." << std::endl;

  const auto exc_on_grid = get_exc(exc_func, features);
  const auto exc = (exc_on_grid * features.at("grid_weights")).sum();

  std::cout << "Exc = " << exc.item() << std::endl;

  std::cout << "Compute Exc and dExc/dfeat..." << std::endl;

  const auto [exc_on_grid2, grad] = get_exc_and_grad(exc_func, features);
  const auto exc2 = (exc_on_grid2 * features.at("grid_weights")).sum();

  std::cout << "Exc = " << exc2.item() << std::endl;
  for (const auto &kv : grad)
  {
    std::cout << "|dExc/d(" << kv.key() << ")| = " << kv.value().norm().item() << std::endl;
  }

  return 0;
}

#include <torch/script.h>
#include <torch/csrc/autograd/autograd.h>

typedef void* torch_jit_script_module_t;
typedef void* torch_tensor_t;
typedef int torch_device_t;

typedef void* skala_dict_t;
typedef void* skala_list_t;

typedef c10::Dict<std::string, std::vector<at::Tensor>> SkalaDict;
typedef std::vector<at::Tensor> SkalaList;
typedef enum SkalaFeature {
  Feature_Density = 1,
  Feature_Grad = 2,
  Feature_Kin = 3,
  Feature_GridCoords = 4,
  Feature_GridWeights = 5,
  Feature_Coarse0AtomicCoords = 6
} SkalaFeature;

static inline
void
ctorch_error(const std::string &message,
             const std::function<void()> &cleanup = nullptr) {
  std::cerr << "[ERROR]: " << message << std::endl;
  if (cleanup) {
    cleanup(); // Perform cleanup actions
  }
  exit(EXIT_FAILURE);
}

extern "C"
torch_tensor_t
skala_tensor_load(const char* filename)
{
  std::ifstream input(std::string(filename), std::ios::binary);
  if (!input.is_open())
  {
    throw std::runtime_error("Failed to open feature file: " + std::string(filename));
  }
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  input.close();
  auto tensor = torch::jit::pickle_load(bytes).toTensor().to(torch::Device(torch::kCPU));
  return new torch::Tensor(std::move(tensor));
}

extern "C"
torch_tensor_t
skala_tensor_sum(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  auto sum = t->sum();
  return new torch::Tensor(std::move(sum));
}

extern "C"
double
skala_tensor_item_double(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->item<double>();
}

extern "C"
torch_jit_script_module_t
skala_model_load(const char *filename,
                 const bool requires_grad,
                 SkalaFeature* features) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::jit::ExtraFilesMap extra_files{{"features", ""}, {"protocol_version", ""}};
  torch::jit::script::Module *module = nullptr;
  try {
    module = new torch::jit::script::Module;
    *module =
        torch::jit::load(filename, torch::Device(torch::kCPU), extra_files);
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete module; });
  }

  if (std::stoi(extra_files.at("protocol_version")) != 2) {
    std::string message = "Unsupported protocol version " + extra_files.at("protocol_version");
    ctorch_error(message, [&]() { delete module; });
  }

  auto feature_str = extra_files.at("features");
  // formatted as ["feature1", "feature2", ...] parse without using a full json library
  std::unordered_map<std::string, SkalaFeature> feature_keys;
  size_t pos = 0;
  while ((pos = feature_str.find('"', pos)) != std::string::npos) {
    size_t end_pos = feature_str.find('"', pos + 1);
    if (end_pos == std::string::npos) break;
    auto feature_key = feature_str.substr(pos + 1, end_pos - pos - 1);
    if (feature_key == "density") {
      feature_keys.insert({feature_key, Feature_Density});
    } else if (feature_key == "grad") {
      feature_keys.insert({feature_key, Feature_Grad});
    } else if (feature_key == "kin") {
      feature_keys.insert({feature_key, Feature_Kin});
    } else if (feature_key == "grid_coords") {
      feature_keys.insert({feature_key, Feature_GridCoords});
    } else if (feature_key == "grid_weights") {
      feature_keys.insert({feature_key, Feature_GridWeights});
    } else if (feature_key == "coarse_0_atomic_coords") {
      feature_keys.insert({feature_key, Feature_Coarse0AtomicCoords});
    }
    pos = end_pos + 1;
  }

  if (features != nullptr) {
    size_t i = 0;
    for (const auto &[key, value] : feature_keys) {
      features[i] = value;
      ++i;
    }
  }

  return module;
}

static inline
at::Tensor
skala_model_forward(torch::jit::script::Module module, const c10::Dict<std::string, at::Tensor> &features)
{
  std::vector<c10::IValue> args;
  std::unordered_map<std::string, c10::IValue> kwargs;
  kwargs["mol"] = features;
  return module.get_method("get_exc_density")(args, kwargs).toTensor();
}

extern "C"
void
skala_model_get_exc(torch_jit_script_module_t module, skala_dict_t input, torch_tensor_t* output)
{
  auto model = static_cast<torch::jit::script::Module *>(module);
  auto dict = static_cast<SkalaDict *>(input);

  c10::Dict<std::string, at::Tensor> features;
  for (const auto& entry : *dict) {
    auto tensor = torch::stack(entry.value());
    features.insert(entry.key(), tensor);
  }

  auto exc_on_grid = skala_model_forward(*model, features);
  *output = new at::Tensor(std::move(exc_on_grid));
}

extern "C"
void
skala_model_get_exc_and_vxc(torch_jit_script_module_t module, skala_dict_t input, torch_tensor_t* exc_output, skala_dict_t* grad_output)
{
  auto model = static_cast<torch::jit::script::Module *>(module);
  auto dict = static_cast<SkalaDict *>(input);

  std::vector<at::Tensor> input_tensors;
  std::vector<std::string> tensor_keys;
  c10::Dict<std::string, at::Tensor> features_with_grad;
  for (const auto& entry : *dict) {
    std::string key = entry.key();
    const auto& values = entry.value();
    std::vector<at::Tensor> tensors;
    for (const auto &value : values) {
      auto tensor_with_grad = value.clone().requires_grad_(true);
      tensors.push_back(tensor_with_grad);
      input_tensors.push_back(tensor_with_grad);
      tensor_keys.push_back(key);
    }
    features_with_grad.insert(key, torch::concat(tensors));
  }

  auto exc_on_grid = skala_model_forward(*model, features_with_grad);
  auto exc = (exc_on_grid * features_with_grad.at("grid_weights")).sum();

  auto grad_tensors = torch::autograd::grad(
      {exc},                  // outputs
      input_tensors,          // inputs
      /*grad_outputs=*/{},    // grad_outputs (defaults to ones)
      /*retain_graph=*/false, // retain_graph, necessary for higher-order grads
      /*create_graph=*/false, // create_graph, necessary for higher-order grads
      /*allow_unused=*/true   // allow_unused
  );

  std::unordered_map<std::string, std::vector<at::Tensor>> grad_map;
  for (size_t i = 0; i < tensor_keys.size(); ++i) {
    grad_map[tensor_keys[i]].push_back(grad_tensors[i]);
  }
  c10::Dict<std::string, std::vector<at::Tensor>> gradients;
  for (auto& [key, value] : grad_map) {
    gradients.insert(key, std::move(value));
  }

  *exc_output = new at::Tensor(std::move(exc_on_grid));
  *grad_output = new SkalaDict(std::move(gradients));
}

extern "C"
skala_dict_t
skala_dict_new()
{
  SkalaDict* input = new SkalaDict();
  return static_cast<skala_dict_t>(input);
}

extern "C"
void
skala_dict_insert(skala_dict_t input, const char* key, const torch_tensor_t* values, size_t size)
{
  auto dict = static_cast<SkalaDict *>(input);
  std::vector<at::Tensor> tensors;
  tensors.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    auto tensor = static_cast<at::Tensor*>(values[i]);
    tensors.push_back(*tensor);
  }
  dict->insert(std::string(key), tensors);
}

extern "C"
skala_list_t
skala_dict_at(skala_dict_t input, const char* key)
{
  auto dict = static_cast<SkalaDict *>(input);
  if (!dict->contains(key)) {
    std::string message = "Key '" + std::string(key) + "' not found in SkalaDict";
    ctorch_error(message, []() {});
  }
  auto tensors = (*dict).at(key);
  if (tensors.empty()) {
    std::string message = "No tensors found for key '" + std::string(key) + "'";
    ctorch_error(message, []() {});
  }

  SkalaList* list = new SkalaList(std::move(tensors));
  return list;
}

extern "C"
size_t
skala_list_size(skala_list_t input)
{
  auto list = static_cast<SkalaList *>(input);
  return list->size();
}

extern "C"
torch_tensor_t
skala_list_at(skala_list_t input, size_t index)
{
  auto list = static_cast<SkalaList *>(input);
  if (index >= list->size()) {
    std::string message = "Index " + std::to_string(index) + " out of bounds for SkalaList of size " + std::to_string(list->size());
    ctorch_error(message, []() {});
  }
  auto tensor = (*list)[index];
  return new at::Tensor(std::move(tensor));
}

extern "C"
torch_tensor_t
skala_tensor_mul(const torch_tensor_t a, const torch_tensor_t b)
{
  auto ta = reinterpret_cast<torch::Tensor *>(a);
  auto tb = reinterpret_cast<torch::Tensor *>(b);
  auto result = (*ta) * (*tb);
  return new torch::Tensor(std::move(result));
}

extern "C"
torch_tensor_t
skala_tensor_mean(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  auto m = t->mean();
  return new torch::Tensor(std::move(m));
}

extern "C"
void*
skala_tensor_data_ptr(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  auto contiguous = t->contiguous().to(torch::kFloat64);
  // Replace the tensor in-place so the pointer stays valid
  *t = contiguous;
  return t->data_ptr();
}

extern "C"
int64_t
skala_tensor_ndim(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->ndimension();
}

extern "C"
int64_t
skala_tensor_size(const torch_tensor_t tensor, int64_t dim)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->size(dim);
}

extern "C"
int64_t
skala_tensor_numel(const torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->numel();
}

extern "C"
void
skala_dict_delete(skala_dict_t input)
{
  if (input == nullptr) return;
  auto dict = static_cast<SkalaDict *>(input);
  delete dict;
}

extern "C"
void
skala_list_delete(skala_list_t input)
{
  if (input == nullptr) return;
  auto list = static_cast<SkalaList *>(input);
  delete list;
}
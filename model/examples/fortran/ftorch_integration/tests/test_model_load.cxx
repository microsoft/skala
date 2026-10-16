#include <torch/script.h>
#include <openssl/evp.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

extern "C" void *skala_model_load(const char *, const char *, bool, int *) noexcept;

static std::string save_model(const std::filesystem::path &path,
                              const std::string &protocol) {
  torch::jit::Module module("test");
  torch::jit::ExtraFilesMap metadata{
      {"features", "[\"density\"]"}, {"protocol_version", protocol}};
  module.save(path.string(), metadata);
  std::ifstream input(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto bytes = buffer.str();
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest, &size, EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Test SHA-256 computation failed");
  }
  std::ostringstream hash;
  for (unsigned int i = 0; i < size; ++i) {
    hash << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<unsigned int>(digest[i]);
  }
  return hash.str();
}

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  const auto path = std::filesystem::current_path() / "model-load-test.fun";
  try {
    const auto hash = save_model(path, "2");
    int features[9] = {};
    auto *module = static_cast<torch::jit::Module *>(
        skala_model_load(path.c_str(), hash.c_str(), false, features));
    require(module != nullptr, "A verified model must load");
    delete module;
    require(features[0] == 1, "Valid feature metadata must survive loading");
    require(skala_model_load(path.c_str(), nullptr, false, nullptr) == nullptr,
            "Missing digest must be rejected");
    require(skala_model_load(path.c_str(), "bad", false, nullptr) == nullptr,
            "Invalid digest must be rejected");
    const std::string wrong_hash(64, '0');
    require(skala_model_load(path.c_str(), wrong_hash.c_str(), false, nullptr) == nullptr,
            "Hash mismatch must be rejected");
    for (const auto &protocol : {"", "garbage", "2junk", "99999999999999999999999", "3"}) {
      const auto malformed_hash = save_model(path, protocol);
      require(skala_model_load(path.c_str(), malformed_hash.c_str(), false, nullptr) == nullptr,
              "Malformed or unsupported protocol must return failure without throwing");
    }
    {
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      output << "not a model";
    }
    require(skala_model_load(path.c_str(), hash.c_str(), false, nullptr) == nullptr,
            "Changed bytes must be rejected");
    std::filesystem::remove(path);
    require(skala_model_load(path.c_str(), hash.c_str(), false, nullptr) == nullptr,
            "Missing model must return failure");
    return 0;
  } catch (const std::exception &error) {
    std::filesystem::remove(path);
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

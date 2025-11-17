#include <gauxc/gauxc_config.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#ifdef GAUXC_HAS_CUDA
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#endif
#include <nlohmann/json.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>

using json = nlohmann::json;
using IValueList = std::vector<c10::IValue>;
using IValueMap = std::unordered_map<std::string, c10::IValue>;
using FeatureDict = c10::Dict<std::string, at::Tensor>;

namespace GauXC {

  //  custom allocator for torch::Tensor  
  // enum store onedft feature keys and our feature keys
  // TODO add laplacian ?
  void print_memory_stats(size_t device_id);
  
  enum ONEDFT_FEATURE { DEN, DDEN, TAU, POINTS, WEIGHTS, COORDS };

  // Mapping enums to string values
  inline const std::map<ONEDFT_FEATURE, std::string> feat_map = {
    {DEN, "density"},
    {DDEN, "grad"},
    {TAU, "kin"},
    {POINTS, "grid_coords"},
    {WEIGHTS, "grid_weights"},
    {COORDS, "coarse_0_atomic_coords"}
  };

  inline const std::map<std::string, ONEDFT_FEATURE> reverse_feat_map = {
    {"density", DEN},
    {"grad", DDEN},
    {"kin", TAU},
    {"grid_coords", POINTS},
    {"grid_weights", WEIGHTS},
    {"coarse_0_atomic_coords", COORDS}
  };
  
int mpi_scatter_onedft_outputs(const FeatureDict features_dict,
                          const int world_rank, const int world_size,
                          std::vector<int> recvcounts, std::vector<int> displs,
                          std::vector<double>& den_eval, std::vector<double>& dden_eval, 
                          std::vector<double>& tau);

int mpi_gather_onedft_inputs(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& sendcounts, std::vector<int>& displs);
                          
int mpi_gather_onedft_inputs_gpu(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& recvcounts, std::vector<int>& displs) ;
  bool valueExists(const std::string& value);

  std::tuple<torch::jit::Method, std::vector<std::string>>
    load_model(const std::string filename, torch::DeviceType device);

  at::Tensor
    get_exc(torch::jit::Method exc_func, FeatureDict features);
} // namespace GauXC
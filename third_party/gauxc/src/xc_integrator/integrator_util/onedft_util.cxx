#include <gauxc/gauxc_config.hpp>
#include "onedft_util.hpp"
#ifdef GAUXC_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <gauxc/exceptions.hpp>
#include <gauxc/util/mpi.hpp>
namespace GauXC {

void print_memory_stats(size_t device_id) {
#ifdef GAUXC_HAS_CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Device ID: " << device_id << std::endl;
    // std::cout << "Total Memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
    // std::cout << "Free Memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
#endif
}
bool valueExists(const std::string& value) {
    for (const auto& pair : feat_map) {
        if (pair.second == value) {
            return true;
        }
    }
    return false;
}

// map model to GAUXC_ONEDFT_MODEL_PATH / model.fun
std::string map_model(const std::string& model, torch::DeviceType device) {
    if (std::filesystem::exists(model)) {
        return model;
    } 
    // find model in GAUXC_ONEDFT_MODEL_PATH or GAUXC_ONEDFT_MODEL_PATH_INSTALL
    std::string model_path = std::string(GAUXC_ONEDFT_MODEL_PATH);
    if (!std::filesystem::exists(model_path)) {
        if (std::filesystem::exists(GAUXC_ONEDFT_MODEL_PATH_INSTALL)) {
            model_path = std::string(GAUXC_ONEDFT_MODEL_PATH_INSTALL);
        } else {
            GAUXC_GENERIC_EXCEPTION("Neither GAUXC_ONEDFT_MODEL_PATH nor GAUXC_ONEDFT_MODEL_PATH_INSTALL exist");
        }
    }
    if (std::filesystem::exists(model_path + "/" + model)) {
        return model_path + "/" + model;
    }
    // check if model is in the form of "PBE", "TPSS", "LDA", "ONEDFT"
    if (model == "PBE") {
        return model_path + "/pbe.fun";
    } else if (model == "TPSS") {
        return model_path + "/tpss.fun";
    } else if (model == "LDA") {
        return model_path + "/lda.fun";
    } else {
        GAUXC_GENERIC_EXCEPTION("Model " + model + " not found in " + model_path);
    }
}

std::tuple<torch::jit::Method, std::vector<std::string>>
load_model(const std::string filename, torch::DeviceType device)
{    
    torch::jit::script::Module mod;
    torch::jit::ExtraFilesMap extra_files{{"features", ""}, {"protocol_version", ""}};
    std::vector<std::string> keys;
    std::string model = map_model(filename, device);
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mod = torch::jit::load(model, device, extra_files);
    }
    catch (const c10::Error& e) {
        GAUXC_GENERIC_EXCEPTION("error loading onedft model: " + std::string(e.what()));
    }

    auto version = json::parse(extra_files.at("protocol_version")).get<int>();
    if (version != 2) {
        GAUXC_GENERIC_EXCEPTION("Unsupported protocol version " + std::to_string(version));
    }

    auto features = json::parse(extra_files.at("features"));
    // check if features is array
    if (!features.is_array()) {
        GAUXC_GENERIC_EXCEPTION("features is not an array");
    }
    for (const auto& feature : features) {
        if (!feature.is_string()) {
            GAUXC_GENERIC_EXCEPTION("feature is not a string");
        }
        keys.push_back(feature.get<std::string>());
    }

    return std::make_tuple(mod.get_method("get_exc_density"), keys);
}

at::Tensor
get_exc(torch::jit::Method exc_func, FeatureDict features) {
    IValueList args;
    IValueMap kwargs;
    kwargs["mol"] = features;
    return exc_func(args, kwargs).toTensor();
}

int mpi_scatter_onedft_outputs(const FeatureDict features_dict, // only exist in rank 0
                          const int world_rank, const int world_size,
                          std::vector<int> recvcounts, std::vector<int> displs,
                          std::vector<double>& den_eval, std::vector<double>& dden_eval, std::vector<double>& tau) {
  // store data
  std::vector<double> recv_den_eval, recv_dden_eval, recv_tau;

  int total_npts;
  bool is_gga, is_mgga;
  if (world_rank == 0) {
    total_npts = features_dict.at(feat_map.at(ONEDFT_FEATURE::DEN)).size(1);
    is_gga = (features_dict.find(feat_map.at(ONEDFT_FEATURE::DDEN)) != features_dict.end());
    is_mgga = (features_dict.find(feat_map.at(ONEDFT_FEATURE::TAU)) != features_dict.end());
    
    recv_den_eval.resize(total_npts * 2);
    if (is_gga | is_mgga) {
      recv_dden_eval.resize(total_npts * 2 * 3);
    }
    if (is_mgga) {
      recv_tau.resize(total_npts * 2);
    }
  }
  #ifdef GAUXC_HAS_MPI
  MPI_Bcast(&is_gga, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&is_mgga, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_npts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  #endif

  double* recv_den_eval_a = recv_den_eval.data();
  double* recv_den_eval_b = recv_den_eval.data() + total_npts;
  double* recv_dden_x_eval_a = recv_dden_eval.data();
  double* recv_dden_y_eval_a = recv_dden_eval.data() + total_npts;
  double* recv_dden_z_eval_a = recv_dden_eval.data() + total_npts * 2;
  double* recv_dden_x_eval_b = recv_dden_eval.data() + total_npts * 3;
  double* recv_dden_y_eval_b = recv_dden_eval.data() + total_npts * 4;
  double* recv_dden_z_eval_b = recv_dden_eval.data() + total_npts * 5;
  double* recv_tau_a = recv_tau.data();
  double* recv_tau_b = recv_tau.data() + total_npts;

  if (world_rank == 0) {
    at::Tensor den_grad_tensor = features_dict.at(feat_map.at(ONEDFT_FEATURE::DEN)).grad().cpu().contiguous();
    std::memcpy(recv_den_eval_a, den_grad_tensor.data_ptr<double>(), total_npts * sizeof(double));
    std::memcpy(recv_den_eval_b, den_grad_tensor.data_ptr<double>() + total_npts, total_npts * sizeof(double));

    if (is_gga || is_mgga) {
      at::Tensor dden_grad_tensor = features_dict.at(feat_map.at(ONEDFT_FEATURE::DDEN)).grad().cpu().contiguous();
      std::memcpy(recv_dden_x_eval_a, dden_grad_tensor.data_ptr<double>(), total_npts * sizeof(double));
      std::memcpy(recv_dden_y_eval_a, dden_grad_tensor.data_ptr<double>() + total_npts, total_npts * sizeof(double));
      std::memcpy(recv_dden_z_eval_a, dden_grad_tensor.data_ptr<double>() + total_npts * 2, total_npts * sizeof(double));
      std::memcpy(recv_dden_x_eval_b, dden_grad_tensor.data_ptr<double>() + total_npts * 3, total_npts * sizeof(double));
      std::memcpy(recv_dden_y_eval_b, dden_grad_tensor.data_ptr<double>() + total_npts * 4, total_npts * sizeof(double));
      std::memcpy(recv_dden_z_eval_b, dden_grad_tensor.data_ptr<double>() + total_npts * 5, total_npts * sizeof(double));
    }
    if (is_mgga) {
      at::Tensor tau_grad_tensor = features_dict.at(feat_map.at(ONEDFT_FEATURE::TAU)).grad().cpu().contiguous();
      std::memcpy(recv_tau_a, tau_grad_tensor.data_ptr<double>(), total_npts * sizeof(double));
      std::memcpy(recv_tau_b, tau_grad_tensor.data_ptr<double>() + total_npts, total_npts * sizeof(double));
    }
  }
  
  if (world_size == 1) {
    // If only one rank, no need to scatter
    den_eval = std::move(recv_den_eval);
    dden_eval = std::move(recv_dden_eval);
    tau = std::move(recv_tau);
    return total_npts;
  }
  // Prepare for scattering
#ifdef GAUXC_HAS_MPI
  MPI_Scatter(recvcounts.data(), 1, MPI_INT, &total_npts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  den_eval.resize(total_npts * 2);
  dden_eval.resize(total_npts * 6);
  tau.resize(total_npts * 2);

  MPI_Scatterv(recv_den_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE,
    den_eval.data(), total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatterv(recv_den_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE,
    den_eval.data() + total_npts, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (is_gga | is_mgga) {
    MPI_Scatterv(recv_dden_x_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data(), total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_dden_y_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data() + total_npts, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_dden_z_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data() + total_npts * 2, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_dden_x_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data() + total_npts * 3, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_dden_y_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data() + total_npts * 4, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_dden_z_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE,
    dden_eval.data() + total_npts * 5, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if (is_mgga) {
    MPI_Scatterv(recv_tau_a, recvcounts.data(), displs.data(), MPI_DOUBLE,
    tau.data(), total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(recv_tau_b, recvcounts.data(), displs.data(), MPI_DOUBLE,
    tau.data() + total_npts, total_npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  return total_npts;
}

int mpi_gather_onedft_inputs_gpu(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& recvcounts, std::vector<int>& displs) {
#ifdef GAUXC_HAS_MPI

    const bool is_gga = (dden_eval.size() > 0);
    const bool is_mgga = (tau.size() > 0);
    std::vector<double> recv_grid_weights, recv_grid_coords, recv_den_eval, recv_dden_eval, 
        recv_tau;
    std::vector<int> recvcounts_coords, displs_coords;
    int total_npts_sum = 0;
    MPI_Allreduce(&total_npts, &total_npts_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (world_rank == 0) {
      recv_grid_weights.resize(total_npts_sum);
      recv_grid_coords.resize(total_npts_sum * 3);
      recv_den_eval.resize(total_npts_sum * 2);
      if (is_gga | is_mgga) {
        recv_dden_eval.resize(total_npts_sum * 2 * 3);
      }
      if (is_mgga) {
        recv_tau.resize(total_npts_sum * 2);
      }
      recvcounts_coords.resize(world_size);
      displs_coords.resize(world_size);
    }

    size_t displ = 0;
    MPI_Scan(&total_npts, &displ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    displ -= total_npts;
    MPI_Gather(&total_npts, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&displ, 1, MPI_INT, displs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
      for (int i = 0; i < world_size; ++i) {
        displs_coords[i] = displs[i] * 3;
        recvcounts_coords[i] = recvcounts[i] * 3;
      }
    }

    double* den_eval_a   = den_eval.data();
    double* den_eval_b   = den_eval_a + total_npts;

    double* recv_den_eval_a = recv_den_eval.data();
    double* recv_den_eval_b = recv_den_eval_a + total_npts_sum;

    MPI_Gatherv(den_eval_a, total_npts, MPI_DOUBLE,
      recv_den_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(den_eval_b, total_npts, MPI_DOUBLE,
      recv_den_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* dden_x_eval_a = dden_eval.data();
    double* dden_y_eval_a = dden_x_eval_a + total_npts;
    double* dden_z_eval_a = dden_x_eval_a + total_npts*2;
    double* dden_x_eval_b = dden_x_eval_a + total_npts*3;
    double* dden_y_eval_b = dden_x_eval_a + total_npts*4;
    double* dden_z_eval_b = dden_x_eval_a + total_npts*5;

    double* recv_dden_x_eval_a = recv_dden_eval.data();
    double* recv_dden_y_eval_a = recv_dden_x_eval_a + total_npts_sum;
    double* recv_dden_z_eval_a = recv_dden_x_eval_a + total_npts_sum*2;
    double* recv_dden_x_eval_b = recv_dden_x_eval_a + total_npts_sum*3;
    double* recv_dden_y_eval_b = recv_dden_x_eval_a + total_npts_sum*4;
    double* recv_dden_z_eval_b = recv_dden_x_eval_a + total_npts_sum*5;

    if (is_gga || is_mgga) {
      MPI_Gatherv(dden_x_eval_a, total_npts, MPI_DOUBLE,
        recv_dden_x_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(dden_y_eval_a, total_npts, MPI_DOUBLE,
        recv_dden_y_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(dden_z_eval_a, total_npts, MPI_DOUBLE,
        recv_dden_z_eval_a, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(dden_x_eval_b, total_npts, MPI_DOUBLE,
        recv_dden_x_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(dden_y_eval_b, total_npts, MPI_DOUBLE,
        recv_dden_y_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(dden_z_eval_b, total_npts, MPI_DOUBLE,
        recv_dden_z_eval_b, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double* tau_a        = tau.data();
    double* tau_b        = tau_a + total_npts;

    double* recv_tau_a   = recv_tau.data();
    double* recv_tau_b   = recv_tau_a + total_npts_sum;
    if (is_mgga) {
      MPI_Gatherv(tau_a, total_npts, MPI_DOUBLE,
        recv_tau_a, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gatherv(tau_b, total_npts, MPI_DOUBLE,
        recv_tau_b, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    MPI_Gatherv(grid_weights.data(), total_npts, MPI_DOUBLE,
     recv_grid_weights.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(grid_coords.data(), total_npts * 3, MPI_DOUBLE,
      recv_grid_coords.data(), recvcounts_coords.data(), displs_coords.data(), 
      MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        den_eval        = std::move(recv_den_eval);
        dden_eval       = std::move(recv_dden_eval);
        tau             = std::move(recv_tau);
        grid_coords     = std::move(recv_grid_coords);
        grid_weights    = std::move(recv_grid_weights);
    }
    return total_npts_sum;
#endif
}

int mpi_gather_onedft_inputs(std::vector<double>& den_eval, std::vector<double>& dden_eval,
                          std::vector<double>& tau, std::vector<double>& grid_coords,
                          std::vector<double>& grid_weights, const int total_npts,
                          const int world_rank, const int world_size,
                          std::vector<int>& sendcounts, std::vector<int>& displs) {
#ifdef GAUXC_HAS_MPI
    const bool is_gga = (dden_eval.size() > 0);
    const bool is_mgga = (tau.size() > 0);
    // store gathered data temporarily
    std::vector<double> recv_grid_weights, recv_grid_coords, recv_den_eval, recv_dden_eval, recv_tau;
    int total_npts_sum = 0;
    MPI_Allreduce(&total_npts, &total_npts_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::vector<int> sendcounts2(world_size), displs2(world_size), sendcounts3(world_size), displs3(world_size);
    std::vector<int> sendcounts_coords(world_size), displs_coords(world_size);
    MPI_Gather(&total_npts, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        displs[0] = 0;
        displs2[0] = 0;
        displs3[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
        for (int i = 0; i < world_size; ++i) {
            sendcounts2[i] = sendcounts[i] * 2;        // den_eval, tau (2 values per pt)
            displs2[i] = displs[i] * 2;
            sendcounts3[i] = sendcounts[i] * 6;        // dden_eval (6 values per pt)
            displs3[i] = displs[i] * 6;
            sendcounts_coords[i] = sendcounts[i] * 3;
            displs_coords[i] = displs[i] * 3;
        }

        recv_grid_weights.resize(total_npts_sum);
        recv_grid_coords.resize(total_npts_sum * 3);
        recv_den_eval.resize(total_npts_sum * 2);
        if (is_gga || is_mgga)
            recv_dden_eval.resize(total_npts_sum * 6);
        if (is_mgga)
            recv_tau.resize(total_npts_sum * 2);
    }
    // grid_weights (1 per point)
    MPI_Gatherv(grid_weights.data(), total_npts, MPI_DOUBLE,
                recv_grid_weights.data(), sendcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // den_eval (2 per point)
    MPI_Gatherv(den_eval.data(), total_npts * 2, MPI_DOUBLE,
                recv_den_eval.data(), sendcounts2.data(), displs2.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // tau (2 per point)
    if (is_mgga) {
    MPI_Gatherv(tau.data(), total_npts * 2, MPI_DOUBLE,
                recv_tau.data(), sendcounts2.data(), displs2.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // dden_eval (6 per point)
    if (is_gga || is_mgga) {
    MPI_Gatherv(dden_eval.data(), total_npts * 6, MPI_DOUBLE,
                recv_dden_eval.data(), sendcounts3.data(), displs3.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Gatherv(grid_coords.data(), total_npts * 3, MPI_DOUBLE,
                recv_grid_coords.data(), sendcounts_coords.data(), displs_coords.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
    if (world_rank == 0) {
        den_eval        = std::move(recv_den_eval);
        dden_eval       = std::move(recv_dden_eval);
        tau             = std::move(recv_tau);
        grid_coords     = std::move(recv_grid_coords);
        grid_weights    = std::move(recv_grid_weights);
    }
  return total_npts_sum;
#endif
}


} // namespace GauXC

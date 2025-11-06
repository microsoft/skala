/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "scheme1_base.hpp"
#include "device/common/zmat_vxc.hpp"
#include "device/common/onedft.hpp"
#include "device/common/zmat_fxc.hpp"
#include "device/common/collocation_device.hpp"
#include "device/common/device_blas.hpp"
#include "device/common/xc_functional_eval_wrapper.hpp"
#include "device/common/uvvars.hpp"
#include "device/common/pack_submat.hpp"
#include "device/common/inc_potential.hpp"
#include "device/common/symmetrize_mat.hpp"
#include "device/common/increment_exc_grad.hpp"
#include "device/common/exx_ek_screening.hpp"

#include "buffer_adaptor.hpp"

#include "device/common/shell_pair_to_task.hpp"
#ifdef GAUXC_HAS_CUDA
#include "device_specific/cuda_util.hpp"
#include "gpu/integral_data_types.hpp"
#include "gpu/obara_saika_integrals.hpp"
#include "gpu/chebyshev_boys_computation.hpp"

#define GAUXC_ENABLE_EXX
#endif

#ifdef GAUXC_ENABLE_EXX
namespace XGPU {
  void integral_0_task_batched(
    size_t ntasks, size_t nsubtask,
    int max_primpairs, size_t max_nsp,
    GauXC::XCDeviceTask*                device_tasks,
    const GauXC::TaskToShellPairDevice* task2sp,
    const std::array<int32_t, 4>*  subtasks,
    const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
    double* sp_X_AB_device,
    double* sp_Y_AB_device,
    double* sp_Z_AB_device,
    double *boys_table,
    cudaStream_t stream);

  void integral_1_task_batched(
    bool sph,
    size_t ntasks, size_t nsubtask,
    int max_primpairs, size_t max_nsp,
    GauXC::XCDeviceTask*                device_tasks,
    const GauXC::TaskToShellPairDevice* task2sp,
    const std::array<int32_t, 4>*  subtasks,
    const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
    double* sp_X_AB_device,
    double* sp_Y_AB_device,
    double* sp_Z_AB_device,
    double *boys_table,
    cudaStream_t stream);

  void integral_2_task_batched(
    bool sph,
    size_t ntasks, size_t nsubtask,
    int max_primpairs, size_t max_nsp,
    GauXC::XCDeviceTask*                device_tasks,
    const GauXC::TaskToShellPairDevice* task2sp,
    const std::array<int32_t, 4>*  subtasks,
    const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
    double* sp_X_AB_device,
    double* sp_Y_AB_device,
    double* sp_Z_AB_device,
    double *boys_table,
    cudaStream_t stream);


  void integral_0_0_task_batched(
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_0_0_shell_batched(
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 

  void integral_1_1_task_batched(
        bool sph,
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_1_1_shell_batched(
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 

  void integral_2_2_task_batched(
        bool sph,
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_2_2_shell_batched(
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 
        
  void integral_1_0_task_batched(
        bool swap,
        bool sph,
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_1_0_shell_batched(
        bool swap,
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 

  void integral_2_0_task_batched(
        bool swap,
        bool sph,
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_2_0_shell_batched(
        bool swap,
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 

  void integral_2_1_task_batched(
        bool swap,
        bool sph_2, bool sph_1,
        size_t ntasks,
        size_t nsubtasks,
        int max_primpairs, size_t max_nsp,
        GauXC::XCDeviceTask*                device_tasks,
        const GauXC::TaskToShellPairDevice* task2sp,
        const std::array<int32_t, 4>*  subtasks,
        const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** pp_ptr_device,
        double* sp_X_AB_device,
        double* sp_Y_AB_device,
        double* sp_Z_AB_device,
        double *boys_table,
        cudaStream_t stream);

  void integral_2_1_shell_batched(
        bool swap,
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
        double *boys_table,
        cudaStream_t stream); 
}
#endif


namespace GauXC {

AoSScheme1Base::AoSScheme1Base() {
#ifdef GAUXC_ENABLE_EXX
  dev_boys_table = XGPU::boys_init();
#endif
}

AoSScheme1Base::~AoSScheme1Base() noexcept {
#ifdef GAUXC_ENABLE_EXX
  XGPU::boys_finalize(dev_boys_table);
#endif
}

void AoSScheme1Base::eval_zmat_lda_vxc( XCDeviceData* _data, integrator_ks_scheme scheme, density_id den ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_lda_vxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks, scheme, den,
    data->device_backend_->queue() );

  data->device_backend_->check_error("zmat_lda" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_gga_vxc( XCDeviceData* _data, integrator_ks_scheme scheme, density_id den ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_gga_vxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks, scheme, den,
    data->device_backend_->queue() );

  data->device_backend_->check_error("zmat_gga" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_mgga_vxc( XCDeviceData* _data, integrator_ks_scheme scheme, bool do_lapl, density_id id){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_mgga_vxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks,
    do_lapl, scheme, id, data->device_backend_->queue() );


  data->device_backend_->check_error("zmat_mgga" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_onedft( XCDeviceData* _data, integrator_term_tracker track, density_id den ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_onedft_vxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks,
    track.xc_approx, den, data->device_backend_->queue() );
  data->device_backend_->check_error("zmat_lda" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::sz_to_ab_onedft( XCDeviceData* _data, size_t offset ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();
  auto backend = data->device_backend_;
  
  auto static_stack = data->static_stack;
  auto base_stack    = data->base_stack;
  size_t npoints = data->total_npts_task_batch;
  size_t total_npts = data->global_dims.total_npts;

  double* dden_x_eval_a = static_stack.dden_eval_device + offset;
  double* dden_y_eval_a = static_stack.dden_eval_device + total_npts + offset;
  double* dden_z_eval_a = static_stack.dden_eval_device + total_npts*2 + offset;

  double* dden_x_eval_b = static_stack.dden_eval_device + total_npts*3 + offset;
  double* dden_y_eval_b = static_stack.dden_eval_device + total_npts*4 + offset;
  double* dden_z_eval_b = static_stack.dden_eval_device + total_npts*5 + offset;

  sz_to_ab(npoints, base_stack.dden_sx_eval_device, base_stack.dden_zx_eval_device, 
    dden_x_eval_a, dden_x_eval_b, data->device_backend_->queue());
  sz_to_ab(npoints, base_stack.dden_sy_eval_device, base_stack.dden_zy_eval_device, 
    dden_y_eval_a, dden_y_eval_b, data->device_backend_->queue());
  sz_to_ab(npoints, base_stack.dden_sz_eval_device, base_stack.dden_zz_eval_device,
    dden_z_eval_a, dden_z_eval_b, data->device_backend_->queue());

  data->device_backend_->check_error("sz_to_ab_onedft" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_lda_fxc( XCDeviceData* _data, density_id den ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_lda_fxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks, den,
    data->device_backend_->queue() );

  data->device_backend_->check_error("zmat_lda_fxc" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_gga_fxc( XCDeviceData* _data, density_id den ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_gga_fxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks, den,
    data->device_backend_->queue() );

  data->device_backend_->check_error("zmat_gga_fxc" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_zmat_mgga_fxc( XCDeviceData* _data, bool do_lapl, density_id id){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  zmat_mgga_fxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks,
    do_lapl, id, data->device_backend_->queue() );


  data->device_backend_->check_error("zmat_mgga_fxc" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_mmat_mgga_vxc( XCDeviceData* _data, integrator_ks_scheme scheme, bool do_lapl, density_id id){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  mmat_mgga_vxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks,
    do_lapl, scheme, id, data->device_backend_->queue() );


  data->device_backend_->check_error("mmat_mgga" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_mmat_mgga_fxc( XCDeviceData* _data, bool do_lapl, density_id id){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  auto aos_stack     = data->aos_stack;
  mmat_mgga_fxc( ntasks, nbe_max, npts_max, aos_stack.device_tasks,
    do_lapl, id, data->device_backend_->queue() );


  data->device_backend_->check_error("mmat_mgga_fxc" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_collocation( XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  size_t npts_max = 0, nshells_max = 0;
  for( auto& task : tasks ) {
    npts_max    = std::max( npts_max, task.npts );
    nshells_max = std::max( nshells_max, task.bfn_screening.nshells );
  }

  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  if( ! static_stack.shells_device )
    GAUXC_GENERIC_EXCEPTION("Shells not Allocated");
  if( ! aos_stack.device_tasks )
    GAUXC_GENERIC_EXCEPTION("Device Tasks not Allocated");

  eval_collocation_masked_combined( ntasks, npts_max, nshells_max,
    static_stack.shells_device, aos_stack.device_tasks, 
    data->device_backend_->queue() );

  data->device_backend_->check_error("collocation" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_collocation_gradient( XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

#ifdef GAUXC_HAS_HIP
  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  size_t npts_max = 0, nshells_max = 0;
  for( auto& task : tasks ) {
    npts_max    = std::max( npts_max, task.npts );
    nshells_max = std::max( nshells_max, task.bfn_screening.nshells );
  }

  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  eval_collocation_masked_combined_deriv1( ntasks, npts_max, nshells_max,
    static_stack.shells_device, aos_stack.device_tasks, 
    data->device_backend_->queue() );
#else
  auto aos_stack     = data->aos_stack;

  auto max_l = data->l_batched_shell_to_task.size() - 1;
  eval_collocation_shell_to_task_gradient( max_l, 
    data->l_batched_shell_to_task.data(), aos_stack.device_tasks,
    data->device_backend_->queue() );
#endif
  
  data->device_backend_->check_error("collocation grad " __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_collocation_hessian( XCDeviceData* _data ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("Hessian NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto aos_stack     = data->aos_stack;

  auto max_l = data->l_batched_shell_to_task.size() - 1;
  eval_collocation_shell_to_task_hessian( max_l, 
    data->l_batched_shell_to_task.data(), aos_stack.device_tasks,
    data->device_backend_->queue() );
#endif
  
  data->device_backend_->check_error("collocation hess" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_collocation_laplacian( XCDeviceData* _data ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("Laplacian NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto aos_stack     = data->aos_stack;

  auto max_l = data->l_batched_shell_to_task.size() - 1;
  eval_collocation_shell_to_task_laplacian( max_l, 
    data->l_batched_shell_to_task.data(), aos_stack.device_tasks,
    data->device_backend_->queue() );
#endif
  
  data->device_backend_->check_error("collocation lapl" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_collocation_lapgrad( XCDeviceData* _data ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("Laplacian Gradient NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto aos_stack     = data->aos_stack;

  auto max_l = data->l_batched_shell_to_task.size() - 1;
  eval_collocation_shell_to_task_lapgrad( max_l, 
    data->l_batched_shell_to_task.data(), aos_stack.device_tasks,
    data->device_backend_->queue() );
#endif
  
  data->device_backend_->check_error("collocation lap grad " __FILE__ ": " + std::to_string(__LINE__));
}





void AoSScheme1Base::inc_exc( XCDeviceData* _data ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto base_stack    = data->base_stack;
  auto static_stack  = data->static_stack;
  const bool is_RKS  = data->allocated_terms.ks_scheme == RKS;
  const bool is_UKS  = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS  = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;
  
  gdot( data->device_backend_->master_blas_handle(), data->total_npts_task_batch,
    base_stack.eps_eval_device, 1, base_stack.den_s_eval_device, 1, 
    static_stack.acc_scr_device, static_stack.exc_device );

  if( is_pol ) {
    gdot( data->device_backend_->master_blas_handle(), data->total_npts_task_batch,
      base_stack.eps_eval_device, 1, base_stack.den_z_eval_device, 1, 
      static_stack.acc_scr_device, static_stack.exc_device );
  }
  
  data->device_backend_->check_error("inc exc" __FILE__ ": " + std::to_string(__LINE__));
}
void AoSScheme1Base::inc_nel( XCDeviceData* _data ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto base_stack    = data->base_stack;
  auto static_stack  = data->static_stack;

  const bool is_RKS  = data->allocated_terms.ks_scheme == RKS;
  const bool is_UKS  = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS  = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;
  
  gdot( data->device_backend_->master_blas_handle(), data->total_npts_task_batch,
    base_stack.weights_device, 1, base_stack.den_s_eval_device, 1, 
    static_stack.acc_scr_device, static_stack.nel_device );

  if( is_pol ) {
    gdot( data->device_backend_->master_blas_handle(), data->total_npts_task_batch,
      base_stack.weights_device, 1, base_stack.den_z_eval_device, 1, 
      static_stack.acc_scr_device, static_stack.nel_device );
  }
  
  data->device_backend_->check_error("inc nel" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_uvars_lda( XCDeviceData* _data, integrator_ks_scheme ks_scheme){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variables
  auto aos_stack     = data->aos_stack;
  GauXC::eval_uvars_lda( ntasks, npts_max, ks_scheme,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar lda" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_uvars_gga( XCDeviceData* _data, integrator_ks_scheme ks_scheme){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_uvars_gga( ntasks, npts_max, ks_scheme,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar gga" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_uvars_mgga( XCDeviceData* _data, integrator_ks_scheme scheme, bool do_lapl ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_uvars_mgga( ntasks, npts_max, scheme, do_lapl,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar mgga" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_vvars_lda( XCDeviceData* _data, density_id den_select){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.den_s_eval_device;
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.den_z_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.den_y_eval_device;
      break;
    case DEN_X:
      den_eval_ptr = base_stack.den_x_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_lda called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );

  // Evaluate V variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_vvars_lda( ntasks, nbe_max, npts_max, den_select,
    aos_stack.device_tasks, data->device_backend_->queue() );

}

void AoSScheme1Base::eval_vvars_gga( XCDeviceData* _data, density_id den_select){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  double* den_x_eval_ptr  = nullptr;
  double* den_y_eval_ptr  = nullptr;
  double* den_z_eval_ptr  = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.den_s_eval_device;
      den_x_eval_ptr = base_stack.dden_sx_eval_device;
      den_y_eval_ptr = base_stack.dden_sy_eval_device;
      den_z_eval_ptr = base_stack.dden_sz_eval_device; 
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.den_z_eval_device;
      den_x_eval_ptr = base_stack.dden_zx_eval_device;
      den_y_eval_ptr = base_stack.dden_zy_eval_device;
      den_z_eval_ptr = base_stack.dden_zz_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.den_y_eval_device;
      den_x_eval_ptr = base_stack.dden_yx_eval_device;
      den_y_eval_ptr = base_stack.dden_yy_eval_device;
      den_z_eval_ptr = base_stack.dden_yz_eval_device; 
      break;
    case DEN_X:
      den_eval_ptr = base_stack.den_x_eval_device;
      den_x_eval_ptr = base_stack.dden_xx_eval_device;
      den_y_eval_ptr = base_stack.dden_xy_eval_device;
      den_z_eval_ptr = base_stack.dden_xz_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_gga called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_x_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_y_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_z_eval_ptr, "Den Zero" );
  
  // Evaluate V variable
  auto aos_stack = data->aos_stack;
  GauXC::eval_vvars_gga( ntasks, nbe_max, npts_max, den_select,
    aos_stack.device_tasks, data->device_backend_->queue() );

}

void AoSScheme1Base::eval_vvars_mgga( XCDeviceData* _data, density_id den_select, bool need_lapl){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  double* den_x_eval_ptr  = nullptr;
  double* den_y_eval_ptr  = nullptr;
  double* den_z_eval_ptr  = nullptr;
  double* tau_eval_ptr    = nullptr;
  double* lapl_eval_ptr   = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.den_s_eval_device;
      den_x_eval_ptr = base_stack.dden_sx_eval_device;
      den_y_eval_ptr = base_stack.dden_sy_eval_device;
      den_z_eval_ptr = base_stack.dden_sz_eval_device; 
      tau_eval_ptr   = base_stack.tau_s_eval_device;
      lapl_eval_ptr  = base_stack.lapl_s_eval_device;
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.den_z_eval_device;
      den_x_eval_ptr = base_stack.dden_zx_eval_device;
      den_y_eval_ptr = base_stack.dden_zy_eval_device;
      den_z_eval_ptr = base_stack.dden_zz_eval_device;
      tau_eval_ptr   = base_stack.tau_z_eval_device;
      lapl_eval_ptr  = base_stack.lapl_z_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.den_y_eval_device;
      den_x_eval_ptr = base_stack.dden_yx_eval_device;
      den_y_eval_ptr = base_stack.dden_yy_eval_device;
      den_z_eval_ptr = base_stack.dden_yz_eval_device; 
      break;
    case DEN_X:
      den_eval_ptr = base_stack.den_x_eval_device;
      den_x_eval_ptr = base_stack.dden_xx_eval_device;
      den_y_eval_ptr = base_stack.dden_xy_eval_device;
      den_z_eval_ptr = base_stack.dden_xz_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_gga called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_x_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_y_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_z_eval_ptr, "Den Zero" );
  if(tau_eval_ptr)
    data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, tau_eval_ptr, "TAU Zero");
  if(lapl_eval_ptr)
    data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, lapl_eval_ptr, "LAPL Zero");
  
  // Evaluate V variable
  auto aos_stack = data->aos_stack;
  GauXC::eval_vvars_mgga( ntasks, nbe_max, npts_max, den_select, need_lapl,
    aos_stack.device_tasks, data->device_backend_->queue() );

}


void AoSScheme1Base::eval_tmat_lda( XCDeviceData* _data, integrator_ks_scheme ks_scheme){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variables
  auto aos_stack     = data->aos_stack;
  GauXC::eval_tmat_lda( ntasks, npts_max, ks_scheme,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar lda trial" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_tmat_gga( XCDeviceData* _data, integrator_ks_scheme ks_scheme){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_tmat_gga( ntasks, npts_max, ks_scheme,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar gga trial" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_tmat_mgga( XCDeviceData* _data, integrator_ks_scheme scheme, bool do_lapl ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    npts_max = std::max( npts_max, task.npts );
  }

  auto base_stack    = data->base_stack;
  
  // Evaluate U variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_tmat_mgga( ntasks, npts_max, scheme, do_lapl,
    aos_stack.device_tasks, data->device_backend_->queue() );

  
  data->device_backend_->check_error("uvvar mgga trial" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::eval_vvars_lda_trial( XCDeviceData* _data, density_id den_select){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.tden_s_eval_device;
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.tden_z_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.tden_y_eval_device;
      break;
    case DEN_X:
      den_eval_ptr = base_stack.tden_x_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_lda_trial called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );

  // Evaluate V variable
  auto aos_stack     = data->aos_stack;
  GauXC::eval_vvars_lda_trial( ntasks, nbe_max, npts_max, den_select,
    aos_stack.device_tasks, data->device_backend_->queue() );

}

void AoSScheme1Base::eval_vvars_gga_trial( XCDeviceData* _data, density_id den_select){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  double* den_x_eval_ptr  = nullptr;
  double* den_y_eval_ptr  = nullptr;
  double* den_z_eval_ptr  = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.tden_s_eval_device;
      den_x_eval_ptr = base_stack.tdden_sx_eval_device;
      den_y_eval_ptr = base_stack.tdden_sy_eval_device;
      den_z_eval_ptr = base_stack.tdden_sz_eval_device; 
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.tden_z_eval_device;
      den_x_eval_ptr = base_stack.tdden_zx_eval_device;
      den_y_eval_ptr = base_stack.tdden_zy_eval_device;
      den_z_eval_ptr = base_stack.tdden_zz_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.tden_y_eval_device;
      den_x_eval_ptr = base_stack.tdden_yx_eval_device;
      den_y_eval_ptr = base_stack.tdden_yy_eval_device;
      den_z_eval_ptr = base_stack.tdden_yz_eval_device; 
      break;
    case DEN_X:
      den_eval_ptr = base_stack.tden_x_eval_device;
      den_x_eval_ptr = base_stack.tdden_xx_eval_device;
      den_y_eval_ptr = base_stack.tdden_xy_eval_device;
      den_z_eval_ptr = base_stack.tdden_xz_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_gga_trial called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_x_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_y_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_z_eval_ptr, "Den Zero" );
  
  // Evaluate V variable
  auto aos_stack = data->aos_stack;
  GauXC::eval_vvars_gga_trial( ntasks, nbe_max, npts_max, den_select,
    aos_stack.device_tasks, data->device_backend_->queue() );

}

void AoSScheme1Base::eval_vvars_mgga_trial( XCDeviceData* _data, density_id den_select, bool need_lapl){
  auto* data = dynamic_cast<Data*>(_data);
  if ( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  size_t nbe_max = 0, npts_max = 0;
  for( auto& task : tasks ) {
    nbe_max  = std::max( nbe_max, task.bfn_screening.nbe );
    npts_max = std::max( npts_max, task.npts );
  }

  // Zero density
  auto base_stack    = data->base_stack;
  double* den_eval_ptr    = nullptr;
  double* den_x_eval_ptr  = nullptr;
  double* den_y_eval_ptr  = nullptr;
  double* den_z_eval_ptr  = nullptr;
  double* tau_eval_ptr    = nullptr;
  double* lapl_eval_ptr   = nullptr;
  switch ( den_select ) {
    case DEN_S:
      den_eval_ptr = base_stack.tden_s_eval_device;
      den_x_eval_ptr = base_stack.tdden_sx_eval_device;
      den_y_eval_ptr = base_stack.tdden_sy_eval_device;
      den_z_eval_ptr = base_stack.tdden_sz_eval_device; 
      tau_eval_ptr   = base_stack.ttau_s_eval_device;
      lapl_eval_ptr  = base_stack.tlapl_s_eval_device;
      break;
    case DEN_Z:
      den_eval_ptr = base_stack.tden_z_eval_device;
      den_x_eval_ptr = base_stack.tdden_zx_eval_device;
      den_y_eval_ptr = base_stack.tdden_zy_eval_device;
      den_z_eval_ptr = base_stack.tdden_zz_eval_device;
      tau_eval_ptr   = base_stack.ttau_z_eval_device;
      lapl_eval_ptr  = base_stack.tlapl_z_eval_device;
      break;
    case DEN_Y:
      den_eval_ptr = base_stack.tden_y_eval_device;
      den_x_eval_ptr = base_stack.tdden_yx_eval_device;
      den_y_eval_ptr = base_stack.tdden_yy_eval_device;
      den_z_eval_ptr = base_stack.tdden_yz_eval_device; 
      break;
    case DEN_X:
      den_eval_ptr = base_stack.tden_x_eval_device;
      den_x_eval_ptr = base_stack.tdden_xx_eval_device;
      den_y_eval_ptr = base_stack.tdden_xy_eval_device;
      den_z_eval_ptr = base_stack.tdden_xz_eval_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_vvars_gga_trial called with invalid density selected!" );
  }

  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_x_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_y_eval_ptr, "Den Zero" );
  data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, den_z_eval_ptr, "Den Zero" );
  if(tau_eval_ptr)
    data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, tau_eval_ptr, "TAU Zero");
  if(lapl_eval_ptr)
    data->device_backend_->set_zero_async_master_queue( data->total_npts_task_batch, lapl_eval_ptr, "LAPL Zero");
  
  // Evaluate V variable
  auto aos_stack = data->aos_stack;
  GauXC::eval_vvars_mgga_trial( ntasks, nbe_max, npts_max, den_select, need_lapl,
    aos_stack.device_tasks, data->device_backend_->queue() );

}


template <typename T>
void interleave_kernel_input(size_t len, const T* src_data, int src_stride, T* tgt_data, int tgt_stride, std::string msg,
                             DeviceBackend* backend) {
  backend->copy_async_2d(1, len, src_data, src_stride, tgt_data, tgt_stride, msg);
}

template <typename T>
void interleave_lda_input(size_t npts, T& base_stack, DeviceBackend* backend) {
  interleave_kernel_input(npts, base_stack.den_s_eval_device, 1, base_stack.den_interleaved_device+0, 2,
    "den_+ - > den_interleaved", backend);
  interleave_kernel_input(npts, base_stack.den_z_eval_device, 1, base_stack.den_interleaved_device+1, 2,
    "den_- - > den_interleaved", backend);
}

template <typename T>
void interleave_gga_input(size_t npts, T& base_stack, DeviceBackend* backend) {
  interleave_lda_input(npts, base_stack, backend);
  interleave_kernel_input(npts, base_stack.gamma_pp_eval_device, 1, base_stack.gamma_eval_device+0, 3,
    "gamma_++ - > gamma_interleaved", backend);
  interleave_kernel_input(npts, base_stack.gamma_pm_eval_device, 1, base_stack.gamma_eval_device+1, 3,
    "gamma_+- - > gamma_interleaved", backend);
  interleave_kernel_input(npts, base_stack.gamma_mm_eval_device, 1, base_stack.gamma_eval_device+2, 3,
    "gamma_-- - > gamma_interleaved", backend);
}

template <typename T>
void interleave_mgga_input(size_t npts, T& base_stack, DeviceBackend* backend, bool need_lapl) {
  interleave_gga_input(npts, base_stack, backend);
  interleave_kernel_input(npts, base_stack.tau_s_eval_device, 1, base_stack.tau_interleaved_device, 2,
    "tau_+ - > tau_interleaved", backend);
  interleave_kernel_input(npts, base_stack.tau_z_eval_device, 1, base_stack.tau_interleaved_device+1, 2,
    "tau_- - > tau_interleaved", backend);
  if(need_lapl) {
    interleave_kernel_input(npts, base_stack.lapl_s_eval_device, 1, base_stack.lapl_interleaved_device, 2,
      "lapl_+ - > lapl_interleaved", backend);
    interleave_kernel_input(npts, base_stack.lapl_z_eval_device, 1, base_stack.lapl_interleaved_device+1, 2,
      "lapl_- - > lapl_interleaved", backend);
  }
}
 


template <typename T>
void deinterleave_lda_output(size_t npts, T& base_stack, DeviceBackend* backend) {
  interleave_kernel_input(npts, base_stack.vrho_eval_device+0, 2, base_stack.vrho_pos_eval_device, 1,
    "vrho -> vrho+", backend);
  interleave_kernel_input(npts, base_stack.vrho_eval_device+1, 2, base_stack.vrho_neg_eval_device, 1,
    "vrho -> vrho-", backend);
}

template <typename T>
void deinterleave_gga_output(size_t npts, T& base_stack, DeviceBackend* backend) {
  deinterleave_lda_output(npts, base_stack, backend);
  interleave_kernel_input(npts, base_stack.vgamma_eval_device+0, 3, base_stack.vgamma_pp_eval_device, 1,
    "vgamma -> vgamma++", backend);
  interleave_kernel_input(npts, base_stack.vgamma_eval_device+1, 3, base_stack.vgamma_pm_eval_device, 1,
    "vgamma -> vgamma+-", backend);
  interleave_kernel_input(npts, base_stack.vgamma_eval_device+2, 3, base_stack.vgamma_mm_eval_device, 1,
    "vgamma -> vgamma--", backend);
}

template <typename T>
void deinterleave_mgga_output(size_t npts, T& base_stack, DeviceBackend* backend, bool need_lapl) {
  deinterleave_gga_output(npts, base_stack, backend);
  interleave_kernel_input(npts, base_stack.vtau_eval_device+0, 2, base_stack.vtau_pos_eval_device, 1,
    "vtau -> vtau+", backend);
  interleave_kernel_input(npts, base_stack.vtau_eval_device+1, 2, base_stack.vtau_neg_eval_device, 1,
    "vtau -> vtau-", backend);
  if(need_lapl) {
    interleave_kernel_input(npts, base_stack.vlapl_eval_device+0, 2, base_stack.vlapl_pos_eval_device, 1,
      "vlapl -> vlapl+", backend);
    interleave_kernel_input(npts, base_stack.vlapl_eval_device+1, 2, base_stack.vlapl_neg_eval_device, 1,
      "vlapl -> vlapl-", backend);
  }
}

template <typename T>
void deinterleave_vxc_fxc_lda(size_t npts, T& base_stack, DeviceBackend* backend) {
  // Deinterleave the lda vxc output
  deinterleave_lda_output(npts, base_stack, backend);
  interleave_kernel_input(npts, base_stack.v2rho2_eval_device+0, 3, base_stack.v2rho2_a_a_eval_device, 1,
    "v2rho2 -> v2rho2_aa", backend);
  interleave_kernel_input(npts, base_stack.v2rho2_eval_device+1, 3, base_stack.v2rho2_a_b_eval_device, 1,
    "v2rho2 -> v2rho2_ab", backend);
  interleave_kernel_input(npts, base_stack.v2rho2_eval_device+2, 3, base_stack.v2rho2_b_b_eval_device, 1,
    "v2rho2 -> v2rho2_bb", backend);
}

template <typename T>
void deinterleave_vxc_fxc_gga(size_t npts, T& base_stack, DeviceBackend* backend) {
  deinterleave_vxc_fxc_lda(npts, base_stack, backend);
  // Deinterleave the gga vxc output
  deinterleave_gga_output(npts, base_stack, backend);
  
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+0, 6, base_stack.v2rhogamma_a_aa_eval_device, 1,
    "v2rhogamma -> v2rhogamma_a_aa", backend);
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+1, 6, base_stack.v2rhogamma_a_ab_eval_device, 1,
    "v2rhogamma -> v2rhogamma_a_ab", backend);
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+2, 6, base_stack.v2rhogamma_a_bb_eval_device, 1,
    "v2rhogamma -> v2rhogamma_a_bb", backend);
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+3, 6, base_stack.v2rhogamma_b_aa_eval_device, 1,
    "v2rhogamma -> v2rhogamma_b_aa", backend);
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+4, 6, base_stack.v2rhogamma_b_ab_eval_device, 1,
    "v2rhogamma -> v2rhogamma_b_ab", backend);
  interleave_kernel_input(npts, base_stack.v2rhogamma_eval_device+5, 6, base_stack.v2rhogamma_b_bb_eval_device, 1,
    "v2rhogamma -> v2rhogamma_b_bb", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+0, 6, base_stack.v2gamma2_aa_aa_eval_device, 1,
    "v2gamma2 -> v2gamma2_aa_aa", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+1, 6, base_stack.v2gamma2_aa_ab_eval_device, 1,
    "v2gamma2 -> v2gamma2_aa_ab", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+2, 6, base_stack.v2gamma2_aa_bb_eval_device, 1,
    "v2gamma2 -> v2gamma2_aa_bb", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+3, 6, base_stack.v2gamma2_ab_ab_eval_device, 1,
    "v2gamma2 -> v2gamma2_ab_ab", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+4, 6, base_stack.v2gamma2_ab_bb_eval_device, 1,
    "v2gamma2 -> v2gamma2_ab_bb", backend);
  interleave_kernel_input(npts, base_stack.v2gamma2_eval_device+5, 6, base_stack.v2gamma2_bb_bb_eval_device, 1,
    "v2gamma2 -> v2gamma2_bb_bb", backend);
}

template <typename T>
void deinterleave_vxc_fxc_mgga(size_t npts, T& base_stack, DeviceBackend* backend, bool need_lapl) {
  deinterleave_vxc_fxc_gga(npts, base_stack, backend);
  // Deinterleave the mgga vxc output
  deinterleave_mgga_output(npts, base_stack, backend, need_lapl);
  
  interleave_kernel_input(npts, base_stack.v2rhotau_eval_device+0, 4, base_stack.v2rhotau_a_a_eval_device, 1,
    "v2rhotau -> v2rhotau_a_a", backend);
  interleave_kernel_input(npts, base_stack.v2rhotau_eval_device+1, 4, base_stack.v2rhotau_a_b_eval_device, 1,
    "v2rhotau -> v2rhotau_a_b", backend);
  interleave_kernel_input(npts, base_stack.v2rhotau_eval_device+2, 4, base_stack.v2rhotau_b_a_eval_device, 1,
    "v2rhotau -> v2rhotau_b_a", backend);
  interleave_kernel_input(npts, base_stack.v2rhotau_eval_device+3, 4, base_stack.v2rhotau_b_b_eval_device, 1,
    "v2rhotau -> v2rhotau_b_b", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+0, 6, base_stack.v2gammatau_aa_a_eval_device, 1,
    "v2gammatau -> v2gammatau_aa_a", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+1, 6, base_stack.v2gammatau_aa_b_eval_device, 1,
    "v2gammatau -> v2gammatau_aa_b", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+2, 6, base_stack.v2gammatau_ab_a_eval_device, 1,
    "v2gammatau -> v2gammatau_ab_a", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+3, 6, base_stack.v2gammatau_ab_b_eval_device, 1,
    "v2gammatau -> v2gammatau_ab_b", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+4, 6, base_stack.v2gammatau_bb_a_eval_device, 1,
    "v2gammatau -> v2gammatau_bb_a", backend);
  interleave_kernel_input(npts, base_stack.v2gammatau_eval_device+5, 6, base_stack.v2gammatau_bb_b_eval_device, 1,
    "v2gammatau -> v2gammatau_bb_b", backend);
  interleave_kernel_input(npts, base_stack.v2tau2_eval_device+0, 3, base_stack.v2tau2_a_a_eval_device, 1,
    "v2tau2 -> v2tau2_a_a", backend);
  interleave_kernel_input(npts, base_stack.v2tau2_eval_device+1, 3, base_stack.v2tau2_a_b_eval_device, 1,
    "v2tau2 -> v2tau2_a_b", backend);
  interleave_kernel_input(npts, base_stack.v2tau2_eval_device+2, 3, base_stack.v2tau2_b_b_eval_device, 1,
    "v2tau2 -> v2tau2_b_b", backend);
  
  if (need_lapl) {
    interleave_kernel_input(npts, base_stack.v2rholapl_eval_device+0, 4, base_stack.v2rholapl_a_a_eval_device, 1,
      "v2rholapl -> v2rholapl_a_a", backend);
    interleave_kernel_input(npts, base_stack.v2rholapl_eval_device+1, 4, base_stack.v2rholapl_a_b_eval_device, 1,
      "v2rholapl -> v2rholapl_a_b", backend);
    interleave_kernel_input(npts, base_stack.v2rholapl_eval_device+2, 4, base_stack.v2rholapl_b_a_eval_device, 1,
      "v2rholapl -> v2rholapl_b_a", backend);
    interleave_kernel_input(npts, base_stack.v2rholapl_eval_device+3, 4, base_stack.v2rholapl_b_b_eval_device, 1,
      "v2rholapl -> v2rholapl_b_b", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+0, 6, base_stack.v2gammalapl_aa_a_eval_device, 1,
      "v2gammalapl -> v2gammalapl_aa_a", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+1, 6, base_stack.v2gammalapl_aa_b_eval_device, 1,
      "v2gammalapl -> v2gammalapl_aa_b", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+2, 6, base_stack.v2gammalapl_ab_a_eval_device, 1,
      "v2gammalapl -> v2gammalapl_ab_a", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+3, 6, base_stack.v2gammalapl_ab_b_eval_device, 1,
      "v2gammalapl -> v2gammalapl_ab_b", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+4, 6, base_stack.v2gammalapl_bb_a_eval_device, 1,
      "v2gammalapl -> v2gammalapl_bb_a", backend);
    interleave_kernel_input(npts, base_stack.v2gammalapl_eval_device+5, 6, base_stack.v2gammalapl_bb_b_eval_device, 1,
      "v2gammalapl -> v2gammalapl_bb_b", backend);
    interleave_kernel_input(npts, base_stack.v2lapl2_eval_device+0, 3, base_stack.v2lapl2_a_a_eval_device, 1,
      "v2lapl2 -> v2lapl2_a_a", backend);
    interleave_kernel_input(npts, base_stack.v2lapl2_eval_device+1, 3, base_stack.v2lapl2_a_b_eval_device, 1,
      "v2lapl2 -> v2lapl2_a_b", backend);
    interleave_kernel_input(npts, base_stack.v2lapl2_eval_device+2, 3, base_stack.v2lapl2_b_b_eval_device, 1,
      "v2lapl2 -> v2lapl2_b_b", backend);
    interleave_kernel_input(npts, base_stack.v2lapltau_eval_device+0, 4, base_stack.v2lapltau_a_a_eval_device, 1,
      "v2lapltau -> v2lapltau_a_a", backend);
    interleave_kernel_input(npts, base_stack.v2lapltau_eval_device+1, 4, base_stack.v2lapltau_a_b_eval_device, 1,
      "v2lapltau -> v2lapltau_a_b", backend);
    interleave_kernel_input(npts, base_stack.v2lapltau_eval_device+2, 4, base_stack.v2lapltau_b_a_eval_device, 1,
      "v2lapltau -> v2lapltau_b_a", backend);
    interleave_kernel_input(npts, base_stack.v2lapltau_eval_device+3, 4, base_stack.v2lapltau_b_b_eval_device, 1,
      "v2lapltau -> v2lapltau_b_b", backend);
  }
}

template <typename T>
void scale_lda_output(size_t npts, T& base_stack, DeviceBackend* backend, bool is_pol) {
  hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
    base_stack.eps_eval_device, 1); 
  if(is_pol) {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vrho_pos_eval_device, 1); 
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vrho_neg_eval_device, 1); 
  } else {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vrho_eval_device, 1); 
  }
}

template <typename T>
void scale_gga_output(size_t npts, T& base_stack, DeviceBackend* backend, bool is_pol) {
  scale_lda_output(npts, base_stack, backend, is_pol);
  if(is_pol) {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vgamma_pp_eval_device, 1); 
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vgamma_pm_eval_device, 1); 
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vgamma_mm_eval_device, 1); 
  } else {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vgamma_eval_device, 1); 
  }
}

template <typename T>
void scale_mgga_output(size_t npts, T& base_stack, DeviceBackend* backend, bool need_lapl, bool is_pol) {
  scale_gga_output(npts, base_stack, backend, is_pol);
  if(is_pol) {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vtau_pos_eval_device, 1); 
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vtau_neg_eval_device, 1); 
    if(need_lapl) {
      hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
        base_stack.vlapl_pos_eval_device, 1); 
      hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
        base_stack.vlapl_neg_eval_device, 1); 
    }
  } else {
    hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
      base_stack.vtau_eval_device, 1); 
    if(need_lapl) {
      hadamard_product(backend->master_blas_handle(), npts, 1, base_stack.weights_device, 1, 
        base_stack.vlapl_eval_device, 1); 
    }
  }
}


void AoSScheme1Base::eval_kern_exc_vxc_lda( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_lda() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not LDA!");

  auto base_stack    = data->base_stack;

  const bool is_RKS = data->allocated_terms.ks_scheme == RKS;
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol = is_UKS or is_GKS;
  const bool is_excgrad = data->allocated_terms.exc_grad;

  const size_t npts = data->total_npts_task_batch ;
  
  auto* den_eval_ptr = base_stack.den_s_eval_device;

  if ( is_pol ) {
    den_eval_ptr = base_stack.den_interleaved_device;
    interleave_lda_input(npts, base_stack, data->device_backend_);
  }

  GauXC::eval_kern_exc_vxc_lda( func, npts,
    den_eval_ptr, base_stack.eps_eval_device, 
    base_stack.vrho_eval_device, data->device_backend_->queue() );

  if(is_pol) deinterleave_lda_output(npts, base_stack, data->device_backend_);
  scale_lda_output(npts, base_stack, data->device_backend_, is_pol);
  
  data->device_backend_->check_error("exc_vxc lda" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_kern_exc_vxc_gga( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_gga() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not GGA!");

  auto base_stack    = data->base_stack;
  double* den_eval_ptr = base_stack.den_s_eval_device;
  
  const bool is_RKS = data->allocated_terms.ks_scheme == RKS;
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;
  const bool is_excgrad = data->allocated_terms.exc_grad;

  const size_t npts = data->total_npts_task_batch ;
  
  if(is_pol) {
    den_eval_ptr = base_stack.den_interleaved_device;
    interleave_gga_input(npts, base_stack, data->device_backend_);
  }

  GauXC::eval_kern_exc_vxc_gga( func, data->total_npts_task_batch, 
    den_eval_ptr, base_stack.gamma_eval_device, 
    base_stack.eps_eval_device, base_stack.vrho_eval_device, 
    base_stack.vgamma_eval_device, data->device_backend_->queue() );

  if(is_pol) deinterleave_gga_output(npts, base_stack, data->device_backend_);
  scale_gga_output(npts, base_stack, data->device_backend_, is_pol);

  data->device_backend_->check_error("exc_vxc gga" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_kern_exc_vxc_mgga( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_mgga() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not MGGA!");

  auto base_stack       = data->base_stack;
  double* den_eval_ptr  = base_stack.den_s_eval_device;
  double* tau_eval_ptr  = base_stack.tau_s_eval_device;
  double* lapl_eval_ptr = base_stack.lapl_s_eval_device;
  
  const bool is_RKS = data->allocated_terms.ks_scheme == RKS;
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;
  const bool is_excgrad = data->allocated_terms.exc_grad;

  const size_t npts = data->total_npts_task_batch ;
  
  if(is_pol) {
    den_eval_ptr = base_stack.den_interleaved_device;
    tau_eval_ptr = base_stack.tau_interleaved_device;
    lapl_eval_ptr = base_stack.lapl_interleaved_device;
    interleave_mgga_input(npts, base_stack, data->device_backend_, func.needs_laplacian());
  }

  GauXC::eval_kern_exc_vxc_mgga( func, data->total_npts_task_batch, 
    den_eval_ptr, base_stack.gamma_eval_device, 
    tau_eval_ptr, lapl_eval_ptr,
    base_stack.eps_eval_device, base_stack.vrho_eval_device, 
    base_stack.vgamma_eval_device, base_stack.vtau_eval_device,
    base_stack.vlapl_eval_device, data->device_backend_->queue() );

  if(is_pol) deinterleave_mgga_output(npts, base_stack, data->device_backend_, func.needs_laplacian());
  scale_mgga_output(npts, base_stack, data->device_backend_, func.needs_laplacian(), is_pol);
  
  data->device_backend_->check_error("exc_vxc mgga" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_kern_vxc_fxc_lda( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_lda() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not LDA!");

  auto base_stack    = data->base_stack;

  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol = is_UKS or is_GKS;

  const size_t npts = data->total_npts_task_batch ;
  
  auto* den_eval_ptr = base_stack.den_s_eval_device;

  if ( is_pol ) {
    den_eval_ptr = base_stack.den_interleaved_device;
    interleave_lda_input(npts, base_stack, data->device_backend_);
  }

  GauXC::eval_kern_vxc_fxc_lda( func, npts,
    den_eval_ptr, base_stack.vrho_eval_device, 
    base_stack.v2rho2_eval_device, data->device_backend_->queue() );

  if(is_pol) deinterleave_vxc_fxc_lda(npts, base_stack, data->device_backend_);
  // For 2nd derivative, we do not scale the output
  // We will multiply it with the weights to the intermediate outputs A, B, C 
  
  data->device_backend_->check_error("exc_vxc_fxc lda" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_kern_vxc_fxc_gga( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_gga() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not GGA!");

  auto base_stack    = data->base_stack;
  double* den_eval_ptr = base_stack.den_s_eval_device;
  
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;

  const size_t npts = data->total_npts_task_batch ;
  
  if(is_pol) {
    den_eval_ptr = base_stack.den_interleaved_device;
    interleave_gga_input(npts, base_stack, data->device_backend_);
  }

  GauXC::eval_kern_vxc_fxc_gga( func, npts, 
    den_eval_ptr, base_stack.gamma_eval_device,
    base_stack.vrho_eval_device, base_stack.vgamma_eval_device,
    base_stack.v2rho2_eval_device, base_stack.v2rhogamma_eval_device, base_stack.v2gamma2_eval_device,
    data->device_backend_->queue() );

  if(is_pol) deinterleave_vxc_fxc_gga(npts, base_stack, data->device_backend_);
  
  // For 2nd derivative, we do not scale the output
  // We will multiply it with the weights to the intermediate outputs A, B, C 

  
  data->device_backend_->check_error("exc_vxc_fxc gga" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::eval_kern_vxc_fxc_mgga( const functional_type& func, 
  XCDeviceData* _data ) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if( !func.is_mgga() ) GAUXC_GENERIC_EXCEPTION("XC Kernel not MGGA!");

  auto base_stack       = data->base_stack;
  double* den_eval_ptr  = base_stack.den_s_eval_device;
  double* tau_eval_ptr  = base_stack.tau_s_eval_device;
  double* lapl_eval_ptr = base_stack.lapl_s_eval_device;
  
  const bool is_UKS = data->allocated_terms.ks_scheme == UKS;
  const bool is_GKS = data->allocated_terms.ks_scheme == GKS;
  const bool is_pol  = is_UKS or is_GKS;

  const size_t npts = data->total_npts_task_batch ;
  
  if(is_pol) {
    den_eval_ptr = base_stack.den_interleaved_device;
    tau_eval_ptr = base_stack.tau_interleaved_device;
    lapl_eval_ptr = base_stack.lapl_interleaved_device;
    interleave_mgga_input(npts, base_stack, data->device_backend_, func.needs_laplacian());
  }

  GauXC::eval_kern_vxc_fxc_mgga( func, npts, 
    den_eval_ptr, base_stack.gamma_eval_device, 
    lapl_eval_ptr, tau_eval_ptr, 
    base_stack.vrho_eval_device, base_stack.vgamma_eval_device, 
    base_stack.vlapl_eval_device, base_stack.vtau_eval_device,
    base_stack.v2rho2_eval_device, base_stack.v2rhogamma_eval_device,
    base_stack.v2rholapl_eval_device, base_stack.v2rhotau_eval_device,
    base_stack.v2gamma2_eval_device, base_stack.v2gammalapl_eval_device,
    base_stack.v2gammatau_eval_device, base_stack.v2lapl2_eval_device,
    base_stack.v2lapltau_eval_device, base_stack.v2tau2_eval_device,
    data->device_backend_->queue() );

  if(is_pol) deinterleave_vxc_fxc_mgga(npts, base_stack, data->device_backend_, func.needs_laplacian());
  
  // For 2nd derivative, we do not scale the output
  // We will multiply it with the weights to the intermediate outputs A, B, C 
  
  data->device_backend_->check_error("exc_vxc_fxc mgga" __FILE__ ": " + std::to_string(__LINE__));
}

template<bool is_trial>
void AoSScheme1Base::eval_xmat_impl( double fac, XCDeviceData* _data, bool do_grad, density_id den_select ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Set correct density matrix pointer on the stack
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  double * dmat_ptr;
  if constexpr (is_trial) {
    dmat_ptr = static_stack.tden_selector(den_select);
    // now screened trial density matrix is stored in aos_stack.device_tasks[itask].nbe_scr
  } else {
    dmat_ptr = static_stack.den_selector(den_select);
  }

  // Pack density matrix 
  sym_pack_submat( ntasks, aos_stack.device_tasks, dmat_ptr, 
    nbf, submat_block_size, data->device_backend_->queue() );


  // Sync blas streams with master stream
  data->device_backend_->sync_blas_pool_with_master();

  auto do_gemm = [&]( auto& handle, size_t npts, size_t nbe, auto* bf_ptr, auto* den_ptr, int ldden, auto* x_ptr ) {
    gemm( handle, DeviceBlasOp::NoTrans, DeviceBlasOp::NoTrans, npts, nbe, nbe, fac, bf_ptr, npts,
      den_ptr, ldden, 0., x_ptr, npts ); 
  };

  // Launch GEMM in round-robin
  const auto n_blas_streams = data->device_backend_->blas_pool_size();
  
   

  //size_t nsingle = 0;
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    auto& task = tasks[iT];
      auto den_ptr = task.bfn_screening.ncut > 1 ? task.nbe_scr : dmat_ptr + task.bfn_screening.ibf_begin*(nbf+1);
      int  ldden   = task.bfn_screening.ncut > 1 ? task.bfn_screening.nbe : nbf;
      auto handle = data->device_backend_->blas_pool_handle( iT % n_blas_streams );
      do_gemm( handle, task.npts, task.bfn_screening.nbe, task.bf, den_ptr, ldden, task.zmat );
      if( do_grad ) {
        do_gemm( handle, task.npts, task.bfn_screening.nbe, task.dbfx, den_ptr, ldden, task.xmat_x );
        do_gemm( handle, task.npts, task.bfn_screening.nbe, task.dbfy, den_ptr, ldden, task.xmat_y );
        do_gemm( handle, task.npts, task.bfn_screening.nbe, task.dbfz, den_ptr, ldden, task.xmat_z );
      }
  }

  
  data->device_backend_->check_error("xmat impl" __FILE__ ": " + std::to_string(__LINE__));
  // Record completion of BLAS ops on master stream
  data->device_backend_->sync_master_with_blas_pool();

}


void AoSScheme1Base::eval_xmat( double fac, XCDeviceData* _data, bool do_grad, density_id den_select ){
  eval_xmat_impl<false>(fac, _data, do_grad, den_select);
}
void AoSScheme1Base::eval_xmat_trial( double fac, XCDeviceData* _data, bool do_grad, density_id den_select ){
  eval_xmat_impl<true>(fac, _data, do_grad, den_select);
}

void AoSScheme1Base::save_xmat( XCDeviceData* _data, bool do_grad, density_id den_select ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();
  auto backend = data->device_backend_;

  auto aos_stack     = data->aos_stack;
  const auto sz = data->total_nbe_bfn_npts_task_batch;

  switch(den_select) {
    case DEN_S:
      backend->copy_async(sz, aos_stack.zmat_vxc_device, aos_stack.xmatS_device, "xmatS copy");
      if(do_grad) {
        backend->copy_async(sz, aos_stack.xmat_dx_device, aos_stack.xmatS_dx_device, "xmatS_dx copy");
        backend->copy_async(sz, aos_stack.xmat_dy_device, aos_stack.xmatS_dy_device, "xmatS_dy copy");
        backend->copy_async(sz, aos_stack.xmat_dz_device, aos_stack.xmatS_dz_device, "xmatS_dz copy");
      }
      break;
    case DEN_Z:
      backend->copy_async(sz, aos_stack.zmat_vxc_device, aos_stack.xmatZ_device, "xmatZ copy");
      if(do_grad) {
        backend->copy_async(sz, aos_stack.xmat_dx_device, aos_stack.xmatZ_dx_device, "xmatZ_dx copy");
        backend->copy_async(sz, aos_stack.xmat_dy_device, aos_stack.xmatZ_dy_device, "xmatZ_dy copy");
        backend->copy_async(sz, aos_stack.xmat_dz_device, aos_stack.xmatZ_dz_device, "xmatZ_dz copy");
      }
      break;
    default:
      GAUXC_GENERIC_EXCEPTION("Save XMat + GKS NYI");
  }
}





template<bool is_fxc>
void AoSScheme1Base::inc_potential_impl( XCDeviceData* _data, density_id den_selector, bool do_m ){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Sync blas streams with master stream
  data->device_backend_->sync_blas_pool_with_master();

  auto do_syr2k = [&]( auto& handle, size_t npts, size_t nbe, auto* bf_ptr, auto* zptr, double fac, auto* v_ptr ) {
    syr2k( handle, DeviceBlasUplo::Lower, DeviceBlasOp::Trans, nbe, npts, 1.0, bf_ptr, npts,
      zptr, npts, fac, v_ptr, nbe ); 
  };

  // Launch SYR2K in round robin
  const auto n_blas_streams = data->device_backend_->blas_pool_size();
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    auto& task = tasks[iT];
    auto handle = data->device_backend_->blas_pool_handle( iT % n_blas_streams );
    do_syr2k(handle, task.npts, task.bfn_screening.nbe, task.bf, task.zmat, 0.0, task.nbe_scr);
    if(do_m) {
      do_syr2k(handle, task.npts, task.bfn_screening.nbe, task.dbfx, task.xmat_x, 1.0, task.nbe_scr);
      do_syr2k(handle, task.npts, task.bfn_screening.nbe, task.dbfy, task.xmat_y, 1.0, task.nbe_scr);
      do_syr2k(handle, task.npts, task.bfn_screening.nbe, task.dbfz, task.xmat_z, 1.0, task.nbe_scr);
    }
  }

  // Record completion of BLAS ops on master stream
  data->device_backend_->sync_master_with_blas_pool();

  // Increment global VXC
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  
  double* potential_ptr;
  if constexpr (is_fxc) {
    potential_ptr = static_stack.fxc_selector(den_selector);
    // cutlass_stack.vmat_array_device points to aos_stack.device_tasks[itask].nbe_scr
  } else {
    potential_ptr = static_stack.vxc_selector(den_selector);
  }

  auto vxc_ptr = static_stack.vxc_selector(den_selector);
  sym_task_inc_potential( ntasks, aos_stack.device_tasks,
    potential_ptr, nbf, submat_block_size,
    data->device_backend_->queue() );
  
  data->device_backend_->check_error("inc_potential_ptr" __FILE__ ": " + std::to_string(__LINE__));
}


void AoSScheme1Base::inc_vxc( XCDeviceData* _data, density_id den_selector, bool do_m ){
  inc_potential_impl<false>(_data, den_selector, do_m);
}

void AoSScheme1Base::inc_fxc( XCDeviceData* _data, density_id den_selector, bool do_m ){
  inc_potential_impl<true>(_data, den_selector, do_m);
}











void AoSScheme1Base::symmetrize_vxc( XCDeviceData* _data, density_id den_selector) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nbf = data->global_dims.nbf;
  auto static_stack  = data->static_stack;
  switch ( den_selector ) {
    case DEN_S:
      symmetrize_matrix( nbf, static_stack.vxc_s_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_Z:
      symmetrize_matrix( nbf, static_stack.vxc_z_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_Y:
      symmetrize_matrix( nbf, static_stack.vxc_y_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_X:
      symmetrize_matrix( nbf, static_stack.vxc_x_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "symmetrize_vxc: invalid density selected" );
  }
  
  data->device_backend_->check_error("symmetrize vxc" __FILE__ ": " + std::to_string(__LINE__));
}

void AoSScheme1Base::symmetrize_fxc( XCDeviceData* _data, density_id den_selector) {

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nbf = data->global_dims.nbf;
  auto static_stack  = data->static_stack;
  switch ( den_selector ) {
    case DEN_S:
      symmetrize_matrix( nbf, static_stack.fxc_s_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_Z:
      symmetrize_matrix( nbf, static_stack.fxc_z_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_Y:
      symmetrize_matrix( nbf, static_stack.fxc_y_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    case DEN_X:
      symmetrize_matrix( nbf, static_stack.fxc_x_device, nbf, 
            data->device_backend_->queue() ); 
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "symmetrize_fxc: invalid density selected" );
  }
  
  data->device_backend_->check_error("symmetrize fxc" __FILE__ ": " + std::to_string(__LINE__));
}




void AoSScheme1Base::inc_exc_grad_lda( XCDeviceData* _data, integrator_ks_scheme ks_scheme, bool with_weight_derivatives ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("LDA Grad NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nshell = data->global_dims.nshells;
  increment_exc_grad_lda( ks_scheme, nshell, 
    data->shell_to_task_stack.shell_to_task_device, 
    data->aos_stack.device_tasks,
    data->static_stack.exc_grad_device,
    with_weight_derivatives,
    data->device_backend_->queue() ); 
#endif
}

void AoSScheme1Base::inc_exc_grad_gga( XCDeviceData* _data, integrator_ks_scheme ks_scheme, bool with_weight_derivatives ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("GGA Grad NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nshell = data->global_dims.nshells;
  increment_exc_grad_gga( ks_scheme, nshell, 
    data->shell_to_task_stack.shell_to_task_device, 
    data->aos_stack.device_tasks,
    data->static_stack.exc_grad_device,
    with_weight_derivatives,
    data->device_backend_->queue() ); 
#endif
}

void AoSScheme1Base::inc_exc_grad_mgga( XCDeviceData* _data, integrator_ks_scheme ks_scheme, bool need_lapl, bool with_weight_derivatives ) {
#ifdef GAUXC_HAS_HIP
  GAUXC_GENERIC_EXCEPTION("MGGA Grad NYI for HIP Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nshell = data->global_dims.nshells;
  increment_exc_grad_mgga( ks_scheme, nshell, need_lapl,
    data->shell_to_task_stack.shell_to_task_device, 
    data->aos_stack.device_tasks,
    data->static_stack.exc_grad_device,
    with_weight_derivatives,
    data->device_backend_->queue() ); 
#endif
}


void AoSScheme1Base::eval_exx_fmat( XCDeviceData* _data ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX F-Matrix NYI for non-CUDA Backends");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();
  const auto nbf = data->global_dims.nbf;
  auto static_stack  = data->static_stack;

  // Pack the density matrix into (bfn, cou) shape
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto aos_stack     = data->aos_stack;
  asym_pack_submat( ntasks, aos_stack.device_tasks, static_stack.dmat_s_device,
    nbf, submat_block_size, data->device_backend_->queue() );

  // Sync blas streams with master stream
  data->device_backend_->sync_blas_pool_with_master();

  // Launch GEMM in round-robin
  const auto n_blas_streams = data->device_backend_->blas_pool_size();
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    auto& task = tasks[iT];
    auto handle = data->device_backend_->blas_pool_handle( iT % n_blas_streams );
    auto npts = task.npts;
    auto nbe_bfn = task.bfn_screening.nbe;
    auto nbe_cou = task.cou_screening.nbe;
    gemm( handle, DeviceBlasOp::NoTrans, DeviceBlasOp::NoTrans, 
      npts, nbe_cou, nbe_bfn, 1., task.bf, npts, task.nbe_scr, nbe_bfn, 
      0., task.fmat, npts );
  }

  // Record completion of BLAS ops on master stream
  data->device_backend_->sync_master_with_blas_pool();
#endif
}

void AoSScheme1Base::eval_exx_gmat( XCDeviceData* _data, 
  const BasisSetMap& basis_map ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX G-Matrix NYI for non-CUDA Backends");
#else

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  //const auto ntasks = tasks.size();
  const size_t nshells = data->global_dims.nshells;
  //auto static_stack  = data->static_stack;

  // XXX: Need to add screening capabilities, packing etc
  //const auto nbf = data->global_dims.nbf;


  if( basis_map.max_l() > 2 ) {
    GAUXC_GENERIC_EXCEPTION("GPU EXX + L>2 NYI");
  }
  
  // Determine purity of shell types
  std::vector<bool> sph_am(basis_map.max_l()+1);
  for( auto i = 0ul; i < nshells; ++i ) {
    sph_am[basis_map.shell_l(i)] =  sph_am[basis_map.shell_l(i)] | basis_map.shell_pure(i);
  }

  // Sanity Check
  for( auto i = 0ul; i < nshells; ++i ) {
    if(basis_map.shell_pure(i) != sph_am[basis_map.shell_l(i)])
      GAUXC_GENERIC_EXCEPTION("GPU EXX requires all shells of the same angular momentum to have the same purity");
  }
  

  // Zero out G
  for( auto& task : tasks ) {
    const size_t sz = task.npts*task.cou_screening.nbe;
    data->device_backend_->set_zero_async_master_queue( 
      sz, task.gmat, "Zero G" );
  }

  // Sync blas streams with master stream
  data->device_backend_->sync_blas_pool_with_master();

  // Launch Shell Pair Kernels in round-robin
  //const auto n_streams = data->device_backend_->blas_pool_size();

  auto& sp_to_task = data->shell_pair_to_task;
  #if 1
  constexpr bool do_batch = true;

  if( do_batch ) { // start batched code

    cudaStream_t stream = 
      data->device_backend_->queue().queue_as<util::cuda_stream>();

    XGPU::integral_0_task_batched(
      tasks.size(), data->subtask.size(),
      data->l_batch_diag_task_to_shell_pair_device[0].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_diag_task_to_shell_pair_device[0].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    if(basis_map.max_l() > 0) {
    XGPU::integral_1_task_batched(
      sph_am[1], tasks.size(), data->subtask.size(),
      data->l_batch_diag_task_to_shell_pair_device[1].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_diag_task_to_shell_pair_device[1].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_1_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }
    if(basis_map.max_l() > 1) {
    XGPU::integral_2_task_batched(
      sph_am[2], tasks.size(), data->subtask.size(),
      data->l_batch_diag_task_to_shell_pair_device[2].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_diag_task_to_shell_pair_device[2].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

  #define SP_LBATCH_IDX(I,J) (I*(basis_map.max_l()+1) + J)

    XGPU::integral_0_0_task_batched(
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[0].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[0].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_0_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));

    if(basis_map.max_l() > 0) {
    XGPU::integral_1_1_task_batched(
      sph_am[1], tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,1)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,1)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_1_1_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 1) {
    XGPU::integral_2_2_task_batched(
      sph_am[2], tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,2)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,2)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_2_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 0) {
    XGPU::integral_1_0_task_batched( true, sph_am[1],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(0,1)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(0,1)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_1_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 0) {
    XGPU::integral_1_0_task_batched( false, sph_am[1],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,0)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,0)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_1_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 1) {
    XGPU::integral_2_0_task_batched( true, sph_am[2],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(0,2)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(0,2)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 1) {
    XGPU::integral_2_0_task_batched( false, sph_am[2],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,0)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,0)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_0_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 1) {
    XGPU::integral_2_1_task_batched( true, sph_am[2], sph_am[1],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,2)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(1,2)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_1_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

    if(basis_map.max_l() > 1) {
    XGPU::integral_2_1_task_batched( false, sph_am[2], sph_am[1],
      tasks.size(), data->subtask.size(),
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,1)].max_prim_pairs, 0,
      data->aos_stack.device_tasks,
      data->l_batch_task_to_shell_pair_device[SP_LBATCH_IDX(2,1)].task_to_shell_pair_device,
      data->task_to_shell_pair_stack.subtask_device,
      data->task_to_shell_pair_stack.nprim_pairs_device,
      data->task_to_shell_pair_stack.pp_ptr_device,
      data->task_to_shell_pair_stack.sp_X_AB_device,
      data->task_to_shell_pair_stack.sp_Y_AB_device,
      data->task_to_shell_pair_stack.sp_Z_AB_device,
      dev_boys_table, stream
    );
    data->device_backend_->check_error("integral_2_1_task_batched" __FILE__ ": " + std::to_string(__LINE__));
    }

  } else { // end batched start unbatched

    cudaStream_t stream = 
      data->device_backend_->queue().queue_as<util::cuda_stream>();
    for( auto& sptt : sp_to_task ) { 
      size_t ntask_sp = sptt.task_idx.size();
      auto ish = sptt.idx_bra;
      auto jsh = sptt.idx_ket;
      for( auto i = 0ul; i < ntask_sp; i++ ) {
        const auto iT = sptt.task_idx[i];
        const auto i_off = sptt.task_shell_off_row[i];
        const auto j_off = sptt.task_shell_off_col[i];

        const auto& task = tasks[iT];
        //cudaStream_t stream = 
          //data->device_backend_->blas_pool_queue(iT % n_streams)
          //  .queue_as<util::cuda_stream>();

        XGPU::compute_integral_shell_pair( ish == jsh,
          task.npts,
          task.points_x,
          task.points_y,
          task.points_z,
          sptt.lA, sptt.lB,
          sptt.rA, sptt.rB,
          sptt.shell_pair_device,
          task.fmat + i_off*task.npts,
          task.fmat + j_off*task.npts,
          task.npts,
          task.gmat + i_off*task.npts,
          task.gmat + j_off*task.npts,
          task.npts,
          task.weights,
          dev_boys_table, stream ); 
      } // Loop over tasks within a shell pair
    } // Loop over shell pair maps
  } // end unbatched
  #else
  size_t isptt = 0;
  for( auto& sptt : sp_to_task ) {
    size_t ntask_sp = sptt.task_idx.size();
    auto ish = sptt.idx_bra;
    auto jsh = sptt.idx_ket;
    //std::cout << "SH " << ish << " " << jsh << std::endl;
    if( true ) {

      cudaStream_t stream = 
        data->device_backend_->queue().queue_as<util::cuda_stream>();
      const auto X_AB = sptt.rA.x - sptt.rB.x;
      const auto Y_AB = sptt.rA.y - sptt.rB.y;
      const auto Z_AB = sptt.rA.z - sptt.rB.z;
      XGPU::compute_integral_shell_pair_batched( ish == jsh, ntask_sp, 
        sptt.lA, sptt.lB, X_AB, Y_AB, Z_AB,
        data->shell_pair_to_task_stack.shell_pair_to_task_device + isptt,
        data->aos_stack.device_tasks, dev_boys_table, stream );

    } else {

      for( auto i = 0ul; i < ntask_sp; i++ ) {
        const auto iT = sptt.task_idx[i];
        const auto i_off = sptt.task_shell_off_row[i];
        const auto j_off = sptt.task_shell_off_col[i];

        const auto& task = tasks[iT];
        cudaStream_t stream = 
          data->device_backend_->queue().queue_as<util::cuda_stream>();
          //data->device_backend_->blas_pool_queue(iT % n_streams)
          //  .queue_as<util::cuda_stream>();

        XGPU::compute_integral_shell_pair( ish == jsh,
          task.npts,
          task.points_x,
          task.points_y,
          task.points_z,
          sptt.lA, sptt.lB,
          sptt.rA, sptt.rB,
          sptt.shell_pair_device,
          task.fmat + i_off*task.npts,
          task.fmat + j_off*task.npts,
          task.npts,
          task.gmat + i_off*task.npts,
          task.gmat + j_off*task.npts,
          task.npts,
          task.weights,
          dev_boys_table, stream ); 
      
      }

    }
    isptt++;
  }
  #endif


  // Record completion of BLAS ops on master stream
  data->device_backend_->sync_master_with_blas_pool();
#endif
}



void AoSScheme1Base::inc_exx_k( XCDeviceData* _data ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Sync blas streams with master stream
  data->device_backend_->sync_blas_pool_with_master();

  // Launch GEMM in round-robin
  const auto n_blas_streams = data->device_backend_->blas_pool_size();
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    auto& task = tasks[iT];
    auto handle = data->device_backend_->blas_pool_handle( iT % n_blas_streams );
    auto npts = task.npts;
    auto nbe_bfn = task.bfn_screening.nbe;
    auto nbe_cou = task.cou_screening.nbe;
    gemm( handle, DeviceBlasOp::Trans, DeviceBlasOp::NoTrans, 
      nbe_bfn, nbe_cou, npts, 1., task.bf, npts, task.gmat, npts, 0., 
      task.nbe_scr, nbe_bfn );
  }

  // Record completion of BLAS ops on master stream
  data->device_backend_->sync_master_with_blas_pool();

  // Increment EXX_K
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  asym_task_inc_potential( ntasks, aos_stack.device_tasks, 
    static_stack.exx_k_device, nbf, submat_block_size, 
    data->device_backend_->queue() );
#endif
}

void AoSScheme1Base::symmetrize_exx_k( XCDeviceData* _data ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto nbf = data->global_dims.nbf;
  auto static_stack  = data->static_stack;
  symmetrize_matrix_inc( nbf, static_stack.exx_k_device, nbf, 
    data->device_backend_->queue() ); 
#endif
}


void AoSScheme1Base::eval_exx_ek_screening_bfn_stats( XCDeviceData* _data ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks_ek = data->global_dims.ntask_ek;
  const auto ntasks = tasks.size();
  //const auto nbf = data->global_dims.nbf;
  auto aos_stack    = data->aos_stack;
  auto static_stack    = data->static_stack;
  GauXC::exx_ek_screening_bfn_stats( ntasks, aos_stack.device_tasks,
    static_stack.ek_max_bfn_sum_device, static_stack.ek_bfn_max_device, 
    ntasks_ek, data->device_backend_->queue() );
#endif
}


void AoSScheme1Base::exx_ek_shellpair_collision( double eps_E, double eps_K,
  XCDeviceData* _data, host_task_iterator tb, host_task_iterator te,
  const ShellPairCollection<double>& shpairs ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  const auto ntasks = std::distance(tb, te);
  if( ntasks > data->global_dims.ntask_ek ) 
    GAUXC_GENERIC_EXCEPTION("EK - Too Many Tasks");

  const auto nshells   = data->global_dims.nshells;
  const auto nbf   = data->global_dims.nbf;
  auto static_stack    = data->static_stack;

  GauXC::exx_ek_shellpair_collision( ntasks, nshells, nbf,
    static_stack.dmat_s_device, nbf,
    static_stack.vshell_max_sparse_device, 
    static_stack.shpair_row_ind_device,
    static_stack.shpair_col_ind_device,
    static_stack.ek_max_bfn_sum_device,
    static_stack.ek_bfn_max_device, data->global_dims.ntask_ek, 
    static_stack.shells_device, static_stack.shell_to_bf_device,
    static_stack.shell_sizes_device, eps_E, eps_K,
    data->dynmem_ptr, data->dynmem_sz,
    tb, te, shpairs,
    data->device_backend_->queue(),
    data->device_backend_->master_blas_handle()
   );
#endif
}


}

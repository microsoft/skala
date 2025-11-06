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
#pragma once
#include "device/device_backend.hpp"
#include <memory>
#include "device_specific/hip_util.hpp"
#include "device_specific/hipblas_util.hpp"

namespace GauXC {

struct HIPBackend : public DeviceBackend {

  device_buffer_t   allocate_device_buffer(int64_t sz) override final;
  size_t            get_available_mem() override final;
  void              free_device_buffer( void* ptr ) override final;
  void              master_queue_synchronize() override final;
  void              create_blas_queue_pool(int32_t)   override final;
  void              sync_master_with_blas_pool() override final;
  void              sync_blas_pool_with_master() override final;
  size_t            blas_pool_size() override final;

  device_queue       queue() override final;
  device_queue       blas_pool_queue(int32_t) override final;
  device_blas_handle blas_pool_handle(int32_t) override final;
  device_blas_handle master_blas_handle() override final;

  void copy_async_( size_t sz, const void* src, void* dest, 
                    std::string msg ) override final;
  void set_zero_( size_t sz, void* data, std::string msg) override final;
  void set_zero_async_master_queue_( size_t sz, void* data, std::string msg) override final;

  void copy_async_2d_( size_t M, size_t N, const void* A, size_t LDA,
    void* B, size_t LDB, std::string msg ) override final;

  void check_error_(std::string msg) override final;

  HIPBackend();
  ~HIPBackend() noexcept;

  // Execution management
  std::shared_ptr<util::hip_stream>   master_stream = nullptr;
  std::shared_ptr<util::hipblas_handle> master_handle = nullptr;

  std::vector<std::shared_ptr<util::hip_stream>>     blas_streams;
  std::vector<std::shared_ptr<util::hipblas_handle>> blas_handles;
};

}

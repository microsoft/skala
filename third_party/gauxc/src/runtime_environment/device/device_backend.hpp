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
#include <tuple>
#include <vector>
#include <memory>
#include <string>
#include "device_queue.hpp"
#include "device_blas_handle.hpp"
#include <gauxc/gauxc_config.hpp>

#ifdef GAUXC_HAS_MAGMA
#include "device_specific/magma_util.hpp"
#endif

namespace GauXC {

class DeviceBackend {

public:

  using device_buffer_t = std::tuple<void*, size_t>;

  virtual device_buffer_t   allocate_device_buffer(int64_t sz) = 0;
  virtual size_t            get_available_mem() = 0;
  virtual void              free_device_buffer( void* ptr ) = 0;
  virtual void              master_queue_synchronize() = 0;
  virtual void              create_blas_queue_pool(int32_t)   = 0;
  virtual void              sync_master_with_blas_pool() = 0;
  virtual void              sync_blas_pool_with_master() = 0;
  virtual size_t            blas_pool_size() = 0;

  virtual device_queue       queue() = 0;
  virtual device_queue       blas_pool_queue(int32_t) = 0;
  virtual device_blas_handle blas_pool_handle(int32_t) = 0;
  virtual device_blas_handle master_blas_handle() = 0;

  #ifdef GAUXC_HAS_MAGMA
  inline util::magma_queue* master_magma_queue(){ return master_magma_queue_.get(); }
  #endif

  virtual ~DeviceBackend() noexcept = default;

  template <typename T>
  void copy_async( size_t sz, const T* src, T* dest, std::string msg ) {
    copy_async_( sz * sizeof(T), src, dest, msg );
  }

  template <typename T>
  void copy_async_2d( size_t M, size_t N, const T* A, size_t LDA,
    T* B, size_t LDB, std::string msg ) {
    copy_async_2d_( M*sizeof(T), N, A, LDA*sizeof(T), B, LDB*sizeof(T), msg );
  }

  template <typename T>
  void set_zero(size_t sz, T* data, std::string msg) {
    set_zero_( sz * sizeof(T), data, msg );
  }

  template <typename T>
  void set_zero_async_master_queue(size_t sz, T* data, std::string msg) {
    set_zero_async_master_queue_( sz * sizeof(T), data, msg );
  }

  void check_error(std::string msg) {
    check_error_(msg);
  }

protected:


  #ifdef GAUXC_HAS_MAGMA
  std::shared_ptr<util::magma_queue> master_magma_queue_;
  #endif

  virtual void copy_async_( size_t sz, const void* src, void* dest, 
                            std::string msg ) = 0;

  virtual void copy_async_2d_( size_t M, size_t N, const void* A, size_t LDA,
    void* B, size_t LDB, std::string msg ) = 0;


  virtual void set_zero_( size_t sz, void* data, std::string msg) = 0;
  virtual void set_zero_async_master_queue_( size_t sz, void* data, 
    std::string msg) = 0;

  virtual void check_error_(std::string msg) = 0;
};



/// Generate the default device backend for this platform
std::unique_ptr<DeviceBackend> make_device_backend();

}

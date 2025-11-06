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
#include "../runtime_environment_impl.hpp"
#include "device_backend.hpp"
#include <gauxc/exceptions.hpp>

namespace GauXC::detail {

size_t memory_cap();

class DeviceRuntimeEnvironmentImpl : public RuntimeEnvironmentImpl {

private:
  using parent_type = RuntimeEnvironmentImpl;

  bool  i_own_this_memory_ = false;
  void* device_memory_;
  size_t device_memory_size_;

  std::unique_ptr<DeviceBackend> device_backend_;

public:

  DeviceRuntimeEnvironmentImpl(GAUXC_MPI_CODE(MPI_Comm c,) void* p,
   size_t sz) : parent_type(GAUXC_MPI_CODE(c)), 
     i_own_this_memory_(false), device_memory_(p), 
     device_memory_size_(sz),
     device_backend_{make_device_backend()} {}


  explicit DeviceRuntimeEnvironmentImpl(GAUXC_MPI_CODE(MPI_Comm c,)
    double fill_fraction) :
    DeviceRuntimeEnvironmentImpl(GAUXC_MPI_CODE(c,) nullptr, 0) {

    // Allocate Device Memory
    auto avail = device_backend_->get_available_mem();
    avail = std::min( avail, detail::memory_cap() );

    std::tie( device_memory_, device_memory_size_ ) = 
      device_backend_->allocate_device_buffer(fill_fraction * avail);

    i_own_this_memory_ = true;

  }

  ~DeviceRuntimeEnvironmentImpl() noexcept {
    if(i_own_this_memory_ and device_memory_ and device_memory_size_) {
      device_backend_->free_device_buffer(device_memory_);
    }
  }

  inline DeviceBackend* device_backend() { return device_backend_.get(); }
  inline const DeviceBackend* device_backend() const { return device_backend_.get(); }

  inline void* device_memory() { return device_memory_; }
  inline void* device_memory() const { return device_memory_; }
  inline size_t device_memory_size() { return device_memory_size_; }
  inline size_t device_memory_size() const { return device_memory_size_; }
  inline bool owns_memory() const { return i_own_this_memory_; }

  inline void release_buffer() {
    if(i_own_this_memory_ and device_memory_ and device_memory_size_) {
      device_backend_->free_device_buffer(device_memory_);
    } else {
      GAUXC_GENERIC_EXCEPTION("GauXC Cannot Release A Buffer It Does Not Own");
    }
  }

  inline void set_buffer(void* p, size_t sz) {
    if(owns_memory()) {
      release_buffer();
      i_own_this_memory_ = false;
    }

    device_memory_ = p;
    device_memory_size_ = sz;
  }
};


template <typename ...Args>
std::unique_ptr<RuntimeEnvironmentImpl> make_device_runtime(Args&&... args) {
  return std::make_unique<DeviceRuntimeEnvironmentImpl>(
    std::forward<Args>(args)...
  );
}

}

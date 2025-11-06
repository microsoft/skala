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
#include <gauxc/runtime_environment/fwd.hpp>
#include <memory>
#include <gauxc/util/mpi.hpp>

namespace GauXC {

namespace detail {
  class RuntimeEnvironmentImpl;
  #ifdef GAUXC_HAS_DEVICE
  DeviceRuntimeEnvironment as_device_runtime( const RuntimeEnvironment& );
  #endif
}

class RuntimeEnvironment {

protected:

#ifdef GAUXC_HAS_DEVICE
  friend DeviceRuntimeEnvironment 
    detail::as_device_runtime(const RuntimeEnvironment&); 
#endif

  using pimpl_type = detail::RuntimeEnvironmentImpl;
  using pimpl_ptr_type = std::shared_ptr<pimpl_type>;
  pimpl_ptr_type pimpl_;
  RuntimeEnvironment( pimpl_ptr_type ptr );

public:

  explicit RuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm comm));
  virtual ~RuntimeEnvironment() noexcept;

  RuntimeEnvironment( const RuntimeEnvironment& );
  RuntimeEnvironment( RuntimeEnvironment&& ) noexcept;

  GAUXC_MPI_CODE(MPI_Comm comm() const;)
  int comm_rank() const;
  int comm_size() const;

  int shared_usage_count() const;

};

#ifdef GAUXC_HAS_DEVICE
class DeviceRuntimeEnvironment : public RuntimeEnvironment {

private:

  using parent_type = RuntimeEnvironment;
  friend DeviceRuntimeEnvironment 
    detail::as_device_runtime(const RuntimeEnvironment&); 

  using parent_type::pimpl_type;
  using parent_type::pimpl_ptr_type;
  DeviceRuntimeEnvironment( pimpl_ptr_type ptr );

public:

  DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm comm,) void* mem, 
    size_t mem_sz);
  DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm,) double fill_fraction);

  ~DeviceRuntimeEnvironment() noexcept;
  DeviceRuntimeEnvironment( const DeviceRuntimeEnvironment& );
  DeviceRuntimeEnvironment( DeviceRuntimeEnvironment&& ) noexcept;

  void* device_memory() const ;
  size_t device_memory_size() const ;
  bool owns_memory() const;
  DeviceBackend* device_backend() const;

  void release_buffer();
  void set_buffer(void* m, size_t sz);
};
#endif

}

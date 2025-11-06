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
#include "device_runtime_environment_impl.hpp"
#include <gauxc/exceptions.hpp>
#include <iostream>

namespace GauXC {

auto* device_runtime_pimpl_cast(detail::RuntimeEnvironmentImpl* ptr) {
  auto dp = dynamic_cast<detail::DeviceRuntimeEnvironmentImpl*>(ptr);
  if(!dp) GAUXC_GENERIC_EXCEPTION("Not A Device Implemention");
  return dp;
}


namespace detail {

DeviceRuntimeEnvironment as_device_runtime(const RuntimeEnvironment& rt) {
  if( auto* p = dynamic_cast<const DeviceRuntimeEnvironment*>(&rt) ) {
    // Instance is actually a DeviceRuntimeEnvironment
    return DeviceRuntimeEnvironment(*p);
  } else {
    // Try a PIMPL cast
    auto pimpl = device_runtime_pimpl_cast(rt.pimpl_.get());
    (void)pimpl;
    return DeviceRuntimeEnvironment(rt.pimpl_);
  }
}

}

DeviceRuntimeEnvironment::DeviceRuntimeEnvironment(pimpl_ptr_type ptr):
  RuntimeEnvironment(ptr) {}

DeviceRuntimeEnvironment::DeviceRuntimeEnvironment(
  GAUXC_MPI_CODE(MPI_Comm c,) void* p, size_t sz ) :
  RuntimeEnvironment(
    detail::make_device_runtime( GAUXC_MPI_CODE(c,) p,sz)
  ) {}

DeviceRuntimeEnvironment::DeviceRuntimeEnvironment(
  GAUXC_MPI_CODE(MPI_Comm c,) double ff) :
  RuntimeEnvironment(detail::make_device_runtime(GAUXC_MPI_CODE(c,)ff)) {}

DeviceRuntimeEnvironment::~DeviceRuntimeEnvironment() noexcept = default;

DeviceRuntimeEnvironment::DeviceRuntimeEnvironment(
  DeviceRuntimeEnvironment&& other) noexcept = default;

DeviceRuntimeEnvironment::DeviceRuntimeEnvironment(
  const DeviceRuntimeEnvironment& other) = default;

void* DeviceRuntimeEnvironment::device_memory() const {
  return device_runtime_pimpl_cast(pimpl_.get())->device_memory();
}
size_t DeviceRuntimeEnvironment::device_memory_size() const {
  return device_runtime_pimpl_cast(pimpl_.get())->device_memory_size();
}
DeviceBackend* DeviceRuntimeEnvironment::device_backend() const {
  return device_runtime_pimpl_cast(pimpl_.get())->device_backend();
}
bool DeviceRuntimeEnvironment::owns_memory() const {
  return device_runtime_pimpl_cast(pimpl_.get())->owns_memory();
}
void DeviceRuntimeEnvironment::release_buffer() {
  device_runtime_pimpl_cast(pimpl_.get())->release_buffer();
}
void DeviceRuntimeEnvironment::set_buffer(void* p, size_t sz) {
  device_runtime_pimpl_cast(pimpl_.get())->set_buffer(p, sz);
}

}

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
#include <gauxc/runtime_environment.hpp>
#include "runtime_environment_impl.hpp"

namespace GauXC {

RuntimeEnvironment::RuntimeEnvironment( pimpl_ptr_type ptr ) :
  pimpl_(ptr) {}

RuntimeEnvironment::RuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm c)) :
  RuntimeEnvironment( std::make_unique<detail::RuntimeEnvironmentImpl>(GAUXC_MPI_CODE(c)) ) {}

RuntimeEnvironment::~RuntimeEnvironment() noexcept = default;

RuntimeEnvironment::RuntimeEnvironment(const RuntimeEnvironment& other) :
  pimpl_(other.pimpl_) {}
RuntimeEnvironment::RuntimeEnvironment(RuntimeEnvironment&& other) noexcept :
  RuntimeEnvironment(std::move(other.pimpl_)) {}

#ifdef GAUXC_HAS_MPI
MPI_Comm RuntimeEnvironment::comm() const {
  return pimpl_->comm();
}
#endif

int RuntimeEnvironment::comm_rank() const {
  return pimpl_->comm_rank();
}

int RuntimeEnvironment::comm_size() const {
  return pimpl_->comm_size();
}

int RuntimeEnvironment::shared_usage_count() const {
  return pimpl_.use_count();
}

}

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
#include "reduction_driver_impl.hpp"
#include <gauxc/exceptions.hpp>


namespace GauXC {

ReductionDriver::ReductionDriver( std::unique_ptr<pimpl_type>&& pimpl ): 
  pimpl_( std::move(pimpl) ) { }

ReductionDriver::ReductionDriver() : ReductionDriver( nullptr ) { }

ReductionDriver::ReductionDriver( const ReductionDriver& other ) :
  ReductionDriver(other.pimpl_->clone()){ }

ReductionDriver::ReductionDriver( ReductionDriver&& ) noexcept = default;
              
ReductionDriver::~ReductionDriver() noexcept = default;


void ReductionDriver::allreduce_typeerased( const void* src, void* dest, size_t size, ReductionOp op, std::type_index idx, std::any optional_args ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  pimpl_->allreduce_typeerased(src, dest, size, op, idx, optional_args);
}

void ReductionDriver::allreduce_inplace_typeerased( void* data, size_t size, ReductionOp op, std::type_index idx, std::any optional_args ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  pimpl_->allreduce_inplace_typeerased(data, size, op, idx, optional_args);
}

bool ReductionDriver::takes_host_memory() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->takes_host_memory();
}
bool ReductionDriver::takes_device_memory() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->takes_device_memory();
}




}

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

namespace GauXC::detail {


ReductionDriverImpl::ReductionDriverImpl( const RuntimeEnvironment& rt ) 
  : runtime_(rt){}

ReductionDriverImpl::~ReductionDriverImpl() noexcept = default;
ReductionDriverImpl::ReductionDriverImpl(const ReductionDriverImpl& ) = default;

}

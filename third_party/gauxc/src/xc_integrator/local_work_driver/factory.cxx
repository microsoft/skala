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
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include "host/reference_local_host_work_driver.hpp"
#ifdef GAUXC_HAS_DEVICE
#include "device/cuda/cuda_aos_scheme1.hpp"
#include "device/hip/hip_aos_scheme1.hpp"
#endif

namespace GauXC {

LocalWorkDriverFactory::ptr_return_t
  LocalWorkDriverFactory::make_local_work_driver( ExecutionSpace ex, 
    std::string name, LocalWorkSettings settings ) {

  std::transform( name.begin(), name.end(), name.begin(), ::toupper );
  (void)(settings);

  switch(ex) {

  case ExecutionSpace::Host:
    if( name == "DEFAULT" ) name = "REFERENCE";

    if( name == "REFERENCE" )
      return std::make_unique<LocalHostWorkDriver>(
        std::make_unique<ReferenceLocalHostWorkDriver>()
      );
    else
      GAUXC_GENERIC_EXCEPTION("LWD Not Recognized: " + name);

  case ExecutionSpace::Device:
    if( name == "DEFAULT" ) name = "SCHEME1";

#ifdef GAUXC_HAS_CUDA
    using scheme1_default = CudaAoSScheme1<>;
#ifdef GAUXC_HAS_MAGMA
    using scheme1_magma   = CudaAoSScheme1<AoSScheme1MAGMABase>;
#endif
#ifdef GAUXC_HAS_CUTLASS
    using scheme1_cutlass   = CudaAoSScheme1<AoSScheme1CUTLASSBase>;
#endif
#elif defined(GAUXC_HAS_HIP)
    using scheme1_default = HipAoSScheme1<>;
#ifdef GAUXC_HAS_MAGMA
    using scheme1_magma   = HipAoSScheme1<AoSScheme1MAGMABase>;
#endif
#endif

#ifdef GAUXC_HAS_DEVICE
    if( name == "SCHEME1" )
      return std::make_unique<LocalDeviceWorkDriver>( std::make_unique<scheme1_default>() );
#ifdef GAUXC_HAS_MAGMA
    else if( name == "SCHEME1-MAGMA" )
      return std::make_unique<LocalDeviceWorkDriver>( std::make_unique<scheme1_magma>() );
#endif
#ifdef GAUXC_HAS_CUTLASS
    else if( name == "SCHEME1-CUTLASS" )
      return std::make_unique<LocalDeviceWorkDriver>( std::make_unique<scheme1_cutlass>() );
#endif
    else
#endif
      GAUXC_GENERIC_EXCEPTION("LWD Not Recognized: " + name);


  default:
    GAUXC_GENERIC_EXCEPTION("Execution Space Not Recognized");

  }

  return nullptr;

}

}

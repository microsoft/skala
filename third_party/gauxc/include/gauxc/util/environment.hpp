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
#include <gauxc/gauxc_config.hpp>
#include <gauxc/enums.hpp>

namespace GauXC {
  
inline int gauxc_max_am(ExecutionSpace ex, SupportedAlg alg) {
  switch(ex) {
    #ifdef GAUXC_HAS_HOST
    case ExecutionSpace::Host:
      switch(alg) {
        case SupportedAlg::XC: 
        case SupportedAlg::DEN: 
          return GAUXC_CPU_XC_MAX_AM;
        case SupportedAlg::SNLINK:
          return GAUXC_CPU_SNLINK_MAX_AM;
        default: return -1;
      }
    #endif
    #ifdef GAUXC_HAS_DEVICE
    case ExecutionSpace::Device:
      switch(alg) {
        case SupportedAlg::XC: 
        case SupportedAlg::DEN: 
          return GAUXC_GPU_XC_MAX_AM;
        case SupportedAlg::SNLINK:
          return GAUXC_GPU_SNLINK_MAX_AM;
        default: return -1;
      }
    #endif
    default: return -1;
  }
}

}

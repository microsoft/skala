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
#include "exceptions/hipblas_exception.hpp"

#ifdef GAUXC_HAS_HIP

namespace GauXC {
namespace util  {

struct hipblas_handle {

  hipblasHandle_t handle;
  inline hipblas_handle() {
    auto stat = hipblasCreate( &handle );
    GAUXC_HIPBLAS_ERROR("HIPBLAS Handle Create Failed", stat);
  }

  inline ~hipblas_handle() noexcept {
    if( handle != 0 ) hipblasDestroy( handle );
  }

  hipblas_handle( const hipblas_handle& ) = delete;
  inline hipblas_handle( hipblas_handle&& other ) noexcept {
    handle = other.handle;
    other.handle = 0;
  };

  inline operator hipblasHandle_t() const { return handle; }

};


inline static hipStream_t get_stream( hipblasHandle_t handle ) {
  hipStream_t stream;
  auto stat = hipblasGetStream(handle, &stream );  
  GAUXC_HIPBLAS_ERROR("HIPBLAS GET STREAM FAILED", stat );
  return stream;
}

}
}

#endif

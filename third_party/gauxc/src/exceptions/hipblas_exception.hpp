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

#include <gauxc/exceptions.hpp>
#include <stdexcept>
#include <string>
#include <sstream>

#ifdef GAUXC_HAS_HIP
#include "hip/hip_runtime.h"
#include <hipblas.h>

namespace GauXC {

namespace detail {

/**
 *  @brief Return a descriptive error string pertaining to a hipBLAS error code
 *
 *  @param[in] error hipBLAS error code
 *  @returns   String pertaining to "error"
 */
static std::string hipblasGetErrorString(hipblasStatus_t error) {
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "HIPBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "HIPBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";

        case HIPBLAS_STATUS_INVALID_ENUM:
            return "HIPBLAS_STATUS_INVALID_ENUM";

        case HIPBLAS_STATUS_UNKNOWN:
            return "HIPBLAS_STATUS_UNKNOWN";
    }
  

    return "<unknown>";
}

}

/**
 *  @brief A class to handle excecptions arising from hipBLAS operations
 */
class hipblas_exception : public std::exception {

  std::string file_;         ///< File which contains the code that threw the exception
  int         line_;         ///< Line number of file_ that threw exception
  std::string msg_prefix_;   ///< General descriptor of task which threw exception
  hipblasStatus_t err_code_;  ///< hipBLAS error code pertaining to the thrown exception

  /**
   *  @brief Get a descriptive message pertaining to the thrown hipBLAS error
   *
   *  @returns a descritive message pertaining to the hipBLAS error represented by
   *  the internal state of the exception object.
   */
  const char* what() const noexcept override {
     std::stringstream ss;
     ss << "HIPBLAS Exception (" << msg_prefix_ << ")" << std::endl
        << "  Error Code " << int(err_code_) << ": \"" 
                           << detail::hipblasGetErrorString( err_code_ ) 
                           << "\"" << std::endl
        << "  File       " << file_ << std::endl
        << "  Line       " << line_ << std::endl;

     auto msg = ss.str();

     return strdup( msg.c_str() );
  }

public:

  /**
   *  @brief Construct a hipblas_exception object
   *
   *  @param[in] file File which contains the code that threw the exception
   *  @param[in] line Line number of file that threw exception
   *  @param[in] msg  General descriptor of task which threw exception
   *  @param[in] err  hipBLAS error code pertaining to the thrown exception
   */
  hipblas_exception( std::string file, int line, std::string msg, 
                    hipblasStatus_t err ) :
    file_(file), line_(line), msg_prefix_(msg), err_code_(err) { }

}; // class hipblas_exception

}


// Macro to wrap hipBLAS error handling
#define GAUXC_HIPBLAS_ERROR( MSG, ERR ) \
  if( ERR != HIPBLAS_STATUS_SUCCESS ) \
    throw hipblas_exception( __FILE__, __LINE__, MSG, ERR );

#endif

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

#ifdef GAUXC_HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GauXC {

namespace detail {

/**
 *  @brief Return a descriptive error string pertaining to a cuBLAS error code
 *
 *  @param[in] error cuBLAS error code
 *  @returns   String pertaining to "error"
 */
static std::string cublasGetErrorString(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

}

/**
 *  @brief A class to handle excecptions arising from cuBLAS operations
 */
class cublas_exception : public std::exception {

  std::string file_;         ///< File which contains the code that threw the exception
  int         line_;         ///< Line number of file_ that threw exception
  std::string msg_prefix_;   ///< General descriptor of task which threw exception
  cublasStatus_t err_code_;  ///< cuBLAS error code pertaining to the thrown exception

  /**
   *  @brief Get a descriptive message pertaining to the thrown cuBLAS error
   *
   *  @returns a descritive message pertaining to the cuBLAS error represented by
   *  the internal state of the exception object.
   */
  const char* what() const noexcept override {
     std::stringstream ss;
     ss << "CUBLAS Exception (" << msg_prefix_ << ")" << std::endl
        << "  Error Code " << int(err_code_) << ": \"" 
                           << detail::cublasGetErrorString( err_code_ ) 
                           << "\"" << std::endl
        << "  File       " << file_ << std::endl
        << "  Line       " << line_ << std::endl;

     auto msg = ss.str();

     return strdup( msg.c_str() );
  }

public:

  /**
   *  @brief Construct a cublas_exception object
   *
   *  @param[in] file File which contains the code that threw the exception
   *  @param[in] line Line number of file that threw exception
   *  @param[in] msg  General descriptor of task which threw exception
   *  @param[in] err  cuBLAS error code pertaining to the thrown exception
   */
  cublas_exception( std::string file, int line, std::string msg, 
                    cublasStatus_t err ) :
    file_(file), line_(line), msg_prefix_(msg), err_code_(err) { }

}; // class cublas_exception

}


// Macro to wrap cuBLAS error handling
#define GAUXC_CUBLAS_ERROR( MSG, ERR ) \
  if( ERR != CUBLAS_STATUS_SUCCESS ) \
    throw cublas_exception( __FILE__, __LINE__, MSG, ERR );

#endif

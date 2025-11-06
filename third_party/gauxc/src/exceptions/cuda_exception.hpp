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
#include <cuda_runtime.h>
#include <string.h>

namespace GauXC {

/**
 *  @brief A class to handle excecptions arising from CUDA operations
 */
class cuda_exception : public std::exception {

  std::string file_;       ///< File which contains the code that threw the exception
  int         line_;       ///< Line number of file_ that threw exception
  std::string msg_prefix_; ///< General descriptor of task which threw exception
  cudaError_t err_code_;   ///< CUDA error code pertaining to the thrown exception

  /**
   *  @brief Get a descriptive message pertaining to the thrown CUDA error
   *
   *  @returns a descritive message pertaining to the CUDA error represented by
   *  the internal state of the exception object.
   */
  const char* what() const noexcept override {
     std::stringstream ss;
     ss << "CUDA Exception (" << msg_prefix_ << ")" << std::endl
        << "  Error Code " << int(err_code_) << ": \"" 
                           << cudaGetErrorString( err_code_ ) << "\"" << std::endl
        << "  File       " << file_ << std::endl
        << "  Line       " << line_ << std::endl;

     auto msg = ss.str();

     return strdup( msg.c_str() );
  }

public:

  /**
   *  @brief Construct a cuda_exception object
   *
   *  @param[in] file File which contains the code that threw the exception
   *  @param[in] line Line number of file that threw exception
   *  @param[in] msg  General descriptor of task which threw exception
   *  @param[in] err  CUDA error code pertaining to the thrown exception
   */
  cuda_exception( std::string file, int line, std::string msg, cudaError_t err ) :
    file_(file), line_(line), msg_prefix_(msg), err_code_(err) { }

}; // class cuda_exception

} // namespace GauXC

// Macro to wrap CUDA error handling
#define GAUXC_CUDA_ERROR( MSG, ERR ) \
  if( ERR != cudaSuccess ) \
    throw cuda_exception( __FILE__, __LINE__, MSG, ERR );

#endif

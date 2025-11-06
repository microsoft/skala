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


namespace GauXC {

/**
 *  @brief A class to handle excecptions arising from HIP operations
 */
class hip_exception : public std::exception {

  std::string file_;       ///< File which contains the code that threw the exception
  int         line_;       ///< Line number of file_ that threw exception
  std::string msg_prefix_; ///< General descriptor of task which threw exception
  hipError_t err_code_;   ///< HIP error code pertaining to the thrown exception

  /**
   *  @brief Get a descriptive message pertaining to the thrown HIP error
   *
   *  @returns a descritive message pertaining to the HIP error represented by
   *  the internal state of the exception object.
   */
  const char* what() const noexcept override {
     std::stringstream ss;
     ss << "HIP Exception (" << msg_prefix_ << ")" << std::endl
        << "  Error Code " << int(err_code_) << ": \"" 
                           << hipGetErrorString( err_code_ ) << "\"" << std::endl
        << "  File       " << file_ << std::endl
        << "  Line       " << line_ << std::endl;

     auto msg = ss.str();

     return strdup( msg.c_str() );
  }

public:

  /**
   *  @brief Construct a hip_exception object
   *
   *  @param[in] file File which contains the code that threw the exception
   *  @param[in] line Line number of file that threw exception
   *  @param[in] msg  General descriptor of task which threw exception
   *  @param[in] err  HIP error code pertaining to the thrown exception
   */
  hip_exception( std::string file, int line, std::string msg, hipError_t err ) :
    file_(file), line_(line), msg_prefix_(msg), err_code_(err) { }

}; // class hip_exception

} // namespace GauXC

// Macro to wrap HIP error handling
#define GAUXC_HIP_ERROR( MSG, ERR ) \
  if( ERR != hipSuccess ) \
    throw hip_exception( __FILE__, __LINE__, MSG, ERR );

#endif

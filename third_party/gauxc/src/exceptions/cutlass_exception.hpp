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

#ifdef GAUXC_HAS_CUTLASS
#include <cutlass/cutlass.h>
#include <string.h>

namespace GauXC {

/**
 *  @brief A class to handle excecptions arising from CUTLASS operations
 */
class cutlass_exception : public std::exception {

  std::string file_;       ///< File which contains the code that threw the exception
  int         line_;       ///< Line number of file_ that threw exception
  std::string msg_prefix_; ///< General descriptor of task which threw exception
  cutlass::Status status_; ///< CUTLASS status pertaining to the thrown exception

  /**
   *  @brief Get a descriptive message pertaining to the thrown CUTLASS error
   *
   *  @returns a descritive message pertaining to the CUTLASS error represented by
   *  the internal state of the exception object.
   */
  const char* what() const noexcept override {
     std::stringstream ss;
     ss << "CUTLASS Exception (" << msg_prefix_ << ")" << std::endl
        << "  Error Code " << int(status_) << ": \"" 
                           << cutlassGetStatusString( status_ ) << "\"" << std::endl
        << "  File       " << file_ << std::endl
        << "  Line       " << line_ << std::endl;

     auto msg = ss.str();

     return strdup( msg.c_str() );
  }

public:

  /**
   *  @brief Construct a cutlass_exception object
   *
   *  @param[in] file File which contains the code that threw the exception
   *  @param[in] line Line number of file that threw exception
   *  @param[in] msg  General descriptor of task which threw exception
   *  @param[in] err  CUTLASS status pertaining to the thrown exception
   */
  cutlass_exception( std::string file, int line, std::string msg, cutlass::Status status ) :
    file_(file), line_(line), msg_prefix_(msg), status_(status) { }

}; // class cutlass_exception

} // namespace GauXC

// Macro to wrap CUTLASS error handling
#define GAUXC_CUTLASS_ERROR( MSG, ERR ) \
  if( ERR != cutlass::Status::kSuccess) \
    throw cutlass_exception( __FILE__, __LINE__, MSG, ERR );

#endif

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
#include <memory>
#include <string>
#include <gauxc/enums.hpp>

namespace GauXC {

/// Base class for all LocalWorkDriver instances
struct LocalWorkDriver { 
  virtual ~LocalWorkDriver() noexcept = default; 
};

/// Base type for all types that specify LWD settings (trivial)
struct LocalWorkSettings { virtual ~LocalWorkSettings() noexcept = default; };





/// Factory to generate LocalWorkDriver instances
class LocalWorkDriverFactory {

public:

  using ptr_return_t = std::unique_ptr<LocalWorkDriver>;

  /** Generate a LWD instance
   * 
   *  @param[in] ex        The Execution space for the LWD driver
   *  @param[in] name      The name of the LWD driver to construct (e.g. "Default" or "Reference")
   *  @param[in] settings  Settings to pass to LWD construction
   */
  static ptr_return_t make_local_work_driver(ExecutionSpace ex, 
    std::string name = "Default", 
    LocalWorkSettings settings = LocalWorkSettings());

private:

  static ptr_return_t make_reference_host_driver();

};

}

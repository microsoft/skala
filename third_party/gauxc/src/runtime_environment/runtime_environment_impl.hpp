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
#include <gauxc/runtime_environment.hpp>

namespace GauXC::detail {

class RuntimeEnvironmentImpl {

protected:
  GAUXC_MPI_CODE(MPI_Comm comm_;)
  int comm_rank_;
  int comm_size_;

public:

  explicit RuntimeEnvironmentImpl(GAUXC_MPI_CODE(MPI_Comm c)) : 
    GAUXC_MPI_CODE(comm_(c),)
    comm_rank_(0), comm_size_(1) {

  #ifdef GAUXC_HAS_MPI
    MPI_Comm_rank( comm_, &comm_rank_ );
    MPI_Comm_size( comm_, &comm_size_ );
  #endif

  }

  virtual ~RuntimeEnvironmentImpl() noexcept = default;

#ifdef GAUXC_HAS_MPI
  inline MPI_Comm comm() const { return comm_; }
#endif

  inline int comm_rank() const { return comm_rank_; }
  inline int comm_size() const { return comm_size_; }

};

}

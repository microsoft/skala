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
#include "device_reduction_driver.hpp"
#include <nccl.h>
#include <memory>

namespace GauXC {

namespace util {
struct nccl_comm {

  ncclComm_t comm;

  inline nccl_comm( MPI_Comm mpi_comm ) { 
    int32_t world_rank, world_size;
    MPI_Comm_rank( mpi_comm, &world_rank );
    MPI_Comm_size( mpi_comm, &world_size );

    ncclUniqueId id;
    if (world_rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclCommInitRank(&comm, world_size, id, world_rank);
  }

  inline ~nccl_comm() noexcept {
    if( comm != 0 ) ncclCommDestroy(comm);
  }

  nccl_comm( const nccl_comm& ) = delete;
  inline nccl_comm( nccl_comm&& other ) noexcept {
    comm = other.comm;
    other.comm = 0;
  };

  inline operator ncclComm_t() const { return comm; }

};
}


struct NCCLReductionDriver : public DeviceReductionDriver {

  std::shared_ptr<util::nccl_comm> nccl_comm_;

  NCCLReductionDriver(const RuntimeEnvironment& rt);
  virtual ~NCCLReductionDriver() noexcept;
  NCCLReductionDriver(const NCCLReductionDriver& );

  void allreduce_typeerased( const void*, void*, size_t, ReductionOp, std::type_index, std::any) override;
  void allreduce_inplace_typeerased( void*, size_t, ReductionOp, std::type_index, std::any ) override;
  
  std::unique_ptr<detail::ReductionDriverImpl> clone() override;

};

}


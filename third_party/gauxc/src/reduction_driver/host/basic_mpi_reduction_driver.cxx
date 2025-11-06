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
#include "basic_mpi_reduction_driver.hpp"
#include <cstring>
#include <memory>
#include <map>
#include <iostream>
#include <cstddef>

namespace GauXC {

#ifdef GAUXC_HAS_MPI
MPI_Datatype get_mpi_datatype( std::type_index idx ) {

  static std::map<std::type_index, MPI_Datatype> map {
    {std::type_index(typeid(double)), MPI_DOUBLE},
    {std::type_index(typeid(float)),  MPI_FLOAT}
  };

  return map.at(idx);

}

MPI_Op get_mpi_op( ReductionOp op ) {

  static std::map< ReductionOp, MPI_Op > map {
    { ReductionOp::Sum, MPI_SUM }
  };

  return map.at(op);

}
#endif

size_t get_dtype_size( std::type_index idx ) {

  static std::map<std::type_index, size_t> map {
    {std::type_index(typeid(double)), sizeof(double)}, 
    {std::type_index(typeid(float)),  sizeof(float)}
  };

  return map.at(idx);
}


BasicMPIReductionDriver::BasicMPIReductionDriver(const RuntimeEnvironment& rt) :
  HostReductionDriver(rt) { }


BasicMPIReductionDriver::~BasicMPIReductionDriver() noexcept = default;
BasicMPIReductionDriver::BasicMPIReductionDriver(const BasicMPIReductionDriver&) = default;


void BasicMPIReductionDriver::allreduce_typeerased( const void* src, void* dest, 
  size_t size, ReductionOp op, std::type_index idx, std::any optional_args )  {

  if( optional_args.has_value() )
    std::cout << "** Warning: Optional Args Are Not Used in BasiMPIReductionDriver::allreduce" << std::endl;

  int world_size = runtime_.comm_size();

  if( world_size == 1 ) {
    std::memcpy( dest, src, size * get_dtype_size(idx)); 
  } else  {
    #ifdef GAUXC_HAS_MPI 
    MPI_Allreduce( src, dest, size, get_mpi_datatype(idx), get_mpi_op(op), runtime_.comm() );
    #endif
  }


}
void BasicMPIReductionDriver::allreduce_inplace_typeerased( void* data, size_t size,
  ReductionOp op, std::type_index idx, std::any optional_args ) {

  if( optional_args.has_value() )
    std::cout << "** Warning: Optional Args Are Not Used in BasiMPIReductionDriver::allreduce" << std::endl;

  int world_size = runtime_.comm_size();

  if(world_size > 1) {
    #ifdef GAUXC_HAS_MPI
    // Test of communicator is an inter-communicator
    int inter_flag;
    MPI_Comm_test_inter( runtime_.comm(), &inter_flag );

    // Reduce in place
    if( not inter_flag ) {
      MPI_Allreduce( MPI_IN_PLACE, data, size, get_mpi_datatype(idx), get_mpi_op(op), runtime_.comm() );

    // Cannot reduce in place
    } else {
      std::allocator<std::byte> alloc;
      auto* tmp = alloc.allocate( size );
      std::memcpy(tmp, data, size);
      allreduce_typeerased( tmp, data, size, op, idx, optional_args );
      alloc.deallocate( tmp, size );
    }
    #endif
  }
}

std::unique_ptr<detail::ReductionDriverImpl> BasicMPIReductionDriver::clone() {
  return std::make_unique<BasicMPIReductionDriver>(*this);
}


}


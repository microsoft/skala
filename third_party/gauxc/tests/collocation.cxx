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
#include "collocation_common.hpp"
#include "collocation_host.hpp"
#include "collocation_cuda.hpp"
#include "collocation_hip.hpp"

//#define GENERATE_TESTS

#if defined(GENERATE_TESTS) && !defined(GAUXC_HAS_HOST)
  #error "Host Integrator Must Be Enabled to Generate Tests"
#endif

TEST_CASE( "Water / cc-pVDZ", "[collocation]" ) {

#ifdef GENERATE_TESTS
#ifdef GAUXC_HAS_MPI
  int world_size;
  MPI_Comm_size( MPI_COMM_WORLD, &world_size );
  if( world_size > 1 ) return;
#endif
#endif

  Molecule mol           = make_water();
  BasisSet<double> basis = make_ccpvdz( mol, SphericalType(true) );

  for( auto& sh : basis ) sh.set_shell_tolerance( 1e-6 );

#ifdef GENERATE_TESTS

  std::ofstream ref_data( "water_cc-pVDZ_collocation.bin", std::ios::binary );
  generate_collocation_data( mol, basis, ref_data );

#else

  std::ifstream ref_data( GAUXC_REF_DATA_PATH "/water_cc-pVDZ_collocation.bin",
                          std::ios::binary );

#ifdef GAUXC_HAS_HOST
  SECTION( "Host Eval" ) {
    test_host_collocation( basis, ref_data );
  }

  SECTION( "Host Eval Grad" ) {
    test_host_collocation_deriv1( basis, ref_data );
  }

  SECTION( "Host Eval Hessian" ) {
    test_host_collocation_deriv2( basis, ref_data );
  }
#endif

#ifdef GAUXC_HAS_CUDA
  BasisSetMap basis_map( basis, mol );
  SECTION( "CUDA Eval" ) {
    test_cuda_collocation( basis, ref_data );
  }
  SECTION( "CUDA Shell to Task Eval" ) {
    test_cuda_collocation_shell_to_task( basis, basis_map, ref_data );
  }

  SECTION( "CUDA Eval Grad" ) {
    test_cuda_collocation_deriv1( basis, ref_data );
  }
  SECTION( "CUDA Shell to Task Eval Grad" ) {
    test_cuda_collocation_shell_to_task_gradient( basis, basis_map, ref_data );
  }

  SECTION( "CUDA Shell to Task Eval Hessian" ) {
    test_cuda_collocation_shell_to_task_hessian( basis, basis_map, ref_data );
  }

  SECTION( "CUDA Shell to Task Eval Laplacian" ) {
    test_cuda_collocation_shell_to_task_laplacian( basis, basis_map, ref_data );
  }

  SECTION( "CUDA Shell to Task Eval Laplacian Gradient" ) {
    test_cuda_collocation_shell_to_task_lapgrad( basis, basis_map, ref_data );
  }
#endif // GAUXC_HAS_CUDA

#ifdef GAUXC_HAS_HIP
  SECTION( "HIP Eval" ) {
    test_hip_collocation( basis, ref_data );
  }

  SECTION( "HIP Eval Grad" ) {
    test_hip_collocation_deriv1( basis, ref_data );
  }
#endif // GAUXC_HAS_HIP




#endif

}

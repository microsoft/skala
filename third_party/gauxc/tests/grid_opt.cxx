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
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/util/div_ceil.hpp>
#include <gauxc/basisset_map.hpp>

#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>
#include "ini_input.hpp"
#include <gauxc/exceptions.hpp>
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Core>

#include <integratorxx/quadratures/muraknowles.hpp>
#include <integratorxx/quadratures/treutleraldrichs.hpp>
#include <integratorxx/quadratures/lebedev_laikov.hpp>
#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <gauxc/grid_factory.hpp>
#include <gauxc/molgrid/defaults.hpp>

#include <chrono>

using namespace GauXC;
using namespace ExchCXX;

int main(int argc, char** argv) {

#ifdef GAUXC_HAS_MPI
  MPI_Init( NULL, NULL );
#endif
  {
    // Set up runtimes
    #ifdef GAUXC_HAS_DEVICE
    auto rt = DeviceRuntimeEnvironment( GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9 );
    #else
    auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
    #endif
    auto world_rank = rt.comm_rank();
    auto world_size = rt.comm_size();

    std::vector< std::string > opts( argc );
    for( int i = 0; i < argc; ++i ) opts[i] = argv[i];

    // Get input file
    auto input_file = opts.at(1);
    INIFile input(input_file);

    // Require Ref file
    auto ref_file = input.getData<std::string>("GAUXC.REF_FILE");

    // Read Molecule
    Molecule mol;
    read_hdf5_record( mol, ref_file, "/MOLECULE" );
    double ref_ne = MolMeta(mol).sum_atomic_charges();

    // Read BasisSet
    BasisSet<double> basis; 
    read_hdf5_record( basis, ref_file, "/BASIS" );

    for( auto& sh : basis ){ sh.set_shell_tolerance( 1e-10 ); }


    // Setup load balancer factory
    LoadBalancerFactory lb_factory( ExecutionSpace::Host, "Replicated");

    // Setup Integrator factory
    using matrix_type = Eigen::MatrixXd;
    XCIntegratorFactory<matrix_type> integrator_factory( ExecutionSpace::Host, 
      "Replicated", "Default", "Default", "Default" );

    // Setup Dummy XC functional
    functional_type func;

    // Read in reference density
    matrix_type P;
    {
      HighFive::File file( ref_file, HighFive::File::ReadOnly );
      auto dset = file.getDataSet("/DENSITY");
      auto dims = dset.getDimensions();
      P       = matrix_type( dims[0], dims[1] );

      if( P.rows() != P.cols() ) 
        throw std::runtime_error("Density Must Be Square");
      if( P.rows() != basis.nbf() ) 
        throw std::runtime_error("Density Not Compatible With Basis");

      dset.read( P.data() );
    }

    auto run_integration = [&](auto scheme){

      auto rq = RadialQuad::MuraKnowles;
      auto rs = RadialSize(100);
      auto as = AngularSize(974);
      auto bs = BatchSize(512);

      #if 0
      auto molmap = MolGridFactory::create_default_gridmap(
        mol, scheme, rq, rs, as );
      MolGrid mg(molmap);
      #else
      auto mg = MolGridFactory::create_default_molgrid(mol, scheme, bs, rq, rs, as);
      #endif

      auto st = std::chrono::high_resolution_clock::now();

      auto lb = lb_factory.get_shared_instance(rt, mol, mg, basis);
      auto integrator = integrator_factory.get_instance( func, lb );

      double N_EL = integrator.integrate_den( P );
      auto en = std::chrono::high_resolution_clock::now();
      std::cout << std::scientific << std::setprecision(16);
      const auto err = std::abs(N_EL-ref_ne);
      std::cout << "NE = " << N_EL << ", " << err << ", " << err/ref_ne << std::endl;
      std::cout << std::chrono::duration<double>(en-st).count() << std::endl;
    };

    // Unpruned Integration
    std::cout << "Unpruned" << std::endl;
    run_integration(PruningScheme::Unpruned);

    // Pruned Integration
    std::cout << "Pruned" << std::endl;
    run_integration(PruningScheme::Robust);


  }
#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif

}

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
#include "ut_common.hpp"
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/molecular_weights.hpp>

#include <gauxc/molgrid/defaults.hpp>

#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>
#include <Eigen/Core>

using namespace GauXC;

void test_dd_psi (
  std::string reference_file, 
  int lmax = 8
) {
    using matrix_type = Eigen::MatrixXd;
    Molecule mol;
    BasisSet<double> basis;
    matrix_type P, ddX, ddPsi_ref, ddPsi_potential_ref;

    read_hdf5_record( mol,   reference_file, "/MOLECULE" );
    read_hdf5_record( basis, reference_file, "/BASIS"    );

    HighFive::File file( reference_file, HighFive::File::ReadOnly );
    std::string den_str = "/DENSITY";
    auto dset = file.getDataSet(den_str);
    auto dims = dset.getDimensions();
    P = matrix_type( dims[0], dims[1] );
    dset.read( P.data() );

    int nharmonics = (lmax + 1) * (lmax + 1);

    ddX = matrix_type( nharmonics, mol.size() );
    dset = file.getDataSet("/DD_X");
    dset.read(ddX.data());

    ddPsi_ref = matrix_type( mol.size(), nharmonics );
    dset = file.getDataSet("/DD_PSI");
    dset.read( ddPsi_ref.data());

    ddPsi_potential_ref = matrix_type( basis.nbf(), basis.nbf() );
    dset = file.getDataSet("/DD_PSI_POTENTIAL");
    dset.read( ddPsi_potential_ref.data() );


    #ifdef GAUXC_HAS_DEVICE
    auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
    #else
    auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
    #endif

    auto mg = MolGridFactory::create_default_molgrid(mol, PruningScheme::Unpruned,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

    auto ex = ExecutionSpace::Host;
    LoadBalancerFactory lb_factory(ex, "Default");
    auto lb = lb_factory.get_instance(rt, mol, mg, basis);

        // Construct Weights Module
    MolecularWeightsFactory mw_factory( ex, "Default", MolecularWeightsSettings{} );
    auto mw = mw_factory.get_instance();

    // Apply partition weights
    mw.modify_weights(lb);

    functional_type func = functional_type( ExchCXX::Backend::builtin, ExchCXX::Functional::PBE0, ExchCXX::Spin::Unpolarized );
        // Construct XCIntegrator
    XCIntegratorFactory<matrix_type> integrator_factory( ex, "Replicated", 
            "Default",  "Default",  "Default" );
    auto integrator = integrator_factory.get_instance( func, lb );

    auto dd_psi = integrator.eval_dd_psi(P, lmax);
    auto ddPsi = Eigen::Map<matrix_type>(dd_psi.data(), mol.size(), nharmonics);
    auto ddPsi_nrm = (ddPsi - ddPsi_ref).norm();
    CHECK( ddPsi_nrm / mol.size() < 1e-10 );

    auto ddPsiPotential = integrator.eval_dd_psi_potential(ddX, lmax);
    auto ddPsiPotential_nrm = (ddPsiPotential - ddPsi_potential_ref).norm();
    CHECK( ddPsiPotential_nrm / basis.nbf() < 1e-10 );

}

TEST_CASE( "DD PSI & PSI POTENTIAL", "[dd]" ) {
    SECTION( " C2H4 / def2-svp / LMAX = 8" ) {
        test_dd_psi( GAUXC_REF_DATA_PATH "/c2h4_l8_dd_psi_potential.hdf5" );
    }
}
 
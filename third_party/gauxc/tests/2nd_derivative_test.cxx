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


void test_fxc_contractioin(ExecutionSpace ex, const RuntimeEnvironment& rt,
  std::string reference_file, 
  functional_type& func, 
  PruningScheme pruning_scheme,
  std::string integrator_kernel = "Default",  
  std::string reduction_kernel  = "Default",
  std::string lwd_kernel        = "Default") {

  // Read the reference file
  using matrix_type = Eigen::MatrixXd;
  Molecule mol;
  BasisSet<double> basis;
  matrix_type P, Pz, tP, tPz, FXC_ref, FXCz_ref;
  bool rks = true, uks = false;
  
  {
    read_hdf5_record( mol,   reference_file, "/MOLECULE" );
    read_hdf5_record( basis, reference_file, "/BASIS"    );

    HighFive::File file( reference_file, HighFive::File::ReadOnly );
    
    std::string den = "/DENSITY";
    std::string tden_str = "/TRIAL_DENSITY";
    std::string fxc_str = "/FXC";
    std::string den2 = "/DENSITY_Z";

    if (file.exist("/DENSITY_Z")) { 
      rks = false; 
      uks = true;
      if (file.exist("/DENSITY_Y") && file.exist("/DENSITY_X")) {
        std::cout << "FXC contraction for GKS is not supported yet. Skipping test." << std::endl;
        return;
      }
    }

    if (uks) {
      tden_str = "/TRIAL_DENSITY_SCALAR";
      den = "/DENSITY_SCALAR";
      fxc_str = "/FXC_SCALAR";
    }
     
    auto dset = file.getDataSet(den);
    auto dims = dset.getDimensions();
    
    P = matrix_type(dims[0], dims[1]);
    dset.read(P.data());
    
    if (not rks) {
      Pz = matrix_type(dims[0], dims[1]);
      dset = file.getDataSet(den2);
      dset.read(Pz.data());
    }
    
    tP = matrix_type(dims[0], dims[1]);
    dset = file.getDataSet(tden_str);
    dset.read(tP.data());
    FXC_ref = matrix_type(dims[0], dims[1]);
    dset = file.getDataSet(fxc_str);
    dset.read(FXC_ref.data());
    
    if (not rks) {
      FXCz_ref = matrix_type(dims[0], dims[1]);
      dset = file.getDataSet("/FXC_Z");
      dset.read(FXCz_ref.data());
      tPz = matrix_type(dims[0], dims[1]);
      dset = file.getDataSet("/TRIAL_DENSITY_Z");
      dset.read(tPz.data());
    }
  }

  // Set shell tolerance
  for (auto& sh : basis) 
    sh.set_shell_tolerance(std::numeric_limits<double>::epsilon());

  // Create molecular grid
  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  // Construct Load Balancer
  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);

  // Construct Weights Module
  MolecularWeightsFactory mw_factory(ex, "Default", MolecularWeightsSettings{});
  auto mw = mw_factory.get_instance();

  // Apply partition weights
  mw.modify_weights(lb);

  // Construct XCIntegrator
  XCIntegratorFactory<matrix_type> integrator_factory(ex, "Replicated", 
    integrator_kernel, lwd_kernel, reduction_kernel);
  auto integrator = integrator_factory.get_instance(func, lb);

  // Test FXC contraction
  if (rks) {
    // Call FXC contraction
    auto FXC = integrator.eval_fxc_contraction(P, tP);
    auto FXC_diff_nrm = (FXC - FXC_ref).norm();
    CHECK(FXC_diff_nrm / basis.nbf() < 1e-10);
  } else if (uks) {
    // Call FXC contraction
    auto [FXCs, FXCz] = integrator.eval_fxc_contraction(P, Pz, tP, tPz);
    
    auto FXCs_diff_nrm = (FXCs - FXC_ref).norm();
    auto FXCz_diff_nrm = (FXCz - FXCz_ref).norm();
    CHECK(FXCs_diff_nrm / basis.nbf() < 1e-10);
    CHECK(FXCz_diff_nrm / basis.nbf() < 1e-10);
  
  }
}

void test_integrator_2nd(std::string reference_file, functional_type& func, PruningScheme pruning_scheme) {

#ifdef GAUXC_HAS_DEVICE
  auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
#else
  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

#ifdef GAUXC_HAS_HOST
    SECTION( "Host" ) {
      SECTION("Reference") {
        test_fxc_contractioin( ExecutionSpace::Host, rt, reference_file, func,
          pruning_scheme, "Default", "Default", "Default" );
      }
    }
#endif

#ifdef GAUXC_HAS_DEVICE
  SECTION( "Device" ) {
    SECTION( "Incore - MPI Reduction" ) {
      test_fxc_contractioin( ExecutionSpace::Device, rt,
        reference_file, func, pruning_scheme,  
        "Default", "Default", "Default" );
    }
    #ifdef GAUXC_HAS_CUTLASS
    SECTION( "Incore - MPI Reduction - CUTLASS" ) {
      test_fxc_contractioin( ExecutionSpace::Device, rt, 
        reference_file, func, pruning_scheme,
        "Default", "Default", "Scheme1-CUTLASS" );
    }
    #endif

  }
#endif

}

functional_type make_functional_2nd(ExchCXX::Functional func_key, ExchCXX::Spin spin) {
  return functional_type(ExchCXX::Backend::builtin, func_key, spin);
}


TEST_CASE( "XC Integrator FXC", "[xc-integrator]" ) {

  auto pol     = ExchCXX::Spin::Polarized;
  auto unpol   = ExchCXX::Spin::Unpolarized;
  auto svwn5   = ExchCXX::Functional::SVWN5;
  auto pbe0    = ExchCXX::Functional::PBE0;
  auto blyp    = ExchCXX::Functional::BLYP;
  auto scan    = ExchCXX::Functional::SCAN;
  auto r2scanl = ExchCXX::Functional::R2SCANL;
  auto m062x   = ExchCXX::Functional::M062X;

  // LDA Test
  SECTION( "Benzene / SVWN5 / cc-pVDZ" ) {
    auto func = make_functional_2nd(svwn5, unpol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf.hdf5", 
        func, PruningScheme::Unpruned );
  }
  SECTION( "Benzene / SVWN5 / cc-pVDZ (Treutler)" ) {
    auto func = make_functional_2nd(svwn5, unpol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf_treutler_prune.hdf5", 
        func, PruningScheme::Treutler );
  }
  SECTION( "Benzene / SVWN5 / cc-pVDZ (Robust)" ) {
    auto func = make_functional_2nd(svwn5, unpol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf_robust_prune.hdf5", 
        func, PruningScheme::Robust );
  }

  // GGA Test
  SECTION( "Benzene / PBE0 / cc-pVDZ" ) {
    auto func = make_functional_2nd(pbe0, unpol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/benzene_pbe0_cc-pvdz_ufg_ssf.hdf5", 
        func, PruningScheme::Unpruned );
  }

  // MGGA Test (TAU Only)
  SECTION( "Cytosine / SCAN / cc-pVDZ") {
    auto func = make_functional_2nd(scan, unpol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust.hdf5", 
        func, PruningScheme::Robust );
  }

  //UKS LDA Test
  SECTION( "Li / SVWN5 / sto-3g" ) {
    auto func = make_functional_2nd(svwn5, pol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/li_svwn5_sto3g_uks.bin",
        func, PruningScheme::Unpruned );
  }

  //UKS GGA Test
  SECTION( "Cytosine (doublet) / BLYP / cc-pVDZ") {
    auto func = make_functional_2nd(blyp, pol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/cytosine_blyp_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }

  // UKS MGGA Test (TAU Only)
  SECTION( "Cytosine (doublet) / SCAN / cc-pVDZ") {
    auto func = make_functional_2nd(scan, pol);
    test_integrator_2nd(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }
}

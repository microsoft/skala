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
#include <gauxc/molecule.hpp>
#include <gauxc/molmeta.hpp>
#include <gauxc/xc_task.hpp>
#include <gauxc/external/hdf5.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>

// Include weights implementation
#include "xc_integrator/local_work_driver/host/reference/weights.hpp"

using namespace GauXC;

// Helper function to compute weights for a task
void compute_weights_task(XCWeightAlg weight_alg, const Molecule& mol, const MolMeta& meta, XCTask& task) {
  // Construct local work driver
  auto lwd = LocalWorkDriverFactory::make_local_work_driver( ExecutionSpace::Host, "Default", LocalWorkSettings() );
  auto* lwd_host = dynamic_cast<LocalHostWorkDriver*>(lwd.get());

  std::vector<XCTask> tasks = {task};
  lwd_host->partition_weights(weight_alg, mol, meta, tasks.begin(), tasks.end());

  // Copy the computed weights back to the original task
  task.weights = tasks[0].weights;
}

// Helper function to compute weights for a task
void compute_int(XCWeightAlg weight_alg, const Molecule& mol, const MolMeta& meta, XCTask& task, 
                 double* f_eval, double* result) {
  std::vector<XCTask> tasks = {task};
  
  auto lwd = LocalWorkDriverFactory::make_local_work_driver( ExecutionSpace::Host, "Default", LocalWorkSettings() );
  auto* lwd_host = dynamic_cast<LocalHostWorkDriver*>(lwd.get());
  lwd_host->partition_weights(weight_alg, mol, meta, tasks.begin(), tasks.end());

  for (size_t i = 0; i < task.points.size(); i++) {
    result[0] += tasks[0].weights[i] * f_eval[i];
  }
}


// Test function that reads molecule and basis from reference file
void test_weight_1st_deri_host_fdiff(const std::string& reference_file, XCWeightAlg weight_alg,
                                        PruningScheme pruning_scheme, double fdiff_step, double fdiff_tolerance) {

  // Create runtime environment
  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
  Molecule mol;
  BasisSet<double> basis;
  
  // Read molecule and basis from HDF5 reference file
  read_hdf5_record(mol, reference_file, "/MOLECULE");
  read_hdf5_record(basis, reference_file, "/BASIS");
  
  // Set shell tolerance for numerical stability
  for(auto& sh : basis) {
    sh.set_shell_tolerance(std::numeric_limits<double>::epsilon());
  }
  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  // Construct Load Balancer
  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);

  
  // Get all XC tasks
  auto& tasks = lb.get_tasks();
  size_t natoms = mol.size();
  size_t ntask = tasks.size();

  auto get_xyz_pointer = [](Atom& atom, size_t i_coord) {
    switch(i_coord) {
      case 0: return &atom.x; // X coordinate
      case 1: return &atom.y; // Y coordinate
      case 2: return &atom.z; // Z coordinate
      default: throw std::out_of_range("Invalid coordinate index");
    }
  };

  // Calculate finite difference derivatives as ref
  std::vector<std::vector<double>> weight_derivatives_ref(ntask);
  for(size_t i_task = 0; i_task < ntask; i_task++) {
    weight_derivatives_ref[i_task].resize(3 * natoms * tasks[i_task].npts);
  }
  for( size_t i_atom = 0; i_atom < mol.size(); i_atom++ ) {
    for( size_t i_coord = 0; i_coord < 3; i_coord++ ) {
      // Create perturbed molecules
      Molecule mol_plus = mol;
      Molecule mol_minus = mol;
      
      // Perturb atom coordinates
      double* coord_ptr_plus = get_xyz_pointer(mol_plus[i_atom], i_coord);
      double* coord_ptr_minus = get_xyz_pointer(mol_minus[i_atom], i_coord);
      double delta = fdiff_step; // Use provided finite difference step
      *coord_ptr_plus += delta;   // Perturb in positive direction
      *coord_ptr_minus -= delta;  // Perturb in negative direction
      
      // Create metadata for perturbed molecules
      MolMeta meta_plus(mol_plus);
      MolMeta meta_minus(mol_minus);
      
      // Compute weights for perturbed geometries
      for(size_t itask = 0; itask < ntask; itask++) {
        XCTask task_plus = tasks[itask];
        XCTask task_minus = tasks[itask];      
        if (i_atom == (size_t)task_plus.iParent) {
          for(size_t ipt = 0; ipt < task_plus.npts; ipt++) {
            task_plus.points[ipt][i_coord] += delta;
            task_minus.points[ipt][i_coord] -= delta;
          }
        }
        task_plus.dist_nearest = meta_plus.dist_nearest()[task_plus.iParent];
        task_minus.dist_nearest = meta_minus.dist_nearest()[task_minus.iParent];

        // Compute weights for perturbed geometries
        compute_weights_task(weight_alg, mol_plus, meta_plus, task_plus);
        compute_weights_task(weight_alg, mol_minus, meta_minus, task_minus);
      
        // Compute centered finite difference
        for(size_t ipt = 0; ipt < task_plus.npts; ipt++) {
          weight_derivatives_ref[itask][3 * natoms * ipt + 3 * i_atom + i_coord] =
            (task_plus.weights[ipt] - task_minus.weights[ipt]) / (2.0 * delta);
        }
      }
    }
  }


  // Test derivatives for all tasks
  for(size_t task_idx = 0; task_idx < ntask; task_idx++) {
    auto& task = tasks[task_idx];
    
    INFO("Testing task " << task_idx << " with " << task.npts << " points");
    
    // Create MolMeta
    MolMeta meta(mol);    // Compute analytical derivatives
    std::vector<double> analytical_derivatives(3 * natoms * task.npts);
    compute_weights_task(weight_alg, mol, meta, task);
  
    switch( weight_alg ) {
      case XCWeightAlg::Becke:
        reference_becke_weights_1st_derivative_host(mol, meta, task, analytical_derivatives.data());
        break;
      case XCWeightAlg::SSF:
        reference_ssf_weights_1st_derivative_host(mol, meta, task, analytical_derivatives.data());
        break;
      default:
        GAUXC_GENERIC_EXCEPTION("Weight Alg Not Supported");
    }

    // Compare with numerical derivatives
    double max_error = 0.0;
    for(size_t ipt = 0; ipt < task.npts; ipt++) {
      for(size_t iatom = 0; iatom < natoms; iatom++) {        
        for(size_t icoord = 0; icoord < 3; icoord++) {
          size_t idx = 3 * natoms * ipt + 3 * iatom + icoord;
          double error = std::abs(analytical_derivatives[idx] - weight_derivatives_ref[task_idx][idx]);
          max_error = std::max(max_error, error);
          
          INFO("Task " << task_idx << ", Point " << ipt << ", Atom " << iatom << ", Coord " << icoord 
                << " iParent: " << task.iParent);
          INFO("Analytical: " << analytical_derivatives[idx]);
          INFO("Numerical: " << weight_derivatives_ref[task_idx][idx]);
          INFO("Error: " << error);
          
          REQUIRE(analytical_derivatives[idx] == Approx(weight_derivatives_ref[task_idx][idx]).margin(fdiff_tolerance));
          
        }
      }
    }
    
    // Report statistics for this task
    INFO("Task " << task_idx << " - Total derivatives tested: " << (task.npts * natoms * 3));
    INFO("Task " << task_idx << " - Maximum error: " << max_error);
  }


}



// Test function that reads molecule and basis from reference file
void test_weight_1st_deri_host_fdiff_contracted(const std::string& reference_file, XCWeightAlg weight_alg,
                                        PruningScheme pruning_scheme, double fdiff_step, double fdiff_tolerance) {

  // Create runtime environment
  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
  Molecule mol;
  BasisSet<double> basis;
  
  // Read molecule and basis from HDF5 reference file
  read_hdf5_record(mol, reference_file, "/MOLECULE");
  read_hdf5_record(basis, reference_file, "/BASIS");
  
  // Set shell tolerance for numerical stability
  for(auto& sh : basis) {
    sh.set_shell_tolerance(std::numeric_limits<double>::epsilon());
  }
  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  // Construct Load Balancer
  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);
  
  // Get all XC tasks
  auto& tasks = lb.get_tasks();
  size_t natoms = mol.size();
  size_t ntask = tasks.size();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::stable_sort( tasks.begin(), tasks.end(), task_comparator );
  
  // generate a random f_eval vector
  std::vector<std::vector<double>> f_evals(ntask);
  for(size_t i_task = 0; i_task < ntask; i_task++) {
    f_evals[i_task].resize(tasks[i_task].npts);
    for(size_t i_pt = 0; i_pt < tasks[i_task].npts; i_pt++) {
      f_evals[i_task][i_pt] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1
    }
  }


  auto get_xyz_pointer = [](Atom& atom, size_t i_coord) {
    switch(i_coord) {
      case 0: return &atom.x; // X coordinate
      case 1: return &atom.y; // Y coordinate
      case 2: return &atom.z; // Z coordinate
      default: throw std::out_of_range("Invalid coordinate index");
    }
  };

  // Calculate finite difference derivatives as ref
  std::vector<std::vector<double>> exc_grad_w_ref(ntask);
  for(size_t i_task = 0; i_task < ntask; i_task++) {
    exc_grad_w_ref[i_task].resize(3 * natoms);
  }
  for( size_t i_atom = 0; i_atom < mol.size(); i_atom++ ) {
    for( size_t i_coord = 0; i_coord < 3; i_coord++ ) {
      // Create perturbed molecules
      Molecule mol_plus = mol;
      Molecule mol_minus = mol;
      
      // Perturb atom coordinates
      double* coord_ptr_plus = get_xyz_pointer(mol_plus[i_atom], i_coord);
      double* coord_ptr_minus = get_xyz_pointer(mol_minus[i_atom], i_coord);
      double delta = fdiff_step; // Use provided finite difference step
      *coord_ptr_plus += delta;   // Perturb in positive direction
      *coord_ptr_minus -= delta;  // Perturb in negative direction
      
      // Create metadata for perturbed molecules
      MolMeta meta_plus(mol_plus);
      MolMeta meta_minus(mol_minus);
      
      // Compute weights for perturbed geometries
      for(size_t itask = 0; itask < ntask; itask++) {
        XCTask task_plus = tasks[itask];
        XCTask task_minus = tasks[itask];      
        if (i_atom == (size_t)task_plus.iParent) {
          for(size_t ipt = 0; ipt < task_plus.npts; ipt++) {
            task_plus.points[ipt][i_coord] += delta;
            task_minus.points[ipt][i_coord] -= delta;
          }
        }
        task_plus.dist_nearest = meta_plus.dist_nearest()[task_plus.iParent];
        task_minus.dist_nearest = meta_minus.dist_nearest()[task_minus.iParent];

        // Compute weights for perturbed geometries
        double result_plus = 0.0, result_minus = 0.0;
        compute_int(weight_alg, mol_plus, meta_plus, task_plus, f_evals[itask].data(), &result_plus);
        compute_int(weight_alg, mol_minus, meta_minus, task_minus, f_evals[itask].data(), &result_minus);
      
        // Compute centered finite difference
        exc_grad_w_ref[itask][3 * i_atom + i_coord] =
          (result_plus - result_minus) / (2.0 * delta);
      }
    }
  }
  
  // Construct Weights Module
  MolecularWeightsFactory mw_factory(ExecutionSpace::Host, "Default", MolecularWeightsSettings{weight_alg, false});
  auto mw = mw_factory.get_instance();
  // Apply partition weights
  mw.modify_weights(lb);

  // check lb.state().xc_weight_alg() == weight_alg;
  REQUIRE(lb.state().weight_alg == weight_alg);

  auto lwd = LocalWorkDriverFactory::make_local_work_driver( ExecutionSpace::Host, "Default", LocalWorkSettings() );
  auto* lwd_host = dynamic_cast<LocalHostWorkDriver*>(lwd.get());

  // Create MolMeta
  MolMeta meta(mol);    
  
  // Test derivatives for all tasks
  std::vector<std::vector<double>> w_times_fs(ntask);
  for(size_t task_idx = 0; task_idx < ntask; task_idx++) {
    auto& task = tasks[task_idx];
    
    INFO("Testing task " << task_idx << " with " << task.npts << " points");
    
    auto w_times_f = w_times_fs[task_idx];
    w_times_f.resize(task.npts);
    for(size_t i = 0; i < task.npts; i++) {
      w_times_f[i] = task.weights[i] * f_evals[task_idx][i];
    }

    // Compute analytical derivatives
    std::vector<double> analytical_derivatives(3 * natoms);
    lwd_host->eval_weight_1st_deriv_contracted(weight_alg, mol, meta, task, w_times_f.data(), analytical_derivatives.data());

    // Compare with numerical derivatives
    double max_error = 0.0;
    for(size_t iatom = 0; iatom < natoms; iatom++) {        
      for(size_t icoord = 0; icoord < 3; icoord++) {
        size_t idx = 3 * iatom + icoord;
        double error = std::abs(analytical_derivatives[idx] - exc_grad_w_ref[task_idx][idx]);
        max_error = std::max(max_error, error);
        
        INFO("Task " << task_idx << ", Atom " << iatom << ", Coord " << icoord 
              << " iParent: " << task.iParent);
        INFO("Analytical: " << analytical_derivatives[idx]);
        INFO("Numerical: " << exc_grad_w_ref[task_idx][idx]);
        INFO("Error: " << error);
        
        REQUIRE(analytical_derivatives[idx] == Approx(exc_grad_w_ref[task_idx][idx]).margin(fdiff_tolerance));
        
      }
    }
    
    // Report statistics for this task
    INFO("Task " << task_idx << " - Total derivatives tested: " << (task.npts * natoms * 3));
    INFO("Task " << task_idx << " - Maximum error: " << max_error);
  }


}

TEST_CASE("Weights First Derivative uncontracted HOST fidiff", "[weights_fdiff]") {
  

  SECTION( "H3 Becke" ) {
  test_weight_1st_deri_host_fdiff(GAUXC_REF_DATA_PATH "/h3_blyp_cc-pvdz_ssf_gks.bin", XCWeightAlg::Becke,
                                      PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}
  SECTION( "H3 SSF" ) {
  test_weight_1st_deri_host_fdiff(GAUXC_REF_DATA_PATH "/h3_blyp_cc-pvdz_ssf_gks.bin", XCWeightAlg::SSF,
                                      PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}
  
}


TEST_CASE("Weights First Derivative contracted HOST fidiff", "[weights_fdiff]") {
  

  SECTION( "H3 Becke" ) {
  test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/h3_blyp_cc-pvdz_ssf_gks.bin", XCWeightAlg::Becke,
                                      PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}

  // SECTION( "Benzene Becke" ) {
  // test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf.hdf5", XCWeightAlg::Becke,
  //                                     PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}

  // SECTION( "Cytosine Becke" ) {
  // test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust.hdf5", XCWeightAlg::Becke,
  //                                     PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}
  

  SECTION( "H3 SSF" ) {
  test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/h3_blyp_cc-pvdz_ssf_gks.bin", XCWeightAlg::SSF,
                                      PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}
  // SECTION( "Benzene SSF" ) {
  // test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf.hdf5", XCWeightAlg::SSF,
  //                                     PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}

  // SECTION( "Cytosine SSF" ) {
  // test_weight_1st_deri_host_fdiff_contracted(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust.hdf5", XCWeightAlg::SSF,
  //                                     PruningScheme::Unpruned, 1.0e-5, 1.0e-6);}
  

}
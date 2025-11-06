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
#include "host/reference/weights.hpp"
#include "common/integrator_constants.hpp"

#include <gauxc/molgrid/defaults.hpp>

namespace GauXC {

// Reference Becke weights impl
void reference_becke_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
) {

  // Becke partition functions
  auto hBecke = [](double x) {return 1.5 * x - 0.5 * x * x * x;}; // Eq. 19
  auto gBecke = [&](double x) {return hBecke(hBecke(hBecke(x)));}; // Eq. 20 f_3

  const size_t ntasks = std::distance(task_begin,task_end);
  const size_t natoms = mol.natoms();

  const auto&  RAB    = meta.rab();

  std::vector<double> slater_radii;
  for( auto& atom : mol ) {
    slater_radii.emplace_back( default_atomic_radius(atom.Z) );
  }


#if 0
  // TODO: Add a pathway for this
  std::vector<double> size_adj(natoms * natoms);
  for( auto i = 0; i < natoms; ++i ) 
  for( auto j = 0; j < natoms; ++j ) {
    const auto si  = slater_radii[i];
    const auto sj  = slater_radii[j];
    const auto chi = std::sqrt(si/sj);
    const auto u   = (chi-1.)/(chi+1.);
    const auto a   = u / (u*u-1.);

    size_adj[i + j*natoms] = a;
  }
#endif

  #pragma omp parallel 
  {

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  #pragma omp for
  for( size_t iT = 0; iT < ntasks;                  ++iT )
  for( size_t i  = 0; i  < (task_begin+iT)->points.size(); ++i  ) {

    auto&       task   = *(task_begin+iT);
    auto&       weight = task.weights[i];
    const auto& point  = task.points[i];

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);
    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ ){

      double mu  = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

#if 0
      // Size Adjustment
      const double a = size_adj[iA + jA*natoms];
      mu = mu + a * ( 1. - mu*mu );
#endif

      const double g = gBecke(mu);

      partitionScratch[iA] *= 0.5 * (1. - g);
      partitionScratch[jA] *= 0.5 * (1. + g);
    }

    // Normalization
    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // Update Weights
    weight *= partitionScratch[task.iParent] / sum;

  } // Collapsed loop over tasks and points

  } // OMP context


}



void reference_ssf_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
) {

  auto gFrisch = [&](double x) {
    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
  };

  const size_t ntasks = std::distance(task_begin,task_end);
  const size_t natoms = mol.natoms();

  const auto&  RAB    = meta.rab();

  #pragma omp parallel 
  {

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  #pragma omp for
  for( size_t iT = 0; iT < ntasks;                  ++iT )
  for( size_t i  = 0; i  < (task_begin+iT)->points.size(); ++i  ) {

    auto&       task   = *(task_begin+iT);
    auto&       weight = task.weights[i];
    const auto& point  = task.points[i];

    const auto dist_cutoff = 0.5 * (1-integrator::magic_ssf_factor<>) * task.dist_nearest;

    // Compute dist to parent atom
    {
      const double da_x = point[0] - mol[task.iParent].x;
      const double da_y = point[1] - mol[task.iParent].y;
      const double da_z = point[2] - mol[task.iParent].z;

      atomDist[task.iParent] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);
    }

    if( atomDist[task.iParent] < dist_cutoff ) continue; // Partition weight = 1

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      if( iA == (size_t)task.iParent ) continue;

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);
#if 1
    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ )
    if( partitionScratch[iA] > integrator::ssf_weight_tol or 
        partitionScratch[jA] > integrator::ssf_weight_tol ) {

      const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

      if( mu <= -integrator::magic_ssf_factor<> ) {

        partitionScratch[jA] = 0.;

      } else if (mu >= integrator::magic_ssf_factor<>) {

        partitionScratch[iA] = 0.;

      } else {

        double g = 0.5 * ( 1. - gFrisch(mu) );
        partitionScratch[iA] *= g;
        partitionScratch[jA] *= 1. - g;

      }

    }
#else
    for(size_t iA = 0; iA < natoms; ++iA)
    for(size_t jA = 0; jA < natoms; ++jA) 
    if(iA != jA and partitionScratch[iA] > integrator::ssf_weight_tol) {
      const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];
      if( fabs(mu) < integrator::magic_ssf_factor<> ) {
        double g = 0.5 * (1. - gFrisch(mu));
        partitionScratch[iA] *= g;
      } else if(mu >= integrator::magic_ssf_factor<>) {
        partitionScratch[iA] = 0.0;
      }
    }

    if(partitionScratch[task.iParent] < std::numeric_limits<double>::epsilon()) {
      weight = 0;
      continue;
    }
#endif

    // Normalization
    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // Update Weights
    weight *= partitionScratch[task.iParent] / sum;

  } // Collapsed loop over tasks and points

  } // OMP context


}

void reference_lko_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
) {


  // Sort on atom index
  std::stable_sort( task_begin, task_end, 
    [](const auto& a, const auto&b ) { return a.iParent < b.iParent; } );

  // Becke partition functions
  auto hBecke = [](double x) {return 1.5 * x - 0.5 * x * x * x;}; // Eq. 19
  auto gBecke = [&](double x) {return hBecke(hBecke(hBecke(x)));}; // Eq. 20 f_3
  constexpr double R_cutoff = 5;

  const size_t natoms = mol.natoms();

  const auto&  RAB    = meta.rab();

  #pragma omp parallel 
  {

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );
  std::vector<size_t> inter_atom_dist_idx( natoms );
  std::vector<size_t> point_dist_idx( natoms );

  #pragma omp for schedule(dynamic)
  for( auto iAtom = 0ul; iAtom < natoms; ++iAtom ) {

    auto atom_begin = std::find_if( task_begin, task_end,
      [&](const auto& t){ return t.iParent == (int)iAtom; } );
    auto atom_end = std::find_if( task_begin, task_end,
      [&](const auto& t){ return t.iParent == (int)(iAtom+1); } );

    auto* RAB_parent = RAB.data() + iAtom*natoms;

    std::iota( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(), 0 );
    std::sort( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(),
      [&](auto i, auto j){ return RAB_parent[i] < RAB_parent[j]; } );

  for( auto task_it = atom_begin; task_it != atom_end; ++task_it ) {

    auto& points  = task_it->points;
    auto& weights = task_it->weights;
    const auto npts = points.size();

  for( auto ipt = 0ul; ipt < npts; ++ipt ) {

    auto& weight = weights[ipt];
    const auto point = points[ipt];

    std::fill( atomDist.begin(), atomDist.end(), std::numeric_limits<double>::infinity() );
    // Parent distance
    {
      const double da_x = point[0] - mol[iAtom].x;
      const double da_y = point[1] - mol[iAtom].y;
      const double da_z = point[2] - mol[iAtom].z;

      atomDist[iAtom] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);
    }

    double r_parent  = atomDist[iAtom];
    double r_nearest = r_parent;
    size_t natoms_keep = 1;
    // Compute distances of each center to point
    for(size_t iA = 1; iA < natoms; iA++) {
      auto idx = inter_atom_dist_idx[iA];
      if( RAB_parent[idx] > (r_parent + r_nearest + 2*R_cutoff) ) break;

      const double da_x = point[0] - mol[idx].x;
      const double da_y = point[1] - mol[idx].y;
      const double da_z = point[2] - mol[idx].z;

      const auto r = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);
      r_nearest = std::min( r_nearest, r );
      atomDist[idx] = r;
      ++natoms_keep;
    }

     // Partition weight is 0
    if( r_parent > r_nearest + R_cutoff ) {
      weight = 0.;
      continue;
    }

    // Partiton atom indices into a petite list of non-negligible centers
    std::iota( point_dist_idx.begin(), point_dist_idx.end(), 0 );
    auto atom_keep_end = std::partition( point_dist_idx.begin(), point_dist_idx.end(), 
      [&](auto i){ return atomDist[i] < std::numeric_limits<double>::infinity(); } );

    // Only sort over non-negligible cetners
    std::sort( point_dist_idx.begin(), atom_keep_end,
      [&](auto i, auto j){ return atomDist[i] < atomDist[j]; } );

    // Get parent index
    auto parent_it  = std::find( point_dist_idx.begin(), atom_keep_end, iAtom );
    auto parent_idx = std::distance( point_dist_idx.begin(), parent_it );

    // Sort atom distances for contiguous reads in weight loop
    auto atom_dist_end = std::partition( atomDist.begin(), atomDist.end(),
      [](auto x){ return x < std::numeric_limits<double>::infinity(); } );
    std::sort( atomDist.begin(), atom_dist_end );



    // Evaluate unnormalized partition functions 
    std::fill_n(partitionScratch.begin(),natoms_keep,0.);
    for( auto i = 0ul; i < natoms_keep; ++i ) {
      auto idx_i = point_dist_idx[i];
      auto r_i = atomDist[i];
      if( r_i > (r_nearest + R_cutoff) ) { break; }
      partitionScratch[i] = 1.;

      const auto* RAB_i_idx = RAB.data() + idx_i*natoms;

    for( auto j = 0ul; j < i; ++j ) {
      auto idx_j = point_dist_idx[j];
      auto r_j = atomDist[j];
      if( r_j > (r_i + R_cutoff) ) { break; }

      const double mu = 
        (r_i - r_j) / std::min(RAB_i_idx[idx_j], R_cutoff);

      const double g = gBecke(mu);
      const auto   s_ij = 0.5 * (1. - g);
      partitionScratch[i] *= s_ij;
      partitionScratch[j] *= 1. - s_ij;
      
    }
    }

    // Normalization
    double sum = 0.;
    for( size_t iA = 0; iA < natoms_keep; iA++ )  sum += partitionScratch[iA];

    // Update Weights
    weight *= partitionScratch[parent_idx] / sum;

  } // Loop over points 
  } // Loop over tasks
  } // Loop over atoms

  } // OMP context

}


/**
 * 1st derivative which expects weight_deri to be preallocated as (ngrid*natoms*3)
 */
void reference_becke_weights_1st_derivative_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  double* weight_deri
){

  // Becke partition functions
  auto hBecke = [](double x) {return 1.5 * x - 0.5 * x * x * x;}; // Eq. 19
  auto gBecke = [&](double x) {return hBecke(hBecke(hBecke(x)));}; // Eq. 20 f_3
  auto tBecke = [&](double x) {
    // for numerical stability (see Jiashu's notes for details)
    if (x > 1.0 - 1e-4) 
      return 0.0; 
    const double p1 = hBecke(x);
    const double p2 = hBecke(p1);
    return - 27.0 * (1. + p2) * (1. + p1) * (1. + x) / (1. - x) / (2. + p2) / (2. + p1) / (2. + x);
  };

  const size_t natoms = mol.natoms();
  const auto&  RAB    = meta.rab();
  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t i  = 0; i  < task.points.size(); ++i  ) {

    auto * weight_deri_ith = weight_deri + 3*natoms*i;
    const size_t iParent = task.iParent;

    //zerofy the derivative
    std::fill(weight_deri_ith, weight_deri_ith + 3*natoms, 0.);
    const auto& point  = task.points[i];
    const auto& weight = task.weights[i];

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);
    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ ){

      double mu  = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];
      const double g = gBecke(mu);

      partitionScratch[iA] *= 0.5 * (1. - g);
      partitionScratch[jA] *= 0.5 * (1. + g);
    }

    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // calculate derivative now
    auto * weight_deri_iParent = weight_deri_ith + 3*iParent;
    for( size_t iB = 0; iB < natoms; iB++ ) {
      if (iB == iParent) continue;
      auto * weight_deri_iB = weight_deri_ith + 3*iB;
      
      const double uB_x = mol[iB].x - point[0];
      const double uB_y = mol[iB].y - point[1];
      const double uB_z = mol[iB].z - point[2];

      const double uBA_x =mol[iB].x - mol[iParent].x;
      const double uBA_y =mol[iB].y - mol[iParent].y;
      const double uBA_z =mol[iB].z - mol[iParent].z;
      const double rAB = RAB[iB + iParent*natoms];

      double mu_AB  = (atomDist[iParent] - atomDist[iB]) / rAB;

      // first term is - coef1 * nabla_B mu_BA
      double coef1 = tBecke(mu_AB);
      weight_deri_iB[0] -= coef1 / rAB * (uB_x / atomDist[iB] + mu_AB * uBA_x /rAB);
      weight_deri_iB[1] -= coef1 / rAB * (uB_y / atomDist[iB] + mu_AB * uBA_y /rAB);
      weight_deri_iB[2] -= coef1 / rAB * (uB_z / atomDist[iB] + mu_AB * uBA_z /rAB);
      
      double term_x = 0.0, term_y = 0.0, term_z = 0.0;
      // second term is 1/Z *  \sum_{C != B} (P(B)t_BC - P(C)t_CB) nabla_B mu_BC
      for( size_t iC = 0; iC < natoms; iC++ ){
        if (iB == iC) continue;

        // coef = (P(B)t_BC - P(C)t_CB)
        double mu_BC = (atomDist[iB] - atomDist[iC]) / RAB[iC + iB*natoms];
        double t_BC = tBecke(mu_BC);
        double t_CB = tBecke(-mu_BC);
        double coef = partitionScratch[iB] *t_BC - partitionScratch[iC] * t_CB;

        const double rBC = RAB[iC + iB*natoms];

        term_x += coef * ((mol[iB].x - point[0]) / atomDist[iB] / rBC - mu_BC * (mol[iB].x - mol[iC].x) / rBC / rBC);
        term_y += coef * ((mol[iB].y - point[1]) / atomDist[iB] / rBC - mu_BC * (mol[iB].y - mol[iC].y) / rBC / rBC);
        term_z += coef * ((mol[iB].z - point[2]) / atomDist[iB] / rBC - mu_BC * (mol[iB].z - mol[iC].z) / rBC / rBC);
      }

      weight_deri_iB[0] -= term_x / sum;
      weight_deri_iB[1] -= term_y / sum;
      weight_deri_iB[2] -= term_z / sum;

      // Use translational invariance to calculate the derivative for the parent atom
      weight_deri_iParent[0] -= weight_deri_iB[0];
      weight_deri_iParent[1] -= weight_deri_iB[1];
      weight_deri_iParent[2] -= weight_deri_iB[2];

    }
    
    // Finally, scale the derivatives by the weight
    for( size_t iB = 0; iB < natoms; iB++ ) 
      for (size_t coord = 0; coord < 3; ++coord) 
        weight_deri_ith[3*iB + coord] *= weight;
      
  } 
}

void reference_ssf_weights_1st_derivative_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  double* weight_deri
){

  const auto safe_magic_ssf_bound = integrator::magic_ssf_factor<> - 1e-4;

  auto gFrisch = [&](double x) {
    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
  };
  auto tFrisch = [&](double x) {
    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double numerator = 35. * (s_x3 + 3. * s_x2 + 3. * s_x + 1.);
    const double denominator = (x - integrator::magic_ssf_factor<>) * (5.*s_x3 + 20.*s_x2 + 29.*s_x + 16.);
    return numerator / denominator ;
  };

  const size_t natoms = mol.natoms();
  const auto&  RAB    = meta.rab();
  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t i  = 0; i  < task.points.size(); ++i  ) {

    auto * weight_deri_ith = weight_deri + 3*natoms*i;

    //zerofy the derivative
    std::fill(weight_deri_ith, weight_deri_ith + 3*natoms, 0.);
    const auto& weight = task.weights[i];

    if (std::abs(weight) < 1.e-12) continue; // weight derivative = 0 when p_A = 0
    const size_t iParent = task.iParent;

    const auto& point  = task.points[i];

    const auto dist_cutoff = 0.5 * (1-integrator::magic_ssf_factor<>) * task.dist_nearest;

    // Compute dist to parent atom
    {
      const double da_x = point[0] - mol[iParent].x;
      const double da_y = point[1] - mol[iParent].y;
      const double da_z = point[2] - mol[iParent].z;

      atomDist[iParent] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);
    }

    if( atomDist[iParent] < dist_cutoff ) continue; // weight derivative = 0 when p_A = 1

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      if( iA == (size_t)iParent ) continue;

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);

    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ )
    if( partitionScratch[iA] > integrator::ssf_weight_tol or 
        partitionScratch[jA] > integrator::ssf_weight_tol ) {

      const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

      if( mu <= -integrator::magic_ssf_factor<> ) {

        partitionScratch[jA] = 0.;

      } else if (mu >= integrator::magic_ssf_factor<>) {

        partitionScratch[iA] = 0.;

      } else {

        double g = 0.5 * ( 1. - gFrisch(mu) );
        partitionScratch[iA] *= g;
        partitionScratch[jA] *= 1. - g;

      }

    }

    // Normalization
    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // calculate derivative now
    auto * weight_deri_iParent = weight_deri_ith + 3*iParent;
    for( size_t iB = 0; iB < natoms; iB++ ) {
      if (iB == iParent) continue;
      auto * weight_deri_iB = weight_deri_ith + 3*iB;
      
      const double rAB = RAB[iB + iParent*natoms];
      double mu_AB  = (atomDist[iParent] - atomDist[iB]) / rAB;
      if(mu_AB > - integrator::magic_ssf_factor<> && mu_AB < safe_magic_ssf_bound){ 
        const double uB_x = mol[iB].x - point[0];
        const double uB_y = mol[iB].y - point[1];
        const double uB_z = mol[iB].z - point[2];

        const double uBA_x =mol[iB].x - mol[iParent].x;
        const double uBA_y =mol[iB].y - mol[iParent].y;
        const double uBA_z =mol[iB].z - mol[iParent].z;

        // first term is - coef1 * nabla_B mu_BA
        double coef1 = tFrisch(mu_AB) * (sum - partitionScratch[iParent])/sum;
        weight_deri_iB[0] -= coef1 / rAB * (uB_x / atomDist[iB] + mu_AB * uBA_x /rAB);
        weight_deri_iB[1] -= coef1 / rAB * (uB_y / atomDist[iB] + mu_AB * uBA_y /rAB);
        weight_deri_iB[2] -= coef1 / rAB * (uB_z / atomDist[iB] + mu_AB * uBA_z /rAB);
      }

      if (std::abs(partitionScratch[iB]) < 1.e-12) continue; // no contribution to the derivative if partition function is zero

      double term_x = 0.0, term_y = 0.0, term_z = 0.0;
      for( size_t iC = 0; iC < natoms; iC++ ){
        if (iB == iC) continue;
        const double rBC = RAB[iC + iB*natoms];
        double mu_BC = (atomDist[iB] - atomDist[iC]) / rBC;
        if(mu_BC > - safe_magic_ssf_bound && mu_BC < safe_magic_ssf_bound){
          double t_BC = tFrisch(mu_BC);
          double coef = partitionScratch[iB] * t_BC / rBC/ sum;

          term_x += coef * ((mol[iB].x - point[0]) / atomDist[iB] - mu_BC * (mol[iB].x - mol[iC].x) / rBC);
          term_y += coef * ((mol[iB].y - point[1]) / atomDist[iB] - mu_BC * (mol[iB].y - mol[iC].y) / rBC);
          term_z += coef * ((mol[iB].z - point[2]) / atomDist[iB] - mu_BC * (mol[iB].z - mol[iC].z) / rBC);

          if(iC != iParent) {
            auto * weight_deri_iC = weight_deri_ith + 3*iC;
            weight_deri_iC[0] += coef * ( (mol[iC].x - point[0]) / atomDist[iC] + mu_BC * (mol[iC].x - mol[iB].x) / rBC );
            weight_deri_iC[1] += coef * ( (mol[iC].y - point[1]) / atomDist[iC] + mu_BC * (mol[iC].y - mol[iB].y) / rBC );
            weight_deri_iC[2] += coef * ( (mol[iC].z - point[2]) / atomDist[iC] + mu_BC * (mol[iC].z - mol[iB].z) / rBC );
          }

        }
      }
        weight_deri_iB[0] -= term_x;
        weight_deri_iB[1] -= term_y;
        weight_deri_iB[2] -= term_z;
    }

    // Use translational invariance to calculate the derivative for the parent atom
    for( size_t iB = 0; iB < natoms; iB++ ) {
      if (iB == iParent) continue;
      auto * weight_deri_iB = weight_deri_ith + 3*iB;
      weight_deri_iParent[0] -= weight_deri_iB[0];
      weight_deri_iParent[1] -= weight_deri_iB[1];
      weight_deri_iParent[2] -= weight_deri_iB[2];
    }
    
    // Finally, scale the derivatives by the weight
    for( size_t iB = 0; iB < natoms; iB++ ) 
      for (size_t coord = 0; coord < 3; ++coord) 
        weight_deri_ith[3*iB + coord] *= weight;

  }
}



/**
 * 1st derivative with contraction 
 */
void reference_becke_weights_1std_contraction_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  const double* w_times_f,
  double* exc_grad_w
){

  // Becke partition functions
  auto hBecke = [](double x) {return 1.5 * x - 0.5 * x * x * x;}; // Eq. 19
  auto gBecke = [&](double x) {return hBecke(hBecke(hBecke(x)));}; // Eq. 20 f_3
  auto tBecke = [&](double x) {
    // for numerical stability (see Jiashu's notes for details)
    if (x > 1.0 - 1e-4) 
      return 0.0; 
    const double p1 = hBecke(x);
    const double p2 = hBecke(p1);
    return - 27.0 * (1. + p2) * (1. + p1) * (1. + x) / (1. - x) / (2. + p2) / (2. + p1) / (2. + x);
  };

  const size_t natoms = mol.natoms();
  const auto&  RAB    = meta.rab();
  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t i  = 0; i  < task.points.size(); ++i  ) {

    const size_t iParent = task.iParent;
    const auto& point  = task.points[i];
    const auto w_times_f_i = w_times_f[i];

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);
    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ ){

      double mu  = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];
      const double g = gBecke(mu);

      partitionScratch[iA] *= 0.5 * (1. - g);
      partitionScratch[jA] *= 0.5 * (1. + g);
    }

    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // calculate derivative now
    for( size_t iB = 0; iB < natoms; iB++ ) {
      if (iB == iParent) continue;
      double exc_grad_w_iBx = 0.0, exc_grad_w_iBy = 0.0, exc_grad_w_iBz = 0.0;
      
      const double uB_x = mol[iB].x - point[0];
      const double uB_y = mol[iB].y - point[1];
      const double uB_z = mol[iB].z - point[2];

      const double uBA_x =mol[iB].x - mol[iParent].x;
      const double uBA_y =mol[iB].y - mol[iParent].y;
      const double uBA_z =mol[iB].z - mol[iParent].z;
      const double rAB = RAB[iB + iParent*natoms];

      double mu_AB  = (atomDist[iParent] - atomDist[iB]) / rAB;

      // first term is - coef1 * nabla_B mu_BA
      double coef1 = tBecke(mu_AB) * w_times_f_i;
      exc_grad_w_iBx = - coef1 / rAB * (uB_x / atomDist[iB] + mu_AB * uBA_x /rAB);
      exc_grad_w_iBy = - coef1 / rAB * (uB_y / atomDist[iB] + mu_AB * uBA_y /rAB);
      exc_grad_w_iBz = - coef1 / rAB * (uB_z / atomDist[iB] + mu_AB * uBA_z /rAB);
      
      // second term is 1/Z *  \sum_{C != B} (P(B)t_BC - P(C)t_CB) nabla_B mu_BC
      for( size_t iC = 0; iC < natoms; iC++ ){
        if (iB == iC) continue;

        // coef = (P(B)t_BC - P(C)t_CB)
        double mu_BC = (atomDist[iB] - atomDist[iC]) / RAB[iC + iB*natoms];
        double t_BC = tBecke(mu_BC);
        double t_CB = tBecke(-mu_BC);
        double coef = (partitionScratch[iB] *t_BC - partitionScratch[iC] * t_CB)/ sum * w_times_f_i;

        const double rBC = RAB[iC + iB*natoms];

        exc_grad_w_iBx -= coef * ((mol[iB].x - point[0]) / atomDist[iB] / rBC - mu_BC * (mol[iB].x - mol[iC].x) / rBC / rBC);
        exc_grad_w_iBy -= coef * ((mol[iB].y - point[1]) / atomDist[iB] / rBC - mu_BC * (mol[iB].y - mol[iC].y) / rBC / rBC);
        exc_grad_w_iBz -= coef * ((mol[iB].z - point[2]) / atomDist[iB] / rBC - mu_BC * (mol[iB].z - mol[iC].z) / rBC / rBC);
      }

      #pragma omp atomic
      exc_grad_w[3*iB + 0] += exc_grad_w_iBx;
      #pragma omp atomic
      exc_grad_w[3*iB + 1] += exc_grad_w_iBy;
      #pragma omp atomic
      exc_grad_w[3*iB + 2] += exc_grad_w_iBz;
      // Use translational invariance to calculate the derivative for the parent atom
      #pragma omp atomic
      exc_grad_w[3*iParent + 0] -= exc_grad_w_iBx;
      #pragma omp atomic
      exc_grad_w[3*iParent + 1] -= exc_grad_w_iBy;
      #pragma omp atomic
      exc_grad_w[3*iParent + 2] -= exc_grad_w_iBz;

    }  
  } 

}


void reference_ssf_weights_1std_contraction_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  const double* w_times_f,
  double* exc_grad_w
){

  const double safe_magic_ssf_bound = integrator::magic_ssf_factor<> - 1.e-4;
  const double w_times_f_thresh = 1.e-12;
  const double weight_tol = integrator::ssf_weight_tol;

  auto gFrisch = [&](double x) {
    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
  };

  auto tFrisch = [&](double x) {
    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double numerator = (35.) * (s_x3 + (3.) * s_x2 + (3.) * s_x + (1.));
    const double denominator = (x - integrator::magic_ssf_factor<>) * ((5.)*s_x3 + (20.)*s_x2 + (29.)*s_x + (16.));
    return numerator / denominator ;
  };

  const size_t natoms = mol.natoms();
  const auto&  RAB    = meta.rab();
  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t i  = 0; i  < task.points.size(); ++i  ) {
    const auto& w_times_f_i = w_times_f[i];
    if (fabs(w_times_f_i) < w_times_f_thresh) continue; // weight derivative = 0 when p_A = 0
    const size_t iParent = task.iParent;
    const auto& point  = task.points[i];

    const auto dist_cutoff = 0.5 * (1-integrator::magic_ssf_factor<>) * task.dist_nearest;

    // Compute dist to parent atom
    {
      const double da_x = point[0] - mol[iParent].x;
      const double da_y = point[1] - mol[iParent].y;
      const double da_z = point[2] - mol[iParent].z;

      atomDist[iParent] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);
    }

    if( atomDist[iParent] < dist_cutoff ) continue; // weight derivative = 0 when p_A = 1

    // Compute distances of each center to point
    for(size_t iA = 0; iA < natoms; iA++) {

      if( iA == iParent ) continue;

      const double da_x = point[0] - mol[iA].x;
      const double da_y = point[1] - mol[iA].y;
      const double da_z = point[2] - mol[iA].z;

      atomDist[iA] = std::sqrt(da_x*da_x + da_y*da_y + da_z*da_z);

    }

    // Evaluate unnormalized partition functions 
    std::fill(partitionScratch.begin(),partitionScratch.end(),1.);

    for( size_t iA = 0; iA < natoms; iA++ ) 
    for( size_t jA = 0; jA < iA;     jA++ )
    if( partitionScratch[iA] > weight_tol or 
        partitionScratch[jA] > weight_tol ) {

      const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

      if( mu <= -integrator::magic_ssf_factor<> ) {

        partitionScratch[jA] = 0.;

      } else if (mu >= integrator::magic_ssf_factor<>) {

        partitionScratch[iA] = 0.;

      } else {

        double g = 0.5 * ( 1. - gFrisch(mu) );
        partitionScratch[iA] *= g;
        partitionScratch[jA] *= 1. - g;

      }

    }

    double sum = 0.;
    for( size_t iA = 0; iA < natoms; iA++ )  sum += partitionScratch[iA];

    // calculate derivative now
    for( size_t iB = 0; iB < natoms; iB++ ) {
      if (iB == iParent) continue;
      double exc_grad_w_iBx = 0.0, exc_grad_w_iBy = 0.0, exc_grad_w_iBz = 0.0;
      
      const double rAB = RAB[iB + iParent*natoms];
      double rAB_inv = 1.0 / rAB;
      double mu_AB  = (atomDist[iParent] - atomDist[iB]) * rAB_inv ;
      if( fabs(mu_AB) < safe_magic_ssf_bound) {
        const double uB_x = mol[iB].x - point[0];
        const double uB_y = mol[iB].y - point[1];
        const double uB_z = mol[iB].z - point[2];

        const double uBA_x =mol[iB].x - mol[iParent].x;
        const double uBA_y =mol[iB].y - mol[iParent].y;
        const double uBA_z =mol[iB].z - mol[iParent].z;

        // first term is - coef1 * nabla_B mu_BA
        double coef1 = tFrisch(mu_AB) / rAB * (partitionScratch[iParent]-sum)/sum * w_times_f_i / atomDist[iB];
        exc_grad_w_iBx = coef1 * (uB_x + mu_AB * uBA_x * rAB_inv * atomDist[iB]);
        exc_grad_w_iBy = coef1 * (uB_y + mu_AB * uBA_y * rAB_inv * atomDist[iB]);
        exc_grad_w_iBz = coef1 * (uB_z + mu_AB * uBA_z * rAB_inv * atomDist[iB]);
      }

      if (partitionScratch[iB] > weight_tol){
        for( size_t iC = 0; iC < natoms; iC++ ){
          if (iB == iC) continue;
          const double rBC = RAB[iC + iB*natoms];
          double mu_BC = (atomDist[iB] - atomDist[iC]) / rBC;
          if(fabs(mu_BC) < safe_magic_ssf_bound){
            double t_BC = tFrisch(mu_BC);
            double coef = partitionScratch[iB] * t_BC / rBC/ sum * w_times_f_i;

            exc_grad_w_iBx -= coef * ((mol[iB].x - point[0]) / atomDist[iB] - mu_BC * (mol[iB].x - mol[iC].x) / rBC);
            exc_grad_w_iBy -= coef * ((mol[iB].y - point[1]) / atomDist[iB] - mu_BC * (mol[iB].y - mol[iC].y) / rBC);
            exc_grad_w_iBz -= coef * ((mol[iB].z - point[2]) / atomDist[iB] - mu_BC * (mol[iB].z - mol[iC].z) / rBC);

            if(iC != iParent) {
              
              double C_x = coef * ((mol[iC].x - point[0]) / atomDist[iC] + mu_BC * (mol[iC].x - mol[iB].x) / rBC);
              double C_y = coef * ((mol[iC].y - point[1]) / atomDist[iC] + mu_BC * (mol[iC].y - mol[iB].y) / rBC);
              double C_z = coef * ((mol[iC].z - point[2]) / atomDist[iC] + mu_BC * (mol[iC].z - mol[iB].z) / rBC);
              // Update exc_grad_w_iC
              #pragma omp atomic
              exc_grad_w[3*iC + 0] += C_x;
              #pragma omp atomic
              exc_grad_w[3*iC + 1] += C_y;
              #pragma omp atomic
              exc_grad_w[3*iC + 2] += C_z;
              // Update exc_grad_w for the parent atom
              #pragma omp atomic
              exc_grad_w[3*iParent + 0] -= C_x;
              #pragma omp atomic
              exc_grad_w[3*iParent + 1] -= C_y;
              #pragma omp atomic
              exc_grad_w[3*iParent + 2] -= C_z;
            }

          }
        }
      } 

      #pragma omp atomic
      exc_grad_w[3*iB + 0] += exc_grad_w_iBx;
      #pragma omp atomic
      exc_grad_w[3*iB + 1] += exc_grad_w_iBy;
      #pragma omp atomic
      exc_grad_w[3*iB + 2] += exc_grad_w_iBz;
      // Use translational invariance to calculate the derivative for the parent atom
      #pragma omp atomic
      exc_grad_w[3*iParent + 0] -= exc_grad_w_iBx;
      #pragma omp atomic
      exc_grad_w[3*iParent + 1] -= exc_grad_w_iBy;
      #pragma omp atomic
      exc_grad_w[3*iParent + 2] -= exc_grad_w_iBz;

    }
  }

}



}

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
#include "weights_generate.hpp"
#include <fstream>
#include <string>
#include <gauxc/util/div_ceil.hpp>

#ifdef GAUXC_HAS_HIP
#include "device_specific/hip_util.hpp"
#include "device/hip/hip_aos_scheme1.hpp"
//#include "device/hip/hip_aos_scheme1_weights.hpp"
#include "device/hip/kernels/grid_to_center.hpp"
#include "device/hip/kernels/hip_ssf_1d.hpp"
#include "device/hip/kernels/hip_ssh_2d.hpp"
      


                        
void test_hip_weights( std::ifstream& in_file ) {

  ref_weights_data ref_data;
  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  //std::vector< std::array<double,3> > points;
  std::vector< double > points_x, points_y, points_z;
  std::vector< double >               weights, weights_ref;
  std::vector< double >               dist_nearest;
  std::vector< int32_t >              iparent;

  for( auto& task : ref_data.tasks_unm ) {
    //points.insert( points.end(),
    //               task.points.begin(),
    //               task.points.end() );
    for( auto pt : task.points ) {
      points_x.emplace_back(pt[0]);
      points_y.emplace_back(pt[1]);
      points_z.emplace_back(pt[2]);
    }
    weights.insert( weights.end(),
                    task.weights.begin(),
                    task.weights.end() );

    size_t npts = task.points.size();
    dist_nearest.insert( dist_nearest.end(), npts,
                         task.dist_nearest );
    iparent.insert( iparent.end(), npts, task.iParent );
  }

  for( auto& task : ref_data.tasks_mod ) {
    weights_ref.insert( weights_ref.end(),
                        task.weights.begin(),
                        task.weights.end() );
  }

  size_t npts   = points_x.size();
  size_t natoms = ref_data.mol.natoms();

  //constexpr auto weight_unroll = alg_constants::HipAoSScheme1::weight_unroll;

  //size_t LDatoms = util::div_ceil( natoms, weight_unroll ) * weight_unroll;
  size_t LDatoms = natoms;

  std::vector< double >  coords( 3 * natoms );
  for( auto iat = 0 ; iat < natoms; ++iat ) {
    coords[ 3*iat + 0 ] = ref_data.mol.at(iat).x;
    coords[ 3*iat + 1 ] = ref_data.mol.at(iat).y;
    coords[ 3*iat + 2 ] = ref_data.mol.at(iat).z;
  }


  //auto* points_d  = util::hip_malloc<double>( 3*npts );
  auto* points_x_d  = util::hip_malloc<double>( npts );
  auto* points_y_d  = util::hip_malloc<double>( npts );
  auto* points_z_d  = util::hip_malloc<double>( npts );
  auto* weights_d = util::hip_malloc<double>( npts   );
  auto* iparent_d = util::hip_malloc<int32_t>( npts  );
  auto* distnea_d = util::hip_malloc<double>( npts   );
  auto* rab_d     = util::hip_malloc<double>( natoms*LDatoms );
  auto* coords_d  = util::hip_malloc<double>( 3*natoms );
  auto* dist_scr_d= util::hip_malloc<double>( npts*LDatoms );

  //util::hip_copy( 3*npts, points_d,  points.data()->data() );
  util::hip_copy( npts, points_x_d,  points_x.data() );
  util::hip_copy( npts, points_y_d,  points_y.data() );
  util::hip_copy( npts, points_z_d,  points_z.data() );
  util::hip_copy( npts,   weights_d, weights.data() );
  util::hip_copy( npts,   iparent_d, iparent.data() );
  util::hip_copy( npts,   distnea_d, dist_nearest.data() );

  std::vector<double> rab_inv(natoms*natoms);
  for( auto i = 0; i < natoms*natoms; ++i )
    rab_inv[i] = 1./ref_data.meta->rab().data()[i];

  util::hip_copy_2d( rab_d, LDatoms * sizeof(double),
                      rab_inv.data(), natoms * sizeof(double),
                      natoms * sizeof(double), natoms, "RAB H2D");

  util::hip_copy( 3*natoms, coords_d, coords.data() );

  hipStream_t stream = 0;
  //hip_aos_scheme1_weights_wrapper( npts, natoms, points_x_d, points_y_d, points_z_d, rab_d,
  //  LDatoms, coords_d, dist_scr_d, LDatoms, iparent_d, distnea_d,
  //  weights_d, stream );
  compute_grid_to_center_dist( npts, natoms, coords_d, points_x_d, points_y_d, points_z_d,
    dist_scr_d, LDatoms, stream );
  partition_weights_ssf_2d( npts, natoms, rab_d, LDatoms, coords_d, dist_scr_d, LDatoms,
    iparent_d, distnea_d, weights_d, stream );

  util::hip_device_sync();
  util::hip_copy( npts, weights.data(), weights_d );
  util::hip_free( points_x_d, points_y_d, points_z_d, weights_d, iparent_d, distnea_d,
                   rab_d, coords_d, dist_scr_d );

  for( auto i = 0ul; i < npts; ++i )
    CHECK( weights.at(i) == Approx( weights_ref.at(i) ) );

}
#endif

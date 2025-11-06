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

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <gauxc/gauxc_config.hpp>
#include <gauxc/shell.hpp>
#include <gauxc/exceptions.hpp>

namespace GauXC {

struct XCTask {

  int32_t                              iParent = -1;
  std::vector< std::array<double,3> >  points;
  std::vector< double  >               weights;
  int32_t                              npts = 0;

  double                               dist_nearest;
  double                               max_weight = std::numeric_limits<double>::infinity();

  struct screening_data {
    using pair_t = std::pair<int32_t,int32_t>;
    std::vector<int32_t>               shell_list;
    std::vector<pair_t>                shell_pair_list;
    std::vector<int32_t>               shell_pair_idx_list;
    std::vector<int32_t>               submat_block;
    std::vector<std::array<int32_t,3>> submat_map;
    int32_t                            nbe = 0;

    bool equiv_with( const screening_data& other ) const {
      return shell_list == other.shell_list and 
        shell_pair_list == other.shell_pair_list;
    }

    inline size_t volume() const {
      return (shell_list.size() + 2*shell_pair_list.size() + submat_block.size() +
              3*submat_map.size() + 1) * sizeof(int32_t);
    }
  };

  struct features {
    // inputs for onedft
    std::vector<double> den_eval;
    std::vector<double> dden_x_eval;
    std::vector<double> dden_y_eval;
    std::vector<double> dden_z_eval;
    std::vector<double> tau;
    // results from onedft
    std::vector<double> vdden_eval_a;
    std::vector<double> vdden_eval_b;
    std::vector<double> vdden_x_eval_a;
    std::vector<double> vdden_x_eval_b;
    std::vector<double> vdden_y_eval_a;
    std::vector<double> vdden_y_eval_b;
    std::vector<double> vdden_z_eval_a;
    std::vector<double> vdden_z_eval_b;
    std::vector<double> vtau;
  };
  features feat;

  inline size_t volume() const {
    return 2 * sizeof(int32_t) +
      (3*points.size() + weights.size() + 2) * sizeof(double) +
      bfn_screening.volume() + cou_screening.volume();
  }

  screening_data bfn_screening;
  screening_data cou_screening;

  void merge_with( const XCTask& other ) {
    if( !equiv_with(other) )
      GAUXC_GENERIC_EXCEPTION("Cannot Perform Requested Merge: Incompatible Tasks");
    points.insert( points.end(), other.points.begin(), other.points.end() );
    weights.insert( weights.end(), other.weights.begin(), other.weights.end() );
    npts = points.size();
  }

  template <typename TaskIt>
  void merge_with( TaskIt begin, TaskIt end ) {

    size_t old_sz = points.size();
    size_t pts_add = std::accumulate( begin, end, 0ul,
      []( const auto &a, const auto &t ) {
        return a + t.points.size();
      });

    size_t new_sz = old_sz + pts_add;
    points.resize( new_sz );
    weights.resize( new_sz );

    auto points_it  = points.begin()  + old_sz;
    auto weights_it = weights.begin() + old_sz;
    for( auto it = begin; it != end; ++it ) {
      if( !equiv_with(*it) )
        GAUXC_GENERIC_EXCEPTION("Cannot Perform Requested Task Merge");
      points_it  = std::copy( it->points.begin(), it->points.end(), points_it );
      weights_it = std::copy( it->weights.begin(), it->weights.end(), weights_it );
    }

    npts = points.size();
  }


  inline bool equiv_with( const XCTask& other ) const {
    return iParent == other.iParent and 
      bfn_screening.equiv_with(other.bfn_screening);
  }

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( iParent, bfn_screening.nbe, npts, dist_nearest, max_weight, 
      bfn_screening.shell_list, points, weights );  
  }


  inline size_t cost(size_t n_deriv, size_t natoms) const {
    return (bfn_screening.nbe * ( 1 + bfn_screening.nbe + n_deriv ) + natoms * natoms) * npts;
  }
  inline size_t cost_exc_vxc(size_t n_deriv) const {
    return bfn_screening.nbe * ( 1 + bfn_screening.nbe + n_deriv ) * npts;
  }
  inline size_t cost_exx() const {
    return ( bfn_screening.nbe + 2*cou_screening.nbe*bfn_screening.nbe +
             2*cou_screening.shell_pair_list.size() ) * npts;
  }
};


}

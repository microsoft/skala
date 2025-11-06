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

#include "shell_batched_xc_integrator.hpp"
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <gauxc/util/misc.hpp>

namespace GauXC::detail {

ShellBatchedXCIntegratorBase::incore_task_data
  ShellBatchedXCIntegratorBase::generate_incore_task( uint32_t nbf_threshold,
    const basis_type& basis, host_task_iterator task_begin,
    host_task_iterator task_end ) {

  // Find task with largest NBE
  auto nbe_comparator = []( const auto& task_a, const auto& task_b ) {
    return task_a.bfn_screening.nbe < task_b.bfn_screening.nbe;
  };
  auto max_task = std::max_element( task_begin, task_end, nbe_comparator );

  const auto max_shell_list = max_task->bfn_screening.shell_list; // copy for reset

  // Init union shell list to max shell list outside of loop
  std::set<int32_t> union_shell_set(max_shell_list.begin(), 
                                    max_shell_list.end());


  // Voodoo: once only Manwe and I knew what was happening here, now
  // only Manwe knows
  int n_overlap_pthresh     = 20;
  double overlap_pthresh_delta = 1. / n_overlap_pthresh;
  std::vector<double> overlap_pthresh;
  for( int i = 1; i < n_overlap_pthresh; ++i )
    overlap_pthresh.emplace_back( i*overlap_pthresh_delta );

  std::vector<int> overlap_pthresh_idx( overlap_pthresh.size() );
  std::iota( overlap_pthresh_idx.begin(), overlap_pthresh_idx.end(), 0 );

  std::map<int, std::pair<host_task_iterator, decltype(union_shell_set)>> 
    cached_task_ends;

  int cur_partition_pthresh_idx = -1;

  auto _it = std::partition_point( overlap_pthresh_idx.rbegin(), 
                                   overlap_pthresh_idx.rend(), 
  [&](int idx) {

    uint32_t overlap_threshold = 
      std::max(1., max_shell_list.size() * overlap_pthresh[idx] );


    host_task_iterator search_st = task_begin;
    host_task_iterator search_en = task_end;

    // Make a local copy of union list
    std::set<int32_t> local_union_shell_set;

    // Attempt to limit task search based on current partition
    if( cur_partition_pthresh_idx >= 0 ) {

      const auto& last_pthresh = 
        cached_task_ends.at(cur_partition_pthresh_idx);

      if( cur_partition_pthresh_idx > idx ) {
        search_st = last_pthresh.first;    
        local_union_shell_set = last_pthresh.second;
      } else {
        search_en = last_pthresh.first;    
        local_union_shell_set = union_shell_set;
      }

    } else {
      local_union_shell_set = union_shell_set;
    }


    // Partition tasks into those which overlap max_task up to
    // specified threshold
    auto local_task_end = std::partition( search_st, search_en, 
      [&](const auto& t) {
        return util::integral_list_intersect( max_shell_list, 
          t.bfn_screening.shell_list, overlap_threshold );
      } );



    // Take union of shell list for all overlapping tasks
    for( auto task_it = search_st; task_it != local_task_end; ++task_it ) {
      local_union_shell_set.insert( task_it->bfn_screening.shell_list.begin(), 
                                    task_it->bfn_screening.shell_list.end() );
    }

    auto cur_nbe = basis.nbf_subset( local_union_shell_set.begin(), 
                                     local_union_shell_set.end() );

    //std::cout << "  Threshold %       = " << std::setw(5)  << overlap_pthresh[idx] << ", ";
    //std::cout << "  Overlap Threshold = " << std::setw(8)  << overlap_threshold    << ", ";
    //std::cout << "  Current NBE       = " << std::setw(8)  << cur_nbe              << std::endl;

    // Cache the data
    cached_task_ends[idx] = std::make_pair( local_task_end, local_union_shell_set );

    // Update partitioned threshold
    cur_partition_pthresh_idx = idx;

    return (uint32_t)cur_nbe < nbf_threshold;

  } );

  host_task_iterator local_task_end;
  auto _idx_partition = (_it == overlap_pthresh_idx.rend()) ? 0 : *_it;
  std::tie( local_task_end, union_shell_set ) = 
    cached_task_ends.at(_idx_partition);





  //std::cout << "FOUND " << std::distance( task_begin, local_task_end ) 
  //                      << " OVERLAPPING TASKS" << std::endl;


  std::vector<int32_t> union_shell_list( union_shell_set.begin(),
                                         union_shell_set.end() );

  // Try to add additional tasks given current union list
  local_task_end = std::partition( local_task_end, task_end, 
    [&]( const auto& t ) {
      return util::list_subset( union_shell_list, t.bfn_screening.shell_list );
    } );

  //std::cout << "FOUND " << std::distance( task_begin, local_task_end ) 
  //                      << " SUBTASKS" << std::endl;


  incore_task_data ex_task;
  ex_task.task_begin = task_begin;
  ex_task.task_end   = local_task_end;
  ex_task.shell_list = std::move( union_shell_list );

  return ex_task;
}

}

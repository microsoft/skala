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
#include "integrator_common.hpp"

#include <tuple>
#include <array>
#include <vector>
#include <cstdint>

namespace GauXC      {

std::tuple< std::vector< std::array<int32_t, 3> > , std::vector< int32_t > >
  gen_compressed_submat_map( const BasisSetMap&       basis_map,
                             const std::vector< int32_t >& shell_mask,
                             const int32_t LDA, const int32_t block_size ) {


  std::vector< std::pair<int32_t, int32_t> > submat_map;

  // Init as if there is no screening
  submat_map.emplace_back(
    basis_map.shell_to_ao_range( shell_mask.front() ).first,
    basis_map.shell_to_ao_range( shell_mask.back()  ).second
  );


  for( auto sh_it =  shell_mask.begin(); sh_it != shell_mask.end()-1; ++sh_it ) {

    if( *(sh_it+1) - *(sh_it) != 1 ) {

      submat_map.back().second = basis_map.shell_to_ao_range(*sh_it).second;
        
      submat_map.emplace_back(
        basis_map.shell_to_ao_range( *(sh_it+1) ).first,
        basis_map.shell_to_ao_range( shell_mask.back()  ).second
      );

    }



  }


  if( shell_mask.size() == 1 )
    submat_map.back().second = 
      basis_map.shell_to_ao_range(shell_mask[0]).second;


  /*
   * This code block does post-processing for the submatrix optimizations
   *
   * It first adds the index within the small matrix as another pair in the vector.
   * This allows the kernel to process multiple cuts concurrently within the same
   * task. Additionally, it adds artificial breaks in the cut at the given interval
   * This is to reduce the amount of bookkeeping that the kernel is required to do.
   *
   * While the small matrix start indices are stored in the additional pair, the second 
   * value is blank as the delta can be reused from the big matrix start and stop points.
   *
   * It also creates an additional vector which stores the mapping from big matrix block 
   * to cut index. As a kernel only processes a single block of the big matrix, it can
   * look up the starting and ending cut indices and ignore all other cuts.
   *
   */
  std::vector< std::array<int32_t, 3> > submat_map_expand;
  std::vector< int32_t > submat_block_idx;
  submat_block_idx.push_back(0);
  const int end_point = LDA; 

  size_t cut_index = 0;
  int cut_expand_index = 0;
  int small_index = 0;
  int delta;
  for (int block_start = 0; block_start < end_point; block_start += block_size) {
    const int block_end = block_start + block_size;
    
    if( cut_index < submat_map.size() ) {

    int cut_start = submat_map[cut_index].first;
    int cut_end   = submat_map[cut_index].second;
    while (cut_index < submat_map.size() && cut_start < block_end) {
      if (cut_start < block_start && cut_end < block_start) {
        // In this case the cut starts and stops before the block starts.
	      // This should never happen as the cut should already have been processed.
	      // But I included this case as a sanity check.
	      std::cout << "Something is wrong constructing the extended cut map " << std::endl;
      } else if (cut_start < block_start && cut_end > block_end) {
        // In this case, the cut spans the entire block. The cut index is not
	      // incremented because we need to process the rest of it.
	      delta = block_end - block_start;
	      submat_map_expand.push_back({block_start, delta, small_index});
        small_index += delta;

	      cut_expand_index++;
	      break;
      } else if (cut_start < block_start) {
	      // In this case the cut begins before the block, but ends within
	      // this block
	      delta = cut_end - block_start;
	      submat_map_expand.push_back({block_start, delta, small_index});
              small_index += delta;

	      cut_index++;
	      cut_expand_index++;
      } else if (cut_end > block_end) {
	      // In this case, the cut starts within the block, but extends
	      // into the next block. Again, the cut index is not incremented
	      delta = block_end - cut_start;
	      submat_map_expand.push_back({cut_start, delta, small_index});
        small_index += delta;

	      cut_expand_index++;
	      break;
      } else {
	      // In this case, the cut starts and ends within the block
	      delta = cut_end - cut_start;
	      submat_map_expand.push_back({cut_start, delta, small_index});
        small_index += delta;

	      cut_index++;
	      cut_expand_index++;
      }

      if( cut_index < submat_map.size() ) {
        cut_start = submat_map[cut_index].first;
        cut_end   = submat_map[cut_index].second;
      }
    }

    } // guard on submat_map access
    submat_block_idx.push_back(cut_expand_index);
  }
  return {submat_map_expand, submat_block_idx};
}



}

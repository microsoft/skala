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

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

namespace GauXC {
namespace util  {

template <typename Integral>
auto ranges_from_list( const std::vector<Integral>& shell_list ) {

  std::vector< std::pair<Integral,Integral> > ranges;
  ranges.emplace_back( shell_list.front(), shell_list.back() );

  for( auto it = shell_list.begin(); it != shell_list.end()-1; ++it ) {
    if( *(it+1) - *it != 1 ) {
      ranges.back().second = *it;
      ranges.emplace_back( *(it+1), shell_list.back() );
    }
  }

  return ranges;

}





// Checks if B is a subset of A
template <typename C1, typename C2>
inline auto list_subset( const C1& A, const C2& B ) {
  return std::includes( A.begin(), A.end(), B.begin(), B.end() );
}

// Check if two lists intersect
template <typename Integral>
inline auto integral_list_intersect( const std::vector<Integral>& A,
                                     const std::vector<Integral>& B ) {


  constexpr size_t sz_ratio = 100;
  const size_t A_sz = A.size();
  const size_t B_sz = B.size();

  const auto A_begin = A.begin();
  const auto A_end   = A.end();
  const auto B_begin = B.begin();
  const auto B_end   = B.end();

  // Fall through if query list is much larger than max list
  if( A_sz * sz_ratio < B_sz ) {
    for( const auto& val : A ) {
      if( std::binary_search( B_begin, B_end, val ) ) 
        return true;
    }
    return false;
  }

  // Fall through if max list is much larger than query list
  if( B_sz * sz_ratio < A_sz ) {
    for( const auto& val : B ) {
      if( std::binary_search( A_begin, A_end, val ) )
        return true;
    }
    return false;
  }

  // Default if lists are about the same size
  auto B_it = B_begin;
  auto A_it = A_begin;

  while( B_it != B_end and A_it != A_end ) {

    if( *B_it < *A_it ) {
      B_it = std::lower_bound( B_it, B_end, *A_it );
      continue;
    }

    if( *A_it < *B_it ) {
      A_it = std::lower_bound( A_it, A_end, *B_it );
      continue;
    }

    return true;

  }

  return false;


}





// Checks if two lists intersect more than a specified threshold
template <typename Integral>
inline auto integral_list_intersect( const std::vector<Integral>& A,
                                     const std::vector<Integral>& B,
                                     const uint32_t overlap_threshold_spec ) {

  const uint32_t max_intersect_sz  = std::min(A.size(), B.size());
  const uint32_t overlap_threshold = std::min( max_intersect_sz, 
                                               overlap_threshold_spec );

  constexpr size_t sz_ratio = 100;
  const size_t A_sz = A.size();
  const size_t B_sz = B.size();

  const auto A_begin = A.begin();
  const auto A_end   = A.end();
  const auto B_begin = B.begin();
  const auto B_end   = B.end();

  uint32_t overlap_count = 0;

  // Fall through if query list is much larger than max list
  if( A_sz * sz_ratio < B_sz ) {

    for( const auto& val : A ) {
      overlap_count += !!std::binary_search( B_begin, B_end, val );
      if( overlap_count == overlap_threshold ) return true;
    }
    return false;

  }

  // Fall through if max list is much larger than query list
  if( B_sz * sz_ratio < A_sz ) {
    for( const auto& val : B ) {
      overlap_count += !!std::binary_search( A_begin, A_end, val );
      if( overlap_count == overlap_threshold ) return true;
    }
    return false;
  }

  // Default if lists are about the same size
  auto B_it = B_begin;
  auto A_it = A_begin;

  while( B_it != B_end and A_it != A_end ) {

    if( *B_it < *A_it ) {
      B_it = std::lower_bound( B_it, B_end, *A_it );
      continue;
    }

    if( *A_it < *B_it ) {
      A_it = std::lower_bound( A_it, A_end, *B_it );
      continue;
    }

    // *A_it == *B_it if code reaches here
    overlap_count++;
    A_it++; B_it++; // Increment iterators
    if( overlap_count == overlap_threshold) return true;

  }

  return false;


}


}
}

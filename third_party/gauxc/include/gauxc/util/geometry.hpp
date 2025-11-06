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
#include <cmath>

namespace GauXC {
namespace geometry {

template <size_t N, typename T>
T euclidean_dist( const T* a, const T* b ) {
  T dist = 0.;
  for( size_t i = 0; i < N; ++i ) {
    auto tmp = a[i] - b[i];
    dist += tmp*tmp;
  }
  return std::sqrt(dist);
}

template <typename T, size_t N>
T euclidean_dist( const std::array<T,N>& a, const std::array<T,N>& b ) {
  return euclidean_dist<N,T>(a.data(), b.data());
}

template <size_t N, typename T>
bool cube_sphere_intersect( const T* lo, const T* up, const T* center, T rad ) {

  T dist = rad * rad;
  for( size_t i = 0; i < N; ++i ) {
    T r = 0.;
    if( center[i] < lo[i] )      r = lo[i] - center[i];
    else if( center[i] > up[i] ) r = center[i] - up[i];
    dist -= r*r;
    if( dist < T(0.) ) return false;
  }

  return true;

}

template <typename T, size_t N>
bool cube_sphere_intersect( const std::array<T,N>& lo, const std::array<T,N>& up,
                            const std::array<T,N>& center, T rad ) {
  return cube_sphere_intersect<N,T>( lo.data(), up.data(), center.data(), rad );
}

template <size_t N, typename T>
T cube_point_dist_closest( const T* lo, const T* up, const T* point ) {

#if 1
  T dist = 0.;
  for( int i = 0; i < N; ++i ) {
    T r = 0.;
    if( point[i] < lo[i] )      r = lo[i] - point[i];
    else if( point[i] > up[i] ) r = point[i] - up[i];
    dist += r*r;
  }

  return std::sqrt(dist);
#else
  std::array<T,N> box_dims;
  for( int i = 0; i < N; ++i ) box_dims[i] = std::abs( up[i] - lo[i] )/2.;
  
  std::array<T,N> pt_tmp;
  // Recenter point width coordinate transformation that sends lo -> -box_dims
  // and scales the box do have dims +-1
  for( int i = 0; i < N; ++i ) {
    pt_tmp[i] = (point[i] - (box_dims[i] + lo[i]))/box_dims[i];
  }

  T dist = 0.;
  for( int i = 0; i < N; ++i ) {
    const T val = box_dims[i] * std::max( T(0.), std::abs(pt_tmp[i])-1 );
    dist += val*val;
  }
  return std::sqrt(dist);

#endif
      
}

template <typename T, size_t N>
T cube_point_dist_closest( const std::array<T,N>& lo, const std::array<T,N>& up, 
                           const std::array<T,N>& point ) {
  return cube_point_dist_closest<N,T>( lo.data(), up.data(), point.data() );
}

template <size_t N, typename T>
std::array<T,N> cube_point_closest_approach( const T* lo, const T* up, 
                                             const T* center ) {
  std::array<T,N> point;
  for( int i = 0; i < N; ++i ) {
    if( center[i] < lo[i] )      point[i] = lo[i];
    else if( center[i] > up[i] ) point[i] = up[i];
    else if( (center[i]-lo[i]) < (up[i]-center[i]) ) point[i] = lo[i];
    else point[i] = up[i];
  }
  return point;
}

template <typename T,size_t N>
std::array<T,N> cube_point_closest_approach( const std::array<T,N>& lo, 
                                             const std::array<T,N>& up, 
                                             const std::array<T,N>& center ) {
  return cube_point_closest_approach( lo.data(), up.data(), center.data() );
}


}
}

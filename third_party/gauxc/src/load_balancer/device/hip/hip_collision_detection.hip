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
#include "hip/hip_runtime.h"
#include <gauxc/util/div_ceil.hpp>
#include <hipcub/hipcub.hpp>

#include "hip_collision_detection.hpp"
#include "device_specific/hip_device_constants.hpp"

namespace GauXC         {
namespace load_balancer {
namespace hip          {

using namespace GauXC::hip;

__device__ __inline__ 
int cube_sphere_intersect( 
  const double3 lo, 
  const double3 up,
  const double3 center,
  const double  rad
) {

  double dist = rad * rad;

  if( center.x < lo.x ) {
    const double r_lo = center.x - lo.x;
    const double dist_lo = r_lo * r_lo;
    dist -= dist_lo;
  } else if( center.x > up.x ) {
    const double r_up = center.x - up.x;
    const double dist_up = r_up * r_up;
    dist -= dist_up;
  }

  if( dist < 0. ) return false;

  if( center.y < lo.y ) {
    const double r_lo = center.y - lo.y;
    const double dist_lo = r_lo * r_lo;
    dist -= dist_lo;
  } else if( center.y > up.y ) {
    const double r_up = center.y - up.y;
    const double dist_up = r_up * r_up;
    dist -= dist_up;
  }

  if( dist < 0. ) return false;


  if( center.z < lo.z ) {
    const double r_lo = center.z - lo.z;
    const double dist_lo = r_lo * r_lo;
    dist -= dist_lo;
  } else if( center.z > up.z ) {
    const double r_up = center.z - up.z;
    const double dist_up = r_up * r_up;
    dist -= dist_up;
  }

  return dist > 0.;

}


__global__ void collision_detection_gpu(
          size_t ncubes,
          size_t nspheres,
          size_t LD_bit,
    const double* low_points,
    const double* high_points,
    const double* centers,
    const double* radii,
         int32_t* collisions,
         int32_t* counts
) {
  const size_t nspheres_block = (nspheres + 31) / 32;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < ncubes; i += blockDim.x * gridDim.x) {
    counts[i] = 0;
    double3 low_point;
    double3 high_point;
    low_point.x = low_points[3*i+0];
    low_point.y = low_points[3*i+1];
    low_point.z = low_points[3*i+2];

    high_point.x = high_points[3*i+0];
    high_point.y = high_points[3*i+1];
    high_point.z = high_points[3*i+2];


    for (int j_block = 0; j_block < nspheres_block; j_block++) {
      int temp_collisions = 0;
      for (int j_inner = 0; j_inner < 32; j_inner++) {
        int j = j_block * 32 + j_inner;
        if (j < nspheres) {
          double3 center;
          double radius; 
          center.x = centers[3*j+0];
          center.y = centers[3*j+1];
          center.z = centers[3*j+2];

          radius = radii[j];
          temp_collisions |= (cube_sphere_intersect(low_point, high_point, center, radius) ? 1 << (j_inner) : 0);
        }
      }
      collisions[i * LD_bit + j_block] = temp_collisions;
      counts[i] += __popc(temp_collisions);
    }
  }
}


static constexpr int32_t buffer_size = 8;
static constexpr int32_t element_size = 32; // number of bits 
static constexpr int32_t buffer_size_bits = buffer_size * element_size;

// This kernel converts the bitvector produced by the collision detection kernel above into a position list.
// For simplicity, the collision detection kernel stores its output as a bitvector. However, the `shell_list`
// of the task is a list of the qualifying indexes, so we must convert the bitvector to a position list. 
//
// We take this chance to compute the nbe value from the shell sizes since the data is already being read in
__global__ void bitvector_to_position_list( 
           size_t  ncubes, 
           size_t  nspheres, 
           size_t  LD_bit,
    const int32_t* collisions, 
    const int32_t* counts, 
    const  size_t* shell_size,
          int32_t* position_list, 
           size_t* nbe_list
) {
  __shared__ int32_t collisions_buffer[max_warps_per_thread_block][warp_size][buffer_size];

  // We are converting a large number of small bitvectors into position lists. For this reason, I am assigning a single thread to each bitvector
  // This avoids having to do popcounts and warp wide reductions, but hurts the memory access pattern

  // All threads in a warp must be active to do shared memory loads, so we seperate out the threadId.x
  for (int i_base = threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y; i_base < ncubes; i_base += blockDim.x * blockDim.y * gridDim.x) {
    const int i = i_base + threadIdx.x;
    int32_t* out = position_list;
    if (i != 0 && i < ncubes) {
      out += counts[i-1];
    } 

    int current = 0;
    size_t nbe = 0;
    size_t nsphere_blocks = (nspheres + buffer_size_bits - 1) / buffer_size_bits;
    for (int j_block = 0; j_block < nsphere_blocks; j_block++) {
      // Each thread has a buffer of length BUFFER_SIZE. All the threads in the warp work to 
      // load this data in a coalesced way (at least as much as possible)
      for (int buffer_loop = 0; buffer_loop < warp_size; buffer_loop += warp_size/buffer_size) {
        const int t_id_x        = threadIdx.x % buffer_size;
        const int buffer_thread = threadIdx.x / buffer_size;
        const int buffer_idx    = buffer_thread + buffer_loop;
        if (j_block * buffer_size_bits + t_id_x * element_size < nspheres && i_base + buffer_idx < ncubes) {
          collisions_buffer[threadIdx.y][buffer_idx][t_id_x] = collisions[(i_base + buffer_idx) * LD_bit + j_block * buffer_size + t_id_x];
        }
      }

      if (i < ncubes) {  // Once the data has been loaded, we exclude the threads not corresponding to a bitvector
        // We have loaded in BUFFER_SIZE_BITS elements to be processed by each warp
        for (int j_inner = 0; j_inner < buffer_size_bits && j_block * buffer_size_bits + j_inner < nspheres; j_inner++) {
          const int j = buffer_size_bits * j_block + j_inner;
          const int j_int = j_inner / element_size;
          const int j_bit = j_inner % element_size;
          if( collisions_buffer[threadIdx.y][threadIdx.x][j_int] & (1 << (j_bit)) ) {
            out[current++] = j;
            nbe += shell_size[j];
          }
        }
      }
    }
    if (i < ncubes) {
      nbe_list[i] = nbe;
    }
  }
}

size_t compute_scratch( size_t ncubes, int32_t* counts_device ) {
    // Computes amount of memory that will be required to do the inclusive sum
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, counts_device, counts_device, ncubes);

    return temp_storage_bytes;
}

void collision_detection( size_t        ncubes,
                          size_t        nspheres,
                          size_t        LD_bit,
                          const double* low_points_device,
                          const double* high_points_device,
                          const double* centers_device,
                          const double* radii_device,
                                size_t  temp_storage_bytes,
                                 void * temp_storage_device,
                               int32_t* collisions_device, 
                               int32_t* counts_device,
                          hipStream_t  stream) {

    dim3 threads( max_threads_per_thread_block );
    dim3 blocks( util::div_ceil( ncubes, threads.x ) );

    collision_detection_gpu<<<blocks, threads, 0, stream>>>(
        ncubes, nspheres, LD_bit, 
        low_points_device, high_points_device, centers_device, radii_device, 
        collisions_device, counts_device
    );

    // Run inclusive prefix sum
    hipcub::DeviceScan::InclusiveSum(temp_storage_device, temp_storage_bytes, counts_device, counts_device, ncubes, stream);

}

void compute_position_list(size_t         ncubes,
                           size_t         nspheres,
                           size_t         LD_bit,
                           const size_t*  shell_sizes_device,
                           const int32_t* collisions_device,
                           const int32_t* counts_device,
                                 int32_t* position_list_device,
                                  size_t* nbe_list_device,
                            hipStream_t  stream) {
    dim3 threads( warp_size, max_warps_per_thread_block );
    dim3 blocks( util::div_ceil( ncubes, threads.x * threads.y ) );

    // convert from bitvector to position list
    bitvector_to_position_list<<<blocks, threads, 0, stream>>>(
        ncubes, nspheres, LD_bit, 
        collisions_device, counts_device, shell_sizes_device, 
        position_list_device, nbe_list_device
    );
}

}
}
}


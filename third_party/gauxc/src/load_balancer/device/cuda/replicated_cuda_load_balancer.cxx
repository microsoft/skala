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
#include "replicated_cuda_load_balancer.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/cuda_util.hpp"

#include "cuda_collision_detection.hpp"

using namespace GauXC::load_balancer::cuda;

namespace GauXC {
namespace detail {

template <typename T>
using pinned_vector = std::vector<T>;

// Helper data struction to keep inputs to collision detection kernels organized
struct CollisionDetectionCudaData {
    // Inputs
    double* low_points_device;
    double* high_points_device;
    double* centers_device;
    double* radii_device;
    size_t* shell_sizes_device;
    // Outputs
    int32_t position_list_length;
    int32_t* position_list_device;
    int32_t* counts_device;
    size_t*  nbe_list_device;
    // Intermediates
    int32_t* collisions_device;
    size_t temp_storage_bytes;
    void * temp_storage_device;
};


DeviceReplicatedLoadBalancer::DeviceReplicatedLoadBalancer( const DeviceReplicatedLoadBalancer& ) = default;
DeviceReplicatedLoadBalancer::DeviceReplicatedLoadBalancer( DeviceReplicatedLoadBalancer&& ) noexcept = default;

DeviceReplicatedLoadBalancer::~DeviceReplicatedLoadBalancer() noexcept = default;

std::unique_ptr<LoadBalancerImpl> DeviceReplicatedLoadBalancer::clone() const {
  return std::make_unique<DeviceReplicatedLoadBalancer>(*this);
}

std::vector<int32_t> inline copy_shell_list(
  const size_t idx,
  const std::vector<int32_t>& counts,
  const pinned_vector<int32_t> &position_list
) {
  int32_t start = 0;
  if ( idx != 0 ) start += counts[idx-1];
  int32_t end = counts[idx];

  std::vector<int32_t> shell_list(end - start);
  std::copy(position_list.begin() + start, position_list.begin() + end, shell_list.begin());

  return shell_list;
}


std::vector< XCTask > DeviceReplicatedLoadBalancer::create_local_tasks_() const  {

  const int32_t n_deriv = 1;
  const size_t atBatchSz = 256;

  int32_t world_rank = runtime_.comm_rank();
  int32_t world_size = runtime_.comm_size();

  std::vector< XCTask > local_work;
  std::vector<size_t> global_workload( world_size, 0 );   

  const auto natoms           = this->mol_->natoms();
  const size_t nspheres       = (*this->basis_).size();
  const size_t num_atom_batch = util::div_ceil(natoms, atBatchSz);
  const size_t max_nbatches   = mg_->max_nbatches() * atBatchSz;
  const size_t LD_bit         = util::div_ceil(nspheres, 32);

  CollisionDetectionCudaData data;
  cudaStream_t master_stream = 0;

  std::vector< XCTask > temp_tasks;              temp_tasks.reserve( max_nbatches );
  std::vector<std::array<double,3>> low_points;  low_points.reserve( max_nbatches );
  std::vector<std::array<double,3>> high_points; high_points.reserve( max_nbatches );
  std::vector<std::array<double,3>> centers;     centers.reserve(nspheres);
  std::vector<double> radii;                     radii.reserve(nspheres);
  std::vector<size_t> shell_sizes;               shell_sizes.reserve(nspheres);
  // These two vectors are populated by cuda memcopies on their data pointer
  // So maybe we should be resizing them instead of just adding capacity?
  std::vector<int32_t> pos_list_idx;             pos_list_idx.reserve(max_nbatches);
  std::vector<size_t> nbe_vec;                   nbe_vec.reserve(max_nbatches);

  // The postion list is the largest struction so I am using pinned memory for the improved bandwidth
  pinned_vector<int32_t> position_list;
  
  data.temp_storage_bytes = compute_scratch(max_nbatches, data.counts_device);
  data.temp_storage_device = util::cuda_malloc<char>(data.temp_storage_bytes); // char is 1 byte

  data.low_points_device   = util::cuda_malloc<double>(max_nbatches * 3);
  data.high_points_device  = util::cuda_malloc<double>(max_nbatches * 3);
  data.collisions_device   = util::cuda_malloc<int32_t>(LD_bit * max_nbatches);
  data.nbe_list_device     = util::cuda_malloc<size_t>(max_nbatches);
  data.counts_device       = util::cuda_malloc<int32_t>(max_nbatches);

  data.centers_device      = util::cuda_malloc<double>(nspheres * 3);
  data.radii_device        = util::cuda_malloc<double>(nspheres);
  data.shell_sizes_device  = util::cuda_malloc<size_t>(nspheres);

  for(auto& shell : (*this->basis_)) {
    centers.push_back(shell.O());
    radii.push_back(shell.cutoff_radius());
    shell_sizes.push_back(shell.size());
  }

  util::cuda_copy(nspheres * 3, data.centers_device, centers[0].data(), "Centers HtoD");
  util::cuda_copy(nspheres, data.radii_device, radii.data(), "Radii HtoD");
  util::cuda_copy(nspheres, data.shell_sizes_device, shell_sizes.data(), "ShellSize HtoD");

  // For batching of multiple atom screening
  for (size_t atom_batch = 0; atom_batch < num_atom_batch; ++atom_batch) {
    //---------------------------------------------------------------------
    // production step 
    int32_t iCurrent  = atom_batch * atBatchSz;
    for ( size_t atom_idx = 0; atom_idx < atBatchSz && atom_batch * atBatchSz + atom_idx < natoms; ++atom_idx ) {

      const auto atom = (*this->mol_)[atom_batch * atBatchSz + atom_idx];
      const std::array<double,3> center = { atom.x, atom.y, atom.z };

      auto& batcher = this->mg_->get_grid(atom.Z).batcher();
      batcher.quadrature().recenter( center );
      const size_t nbatches = batcher.nbatches();

      for( size_t ibatch = 0; ibatch < nbatches; ++ibatch ) {
        // Generate the batch (non-negligible cost)
        auto [ npts, pts_b, pts_en, w_b, w_en ] = (batcher.begin() + ibatch).range();
        auto [lo, up] = IntegratorXX::detail::get_box_bounds_points(pts_b, pts_en);

        if( npts == 0 ) continue;

        // Partially copy task data
        XCTask task;
        task.iParent      = iCurrent;
        task.npts         = npts;
        task.dist_nearest = this->molmeta_->dist_nearest()[iCurrent];
        temp_tasks.push_back( std::move( task ) );
        low_points.push_back( std::move( lo ) );
        high_points.push_back( std::move( up ) );
      }
      iCurrent++;
    }

    //---------------------------------------------------------------------
    // Device collision detection step  
    const size_t ncubes = low_points.size();
    util::cuda_copy(ncubes * 3, data.low_points_device, low_points[0].data(), "Low points HtoD");
    util::cuda_copy(ncubes * 3, data.high_points_device, high_points[0].data(), "High points HtoD");

    collision_detection(
      ncubes, nspheres, LD_bit,
      data.low_points_device, data.high_points_device,
      data.centers_device, data.radii_device, 
      data.temp_storage_bytes, data.temp_storage_device,
      data.collisions_device, data.counts_device,
      master_stream
    );

    // Copy total number of collisions back to host to allocate result array
    int32_t total_collisions;
    util::cuda_copy(1, &total_collisions, data.counts_device + ncubes - 1, "Total collisions DtoH");
    data.position_list_device = util::cuda_malloc<int32_t>(total_collisions);

    compute_position_list(
      ncubes, nspheres, LD_bit,
      data.shell_sizes_device,
      data.collisions_device,
      data.counts_device,
      data.position_list_device,
      data.nbe_list_device,
      master_stream
    );

    position_list.reserve(total_collisions);

    util::cuda_device_sync();
    // Copy results back to host
    util::cuda_copy(total_collisions, position_list.data(), data.position_list_device, "Position List DtoH");
    util::cuda_copy(ncubes, pos_list_idx.data(), data.counts_device, "Position List Idx DtoH");
    util::cuda_copy(ncubes, nbe_vec.data(), data.nbe_list_device, "NBE counts DtoH");
    util::cuda_free(data.position_list_device);

    low_points.clear();
    high_points.clear();

    //---------------------------------------------------------------------
    // Assign batches to MPI ranks
    size_t idx = 0;
    for ( size_t atom_idx = 0; atom_idx < atBatchSz && atom_batch * atBatchSz + atom_idx < natoms; ++atom_idx ) {

      const auto atom = (*this->mol_)[atom_batch * atBatchSz + atom_idx];
      const std::array<double,3> center = { atom.x, atom.y, atom.z };

      auto& batcher = this->mg_->get_grid(atom.Z).batcher();
      batcher.quadrature().recenter( center );
      const size_t nbatches = batcher.nbatches();

      for( size_t ibatch = 0; ibatch < nbatches; ++ibatch ) {
        auto [ npts, pts_b, pts_en, w_b, w_en ] = (batcher.begin() + ibatch).range();
        XCTask task = std::move( temp_tasks.at( idx ) );
        task.bfn_screening.nbe  = nbe_vec[idx];

        // Update npts
        task.npts = npts;

        // Get rank with minimum work
        auto min_rank_it = 
          std::min_element( global_workload.begin(), global_workload.end() );
        int64_t min_rank = std::distance( global_workload.begin(), min_rank_it );

        global_workload[ min_rank ] += task.cost( n_deriv, natoms );

        if( world_rank == min_rank ) {
          auto shell_list = std::move( copy_shell_list(idx, pos_list_idx, position_list) );
          // Course grain screening
          if( shell_list.size() ) {
            task.bfn_screening.shell_list = shell_list;

            // Get local copy of points weights
            std::vector<std::array<double,3>> points(pts_b, pts_en);
            std::vector<double>               weights(w_b, w_en);

            task.points  = std::move(points);
            task.weights = std::move(weights);
            local_work.push_back( std::move(task) );
          }
        }
        idx++;
      }
    }
    temp_tasks.clear();

  }
  
  // Lexicographic ordering of tasks
  auto task_order = []( const auto& a, const auto& b ) {

    // Sort by iParent first
    if( a.iParent < b.iParent )      return true;
    else if( a.iParent > b.iParent ) return false;

    // Equal iParent: lex sort on shell list
    else return a.bfn_screening.shell_list < b.bfn_screening.shell_list;

  };

  std::sort( local_work.begin(), local_work.end(),
    task_order ); 


  // Get unique tasks
  auto task_equiv = []( const auto& a, const auto& b ) {
    return a.equiv_with(b);
  };

  auto local_work_unique = local_work;
  auto last_unique = 
    std::unique( local_work_unique.begin(),
                 local_work_unique.end(),
                 task_equiv );
  local_work_unique.erase( last_unique, local_work_unique.end() );
  

  // Merge tasks
  for( auto&& t : local_work_unique ) {
    t.points.clear();
    t.weights.clear();
  }

  auto cur_lw_begin = local_work.begin();
  auto cur_uniq_it  = local_work_unique.begin();

  for( auto lw_it = local_work.begin(); lw_it != local_work.end(); ++lw_it ) 
  if( not task_equiv( *lw_it, *cur_uniq_it ) ) {

    if( cur_uniq_it == local_work_unique.end() )
      GAUXC_GENERIC_EXCEPTION("Messed up in unique");

    cur_uniq_it->merge_with( cur_lw_begin, lw_it );

    cur_lw_begin = lw_it;
    cur_uniq_it++;

  }

  // Merge the last set of batches
  for( ; cur_lw_begin != local_work.end(); ++cur_lw_begin )
    cur_uniq_it->merge_with( *cur_lw_begin );
  cur_uniq_it++;
  

  local_work = std::move(local_work_unique);
  
  // Free all device memory
  util::cuda_free(data.low_points_device);
  util::cuda_free(data.high_points_device);
  util::cuda_free(data.centers_device);
  util::cuda_free(data.radii_device);
  util::cuda_free(data.shell_sizes_device);
  util::cuda_free(data.collisions_device);
  util::cuda_free(data.nbe_list_device);
  util::cuda_free(data.counts_device);
  util::cuda_free(data.temp_storage_device);

  return local_work;
}







}
}

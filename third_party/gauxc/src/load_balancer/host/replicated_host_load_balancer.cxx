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
#include "replicated_host_load_balancer.hpp"

namespace GauXC {
namespace detail {

HostReplicatedLoadBalancer::HostReplicatedLoadBalancer( const HostReplicatedLoadBalancer& ) = default;
HostReplicatedLoadBalancer::HostReplicatedLoadBalancer( HostReplicatedLoadBalancer&& ) noexcept = default;

HostReplicatedLoadBalancer::~HostReplicatedLoadBalancer() noexcept = default;

std::vector< XCTask > HostReplicatedLoadBalancer::create_local_tasks_() const  {

  const int32_t n_deriv = 1; // Effects cost heuristic

  int32_t world_rank = runtime_.comm_rank();
  int32_t world_size = runtime_.comm_size();

  std::vector< XCTask > local_work;
  std::vector<size_t> global_workload( world_size, 0 );   

  const auto natoms = this->mol_->natoms();
  int32_t iCurrent  = 0;
  int32_t atBatchSz = 1;

  const size_t max_nbatches = mg_->max_nbatches();
  std::vector< std::pair<size_t, XCTask> > temp_tasks;
  temp_tasks.reserve( max_nbatches );

  // For batching of multiple atom screening
  size_t batch_idx_offset = 0;

  // Loop over Atoms
  for( const auto& atom : *this->mol_ ) {

    const std::array<double,3> center = { atom.x, atom.y, atom.z };

    auto& batcher = mg_->get_grid(atom.Z).batcher();
    batcher.quadrature().recenter( center );
    const size_t nbatches = batcher.nbatches();

    #pragma omp parallel for
    for( size_t ibatch = 0; ibatch < nbatches; ++ibatch ) {
    
      size_t batch_idx = ibatch + batch_idx_offset;

      // Generate the batch (non-negligible cost)
      auto [lo, up, points, weights] = batcher.at(ibatch);

      if( points.size() == 0 ) continue;

      // Microbatch Screening
      auto [shell_list, nbe] = micro_batch_screen( (*this->basis_), lo, up );

      // Course grain screening
      if( not shell_list.size() ) continue; 

      // Copy task data
      XCTask task;
      task.iParent    = iCurrent;
      // This enables lazy assignment of points vector (see CUDA impl)
      task.npts       = points.size(); 
      task.points     = std::move( points );
      task.weights    = std::move( weights );
      task.bfn_screening.shell_list = std::move(shell_list);
      task.bfn_screening.nbe        = nbe;
      task.dist_nearest = molmeta_->dist_nearest()[iCurrent];

      #pragma omp critical
      temp_tasks.push_back( 
        std::pair(batch_idx,std::move( task )) 
      );

    } // omp parallel for over batches





    // Assign Tasks to MPI ranks
    if( (iCurrent+1) % atBatchSz == 0 or iCurrent == ((int32_t)natoms-1) ) {

      // Sort based on task index for deterministic assignment
      std::sort( temp_tasks.begin(), temp_tasks.end(), 
        []( const auto& a, const auto& b ) {
          return a.first < b.first;
        } );

      // Assign batches to MPI ranks
      for( size_t ibatch = 0; ibatch < temp_tasks.size(); ++ibatch ) {

        XCTask task = std::move(temp_tasks.at(ibatch).second);
        //auto& points = task.points;
        //auto  nbe    = task.nbe;

        // Get rank with minimum work
        auto min_rank_it = 
          std::min_element( global_workload.begin(), global_workload.end() );
        int64_t min_rank = std::distance( global_workload.begin(), min_rank_it );

        // Compute cost heuristic and increment total work
        global_workload[ min_rank ] += task.cost( n_deriv, natoms );

        if( world_rank == min_rank ) 
          local_work.push_back( std::move(task) );

      }

      temp_tasks.clear();

    }


    // Update counters and offsets
    iCurrent++;
    batch_idx_offset += nbatches;

  } // Loop over Atoms

// return local_work;

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
    t.npts = 0;
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

  return local_work;
}







}
}

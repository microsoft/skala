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
#include "load_balancer_impl.hpp"
#include <gauxc/util/mpi.hpp>
#include <gauxc/util/div_ceil.hpp>
#include <fstream>

namespace GauXC::detail {

#ifdef GAUXC_HAS_MPI
template <typename TaskIterator, typename CostFunctor>
auto rebalance(TaskIterator begin, TaskIterator end, const CostFunctor& cost, MPI_Comm comm) {

  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  int world_rank, world_size;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  MPI_Barrier(MPI_COMM_WORLD);
  printf(
  "RANK %d BEFORE REBALNACE: LW = %lu\n",
    world_rank,
    std::accumulate(begin, end, 0ul,
      [=](const auto& a, const auto& b){ return a + cost(b); })
  );

  // Compute local task costs
  size_t ntask_local = std::distance(begin, end);
  std::vector<size_t> local_task_cost(ntask_local);
  std::transform(begin, end, local_task_cost.begin(),
    [&](const auto& task){ return cost(task); });

  // Compute task prefix sum
  auto prefix_st = hrt_t::now();
  std::vector<size_t> local_prefix_sum(ntask_local);
  auto [local_task_sum, prefix_seed] =
    mpi_prefix_sum(local_task_cost.begin(), local_task_cost.end(),
      local_prefix_sum.begin(), comm);

  // Compute total/avg cost
  auto total_task_sum = allreduce( local_task_sum, MPI_SUM, comm );
  size_t task_avg = util::div_ceil(total_task_sum, world_size);


  // Generate outgoing messages
  struct task_message {
    int dst;
    size_t idx_st, idx_en, vol;
  };

  std::vector<task_message> task_outgoing;
  auto it = local_prefix_sum.begin();
  for( int i = 0; i < world_size; ++i) {
    auto n_it = std::lower_bound(it, local_prefix_sum.end(), (size_t)i,
      [=](auto a, auto b) { return a / task_avg < b+1; });
    size_t st_idx = std::distance(local_prefix_sum.begin(), it  );
    size_t en_idx = std::distance(local_prefix_sum.begin(), n_it);
    if(st_idx != en_idx and i != world_rank) {
      size_t vol = 0;
      for( size_t t = st_idx; t < en_idx; ++t ) {
        vol += (begin + t)->volume();
      }
      task_outgoing.push_back({i,st_idx,en_idx,vol});
      size_t work_to_send = 0;
      for( size_t t = st_idx; t < en_idx; ++t ) {
        work_to_send += cost(*(begin + t));
      }
      printf("RANK %d SENDING %lu to RANK %d\n", world_rank, work_to_send, i);
    }
    it = n_it;
  }

  auto prefix_en = hrt_t::now();

  // Sanity check
  if(task_outgoing.size() > 2 )
    GAUXC_GENERIC_EXCEPTION("Incorrect Outgoing Task Message Size RANK = " + std::to_string(world_rank) + " SZ = " + std::to_string(task_outgoing.size()));

  MPI_Barrier(comm);
  ring_execute(
  [&]() {
    printf("RANK %d MESSAGES:\n", world_rank);
    for(auto& msg : task_outgoing) {
      printf("  DST %d ST %lu EN %lu V %lu\n",
        msg.dst, msg.idx_st, msg.idx_en, msg.vol);
    }
  } , comm);
  MPI_Barrier(comm);

  std::vector<MPI_Request> packed_req; packed_req.reserve(8);

  // Ask neighbors if they're sending messages
  int recv_from_backward = 0;
  int recv_from_forward = 0;
  int send_forward = 0;
  int send_backward = 0;
  if(world_rank) {
    auto& req = packed_req.emplace_back();
    MPI_Irecv( &recv_from_backward, 1, MPI_INT, world_rank-1, 0, comm, &req );
  }
  if(world_rank < world_size-1) {
    auto& req = packed_req.emplace_back();
    MPI_Irecv( &recv_from_forward, 1, MPI_INT, world_rank+1, 1, comm, &req );
  }
  if(world_rank < world_size-1) {
    auto& req = packed_req.emplace_back();
    for(auto& msg : task_outgoing) send_forward |= (msg.dst == world_rank+1);
    MPI_Isend( &send_forward, 1, MPI_INT, world_rank+1, 0, comm, &req );
  }
  if(world_rank) {
    auto& req = packed_req.emplace_back();
    for(auto& msg : task_outgoing) send_backward |= (msg.dst == world_rank-1);
    MPI_Isend( &send_backward, 1, MPI_INT, world_rank-1, 1, comm, &req );
  }



  // Wait for messages to complete
  if(packed_req.size()) {
    MPI_Waitall(packed_req.size(), packed_req.data(), MPI_STATUS_IGNORE);
  }
  // Reset messages
  packed_req.clear();

  printf("RANK %d, BW %d, FW %d\n", world_rank, recv_from_backward, recv_from_forward );
  MPI_Barrier(MPI_COMM_WORLD);


#if 0
  // Sanity check
  if( world_rank < world_size - 1 and task_outgoing.size() != 1 )
    GAUXC_GENERIC_EXCEPTION("Incorrect Outgoing Task Message Size RANK = " + std::to_string(world_rank));
  if( world_rank == world_size - 1 and task_outgoing.size() )
    GAUXC_GENERIC_EXCEPTION("Incorrect Outgoing Task Message Size RANK = " + std::to_string(world_rank));
#endif

  // Allocate incoming and outgoing buffers = 64 MiB
  constexpr size_t packed_buffer_size = 64 * 1024 * 1024;
  MPI_Packed_Buffer packed_outgoing_forward( packed_buffer_size, comm );
  MPI_Packed_Buffer packed_incoming_forward( packed_buffer_size, comm );
  MPI_Packed_Buffer packed_outgoing_backward( packed_buffer_size, comm );
  MPI_Packed_Buffer packed_incoming_backward( packed_buffer_size, comm );

  // Post receives
  if(recv_from_backward) {
    auto& packed_recv_req = packed_req.emplace_back();
    MPI_Irecv( packed_incoming_backward.buffer(), packed_buffer_size, MPI_PACKED,
      world_rank-1, 0, comm, &packed_recv_req );
  }
  if(recv_from_forward) {
    auto& packed_recv_req = packed_req.emplace_back();
    MPI_Irecv( packed_incoming_forward.buffer(), packed_buffer_size, MPI_PACKED,
      world_rank+1, 0, comm, &packed_recv_req );
  }

  auto pack_st = hrt_t::now();
  {

  //std::ofstream ofile("task_send." + std::to_string(world_rank) + ".txt");

  auto pack_msg = [&](const auto& msg, auto& mpi_buffer) {
    size_t ntask_send = msg.idx_en - msg.idx_st;
    //ofile << "DEST " << msg.dst << std::endl;
    mpi_buffer.pack(ntask_send); //ofile << ntask_send << std::endl;
    for(size_t i = msg.idx_st; i < msg.idx_en; ++i) {
      const auto& task = *(begin + i);
      mpi_buffer.pack(task.iParent);
      mpi_buffer.pack(task.npts);
      mpi_buffer.pack(task.points);
      mpi_buffer.pack(task.weights);
      mpi_buffer.pack(task.bfn_screening.shell_list);
      mpi_buffer.pack(task.bfn_screening.nbe);
      mpi_buffer.pack(task.cou_screening.shell_list);
      mpi_buffer.pack(task.cou_screening.shell_pair_list);
      mpi_buffer.pack(task.cou_screening.shell_pair_idx_list);
      mpi_buffer.pack(task.cou_screening.nbe);
      mpi_buffer.pack(task.dist_nearest);
      //ofile << task.iParent << ", " << task.npts << ", " << task.dist_nearest <<
      //  ", " << task.bfn_screening.nbe << ", " << task.points.size() <<
      //  ", " << task.cou_screening.nbe << ", " << task.cou_screening.shell_list.size() <<
      //  ", " << task.cou_screening.shell_pair_list.size() << ", " << task.cou_screening.shell_pair_idx_list.size()
      //  << std::endl;
    }

    // Send data to neighbor
    auto& packed_send_req = packed_req.emplace_back();
    auto* buffer = mpi_buffer.buffer();
    MPI_Isend(buffer, packed_buffer_size, MPI_PACKED, msg.dst, 0,
      comm, &packed_send_req);
  };

  // Pack and send to neighbor
  #if 0
  if(send_backward or send_forward) {
    // Pack Data
    auto& msg = task_outgoing[0];
    if( msg.dst != world_rank+1 )
      GAUXC_GENERIC_EXCEPTION("Invalid Destination Rank");

    size_t ntask_send = msg.idx_en - msg.idx_st;

    //std::ofstream ofile("task_send." + std::to_string(world_rank) + ".txt");

    packed_outgoing.pack(ntask_send); //ofile << ntask_send << std::endl;
    //printf("RANK = %d SEND = %lu\n", world_rank, ntask_send);
    for(size_t i = msg.idx_st; i < msg.idx_en; ++i) {
      const auto& task = *(begin + i);
      packed_outgoing.pack(task.iParent);
      packed_outgoing.pack(task.npts);
      packed_outgoing.pack(task.points);
      packed_outgoing.pack(task.weights);
      packed_outgoing.pack(task.bfn_screening.shell_list);
      packed_outgoing.pack(task.bfn_screening.nbe);
      packed_outgoing.pack(task.cou_screening.shell_list);
      packed_outgoing.pack(task.cou_screening.shell_pair_list);
      packed_outgoing.pack(task.cou_screening.shell_pair_idx_list);
      packed_outgoing.pack(task.cou_screening.nbe);
      packed_outgoing.pack(task.dist_nearest);
      //ofile << task.iParent << ", " << task.npts << ", " << task.dist_nearest <<
      //  ", " << task.bfn_screening.nbe << ", " << task.points.size() << std::endl;
    }

    // Send data to neighbor
    auto& packed_send_req = packed_req.emplace_back();
    auto* buffer = packed_outgoing.buffer();
    MPI_Isend(buffer, packed_buffer_size, MPI_PACKED, world_rank+1, 0,
      comm, &packed_send_req);

  }
  #else
  for( auto& msg : task_outgoing ) {
    if(msg.dst == world_rank + 1)
      pack_msg(msg, packed_outgoing_forward);
    else
      pack_msg(msg, packed_outgoing_backward);
  }
  #endif
  }

  auto pack_en = hrt_t::now();

  // Local task storage
  std::vector< XCTask > local_work;

  // Move local tasks to task storage
  {
    auto func = [=](auto a, auto b) { return a / task_avg < b+1; };
    auto lps_begin = local_prefix_sum.begin();
    auto lps_end   = local_prefix_sum.end();
    auto local_begin =
      world_rank ? std::lower_bound(lps_begin,lps_end,(size_t)(world_rank-1),func) : lps_begin;
    auto local_end = std::lower_bound(local_begin, lps_end, (size_t)world_rank,func);
    size_t st_idx = std::distance(lps_begin, local_begin);
    size_t en_idx = std::distance(lps_begin, local_end  );

    size_t local_cost = 0;
    for(size_t i = st_idx; i < en_idx; ++i) {
      local_work.emplace_back( std::move(*(begin + i)) );
      local_cost += cost(local_work.back());
    }
    printf("RANK %d LC %lu\n", world_rank, local_cost);
  }


  // Wait on sends and receives
  auto wait_st = hrt_t::now();
  if(packed_req.size()) {
    MPI_Waitall(packed_req.size(), packed_req.data(), MPI_STATUS_IGNORE);
  }
  auto wait_en = hrt_t::now();


  auto unpack_st  = hrt_t::now();

  {

  //std::ofstream ofile("task_recv." + std::to_string(world_rank) + ".txt");
  auto unpack_msg = [&](auto& mpi_buffer) {
    size_t ntask_recv = 0;
    //if(&mpi_buffer == &packed_incoming_backward) ofile << "SRC " << world_rank - 1 << std::endl;
    //else ofile << "SRC " << world_rank + 1 << std::endl;

    mpi_buffer.unpack(ntask_recv); //ofile << ntask_recv << std::endl;
    for(size_t i = 0; i < ntask_recv; ++i) {
      auto& task = local_work.emplace_back();
      mpi_buffer.unpack(task.iParent);
      mpi_buffer.unpack(task.npts);
      mpi_buffer.unpack(task.points);
      mpi_buffer.unpack(task.weights);
      mpi_buffer.unpack(task.bfn_screening.shell_list);
      mpi_buffer.unpack(task.bfn_screening.nbe);
      mpi_buffer.unpack(task.cou_screening.shell_list);
      mpi_buffer.unpack(task.cou_screening.shell_pair_list);
      mpi_buffer.unpack(task.cou_screening.shell_pair_idx_list);
      mpi_buffer.unpack(task.cou_screening.nbe);
      mpi_buffer.unpack(task.dist_nearest);
      //ofile << task.iParent << ", " << task.npts << ", " << task.dist_nearest <<
      //  ", " << task.bfn_screening.nbe << ", " << task.points.size() <<
      //  ", " << task.cou_screening.nbe << ", " << task.cou_screening.shell_list.size() <<
      //  ", " << task.cou_screening.shell_pair_list.size() << ", " << task.cou_screening.shell_pair_idx_list.size()
      //  << std::endl;
    }
  };
  #if 0
  // Unpack
  if(world_rank) {
    size_t ntask_recv = 0;
    packed_incoming.unpack(ntask_recv); 
    for(size_t i = 0; i < ntask_recv; ++i) {
      auto& task = local_work.emplace_back();
      packed_incoming.unpack(task.iParent);
      packed_incoming.unpack(task.npts);
      packed_incoming.unpack(task.points);
      packed_incoming.unpack(task.weights);
      packed_incoming.unpack(task.bfn_screening.shell_list);
      packed_incoming.unpack(task.bfn_screening.nbe);
      packed_incoming.unpack(task.cou_screening.shell_list);
      packed_incoming.unpack(task.cou_screening.shell_pair_list);
      packed_incoming.unpack(task.cou_screening.shell_pair_idx_list);
      packed_incoming.unpack(task.cou_screening.nbe);
      packed_incoming.unpack(task.dist_nearest);
      //ofile << task.iParent << ", " << task.npts << ", " << task.dist_nearest <<
      //  ", " << task.bfn_screening.nbe << ", " << task.points.size() << std::endl;
    }

  }
  #else
  if(recv_from_backward) unpack_msg(packed_incoming_backward);
  if(recv_from_forward ) unpack_msg(packed_incoming_forward );
  #endif
  }
  auto unpack_en  = hrt_t::now();


  MPI_Barrier(MPI_COMM_WORLD);
  printf(
  "RANK %d AFTER REBALNACE: PREFIX_DUR = %f PACK_DUR = %f WAIT_DUR = %f UNPACK_DUR = %f LW = %lu\n",
    world_rank,
    dur_t(prefix_en-prefix_st).count(),
    dur_t(pack_en-pack_st).count(),
    dur_t(wait_en-wait_st).count(),
    dur_t(unpack_en-unpack_st).count(),
    //local_work.size()
    std::accumulate(local_work.begin(),local_work.end(),0ul,
      [=](const auto& a, const auto& b){ return a + cost(b); })
  );

  return local_work;

}
#endif


void LoadBalancerImpl::rebalance_weights() {
#ifdef GAUXC_HAS_MPI
  auto& tasks = get_tasks();
  const size_t natoms = molecule().natoms();
  auto cost = [=](const auto& task){ return task.cost(1,natoms); };
  auto new_tasks = rebalance( tasks.begin(), tasks.end(), cost, runtime_.comm());
  tasks = std::move(new_tasks);
#endif
}

void LoadBalancerImpl::rebalance_exc_vxc() {
#ifdef GAUXC_HAS_MPI
  auto& tasks = get_tasks();
  auto cost = [=](const auto& task){ return task.cost_exc_vxc(1); };
  auto new_tasks = rebalance( tasks.begin(), tasks.end(), cost, runtime_.comm());
  tasks = std::move(new_tasks);
#endif
}

void LoadBalancerImpl::rebalance_exx() {
#ifdef GAUXC_HAS_MPI
  auto& tasks = get_tasks();
  auto cost = [=](const auto& task){ return task.cost_exx(); };
  auto new_tasks = rebalance( tasks.begin(), tasks.end(), cost, runtime_.comm());
  local_tasks_ = std::move(new_tasks);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

}

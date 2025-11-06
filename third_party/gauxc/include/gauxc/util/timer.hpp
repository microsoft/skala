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
#include <chrono>
#include <map>
#include <string>
#include <type_traits>

#include <gauxc/gauxc_config.hpp>
#ifdef GAUXC_HAS_MPI
#include <mpi.h>
#endif

//#define GAUXC_DISABLE_TIMINGS

namespace GauXC {
namespace util  {

namespace detail {
  // TODO: write type_traits for GauXC
 
  template <class F, class... Args>
  struct has_void_return_type {

    // TODO: Write C++20 friendly version with removal of std::result_of
    static constexpr bool value = 
      std::is_same< std::invoke_result_t<F,Args...>, void >::value;

  };
  
}

class Timer {

  template <class Rep, class Period>
  using duration = std::chrono::duration<Rep,Period>;

  std::map< std::string, duration<double, std::milli>> timings_;

public:


  Timer()                               = default;
  Timer( const Timer& )                 = default;
  Timer( Timer&& ) noexcept             = default;
  Timer& operator=( const Timer& )      = default;
  Timer& operator=( Timer&& ) noexcept  = default;

  template <class OtherRep, class OtherPeriod>
  void add_timing( std::string name, duration<OtherRep,OtherPeriod> dur ) {
    timings_.insert_or_assign( name, duration<double,std::milli>(dur) );
  }

  template <class OtherRep, class OtherPeriod>
  void add_or_accumulate_timing( std::string name, 
                                 duration<OtherRep,OtherPeriod> dur ) {

    if( timings_.find( name ) != timings_.end() ) {
      timings_.at( name ) += dur;
    } else {
      add_timing( name, dur );
    }

  }

  template <typename Op>
  inline 
  std::enable_if_t< detail::has_void_return_type<Op>::value > 
    time_op( std::string name, const Op& op ) {

#ifndef GAUXC_DISABLE_TIMINGS
    auto st = std::chrono::high_resolution_clock::now();
    op();
    auto en = std::chrono::high_resolution_clock::now();

    duration< double, std::milli > dur( en - st );
    add_timing( name, dur );
#else
    op();
#endif

  }

  template <typename Op>
  inline 
  std::enable_if_t< not detail::has_void_return_type<Op>::value, 
                    std::invoke_result_t<Op>
                  > time_op( std::string name, const Op& op ) {
#ifndef GAUXC_DISABLE_TIMINGS
    auto st = std::chrono::high_resolution_clock::now();
    auto res = op();
    auto en = std::chrono::high_resolution_clock::now();

    duration< double, std::milli > dur( en - st );
    add_timing( name, dur );

    return res;
#else
    return op();
#endif
  }






  template <typename Op>
  inline 
  std::enable_if_t< detail::has_void_return_type<Op>::value > 
    time_op_accumulate( std::string name, const Op& op ) {

#ifndef GAUXC_DISABLE_TIMINGS
    auto st = std::chrono::high_resolution_clock::now();
    op();
    auto en = std::chrono::high_resolution_clock::now();

    duration< double, std::milli > dur( en - st );
    add_or_accumulate_timing( name, dur );
#else
    op();
#endif

  }

  template <typename Op>
  inline 
  std::enable_if_t< not detail::has_void_return_type<Op>::value, 
                    std::invoke_result_t<Op>
                  > time_op_accumulate( std::string name, const Op& op ) {

#ifndef GAUXC_DISABLE_TIMINGS
    auto st = std::chrono::high_resolution_clock::now();
    auto res = op();
    auto en = std::chrono::high_resolution_clock::now();

    duration< double, std::milli > dur( en - st );
    add_or_accumulate_timing( name, dur );

    return res;
#else
    return op();
#endif
  }





  template <class Rep = double, class Period = std::milli>
  inline duration<Rep,Period> get_duration( std::string name ) {
    return timings_.at(name);
  }

  inline const auto& all_timings() const { return timings_; }

};


#ifdef GAUXC_HAS_MPI
class MPITimer {

  template <class Rep, class Period>
  using duration = std::chrono::duration<Rep,Period>;

  Timer    rank_timer_;
  MPI_Comm comm_;
  std::map<std::string, duration<double,std::milli>> avg_timings_;
  std::map<std::string, duration<double,std::milli>> min_timings_;
  std::map<std::string, duration<double,std::milli>> max_timings_;
  std::map<std::string, duration<double,std::milli>> std_dev_timings_;

public:

  MPITimer() = delete;
  inline MPITimer( MPI_Comm comm, const Timer& timer ) : 
    rank_timer_(timer), comm_(comm) { get_stats(); }

  inline void get_stats( std::string key ) {
    int world_size; MPI_Comm_size( comm_, &world_size );
    double dur = rank_timer_.get_duration<double,std::nano>( key ).count();

    std::vector<double> durs_mpi( world_size );
    MPI_Allgather( &dur, 1, MPI_DOUBLE, durs_mpi.data(), 1, MPI_DOUBLE, comm_ );

#if 1
    double min_dur = *std::min_element( durs_mpi.begin(), durs_mpi.end() );
    double max_dur = *std::max_element( durs_mpi.begin(), durs_mpi.end() );
    double avg_dur = std::accumulate( durs_mpi.begin(), durs_mpi.end(), 0.0 );
    avg_dur = avg_dur / world_size;

    double std_dev = std::accumulate( durs_mpi.begin(), durs_mpi.end(), 0.0,
      [=]( auto a, auto b ) { 
        const auto diff = ( b - avg_dur );
        return a + diff*diff;
      });
    std_dev = std::sqrt( std_dev / world_size );
#else
    double min_dur, max_dur, avg_dur, std_dev;
#endif

    avg_timings_[ key ]     = std::chrono::nanoseconds( (size_t)std::ceil(avg_dur) );
    min_timings_[ key ]     = std::chrono::nanoseconds( (size_t)std::ceil(min_dur) );
    max_timings_[ key ]     = std::chrono::nanoseconds( (size_t)std::ceil(max_dur) );
    std_dev_timings_[ key ] = std::chrono::nanoseconds( (size_t)std::ceil(std_dev) );
  }

  inline void get_stats() {
    for( auto& [key, val] : rank_timer_.all_timings() ) get_stats(key);
  }


  template <class Rep = double, class Period = std::milli>
  inline duration<Rep,Period> get_avg_duration( std::string name ) {
    return avg_timings_.at(name);
  }

  template <class Rep = double, class Period = std::milli>
  inline duration<Rep,Period> get_min_duration( std::string name ) {
    return min_timings_.at(name);
  }

  template <class Rep = double, class Period = std::milli>
  inline duration<Rep,Period> get_max_duration( std::string name ) {
    return max_timings_.at(name);
  }

  template <class Rep = double, class Period = std::milli>
  inline duration<Rep,Period> get_std_dev( std::string name ) {
    return std_dev_timings_.at(name);
  }

};
#endif

}
}

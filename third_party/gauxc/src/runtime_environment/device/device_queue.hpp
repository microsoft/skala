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
#include <any>
#include <memory>
#include <type_traits>

namespace GauXC {

class device_queue {

  std::any queue_;

public:

  device_queue() = default;

  template <typename T>
  device_queue( std::shared_ptr<T> q ) : queue_( std::move(q) ) { }

  template <typename T, typename... Args>
  static device_queue generate( Args&&... args ) {
    return device_queue( std::make_shared<T>( std::forward<Args>(args)... ) );
  }

  template <typename T>
  inline const T* queue_as_ptr() const { 
    if( !queue_.has_value() ) return nullptr;
    if( auto q_ptr = std::any_cast< std::shared_ptr<T> >( &queue_ ) ) {
      return q_ptr->get();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  inline T* queue_as_ptr() { 
    if( !queue_.has_value() ) return nullptr;
    if( auto q_ptr = std::any_cast< std::shared_ptr<T> >( &queue_ ) ) {
      return q_ptr->get();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  inline const T& queue_as() const {
    auto ptr = queue_as_ptr<T>();
    if( not ptr ) throw std::bad_any_cast();
    return *ptr;
  }

  template <typename T>
  inline T& queue_as() {
    auto ptr = queue_as_ptr<T>();
    if( not ptr ) throw std::bad_any_cast();
    return *ptr;
  }

};



#if 0
class device_queue_pool {

  std::vector<device_queue> queues_;

public:

  device_queue_pool() = default;
  template <typename T, template... Args>
  device_queue_pool(size_t nq, Args&&... args) {
    for( auto i = 0ul; i < nq; ++i )
      queues_.emplace_back( std::make_shared<T>(std::forward<Args>(args)...) );
  }

  size_t size() const { return queues_.size(); }

  const device_queue& operator[]( size_t i ) const { return queues_.at(i); }
  device_queue&       operator[]( size_t i )       { return queues_.at(i); }

  template <typename T>
  const T& at_as( size_t i ) const { return queues_.at(i).queue_as<T>(); }
  template <typename T>
  T& at_as( size_t i ) { return queues_.at(i).queue_as<T>(); }

  template <typename T>
  const T* at_as_ptr( size_t i ) const { return queues_.at(i).queue_as_ptr<T>(); }
  template <typename T>
  T* at_as_ptr( size_t i ) { return queues_.at(i).queue_as_ptr<T>(); }

};
#endif

}

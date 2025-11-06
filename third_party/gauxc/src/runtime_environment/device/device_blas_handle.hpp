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

class device_blas_handle {

  std::any blas_handle_;

public:

  device_blas_handle() = default;

  template <typename T>
  device_blas_handle( std::shared_ptr<T> q ) : blas_handle_( std::move(q) ) { }

  template <typename T>
  inline const T* blas_handle_as_ptr() const { 
    if( !blas_handle_.has_value() ) return nullptr;
    if( auto q_ptr = std::any_cast< std::shared_ptr<T> >( &blas_handle_ ) ) {
      return q_ptr->get();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  inline T* blas_handle_as_ptr() { 
    if( auto q_ptr = std::any_cast< std::shared_ptr<T> >( &blas_handle_ ) ) {
      return q_ptr->get();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  inline const T& blas_handle_as() const {
    auto ptr = blas_handle_as_ptr<T>();
    if( not ptr ) throw std::bad_any_cast();
    return *ptr;
  }

  template <typename T>
  inline T& blas_handle_as() {
    auto ptr = blas_handle_as_ptr<T>();
    if( not ptr ) throw std::bad_any_cast();
    return *ptr;
  }

};

}

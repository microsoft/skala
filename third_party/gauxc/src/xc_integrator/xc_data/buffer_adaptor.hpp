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
#include <memory>
#include <gauxc/exceptions.hpp>

//#define csl __PRETTY_FUNCTION__
#define csl std::string(__FILE__) + ": " + std::to_string(__LINE__)

namespace GauXC {

template <typename T>
struct buffer {
  T*     ptr = nullptr;
  size_t length = 0;
  size_t alignment = 0;

  operator T*() { return ptr; }
  //buffer( nullptr_t ) : ptr(nullptr), len(0), alignment(0) { }
};

class buffer_adaptor {

  //size_t nalloc_;
  size_t nleft_;
  //void*  top_;
  void*  stack_;

public:

  buffer_adaptor() = delete;

  inline buffer_adaptor( void* ptr, size_t len ) :
    //nalloc_(len), 
    nleft_(len), 
    //top_(ptr), 
    stack_(ptr) { }

  template <typename T>
  buffer<T> aligned_alloc( size_t len, 
                           size_t align = alignof(T),
                           std::string msg = "" ) {

    if(len == 0ul) return buffer<T>{nullptr, 0, align};

    char* old_stack = (char*)stack_;
    if( std::align( align, 
                    len*sizeof(T), 
                    stack_, 
                    nleft_          ) ) {

      T* result = reinterpret_cast<T*>(stack_);
      stack_ = (char*)stack_ + len*sizeof(T);
      nleft_ -= std::distance( old_stack, 
                               (char*)stack_ );
      //return result;
      return buffer<T>{result, len, align};

    }

    GAUXC_GENERIC_EXCEPTION("device std::bad_alloc " + msg 
      + " nalloc = " + std::to_string(len*sizeof(T)) 
      + " nleft = " +std::to_string(nleft_));

  }

  template <typename T>
  auto aligned_alloc( size_t len, std::string msg ) {
    return aligned_alloc<T>( len, alignof(T), msg );
  }


  inline void* stack() const {return stack_;}
  inline size_t nleft() const { return nleft_; }

};


}

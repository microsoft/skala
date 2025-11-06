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
#include <type_traits>
#include <gauxc/gauxc_config.hpp>

namespace GauXC {
namespace detail {

template <typename T, typename ParameterType>
class NamedType {

public:

  constexpr explicit NamedType() : value_() { }
  constexpr explicit NamedType(T const& value) : value_(value) {}
  constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}

  constexpr NamedType( const NamedType& other ) : value_(other.get()) { }
  constexpr NamedType( NamedType&& other ) noexcept : 
    value_(std::move(other.get())) { };

  constexpr NamedType& operator=( const NamedType& other ) {
    value_ = other.get();
    return *this;
  }
  constexpr NamedType& operator=( NamedType&& other ) noexcept {
    value_ = std::move(other.get());
    return *this;
  }

  constexpr T& get() { return value_; }
  constexpr T const& get() const {return value_; }

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( value_ );
  }

private:

  T value_;

};

template <typename T, typename ParameterType>
inline bool operator==( 
  const NamedType<T,ParameterType>& n1,
  const NamedType<T,ParameterType>& n2
) { return n1.get() == n2.get(); }

template <typename T, typename ParameterType>
inline bool operator==( 
  const NamedType<T,ParameterType>& n1,
  const           T               & n2
) { return n1.get() == n2; }

template <typename T, typename ParameterType>
inline bool operator==( 
  const           T               & n1,
  const NamedType<T,ParameterType>& n2
) { return n2 == n1; }

template <typename T, typename ParameterType, typename U,
  typename = std::enable_if_t<std::is_convertible<U,T>::value>
>
inline bool operator==(
  const NamedType<T,ParameterType>& n1,
  const           U               & n2
) { return n1.get() == T(n2); }

template <typename T, typename ParameterType, typename U,
  typename = std::enable_if_t<std::is_convertible<U,T>::value>
>
inline bool operator==(
  const           U               & n1,
  const NamedType<T,ParameterType>& n2
) { return n2 == n1; }






template <typename T, typename ParameterType>
inline bool operator!=( 
  const NamedType<T,ParameterType>& n1,
  const NamedType<T,ParameterType>& n2
) { return not(n1 == n2); }

template <typename T, typename ParameterType>
inline bool operator!=( 
  const NamedType<T,ParameterType>& n1,
  const           T               & n2
) { return not( n1 == n2 ); }

template <typename T, typename ParameterType>
inline bool operator!=( 
  const           T               & n1,
  const NamedType<T,ParameterType>& n2
) { return not( n1 == n2 ); }

template <typename T, typename ParameterType, typename U,
  typename = std::enable_if_t<std::is_convertible<U,T>::value>
>
inline bool operator!=(
  const NamedType<T,ParameterType>& n1,
  const           U               & n2
) { return not( n1 == n2 ); }

template <typename T, typename ParameterType, typename U,
  typename = std::enable_if_t<std::is_convertible<U,T>::value>
>
inline bool operator!=(
  const           U               & n1,
  const NamedType<T,ParameterType>& n2
) { return not( n1 == n2 ); }

template <typename T, typename ParameterType>
inline std::ostream& operator<<( std::ostream&                     out, 
                                 const NamedType<T,ParameterType>& n ) {

  out << n.get();
  return out;
}

}
}

namespace std {

template <typename T, typename ParameterType>
struct hash< GauXC::detail::NamedType<T,ParameterType> > {

  std::size_t 
    operator()( const GauXC::detail::NamedType<T,ParameterType>& key ) const {
    return hash<T>()(key.get());
  }

};

}

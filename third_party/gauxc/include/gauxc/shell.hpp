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
#include <iostream>
#include <cassert>
#include <algorithm>
#include <tuple>

#include <gauxc/named_type.hpp>
#include <gauxc/gauxc_config.hpp>
#include <gauxc/util/contiguous_container_data.hpp>
#include <gauxc/util/gau_rad_eval.hpp>


namespace GauXC {

namespace detail {

  static constexpr size_t shell_nprim_max = 32ul;

  static constexpr std::array<int64_t,31> df_Kminus1 = 
    {{ 1LL, 1LL, 1LL, 2LL, 3LL, 8LL, 15LL, 48LL, 105LL, 384LL, 945LL, 3840LL, 
       10395LL, 46080LL, 135135LL, 645120LL, 2027025LL, 10321920LL, 34459425LL, 
       185794560LL, 654729075LL, 3715891200LL, 13749310575LL, 81749606400LL, 
       316234143225LL, 1961990553600LL, 7905853580625LL, 51011754393600LL, 
       213458046676875LL, 1428329123020800LL, 6190283353629375LL }};

  static constexpr double default_shell_tolerance = 1e-10;

}

using PrimSize        = detail::NamedType< int32_t, struct PrimSizeType >;
using AngularMomentum = detail::NamedType< int32_t, struct AngularMomentumType >;
using SphericalType   = detail::NamedType< int32_t, struct SphericalTypeType >;

template <typename F>
class alignas(256) Shell {

public:

  using prim_array = std::array< F, detail::shell_nprim_max >;
  using cart_array = std::array< double, 3 >;

private:

  prim_array alpha_;
  prim_array coeff_;
  cart_array O_;

  int32_t nprim_;
  int32_t l_;
  int32_t pure_;

  double cutoff_radius_;
  double shell_tolerance_{detail::default_shell_tolerance}; 

  //double _pad_; // Pad to be a multiple of 16
    
  // Shamelessly adapted from Libint...
  void normalize() {

    assert( l_ <= 15 );

    constexpr auto sqrt_Pi_cubed = F{5.56832799683170784528481798212};

    const auto two_to_l = std::pow(2, l_);
    const auto df_term  = two_to_l / sqrt_Pi_cubed / detail::df_Kminus1[2*l_];

    for( int32_t i = 0; i < nprim_; ++i ) {
      assert( alpha_[i] >= 0. );
      if( alpha_[i] != 0. ) {
        const auto two_alpha = 2 * alpha_[i];
        const auto two_alpha_to_am32 = 
          std::pow(two_alpha,l_+1) * std::sqrt(two_alpha);
        const auto normalization_factor = std::sqrt(df_term * two_alpha_to_am32);

        coeff_[i] *= normalization_factor;
      }
    }

    double norm{0};
    for(int32_t i = 0; i < nprim_; ++i ) {
    for(int32_t j = 0; j <= i;     ++j ) {
      const auto gamma = alpha_[i] + alpha_[j];
      const auto gamma_to_am32 = std::pow(gamma, l_+1) * std::sqrt(gamma);
      norm += (i==j ? 1 : 2) * coeff_[i] * coeff_[j] /
              (df_term * gamma_to_am32 );
    }
    }

    auto normalization_factor = 1. / std::sqrt(norm);
    for(int32_t i = 0; i < nprim_; ++i ) {
      coeff_[i] *= normalization_factor;
    }


  }

  void compute_shell_cutoff() {

#if 0
    // Cutoff radius according to Eq.20 in J. Chem. Theory Comput. 2011, 7, 3097-3104
    auto cutFunc = [tol=shell_tolerance_] (double alpha) -> double {
      const double log_tol  = -std::log(tol);
      const double log_alph =  std::log(alpha);
      return std::sqrt( (log_tol + 0.5 * log_alph)/alpha );
    };

    cutoff_radius_ = cutFunc(
      *std::max_element( alpha_.begin(), alpha_.begin() + nprim_, 
        [&](F x, F y){ return cutFunc(x) < cutFunc(y); }
      )
    );
#else
    cutoff_radius_ = util::gau_rad_cutoff( l_, nprim_, alpha_.data(), 
      coeff_.data(), shell_tolerance_ );
#endif

  }
public:

  Shell() : nprim_(0), l_(0), pure_(false) { };

  Shell( PrimSize nprim, AngularMomentum l, SphericalType pure,
    prim_array alpha, prim_array coeff, cart_array O, bool _normalize = true ) :
    alpha_( alpha ), coeff_( coeff ), O_( O ),
    nprim_( nprim.get() ), l_( l.get() ), pure_( pure.get() ) {

    if( _normalize ) normalize();
    compute_shell_cutoff();

  }
  
  void set_shell_tolerance( double tol ) {
    if( tol != shell_tolerance_ ) {
      shell_tolerance_ = tol;
      compute_shell_cutoff();
    }
  }


  ~Shell() noexcept = default;

  Shell( const Shell& )          = default;
  Shell( Shell&&      ) noexcept = default;

  Shell& operator=(const Shell&)     = default;
  Shell& operator=(Shell&&) noexcept = default;


  inline HOST_DEVICE_ACCESSIBLE int32_t nprim() const { return nprim_; }
  inline HOST_DEVICE_ACCESSIBLE int32_t l()     const { return l_;     }
  inline HOST_DEVICE_ACCESSIBLE int32_t pure()  const { return pure_;  }

  inline HOST_DEVICE_ACCESSIBLE const F* alpha_data()  const { 
    return detail::contiguous_data(alpha_); 
  }
  inline HOST_DEVICE_ACCESSIBLE const F* coeff_data()  const { 
    return detail::contiguous_data(coeff_); 
  }
  inline HOST_DEVICE_ACCESSIBLE const double* O_data() const { 
    return detail::contiguous_data(O_);     
  }
  inline HOST_DEVICE_ACCESSIBLE  F* alpha_data()   { 
    return detail::contiguous_data(alpha_); 
  }
  inline HOST_DEVICE_ACCESSIBLE  F* coeff_data()   { 
    return detail::contiguous_data(coeff_); 
  }
  inline HOST_DEVICE_ACCESSIBLE  double* O_data()  { 
    return detail::contiguous_data(O_);     
  }

  inline HOST_DEVICE_ACCESSIBLE double cutoff_radius() const { 
    return cutoff_radius_;
  }

  inline HOST_DEVICE_ACCESSIBLE int32_t cart_size() const {
    return (l_+1)*(l_+2)/2;
  }
  inline HOST_DEVICE_ACCESSIBLE int32_t pure_size() const {
    return 2*l_ + 1;
  }
  inline HOST_DEVICE_ACCESSIBLE int32_t size() const {;
    return pure_ ? pure_size() : cart_size();
  }

  inline const prim_array& alpha()  const { return alpha_; }
  inline const prim_array& coeff()  const { return coeff_; }
  inline const cart_array& O()      const { return O_;     }
  inline       prim_array& alpha()        { return alpha_; }
  inline       prim_array& coeff()        { return coeff_; }
  inline       cart_array& O()            { return O_;     }

  inline void set_pure(bool p) { pure_ = p; }

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( nprim_, l_, pure_, alpha_, coeff_, O_, cutoff_radius_, shell_tolerance_ );
  }


  bool operator==( const Shell& other ) const {
    if( other.nprim_ != nprim_ ) return false;
    if( other.l_ != l_ ) return false;
    if( other.pure_ != pure_ ) return false;
    if( other.O_ != O_ ) return false;

    for( auto i = 0; i < nprim_; ++i ) {
      if( alpha_[i] != other.alpha_[i] ) return false;
      if( coeff_[i] != other.coeff_[i] ) return false;
    }

    return true;
  }

};


template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Shell<T>& sh ) {
    os << "GauXC::Shell:( O={" 
	<< sh.O()[0] << "," << sh.O()[1] << "," << sh.O()[2] 
	<< "}" << std::endl;
    os << "  ";
    os << " {l=" << sh.l() << ",sph=" << sh.pure() << "}";
    os << std::endl;

    for(auto i=0ul; i<sh.nprim(); ++i) {
      os << "  " << sh.alpha()[i];
      os << " "  << sh.coeff().at(i);
      os << std::endl;
    }

    return os;
}

}

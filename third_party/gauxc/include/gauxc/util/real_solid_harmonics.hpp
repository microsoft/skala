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

namespace GauXC {
namespace util {

inline constexpr intmax_t integral_falling_factorial( intmax_t n, intmax_t k ) {
  if( n == 0 or n == 1) return 1;
  intmax_t fact = 1;
  for( intmax_t i = k; i <= n; ++i ) fact *= i;
  return fact;
}

inline constexpr intmax_t integral_factorial( intmax_t n ) {
  if( n == 0 or n == 1 ) return 1;
  intmax_t fact = 1;
  for( intmax_t i = 2; i <= n; ++i ) fact *= i;
  return fact;
}

inline constexpr intmax_t integral_double_factorial( intmax_t n ) {
  if( n == 0 or n == 1 ) return 1;
  intmax_t fact = 1;
  if( n % 2 ) {
    // Odd
    for( intmax_t i = 3; i <= n; i += 2 ) fact *= i;
  } else {
    // Even
    for( intmax_t i = 2; i <= n; i += 2 ) fact *= i;
  }
  return fact;
}

inline constexpr intmax_t binomial_coefficient( intmax_t n, intmax_t k ) {
  assert( n >= k );
  if( n == 0 or n == 1 ) return 1;
  if( k == 0 )           return 1;
  if( k == n )           return 1;

  return integral_falling_factorial(n, k+1) / integral_factorial(n-k);
}


inline constexpr auto parity( int i ) {
  return (i%2) ? -1 : 1;
}

inline constexpr double real_solid_harmonic_coeff( int l, int m, int lx, int ly, int lz ) {
  const auto abs_m = m < 0 ? -m : m;
  auto j           = (lx + ly - abs_m);

  if( j % 2 or j < 0 ) return 0.;
  j = j / 2;

  const auto comp = (m >= 0) ? 1 : -1;
  auto i    = abs_m - lx;
  if( comp != parity( std::abs(i) ) ) return 0.;

  double pfac = integral_falling_factorial( 2*lx, lx+1 ) *
                integral_falling_factorial( 2*ly, ly+1 ) *
                integral_falling_factorial( 2*lz, lz+1 );
  const double factorial_l = integral_factorial(l);
  pfac = pfac / ( factorial_l * factorial_l * integral_falling_factorial(2*l,l+1) *
                  integral_falling_factorial(l+abs_m,l-abs_m+1) );
  pfac = std::sqrt(pfac);

  pfac /= (1L << l);
  if (m < 0)
    pfac *= parity((i-1)/2);
  else
    pfac *= parity(i/2);

  auto i_min = j;
  auto i_max = (l-abs_m)/2;
  double sum = 0;
  for(i=i_min;i<=i_max;i++) {
    double pfac1 = parity(i) * binomial_coefficient(l,i) * binomial_coefficient(i,j);
    pfac1 *= integral_factorial(2*(l-i));
    pfac1 /= integral_factorial(l-abs_m-2*i);
    double sum1 = 0.0;
    const int k_min = std::max((lx-abs_m)/2,0);
    const int k_max = std::min(j,lx/2);
    for(int k=k_min;k<=k_max;k++) {
      if (lx-2*k <= abs_m)
        sum1 += parity(k) * 
          binomial_coefficient(j,k) *
          binomial_coefficient(abs_m,lx-2*k);
    }
    sum += pfac1*sum1;
  }

  double pfac2 =  integral_double_factorial( 2*l  - 1 );
  pfac2 = pfac2 / integral_double_factorial( 2*lx - 1 );
  pfac2 = pfac2 / integral_double_factorial( 2*ly - 1 );
  pfac2 = pfac2 / integral_double_factorial( 2*lz - 1 );

  sum *= std::sqrt(pfac2);

  double result = (m == 0) ? pfac*sum : M_SQRT2*pfac*sum;
  return result;
}

class SphericalHarmonicTransform {

  std::vector< std::vector<double> > table_;

public:

  inline SphericalHarmonicTransform( int max_l ) {

    table_.resize(max_l+1);
    for( auto l = 0; l <= max_l; ++ l ) {
      const int nsph  = 2*l + 1;
      const int ncart = (l+1)*(l+2)/2;
      table_[l].resize( nsph * ncart );

      for( int m = -l, isph = 0; m <= l; ++m, ++isph ) {
        for( int ix = l, icart = 0; ix >= 0; --ix )
        for( int iy = l-ix;         iy >= 0; --iy, ++icart ) {
          int iz = l - (ix+iy);
          table_[l][ isph + icart*nsph ] = 
            real_solid_harmonic_coeff(l,m,ix,iy,iz);
        }
      }
    }

  }

  inline void tform_bra_rm( int bra_l, int nket, const double* cart,
    int ldc, double* sph, int lds ) {

    const int bra_cart_sz = (bra_l+1) * (bra_l+2)/2;
    const int bra_sph_sz  = 2*bra_l + 1;
    for( int i = 0; i < bra_sph_sz; ++i )
    for( int j = 0; j < nket;       ++j ) {
      double tmp = 0.;
      for( int k = 0; k < bra_cart_sz; ++k ) {
        tmp += table_.at(bra_l)[ i + k*bra_sph_sz ] * cart[ k*ldc + j ];
      }
      sph[ i*lds + j ] = tmp;
    }

  }

  inline void tform_bra_cm( int bra_l, int nket, const double* cart,
    int ldc, double* sph, int lds ) {

    const int bra_cart_sz = (bra_l+1) * (bra_l+2)/2;
    const int bra_sph_sz  = 2*bra_l + 1;
    for( int i = 0; i < bra_sph_sz; ++i )
    for( int j = 0; j < nket;       ++j ) {
      double tmp = 0.;
      for( int k = 0; k < bra_cart_sz; ++k ) {
        tmp += table_.at(bra_l)[ i + k*bra_sph_sz ] * cart[ k + j*ldc ];
      }
      sph[ i + j*lds ] = tmp;
    }

  }

  inline void itform_bra_rm( int bra_l, int nket, const double* sph,
    int lds, double* cart, int ldc ) {

    const int bra_cart_sz = (bra_l+1) * (bra_l+2)/2;
    const int bra_sph_sz  = 2*bra_l + 1;
    for( int i = 0; i < bra_cart_sz; ++i )
    for( int j = 0; j < nket;        ++j ) {
      double tmp = 0.;
      for(int k = 0; k < bra_sph_sz; ++k ) {
        tmp += table_.at(bra_l)[ k + i*bra_sph_sz] * sph[ k*lds + j ];
      }
      cart[ i*ldc + j ] = tmp;
    }

  }

  inline void itform_bra_cm( int bra_l, int nket, const double* sph,
    int lds, double* cart, int ldc ) {

    const int bra_cart_sz = (bra_l+1) * (bra_l+2)/2;
    const int bra_sph_sz  = 2*bra_l + 1;
    for( int i = 0; i < bra_cart_sz; ++i )
    for( int j = 0; j < nket;        ++j ) {
      double tmp = 0.;
      for(int k = 0; k < bra_sph_sz; ++k ) {
        tmp += table_.at(bra_l)[ k + i*bra_sph_sz] * sph[ k + j*lds ];
      }
      cart[ i + j*ldc ] = tmp;
    }

  }

  inline void tform_ket_rm( int nbra, int ket_l, const double* cart,
    int ldc, double* sph, int lds ) {

    const int ket_cart_sz = (ket_l+1) * (ket_l+2)/2;
    const int ket_sph_sz  = 2*ket_l + 1;
    for( int i = 0; i < nbra;       ++i )
    for( int j = 0; j < ket_sph_sz; ++j ) {
      double tmp = 0.;
      for( int k = 0; k < ket_cart_sz; ++k ) {
        tmp += cart[ i*ldc + k ] * table_.at(ket_l)[ j + k*ket_sph_sz ]; 
      }
      sph[ i*lds + j ] = tmp;
    }

  }

  inline void tform_both_rm( int bra_l, int ket_l, const double* cart,
    int ldc, double* sph, int lds ) {

    //const int bra_cart_sz = (bra_l+1) * (bra_l+2)/2;
    const int ket_cart_sz = (ket_l+1) * (ket_l+2)/2;
    const int bra_sph_sz  = 2*bra_l + 1;
    //const int ket_sph_sz  = 2*ket_l + 1;
    std::vector<double> row_tmp( bra_sph_sz * ket_cart_sz );
    tform_bra_rm( bra_l, ket_cart_sz, cart, ldc, row_tmp.data(), ket_cart_sz );
    tform_ket_rm( bra_sph_sz, ket_l,  row_tmp.data(), ket_cart_sz, sph, lds  );

  }

};


}
}

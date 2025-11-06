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
#include "integral_bounds.hpp"
#include <vector>
#include <gauxc/util/geometry.hpp>
#include <gauxc/util/constexpr_math.hpp>
#include <gauxc/exceptions.hpp>
#include <gauxc/shell_pair.hpp>


namespace GauXC {
namespace util  {


inline constexpr double max_coulomb_20( double Rab, double alpha, double beta, 
  double gamma ) {
  (void)alpha;
  return 1.0 * ( gamma + Rab * integral_pow<2>(beta) ) / integral_pow<2>(gamma);
}

inline constexpr double max_coulomb_22( double Rab, double alpha, double beta, 
  double gamma ) {
  return 1.0 / integral_pow<4>(gamma) * 
    ( Rab * integral_pow<3>(alpha) +
      alpha * beta * (4. - Rab*beta ) +
      integral_pow<2>(beta) * (2.  + Rab * beta ) +
      integral_pow<2>(alpha) * (2. - Rab * beta + Rab*Rab * beta*beta)
    );
}

inline constexpr double max_coulomb_40( double Rab, double alpha, double beta, 
  double gamma ) {
  return 1.0 / integral_pow<4>(gamma) *
  (
    2.*gamma*gamma +
    4.*beta*beta * gamma * Rab +
    beta*beta* integral_pow<2>(alpha - gamma) * Rab*Rab
  );
}
inline constexpr double max_coulomb_42( double Rab, double alpha, double beta, 
  double gamma ) {
  return -1.0 / integral_pow<6>(gamma) *
  (
    -6.   * integral_pow<3>(gamma) +
    -2.   * (3.*alpha - 2.*gamma) * gamma*gamma * (gamma - 3.*beta) * Rab +
    -beta * (3.*beta  - 2.*gamma) * gamma * (3.*alpha*alpha - 4.*alpha*gamma + gamma*gamma) * Rab*Rab +
    alpha * beta*beta * integral_pow<2>(alpha - gamma) * (beta - gamma) * Rab*Rab*Rab
  );
}

inline constexpr double max_coulomb_44( double Rab, double alpha, double beta, 
  double gamma ) {
  return 1.0 / integral_pow<8>(gamma) *
  (
    24. * integral_pow<4>(gamma) +
    24. * (2.*alpha - gamma) * integral_pow<3>(gamma) * (gamma - 2.*beta) * Rab +
    2. * gamma*gamma * (6.*alpha*alpha - 6.*alpha*gamma + gamma*gamma) *
      (6.*beta*beta - 6.*beta*gamma + gamma*gamma) *Rab*Rab +
    -4. * alpha * beta * gamma * (2.*alpha*alpha - 3.*alpha*gamma + gamma*gamma) *
      (2.*beta*beta - 3.*beta*gamma + gamma*gamma) * Rab*Rab*Rab +
    alpha*alpha*beta*beta * integral_pow<2>(alpha-gamma) * integral_pow<2>(beta-gamma) *
      Rab*Rab*Rab*Rab
  );
}


inline double max_coulomb( int l_a, int l_b, double Rab, double alpha, 
  double beta, double gamma ) {

  if( l_a == 0 and l_b == 0 ) return 1.0;
  if( l_a == 2 and l_b == 2 ) return max_coulomb_22( Rab, alpha, beta, gamma );
  if( l_a == 2 and l_b == 0 ) return max_coulomb_20( Rab, alpha, beta, gamma );
  if( l_a == 0 and l_b == 2 ) return max_coulomb_20( Rab, beta, alpha, gamma );
  if( l_a == 4 and l_b == 4 ) return max_coulomb_44( Rab, alpha, beta, gamma );
  if( l_a == 4 and l_b == 0 ) return max_coulomb_40( Rab, alpha, beta, gamma );
  if( l_a == 0 and l_b == 4 ) return max_coulomb_40( Rab, beta, alpha, gamma );
  if( l_a == 4 and l_b == 2 ) return max_coulomb_42( Rab, alpha, beta, gamma );
  if( l_a == 2 and l_b == 4 ) return max_coulomb_42( Rab, beta, alpha, gamma );

  const int l_a_p = l_a + (l_a % 2);
  const int l_b_p = l_b + (l_b % 2);

  const int l_a_m = l_a - (l_a % 2);
  const int l_b_m = l_b - (l_b % 2);

  if( l_a_p > 4 or l_b_p > 4 ) GAUXC_GENERIC_EXCEPTION("Case Not Handled"); 

  double V_pm = std::numeric_limits<double>::infinity();
  if( l_a_p == 0 and l_b_m == 0 ) 
    V_pm = 1.0;
  else if( l_a_p == 2 and l_b_m == 0 ) 
    V_pm = max_coulomb_20( Rab, alpha, beta, gamma );
  else if( l_a_p == 0 and l_b_m == 2 ) 
    V_pm = max_coulomb_20( Rab, beta, alpha, gamma );
  else if( l_a_p == 2 and l_b_m == 2 )
    V_pm = max_coulomb_22( Rab, alpha, beta, gamma );
  else if( l_a_p == 4 and l_b_m == 0 ) 
    V_pm = max_coulomb_40( Rab, alpha, beta, gamma );
  else if( l_a_p == 0 and l_b_m == 4 ) 
    V_pm = max_coulomb_40( Rab, beta, alpha, gamma );
  else if( l_a_p == 4 and l_b_m == 2 ) 
    V_pm = max_coulomb_42( Rab, alpha, beta, gamma );
  else if( l_a_p == 2 and l_b_m == 4 ) 
    V_pm = max_coulomb_42( Rab, beta, alpha, gamma );
  else if( l_a_p == 4 and l_b_m == 4 )
    V_pm = max_coulomb_44( Rab, alpha, beta, gamma );

  double V_mp = std::numeric_limits<double>::infinity();
  if( l_a_m == 0 and l_b_p == 0 ) 
    V_mp = 1.0;
  else if( l_a_m == 2 and l_b_p == 0 ) 
    V_mp = max_coulomb_20( Rab, alpha, beta, gamma );
  else if( l_a_m == 0 and l_b_p == 2 ) 
    V_mp = max_coulomb_20( Rab, beta, alpha, gamma );
  else if( l_a_m == 2 and l_b_p == 2 )
    V_mp = max_coulomb_22( Rab, alpha, beta, gamma );
  else if( l_a_m == 4 and l_b_p == 0 ) 
    V_mp = max_coulomb_40( Rab, alpha, beta, gamma );
  else if( l_a_m == 0 and l_b_p == 4 ) 
    V_mp = max_coulomb_40( Rab, beta, alpha, gamma );
  else if( l_a_m == 4 and l_b_p == 2 ) 
    V_mp = max_coulomb_42( Rab, alpha, beta, gamma );
  else if( l_a_m == 2 and l_b_p == 4 ) 
    V_mp = max_coulomb_42( Rab, beta, alpha, gamma );
  else if( l_a_m == 4 and l_b_p == 4 )
    V_mp = max_coulomb_44( Rab, alpha, beta, gamma );

  return std::sqrt(V_pm * V_mp);
}


template <typename T>
T max_coulomb( const Shell<T>& bra, const Shell<T>& ket) {

  const auto A = bra.O();
  const auto B = ket.O();
  const auto RAB = std::pow(geometry::euclidean_dist( A, B ),2);

  double max_val = 0.;
  for( auto i = 0; i < bra.nprim(); ++i )
  for( auto j = 0; j < ket.nprim(); ++j ) {
    const auto alpha = bra.alpha()[i];
    const auto beta  = ket.alpha()[j];
    const auto gamma = alpha + beta;

    const auto Kab = std::exp( - alpha*beta*RAB / gamma );

    const auto c_a = bra.coeff()[i];
    const auto c_b = ket.coeff()[j];
    const auto c = 2 * M_PI * Kab * std::abs( c_a * c_b / gamma );

    max_val += c * max_coulomb( bra.l(), ket.l(), RAB, alpha, beta, gamma );
  }

  return max_val;
}


template double max_coulomb( const Shell<double>&, const Shell<double>& );

}
}

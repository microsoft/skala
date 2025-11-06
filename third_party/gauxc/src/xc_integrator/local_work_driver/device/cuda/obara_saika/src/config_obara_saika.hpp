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

#include <gauxc/util/constexpr_math.hpp>

#define NPTS_LOCAL 64

#define DEFAULT_NCHEB  7
#define DEFAULT_MAX_M  8
#define DEFAULT_MAX_T 30

#define DEFAULT_NSEGMENT ((DEFAULT_MAX_T * DEFAULT_NCHEB) / 2)
#define DEFAULT_LD_TABLE (DEFAULT_NCHEB + 1)
 
namespace XGPU {
  template <int M>
  __device__ inline void boys_element(double *T, double *T_inv_e, double *eval, double *boys_table) {
    if((*T) < DEFAULT_MAX_T) {
      if constexpr (M == 0) {
	const double sqrt_t = std::sqrt((*T));
	const double inv_sqrt_t = 1./sqrt_t;
	*(T_inv_e) = 0.0;
	*(eval) = GauXC::constants::sqrt_pi_ov_2<> * std::erf(sqrt_t) * inv_sqrt_t;
      } else {
	const double* boys_m = (boys_table + M * DEFAULT_LD_TABLE * DEFAULT_NSEGMENT);
	constexpr double deltaT = double(DEFAULT_MAX_T) / DEFAULT_NSEGMENT;
	constexpr double one_over_deltaT = 1 / deltaT;
	
	int iseg = std::floor((*T) * one_over_deltaT);
	const double* boys_seg = boys_m + iseg * DEFAULT_LD_TABLE;
	
	const double ratio = (2 * iseg + 1);
	const double fact  = 2.0 / deltaT;
	
	double xt = (*T) * fact - ratio;
	double _rec = 1.0;
	double _val = boys_seg[0];
	
	for(int i = 1; i < DEFAULT_NCHEB + 1; ++i) {
	  _rec = _rec * xt;
	  _val += _rec * boys_seg[i];
	}

	*(T_inv_e) = 0.5 * std::exp(-(*T));
	*(eval) = _val;
      }
    } else {
      const double t_inv = 1./(*T);
      //double _val = GauXC::constants::sqrt_pi_ov_2<> * std::sqrt(t_inv);
      double _val = GauXC::constants::sqrt_pi_ov_2<> * rsqrt(*T);
    
      for(int i = 1; i < M + 1; ++i) {
	_val *= ((i - 0.5) * t_inv);
      }

      *(T_inv_e) = 0.0;
      *(eval) = _val;
    }
  }

  __device__ __inline__ double boys_element_0( double T ) {
  #if 0
    if( T < DEFAULT_MAX_T ) {
      const double sqrt_t = std::sqrt(T);
      const double inv_sqrt_t = 1.0 / sqrt_t;
      return 0.88622692545275801364 * std::erf(sqrt_t) * inv_sqrt_t;
    } else {
      return 0.88622692545275801364 * rsqrt(T);
    }
  #else
    if( T > 26.0 ) {
      return 0.88622692545275801364 * rsqrt(T);
    } else if( T < 13.0 ) {
      const auto exp_t = exp( - T * 0.33333333333333333333 );

      double b =  4.014103057876808e-23;
      b = fma( T, b, -5.822235306869006e-21 );
      b = fma( T, b,  4.093796011592500e-19 );
      b = fma( T, b, -1.869382772172656e-17 );
      b = fma( T, b,  6.338163538927402e-16 );
      b = fma( T, b, -1.721896819094452e-14 );
      b = fma( T, b,  3.984232174194261e-13 );
      b = fma( T, b, -8.072677948936458e-12 );
      b = fma( T, b,  1.489767929273334e-10 );
      b = fma( T, b, -2.441928489146782e-09 );
      b = fma( T, b,  3.780445468547986e-08 );
      b = fma( T, b, -4.872128794416657e-07 );
      b = fma( T, b,  6.455920003140367e-06 );
      b = fma( T, b, -5.700739807688489e-05 );
      b = fma( T, b,  7.054673174084430e-04 );
      b = fma( T, b, -2.821869460954601e-03 );
      b = fma( T, b,  4.444444443709288e-02 );
      b = fma( T, b,  7.778049953252520e-13 );
      b = fma( T, b,  9.999999999999863e-01 );
      return b * exp_t;

    } else {
    #if 0
      const double sqrt_t = std::sqrt(T);
      const double inv_sqrt_t = 1.0 / sqrt_t;
      return 0.88622692545275801364 * std::erf(sqrt_t) * inv_sqrt_t;
    #else
      const auto exp_t = exp( - T * 0.33333333333333333333 );

      double b = 1.153599464241947e-26;
      b = fma( T, b, -4.025061230220665e-24);
      b = fma( T, b,  6.845330692919496e-22);
      b = fma( T, b, -7.455104439417363e-20);
      b = fma( T, b,  5.806227138295288e-18);
      b = fma( T, b, -3.426510194853584e-16);
      b = fma( T, b,  1.587043680665803e-14);
      b = fma( T, b, -5.898342915599428e-13);
      b = fma( T, b,  1.785040325720807e-11);
      b = fma( T, b, -4.437916159483046e-10);
      b = fma( T, b,  9.111870867088944e-09);
      b = fma( T, b, -1.546337818112499e-07);
      b = fma( T, b,  2.167268088592726e-06);
      b = fma( T, b, -2.490299656562666e-05);
      b = fma( T, b,  2.335812755969758e-04);
      b = fma( T, b, -1.744532113923084e-03);
      b = fma( T, b,  1.048354410615184e-02);
      b = fma( T, b, -4.539934464926983e-02);
      b = fma( T, b,  1.754968961724573e-01);
      b = fma( T, b, -2.542050397037139e-01);
      b = fma( T, b,  1.233675832421592e+00);
      return b * exp_t;
    #endif
    }
  #endif
  }
}

#define SCALAR_TYPE double

#define SCALAR_LENGTH 1

#define SCALAR_SET1(x) (x)
#define SCALAR_ZERO() (0.0)

#define SCALAR_LOAD(x) *(x)
#define SCALAR_STORE(x, y) *(x) = y

#define SCALAR_ADD(x, y) (x + y)
#define SCALAR_SUB(x, y) (x - y)

#define SCALAR_MUL(x, y) (x * y)
#define SCALAR_FMA(x, y, z) (z + x * y)
#define SCALAR_FNMA(x, y, z) (z - x * y)

#define SCALAR_RECIPROCAL(x) (1.0 / (1.0 * x))

/*
  __device__  inline double monomial_expand(const double* coeff, const double x, double a, double b) {
  //const int n = DEFAULT_NCHEB + 1;
  const double sum = a+b;
  const double diff = b-a;
  const double ratio = sum / diff;
  const double fact = 2. / diff;

  //double xp[n]; xp[0] = 1.;
  double xp[DEFAULT_NCHEB + 1]; xp[0] = 1.;

  double xt = fact * x - ratio;

  //for(int i = 1; i < n; ++i) xp[i] = xp[i-1] * xt;
  for(int i = 1; i < DEFAULT_NCHEB + 1; ++i) xp[i] = xp[i-1] * xt;

  double _val = 0.;
  //for(int i = 0; i < n; ++i) _val += xp[i] * coeff[i];
  for(int i = 0; i < DEFAULT_NCHEB + 1; ++i) _val += xp[i] * coeff[i];

  return _val;
  }

  template <int M>
  __device__  inline double boys_asymp_element( double x ) {
  const auto x_inv = 1./x;

  if constexpr (M != 0) {
  constexpr double const_coeff = (constants::sqrt_pi<> / integral_pow_two<2*M+1>::value) * (integral_factorial<2*M>::value / integral_factorial<M>::value);
  return const_coeff * std::sqrt(integral_pow<2*M+1>(x_inv));
  }
    
  return constants::sqrt_pi_ov_2<> * std::sqrt( x_inv ); 
  }
  
  template <int M>
  __device__  inline double gauxc_boys_element(double *boys_table, double T) {

  if(T < DEFAULT_MAX_T) {
  if constexpr (M != 0) {
  const double* boys_m = (boys_table + M * DEFAULT_LD_TABLE * DEFAULT_NSEGMENT);
  constexpr double deltaT = double(DEFAULT_MAX_T) / DEFAULT_NSEGMENT;

  int iseg = std::floor(T/ deltaT);
  const double* boys_seg = boys_m + iseg * DEFAULT_LD_TABLE;

  const double a = iseg * deltaT;
  const double b = a + deltaT;
  return monomial_expand(boys_seg, T, a, b);
  }

  const double sqrt_t = std::sqrt(T);
  const double inv_sqrt_t = 1./sqrt_t;
  return constants::sqrt_pi_ov_2<> * std::erf(sqrt_t) * inv_sqrt_t;
  }

  return boys_asymp_element<M>(T);
  }
*/

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

namespace XCPU {

  constexpr double shpair_screen_tol = 1e-12;

  template <int M>
  inline void boys_element(double *T, double *T_inv_e, double *eval, double *boys_table) {
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
      double _val = GauXC::constants::sqrt_pi_ov_2<> * GauXC::rsqrt(*T);
    
      for(int i = 1; i < M + 1; ++i) {
	_val *= ((i - 0.5) * t_inv);
      }

      *(T_inv_e) = 0.0;
      *(eval) = _val;
    }
  }

  template <int M>
  inline void boys_elements(size_t npts, double* T, double *T_inv_e, double* eval, double *boys_table) {    
    for(size_t i = 0; i < npts; ++i) {
      if(T[i] < DEFAULT_MAX_T) {
	if constexpr (M == 0) {
	  const double sqrt_t = std::sqrt(T[i]);
	  const double inv_sqrt_t = 1./sqrt_t;
	  
	  T_inv_e[i] = 0.0;
	  eval[i] = GauXC::constants::sqrt_pi_ov_2<> * std::erf(sqrt_t) * inv_sqrt_t;
	} else {
	  const double* boys_m = (boys_table + M * DEFAULT_LD_TABLE * DEFAULT_NSEGMENT);
	  constexpr double deltaT = double(DEFAULT_MAX_T) / DEFAULT_NSEGMENT;
	  constexpr double one_over_deltaT = 1 / deltaT;

	  int iseg = std::floor(T[i] * one_over_deltaT);
	  const double* boys_seg = boys_m + iseg * DEFAULT_LD_TABLE;
	  
	  const double ratio = (2 * iseg + 1);
	  const double fact  = 2.0 / deltaT;
	  
	  double xt = fact * T[i] - ratio;
	  double _rec = 1.0;
	  double _val = boys_seg[0];
	  for(int j = 1; j < DEFAULT_NCHEB + 1; ++j) {
	    _rec = _rec * xt;
	    _val += _rec * boys_seg[j];
	  }

	  T_inv_e[i] = 0.5 * std::exp(-T[i]);
	  eval[i] = _val;
	}
      } else {
	const double t_inv = 1./T[i];
	//double _val = GauXC::constants::sqrt_pi_ov_2<> * std::sqrt(t_inv);
    double _val = GauXC::constants::sqrt_pi_ov_2<> * GauXC::rsqrt(T[i]);
      
	for(int j = 1; j < M + 1; ++j) {
	  _val *= ((j - 0.5) * t_inv);
	}

	T_inv_e[i] = 0.0;
	eval[i] = _val;
      }
    }
  }


  inline double boys_element_0( double T ) {
    if( T > 26.0 ) {
      return 0.88622692545275801364 * GauXC::rsqrt(T);
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
    }
  }
  inline void boys_elements_0( int npts, const double* T, double* FmT ) {
    for(int i = 0; i < npts; ++i) FmT[i] = boys_element_0(T[i]);
  }

}

// Scalar types
#define SCALAR_TYPE double
#define SCALAR_LENGTH 1

#define SCALAR_SET1(x) (x)

#define SCALAR_LOAD(x) *(x)
#define SCALAR_STORE(x, y) *(x) = y

#define SCALAR_ADD(x, y) (x + y)
#define SCALAR_SUB(x, y) (x - y)

#define SCALAR_MUL(x, y) (x * y)
#define SCALAR_FMA(x, y, z) (z + x * y)
#define SCALAR_FNMA(x, y, z) (z - x * y)

#define SCALAR_RECIPROCAL(x) (1.0 / (1.0 * x))

#define SCALAR_DUPLICATE(x) (*(x))

// AVX-512 SIMD Types
#if __AVX512F__ && __has_include(<zmmintrin.h>)

  #include <zmmintrin.h>
  
  #define SIMD_TYPE __m512d
  
  #define SIMD_LENGTH 8
  
  #define SIMD_ZERO() _mm512_setzero_pd()
  #define SIMD_SET1(x) _mm512_set1_pd(x)
  
  #define SIMD_ALIGNED_LOAD(x) _mm512_load_pd(x)
  #define SIMD_UNALIGNED_LOAD(x) _mm512_loadu_pd(x)
  
  #define SIMD_ALIGNED_STORE(x, y) _mm512_store_pd(x, y)
  #define SIMD_UNALIGNED_STORE(x, y) _mm512_storeu_pd(x, y)
  
  #define SIMD_ADD(x, y) _mm512_add_pd(x, y)
  #define SIMD_SUB(x, y) _mm512_sub_pd(x, y)
  
  #define SIMD_MUL(x, y) _mm512_mul_pd(x, y)
  #define SIMD_FMA(x, y, z) _mm512_fmadd_pd(x, y, z)
  #define SIMD_FNMA(x, y, z) _mm512_fnmadd_pd(x, y, z)
  
  #define SIMD_DUPLICATE(x) _mm512_broadcast_f64x4(_mm256_broadcast_sd(x))

// AVX-256 SIMD Types
#elif __AVX__ || __AVX2__

  #include <immintrin.h>
  
  #define SIMD_TYPE __m256d
  
  #define SIMD_LENGTH 4
  
  #define SIMD_ZERO() _mm256_setzero_pd()
  #define SIMD_SET1(x) _mm256_set1_pd(x)
  
  #define SIMD_ALIGNED_LOAD(x) _mm256_load_pd(x)
  #define SIMD_UNALIGNED_LOAD(x) _mm256_loadu_pd(x)
  
  #define SIMD_ALIGNED_STORE(x, y) _mm256_store_pd(x, y)
  #define SIMD_UNALIGNED_STORE(x, y) _mm256_storeu_pd(x, y)
  
  #define SIMD_ADD(x, y) _mm256_add_pd(x, y)
  #define SIMD_SUB(x, y) _mm256_sub_pd(x, y)
  
  #define SIMD_MUL(x, y) _mm256_mul_pd(x, y)
  #define SIMD_FMA(x, y, z) _mm256_fmadd_pd(x, y, z)
  #define SIMD_FNMA(x, y, z) _mm256_fnmadd_pd(x, y, z)
  
  #define SIMD_DUPLICATE(x) _mm256_broadcast_sd(x)

// Scalar SIMD Emulation
#else
  #define SIMD_TYPE double
  
  #define SIMD_LENGTH 1
  
  #define SIMD_ZERO() 0.0
  #define SIMD_SET1(x) SCALAR_SET1(x)
  
  #define SIMD_ALIGNED_LOAD(x) SCALAR_LOAD(x)
  #define SIMD_UNALIGNED_LOAD(x) SCALAR_LOAD(x)
  
  #define SIMD_ALIGNED_STORE(x, y) SCALAR_STORE(x, y)
  #define SIMD_UNALIGNED_STORE(x, y) SCALAR_STORE(x, y)
  
  #define SIMD_ADD(x, y) SCALAR_ADD(x, y)
  #define SIMD_SUB(x, y) SCALAR_SUB(x, y)
  
  #define SIMD_MUL(x, y) SCALAR_MUL(x, y)
  #define SIMD_FMA(x, y, z) SCALAR_FMA(x, y, z)
  #define SIMD_FNMA(x, y, z) SCALAR_FNMA(x, y, z)
  
  #define SIMD_DUPLICATE(x) SCALAR_DUPLICATE(x)

#endif

#if 0
#ifdef X86_SCALAR
#elif defined(X86_SSE)
#elif defined(X86_AVX)
#elif defined(X86_AVX512)
#else
  #error "That ISA is not recognized!!!\n"
#endif
#endif


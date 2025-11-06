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
#include "cpu/chebyshev_boys_computation.hpp"

#include <gauxc/util/constexpr_math.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <vector>


namespace XCPU {
  double boys_reference(int m, double T) {
    double denom = m + 0.5;
    double term  = std::exp(-T) / (2 * denom);
    double old_term = term;
    double sum = old_term;

    double eps = std::numeric_limits<double>::epsilon();
    double eps_10 = eps / 10;

    while( term > sum * eps_10 || old_term < term ) {
      denom = denom + 1;
      old_term = term;
      term = old_term * T / denom;
      sum = sum + term;
    }

    return sum;
  }
  
  // create table - so this should be done on the host
  void cheby_coeff(int m, int ncheb, double a, double b, double* c) {
    const int n = ncheb+1;
    const double pi_ov_2n = M_PI / (2 * n);
    
    std::vector<double> f_table(n);
    for( int i = 0; i < n; ++i ) {
      double x = std::cos( (2.*(i+1)-1) * pi_ov_2n );
      x = 0.5 * ( a+b + (b-a)*x );
      f_table[i] = boys_reference(m, x);
    }

    c[0] = std::accumulate( f_table.begin(), f_table.end(),0. ) / n;
    for( int i = 1; i < n; ++i ) {
      double _val = 0.;
      for( int j = 0; j < n; ++j ) {
	_val += f_table[j] * std::cos( i * (2*(j+1)-1) * pi_ov_2n );
      }
      c[i] = 2.0 * _val / n;
    }
  }

  void cheby_to_monomial_coeff( int ncheb, double *coeff ) {
    const int n = ncheb+1;
    int64_t i_fact = 1;
    int64_t t_fact = 1;
    for(int i = 0; i < n; ++i) {
      if(i)     i_fact *= i;
      if(i > 1) t_fact *= 2;

      double _val = 0;
      if(!i) {
	int m1_fact = 1;
	for( int j = i; j < n; j += 2 ) {
	  _val += m1_fact * coeff[j];
	  m1_fact *= -1;
	}
      } else {
	int m1_term = 1;
	for( int j = i; j < n; j += 2 ) {
	  const int f_up = (i+j)/2 - 1;
	  const int f_lo = (j-i)/2;
	  int f_term = 1;
	  for( int k = f_lo+1; k <= f_up; ++k ) f_term *= k;
	  _val += t_fact * j * m1_term * double(f_term) / double(i_fact) * coeff[j];
	  m1_term *= -1;
	}

      }
      coeff[i] = _val;
    }
  }
  
  void generate_boys_table(int ncheb, int maxM, double maxT, int nseg, double* cheb_coeff_table, int ld) {
    const double deltaT = maxT / nseg;
    for( int m = 0; m <= maxM; ++m ) {
      double* coeff_m = cheb_coeff_table + m * ld * nseg; // table offset for current m
      for( int iseg = 0; iseg < nseg; ++iseg ) {
	double* coeff_seg = coeff_m + iseg * ld;

	const double a = iseg * deltaT;
	const double b = a + deltaT;

	cheby_coeff( m, ncheb, a, b, coeff_seg ); // Generate coeff in Chebyshev basis
	cheby_to_monomial_coeff( ncheb, coeff_seg );   // Convert to monomial basis
      }
    }
  }

  double *boys_init() {
    double *boys_table = (double*) malloc(DEFAULT_LD_TABLE * DEFAULT_NSEGMENT * (DEFAULT_MAX_M + 1) * sizeof(double));
    generate_boys_table(DEFAULT_NCHEB, DEFAULT_MAX_M, DEFAULT_MAX_T, DEFAULT_NSEGMENT, boys_table, DEFAULT_LD_TABLE);

    return boys_table;
  }
  
  void boys_finalize(double *boys_table) {
    free(boys_table);
  }
}

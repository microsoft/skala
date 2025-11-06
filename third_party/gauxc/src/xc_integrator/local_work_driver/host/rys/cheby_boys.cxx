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
#include <iostream>
#include <limits>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <array>
#include <iomanip>

int64_t ifact( int64_t i ) {
  if( i == 0 or i == 1 ) return 1;
  int64_t v = 1;
  for( int k = 1; k <= i; ++k ) v += k;
  return v;
}

int64_t difact( int64_t i ) {
  int64_t v = 1;
  for( int k = 0; k < (i/2); ++k ) v *= i - 2*k;
  return v;
}


double boys_reference(int m, double T) {
  double denom = m + 0.5;
  double term  = std::exp(-T) / (2 * denom);
  double old_term = term;
  double sum = old_term;

  constexpr auto eps = std::numeric_limits<double>::epsilon();
  constexpr auto eps_10 = eps / 10;

  while( term > sum * eps_10 || old_term < term ) {
    denom = denom + 1;
    old_term = term;
    term = old_term * T / denom;
    sum = sum + term;
  }

  return sum;
}

double boys_asymp(int m, double T) {
  return difact(2*m-1) / std::pow(2.,m+1) * std::sqrt(M_PI/std::pow(T,2*m+1));
}


template <typename Op>
void cheby_coeff( int ncheb, const Op& f, double a, double b, double* c ) {

  const int n = ncheb+1;
  const double pi_ov_2n = M_PI / (2 * n);

  // Generate function table
  std::vector<double> f_table(n);
  for( int i = 0; i < n; ++i ) {
    double x = std::cos( (2.*(i+1)-1) * pi_ov_2n ); // Chebyshev node
    x = 0.5 * ( a+b + (b-a)*x );
    f_table[i] = f(x);
  }

  c[0] = std::accumulate( f_table.begin(), f_table.end(),0. ) / n;
  for( int i = 1; i < n; ++i ) {
    double _val = 0.;
    for( int j = 0; j < n; ++j ) {
      // f(x_j) * T_i(x_j)
      _val += f_table[j] * std::cos( i * (2*(j+1)-1) * pi_ov_2n );
    }
    c[i] = 2.0 * _val / n;
  }

};

void cheby_to_monomial_coeff( int ncheb, double *coeff ) {

  const int n = ncheb+1;
  int64_t i_fact = 1; // i!
  int64_t t_fact = 1; // 2^(i-1)
  for(int i = 0; i < n; ++i) {
    if(i)     i_fact *= i; // Update i!
    if(i > 1) t_fact *= 2; // Update 2^(i-1)

    double _val = 0;
    if(!i) {
      int m1_fact = 1; // (-1)^(j/2)
      for( int j = i; j < n; j += 2 ) {
        _val += m1_fact * coeff[j];
	m1_fact *= -1; // Update (-1)^(j/2)
      }
    } else {

      // OEIS A008310
      // a(i,j) = 2^(i-1) * j * (-1)^((j-i)/2) * ((i+j)/2-1)! / (i! * ((j-i)/2)!)
      // Coeff tform is UT, able to be performed in place
#if 1
      int m1_term = 1;
      for( int j = i; j < n; j += 2 ) {
	const int f_up = (i+j)/2 - 1;
        const int f_lo = (j-i)/2;
	// This comptes the ratio of factorials as a 
	// Pochhammer falling factorial
	int f_term = 1;
	for( int k = f_lo+1; k <= f_up; ++k ) f_term *= k;
        _val += t_fact * j * m1_term * double(f_term) / double(i_fact) * coeff[j];
	m1_term *= -1;
      }
#else
      // Attempt at f term optimization, minor errors that are too tedious
      // to figure out for only minor performance improvements
      int f_term  = 1;
      int m1_term = 1;
      for( int j = 1; j < i; ++j ) f_term *= j;
      for( int j = i, f_div_val = 1; j < n; j += 2, f_div_val++ ) {
        const int f_lo = (j-i)/2;
        _val += t_fact * j * std::pow(-1,f_lo) * double(f_term) / double(i_fact) * coeff[j];
	//printf("%d, %d, %d, %d, %d\n",i,j,t_fact,m1_term,f_term);
        if(i>1) f_term = f_term * (i - f_div_val + 1);
	m1_term *= -1;
      }
#endif
    }

    coeff[i] = _val;
  }

}

void cheby_expand( size_t npts, int ncheb, const double* coeff, const double *x, 
		   double a, double b, double* eval ) {

  const int n = ncheb+1;
  for( size_t j = 0; j < npts; ++j ) {
    double xt = (2*x[j] - (a+b)) / (b-a);
    double wm2 = 1;
    double wm1 = xt;
    double _val = coeff[0] + coeff[1] * wm1;
    for( int i = 2; i < n; ++i ) {
      double w = 2 * xt * wm1 - wm2;
      _val += coeff[i] * w;
      wm2 = wm1;
      wm1 = w;
    }
    eval[j] = _val;
  }

}

void monomial_expand( size_t npts, int npoly, const double* coeff, const double *x, 
		      double a, double b, double* eval ) {

  const int n = npoly+1;
  const double sum = a+b;
  const double diff = b-a;
  const double ratio = sum / diff;
  const double fact = 2. / diff;
  for( size_t j = 0; j < npts; ++j ) {
    double xt = fact * x[j] - ratio;
    double xt_use = xt;
    double _val = coeff[0];
    for( int i = 1; i < n; ++i ) {
      _val += xt_use * coeff[i];
      xt_use *= xt;
    }
    eval[j] = _val;
  }

}



void generate_boys_table(int ncheb, int maxM, double maxT, int nseg, double* cheb_coeff_table, int ld) {
  // cheb_coeff_table is (ld, nseg, maxM+1) ld >= (ncheb+1)

  const double deltaT = maxT / nseg;
  for( int m = 0; m <= maxM; ++m ) {
    double* coeff_m = cheb_coeff_table + m * ld * nseg; // table offset for current m
    for( int iseg = 0; iseg < nseg; ++iseg ) {
      double* coeff_seg = coeff_m + iseg * ld;

      const double a = iseg * deltaT;
      const double b = a + deltaT;

      auto f = [=](double x){ return boys_reference(m,x); };
      cheby_coeff( ncheb, f, a, b, coeff_seg ); // Generate coeff in Chebyshev basis
      cheby_to_monomial_coeff( ncheb, coeff_seg );   // Convert to monomial basis
    }
  }

}

void boys_chebyshev( int npts, int m, const double* T, int ncheb, int nseg, double maxT, const double* boys_table, int ld, double* eval ) {
  const double* boys_m = boys_table + m * ld * nseg;

  const double deltaT = maxT / nseg;
  for( int i = 0; i < npts; ++i ) {
    const double tval = T[i];
    if( tval > maxT ) eval[i] = boys_asymp(m,tval);
    else {
      int iseg = std::floor( tval / deltaT);
      const double* boys_seg = boys_m + iseg * ld;

      const double a = iseg * deltaT;
      const double b = a + deltaT;
      monomial_expand( 1, ncheb, boys_seg, T+i, a, b, eval+i );
    }
  }
}

int main() {

#if 0
  const double maxT   = 117;
  const int    nseg   = 60;//maxT;
  const double deltaT = maxT / nseg;
  const int    ncheb  = 13;
  const int    ntest  = 100;


  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.,1.);
  auto rand_gen = [&](double a, double b) { return a + (b-a)*dist(gen); };

  double max_diff = -1;
  for( int m = 0;    m < 10;      ++m    ) 
  for( int iseg = 0; iseg < nseg; ++iseg ) {
    const double a = iseg * deltaT;
    const double b = a + deltaT;

    auto f = [=](double x){ return boys_reference(m,x); };
    auto coeff = cheby_coeff( ncheb, f, a, b );

    // Generate test points
    std::vector<double> x_test(ntest);
    auto rand_gen_ab = [&](){ return rand_gen(a,b); };
    std::generate( x_test.begin(), x_test.end(), rand_gen_ab );

    std::vector<double> f_cheb(ntest), f_ref(ntest);
    cheby_expand( ntest, ncheb, coeff.data(), x_test.data(), a, b, f_cheb.data() );
    std::transform( x_test.begin(), x_test.end(), f_ref.begin(), f );

    const auto cheby_diff = std::abs(f_cheb[0] - f_ref[0]) / std::abs(f_ref[0]);

    // Transform into monomial basis
    cheby_to_monomial_coeff( ncheb, coeff.data() );
    std::vector<double> f_monomial(ntest);
    monomial_expand( ntest, ncheb, coeff.data(), x_test.data(), a, b, f_monomial.data() );
    
    const auto monomial_diff = std::abs(f_monomial[0] - f_ref[0]) / std::abs(f_ref[0]);


    std::cout << m << ", " << a << ", " << b << ", " << cheby_diff << ", " << monomial_diff << std::endl;
    max_diff = std::max( cheby_diff, max_diff );
  }
  std::cout << max_diff << std::endl;

#else

  const double maxT   = 117;
  const int    maxM   = 10;
  const int    nseg   = 60;
  const int    ncheb  = 13;

  std::vector<double> boys_table( (ncheb+1)*nseg*(maxM+1) );
  generate_boys_table( ncheb, maxM, maxT, nseg, boys_table.data(), ncheb+1 );

  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.,1.);
  auto rand_gen = [&](double a, double b) { return a + (b-a)*dist(gen); };

  auto rand_gen_seg = [&](){ return rand_gen(0,maxT); };
  auto rand_gen_asy = [&](){ return rand_gen(maxT,maxT+20); };
  auto rand_gen_ran = [&](){ return rand_gen(0,maxT+20); };

  size_t ntest = 1000;
  std::vector<double> seg_pts( ntest ), asy_pts( ntest ), ran_pts( ntest );
  std::generate( seg_pts.begin(), seg_pts.end(), rand_gen_seg );
  std::generate( asy_pts.begin(), asy_pts.end(), rand_gen_asy );
  std::generate( ran_pts.begin(), ran_pts.end(), rand_gen_ran );



  std::vector<double> seg_eval( ntest ), asy_eval( ntest ), ran_eval( ntest );
  for( int m = 0; m <= maxM; ++m ) {
    boys_chebyshev( ntest, m, seg_pts.data(), ncheb, nseg, maxT, boys_table.data(), ncheb+1, seg_eval.data() );
    boys_chebyshev( ntest, m, asy_pts.data(), ncheb, nseg, maxT, boys_table.data(), ncheb+1, asy_eval.data() );
    boys_chebyshev( ntest, m, ran_pts.data(), ncheb, nseg, maxT, boys_table.data(), ncheb+1, ran_eval.data() );

    for( int i = 0; i < ntest; ++i ) {
      const auto seg_ref_val = boys_reference( m, seg_pts[i] );
      const auto asy_ref_val = boys_reference( m, asy_pts[i] );
      const auto ran_ref_val = boys_reference( m, ran_pts[i] );

      seg_eval[i] = std::abs( seg_eval[i] - seg_ref_val ) / seg_ref_val;
      asy_eval[i] = std::abs( asy_eval[i] - asy_ref_val ) / asy_ref_val;
      ran_eval[i] = std::abs( ran_eval[i] - ran_ref_val ) / ran_ref_val;
    }

    auto max_seg_diff = *std::max_element( seg_eval.begin(), seg_eval.end() );
    auto max_asy_diff = *std::max_element( asy_eval.begin(), asy_eval.end() );
    auto max_ran_diff = *std::max_element( ran_eval.begin(), ran_eval.end() );

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "m = " << std::setw(4) << m 
	    << " seg max = " << max_seg_diff
	    << " asy max = " << max_asy_diff
	    << " ran max = " << max_ran_diff << std::endl;
  }

#endif
}

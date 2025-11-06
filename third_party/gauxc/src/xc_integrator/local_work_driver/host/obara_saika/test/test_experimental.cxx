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
#include <libint2.hpp>
#include <integral_data_types.hpp>
#include <obara_saika_integrals.hpp>
#include <chebyshev_boys_computation.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <sys/time.h>
#include <float.h>

int main(int argc, char** argv) {
  libint2::initialize();

  double *boys_table = XCPU::boys_init();

  // Benzene
  std::vector<libint2::Atom> atoms = {
    libint2::Atom{ 6,  6.92768e-01,  -1.77656e+00,   1.40218e-03},
    libint2::Atom{ 6,  3.35108e+00,  -1.77668e+00,   2.21098e-03},
    libint2::Atom{ 6,  4.68035e+00,   5.25219e-01,   1.22454e-03},
    libint2::Atom{ 6,  3.35121e+00,   2.82744e+00,  -7.02978e-04},
    libint2::Atom{ 6,  6.93087e-01,   2.82756e+00,  -1.55902e-03},
    libint2::Atom{ 6, -6.36278e-01,   5.25491e-01,  -4.68652e-04},
    libint2::Atom{ 1, -3.41271e-01,  -3.56759e+00,   2.21287e-03},
    libint2::Atom{ 1,  4.38492e+00,  -3.56783e+00,   3.73599e-03},
    libint2::Atom{ 1,  6.74844e+00,   5.25274e-01,   1.88028e-03},
    libint2::Atom{ 1,  4.38551e+00,   4.61832e+00,  -1.48721e-03},
    libint2::Atom{ 1, -3.41001e-01,   4.61857e+00,  -3.05569e-03},
    libint2::Atom{ 1, -2.70437e+00,   5.25727e-01,  -1.09793e-03}
  };

  // Create cc-pVDZ BasisSet
  const std::string basis_name = "cc-pVDZ";
  libint2::BasisSet basis( basis_name, atoms );
  basis.set_pure(false); // Reset to Cartesian
  auto shell2bf = basis.shell2bf();

  auto [min_x, max_x] = std::minmax_element( atoms.begin(), atoms.end(),
    []( const auto& a, const auto& b) { return a.x < b.x; } );
  auto [min_y, max_y] = std::minmax_element( atoms.begin(), atoms.end(),
    []( const auto& a, const auto& b) { return a.y < b.y; } );
  auto [min_z, max_z] = std::minmax_element( atoms.begin(), atoms.end(),
    []( const auto& a, const auto& b) { return a.z < b.z; } );

  std::array<double,3> box_lo = { min_x->x, min_y->y, min_z->z };
  std::array<double,3> box_hi = { max_x->x, max_y->y, max_z->z };

  std::default_random_engine gen;
  std::uniform_real_distribution<double>
    dist_x( box_lo[0], box_hi[0] ),
    dist_y( box_lo[1], box_hi[1] ),
    dist_z( box_lo[2], box_hi[2] );

  auto gen_grid_point = [&]() {
    return std::array<double,3>{ dist_x(gen), dist_y(gen), dist_z(gen) };
  };

  if( argc != 2 ) throw std::runtime_error("Must Specify NGrid");

  const int ngrid = std::stoll( std::string(argv[1]) );

  std::vector< std::array<double,3> > grid_points( ngrid );
  std::generate( grid_points.begin(), grid_points.end(), gen_grid_point );

  const size_t nbf = basis.nbf();
  std::cout << "Running sn-LinK Proxy App with Settings:" << std::endl
	    << "  * NBF   = " << nbf << std::endl
	    << "  * NGRID = " << ngrid << std::endl
	    << std::endl;

  std::vector<libint2::Engine> engines;
  engines.reserve(ngrid);
  for( const auto& g : grid_points ) {
    engines.emplace_back( libint2::Operator::nuclear, basis.max_nprim(),
		          basis.max_l(), 0 );
    std::vector< std::pair<double, std::array<double,3>> > q = { {-1., g} };
    engines.back().set_params(q);
  }

  // Generate a random F matrix
  std::vector<double> F( ngrid * nbf );
  std::generate( F.begin(), F.end(), [&](){ return dist_x(gen); } );

  // Generate random grid weights
  std::vector<double> w( ngrid );
  std::generate( w.begin(), w.end(), [&](){ return dist_x(gen); } );

  // Compute A
  std::vector<double> A( nbf * nbf * ngrid );
  memset(&A[0], 0, nbf * nbf * ngrid * sizeof(double));

  using row_major_mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using const_row_major_map = Eigen::Map< const row_major_mat >;

  using col_major_mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using col_major_map = Eigen::Map< col_major_mat >;

  // correctness - libint implementation

  for( int k = 0; k < ngrid; ++k ) {
    auto& engine = engines.at(k);
    const auto& engine_buf = engine.results();

    col_major_map A_k( A.data() + nbf * nbf * k, nbf, nbf );

    for( int j = 0; j < basis.size(); ++j) {
      auto bf_j = shell2bf[j];
      auto nj   = basis[j].size();

      for( int i = 0; i < basis.size(); ++i) {
        auto bf_i = shell2bf[i];
        auto ni   = basis[i].size();

	engine.compute( basis[i], basis[j] );
	const_row_major_map buf_map( engine_buf[0], ni, nj );
	A_k.block( bf_i, bf_j, ni, nj ) = buf_map;
      }
    }
  }

  std::vector<double> G_libint( ngrid * nbf );
  for(int i = 0; i < ngrid * nbf; ++i) {
    G_libint[i] = 0.0;
  }
  for( int k = 0; k < ngrid; ++k ) {
    for( int i = 0; i < nbf; ++i ) {
      double tmp = 0.0;

      for( int j = 0; j < nbf; ++j )
        tmp += A[i + j * nbf + k * nbf * nbf] * F[j * ngrid + k];

      G_libint[ i * ngrid + k ] = w[k] * tmp;
    }
  }

  // correctness - own implementation

  std::vector<XCPU::point>  _points(ngrid);
  std::vector<double> _points_transposed(3 * ngrid);

  _points.resize(ngrid);

  for( int i = 0; i < ngrid; ++i ){
    _points[i].x = grid_points[i][0];
    _points[i].y = grid_points[i][1];
    _points[i].z = grid_points[i][2];

    _points_transposed[i + 0 * ngrid] = grid_points[i][0];
    _points_transposed[i + 1 * ngrid] = grid_points[i][1];
    _points_transposed[i + 2 * ngrid] = grid_points[i][2];
  }

  std::vector<XCPU::shells> _shells;

  int nshells = basis.size();

  _shells.resize(nshells);

  for( int i = 0; i < nshells; ++i ) {
    _shells[i].origin.x = basis[i].O[0];
    _shells[i].origin.y = basis[i].O[1];
    _shells[i].origin.z = basis[i].O[2];

    _shells[i].m = basis[i].alpha.size();
    _shells[i].L = basis[i].contr[0].l;

    _shells[i].coeff = new XCPU::coefficients[_shells[i].m];
    for( int j = 0; j < _shells[i].m; ++j ) {
      _shells[i].coeff[j].alpha = basis[i].alpha[j];
      _shells[i].coeff[j].coeff = basis[i].contr[0].coeff[j];
    }
  }

  int total_prim_pairs = 0;
  for( int i = 0; i < nshells; ++i) {
    for( int j = 0; j <= i; ++j) {
      total_prim_pairs += (_shells[i].m * _shells[j].m);
    }
  }

  XCPU::prim_pair *prim_pairs = new XCPU::prim_pair[total_prim_pairs];

  int offset = 0;
  for( int i = 0; i < nshells; ++i) {
    for( int j = 0; j <= i; ++j) {
      if( _shells[i].L >= _shells[j].L )
	XCPU::generate_shell_pair(_shells[i], _shells[j], (prim_pairs + offset));
      else
	XCPU::generate_shell_pair(_shells[j], _shells[i], (prim_pairs + offset));

      offset += (_shells[i].m * _shells[j].m);
    }
  }

  std::vector<double> G_own( ngrid * nbf );
  for(int i = 0; i < ngrid * nbf; ++i) {
    G_own[i] = 0.0;
  }
  double *Xi = F.data();
  double *Xj = F.data();

  double *Gi = G_own.data();
  double *Gj = G_own.data();

  struct timeval start, end;

  gettimeofday(&start, NULL);
  offset = 0;
  int ioff_cart = 0;
  for( int i = 0; i < nshells; ++i) {
    XCPU::shells bra_shell = _shells[i];
    int bra_cart_size = (bra_shell.L + 1) * (bra_shell.L + 2) / 2;

    int joff_cart = 0;
    for( int j = 0; j <= i; ++j) {
      XCPU::shells ket_shell = _shells[j];
      int ket_cart_size = (ket_shell.L + 1) * (ket_shell.L + 2) / 2;

      XCPU::compute_integral_shell_pair(i == j,
					ngrid,
					_points_transposed.data(),
					_shells[i].L,
					_shells[j].L,
					_shells[i].origin,
					_shells[j].origin,
					(_shells[i].m * _shells[j].m),
					(prim_pairs + offset),
					(Xi + ioff_cart * ngrid),
					(Xj + joff_cart * ngrid),
					ngrid,
					(Gi + ioff_cart * ngrid),
					(Gj + joff_cart * ngrid),
					ngrid,
					w.data(),
					boys_table);

      offset += (_shells[i].m * _shells[j].m);
      joff_cart += ket_cart_size;
    }

    ioff_cart += bra_cart_size;
  }

  gettimeofday(&end, NULL);

  int correct = 1;
  for( int i = 0; i < nbf * ngrid; ++i) {
    if((fabs(G_libint[i] - G_own[i]) > 1e-6) || std::isnan(G_own[i])) {
      printf("%lf %lf\n", G_libint[i], G_own[i]);
      correct = 0;
    }
  }

  std::cout << "Correctness: " << correct << "\tExecution: "<< 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) << std::endl;

  delete prim_pairs;

  libint2::finalize();  // done with libint
  XCPU::boys_finalize(boys_table);
}

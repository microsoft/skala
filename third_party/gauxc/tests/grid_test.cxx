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
#include "catch2/catch.hpp"
#include <gauxc/grid.hpp>

#include <integratorxx/quadratures/radial/muraknowles.hpp>
#include <integratorxx/quadratures/radial/mhl.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>

#include <random>

using namespace GauXC;

TEST_CASE("Grid", "[grid]") {

  std::default_random_engine gen;
  std::uniform_real_distribution<> pos_real_dist( 0., 10. );

  int64_t n_rad    = 99;
  int64_t n_ang    = 770;
  int64_t batch_sz = 128;

  double r_scal = pos_real_dist(gen);

  IntegratorXX::LebedevLaikov<double>            ang_quad( n_ang         );
  IntegratorXX::MuraKnowles<double,double>       mk_quad ( n_rad, r_scal );

  using sphere_type = IntegratorXX::SphericalQuadrature<
    IntegratorXX::MuraKnowles<double,double>, IntegratorXX::LebedevLaikov<double>
  >;

  RadialQuad rquad = RadialQuad::MuraKnowles;
  auto mk_sphere = std::make_shared<sphere_type>( mk_quad, ang_quad  );
  auto mk_batch  = IntegratorXX::make_batcher( batch_sz, *mk_sphere  );

  SECTION("Full Construction") {

    Grid grid( mk_sphere, BatchSize(batch_sz) );
    CHECK( grid.batcher().max_batch_size() == batch_sz );

    for( auto i = 0; i < mk_batch.nbatches(); ++i ) {

      auto&& [box_lo_ref, box_up_ref, points_ref, weights_ref] = mk_batch.at(i);
      auto&& [box_lo, box_up, points, weights] = grid.batcher().at(i);

      CHECK( box_lo_ref == box_lo );
      CHECK( box_up_ref == box_up );

      CHECK( points_ref  == points  );
      CHECK( weights_ref == weights );

    }

  }

#if 0
    SECTION("Default Batch Size") {
      Grid grid( rquad, RadialSize(n_rad), AngularSize(n_ang), 
                 RadialScale(r_scal) );

      CHECK( grid.max_batch_sz() == 512ll );
    }
#endif

}

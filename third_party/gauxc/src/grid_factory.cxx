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
#include <gauxc/grid_factory.hpp>

#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <integratorxx/quadratures/radial/muraknowles.hpp>
#include <integratorxx/quadratures/radial/mhl.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/radial/becke.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <gauxc/exceptions.hpp>

namespace GauXC {

/*****************/
/**** Visitor ****/
/*****************/
Grid AtomicGridFactory::generate_grid( atomic_grid_variant gs, BatchSize bsz ) {
  return std::visit( [=](auto&& s){ return generate_grid(s, bsz); }, gs );
}

/************************/
/**** Unpruned Grids ****/
/************************/

Grid AtomicGridFactory::generate_unpruned_grid( RadialQuad rq, RadialSize nrad, 
  AngularSize nang, RadialScale rscal, BatchSize bsz) {

  using bk_type  = IntegratorXX::Becke<double, double>;
  using mk_type  = IntegratorXX::MuraKnowles<double,double>;
  using mhl_type = IntegratorXX::MurrayHandyLaming<double,double>;
  using ta_type  = IntegratorXX::TreutlerAhlrichs<double,double>;
  using ll_type  = IntegratorXX::LebedevLaikov<double>;

  ll_type ang_quad( nang.get() );

  switch( rq ) {
    case RadialQuad::Becke:
      return generate_unpruned_grid( bk_type(nrad.get(), rscal.get()),
        std::move(ang_quad), bsz );

    case RadialQuad::MuraKnowles:
      return generate_unpruned_grid( mk_type(nrad.get(), rscal.get()),
        std::move(ang_quad), bsz );

    case RadialQuad::MurrayHandyLaming:
      return generate_unpruned_grid( mhl_type(nrad.get(), rscal.get()),
        std::move(ang_quad), bsz );

    case RadialQuad::TreutlerAhlrichs:
      return generate_unpruned_grid( ta_type(nrad.get(), rscal.get()),
        std::move(ang_quad), bsz );

    default:
      GAUXC_GENERIC_EXCEPTION("Unsupported Radial Quadrature");
      abort();

  }

}

Grid AtomicGridFactory::generate_grid( UnprunedAtomicGridSpecification gs, BatchSize bsz ) {
  return generate_unpruned_grid( gs.radial_quad, gs.radial_size, gs.angular_size,
    gs.radial_scale, bsz );
}




/**********************/
/**** Pruned Grids ****/
/**********************/

template <typename RadialQuad, 
  typename AngularQuad = IntegratorXX::LebedevLaikov<double>>
auto make_pruned_grid(RadialSize nrad, 
  const std::vector<PruningRegion>& pruning_regions,
  RadialScale rscal ) {

  RadialQuad rq(nrad.get(), rscal.get());
  IntegratorXX::RadialGridPartition<AngularQuad> rgp;
  for( auto& region : pruning_regions ) {
    rgp.add_quad( rq, region.idx_st, 
      AngularQuad(region.angular_size.get()) );
  }
  rgp.finalize(rq);

  return std::make_tuple( rq, rgp );

}

Grid AtomicGridFactory::generate_pruned_grid( RadialQuad rq, 
  RadialSize nrad, const std::vector<PruningRegion>& pruning_regions, 
  RadialScale rscal, BatchSize bsz) {

  using mk_type  = IntegratorXX::MuraKnowles<double,double>;
  using mhl_type = IntegratorXX::MurrayHandyLaming<double,double>;
  using ta_type  = IntegratorXX::TreutlerAhlrichs<double,double>;

  switch( rq ) {

    case RadialQuad::MuraKnowles:
    {
      auto [rg, rgp] = 
        make_pruned_grid<mk_type>( nrad, pruning_regions, rscal );
      return generate_pruned_grid(std::move(rg), std::move(rgp), bsz);
    }

    case RadialQuad::MurrayHandyLaming:
    {
      auto [rg, rgp] = 
        make_pruned_grid<mhl_type>( nrad, pruning_regions, rscal );
      return generate_pruned_grid(std::move(rg), std::move(rgp), bsz);
    }

    case RadialQuad::TreutlerAhlrichs:
    {
      auto [rg, rgp] = 
        make_pruned_grid<ta_type>( nrad, pruning_regions, rscal );
      return generate_pruned_grid(std::move(rg), std::move(rgp), bsz);
    }
    case RadialQuad::Becke:
    {
      auto[rg, rgp] = 
        make_pruned_grid<IntegratorXX::Becke<double,double>>( nrad, pruning_regions, rscal );
        return generate_pruned_grid(std::move(rg), std::move(rgp), bsz);
    }

    default:
      GAUXC_GENERIC_EXCEPTION("Unsupported Radial Quadrature");
      abort();

  }

}

Grid AtomicGridFactory::generate_grid( PrunedAtomicGridSpecification gs, BatchSize bsz ) {
  return generate_pruned_grid( gs.radial_quad, gs.radial_size, 
    gs.pruning_regions, gs.radial_scale, bsz );
}



/***************************
 * Default Pruning Schemes *
 ***************************/


PrunedAtomicGridSpecification robust_psi4_pruning_scheme(
  UnprunedAtomicGridSpecification unp ) {

  // Look up order
  // XXX: THIS ONLY WORKS FOR LEBEDEV
  using angular_type = IntegratorXX::LebedevLaikov<double>; 
  using traits = IntegratorXX::quadrature_traits<angular_type>;
  const auto asz = unp.angular_size.get();
  const auto base_order = traits::algebraic_order_by_npts(asz);
  if( base_order < 0 ) GAUXC_GENERIC_EXCEPTION("Invalid Base Grid");

  const auto med_order = 
    traits::next_algebraic_order(base_order > 6 ? base_order-6 : base_order);
  const auto low_order = 7;

  AngularSize med_sz(traits::npts_by_algebraic_order(med_order));
  AngularSize low_sz(traits::npts_by_algebraic_order(low_order));

  // Create Pruning Regions
  const size_t rsz = unp.radial_size.get();
  const size_t r_div_4 = rsz / 4ul + 1ul;
  const size_t r_div_2 = rsz / 2ul + 1ul;
  std::vector<PruningRegion> pruning_regions = {
    {0ul,     r_div_4, low_sz},
    {r_div_4, r_div_2, med_sz},
    {r_div_2,     rsz, unp.angular_size}
  };

  return PrunedAtomicGridSpecification{
    unp.radial_quad, unp.radial_size, unp.radial_scale, pruning_regions
  };
  
}



PrunedAtomicGridSpecification treutler_pruning_scheme(
  UnprunedAtomicGridSpecification unp ) {

  const size_t med_order = 11;
  const size_t low_order = 7;

  // Look up order
  // XXX: THIS ONLY WORKS FOR LEBEDEV
  using angular_type = IntegratorXX::LebedevLaikov<double>;
  using traits = IntegratorXX::quadrature_traits<angular_type>;

  AngularSize med_sz(traits::npts_by_algebraic_order(med_order));
  AngularSize low_sz(traits::npts_by_algebraic_order(low_order));

  // Create Pruning Regions
  const size_t rsz = unp.radial_size.get();
  const size_t r_div_3 = rsz / 3ul + 1ul;
  const size_t r_div_2 = rsz / 2ul + 1ul;
  std::vector<PruningRegion> pruning_regions = {
    {0ul,     r_div_3, low_sz},
    {r_div_3, r_div_2, med_sz},
    {r_div_2, rsz,     unp.angular_size}
  };

  return PrunedAtomicGridSpecification{
    unp.radial_quad, unp.radial_size, unp.radial_scale, pruning_regions
  };
  
}


PrunedAtomicGridSpecification create_pruned_spec(
  PruningScheme scheme, UnprunedAtomicGridSpecification unp
) {

  switch(scheme) {
    case PruningScheme::Robust:
      return robust_psi4_pruning_scheme(unp);
    case PruningScheme::Treutler:
      return treutler_pruning_scheme(unp);
    
    // Default to Unpruned Grid
    case PruningScheme::Unpruned:
    default:
      std::vector<PruningRegion> pruning_regions = {
        {0ul, (size_t)unp.radial_size.get(), unp.angular_size}
      };
      return PrunedAtomicGridSpecification{
        unp.radial_quad, unp.radial_size, unp.radial_scale, pruning_regions
      };
  }

}

}

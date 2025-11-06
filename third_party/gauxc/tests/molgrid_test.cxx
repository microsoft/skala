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
#include <gauxc/molgrid.hpp>
#include <gauxc/molgrid/defaults.hpp>


#include <random>

using namespace GauXC;

TEST_CASE("MolGrid Defaults", "[molgrid]") {

  SECTION("MK Defaults") {

#if 0
    auto default_scaling = 
      get_default_scaling_factors( RadialQuad::MuraKnowles, AtomicNumber(100) );
     
    for( auto [Z, alpha] : default_scaling ) {

      switch( Z.get() ) {
        case 3:
        case 4:
        case 11:
        case 12:
        case 19:
        case 20:
        case 37:
        case 38:
        case 55:
        case 56:
        case 87:
        case 88:
          CHECK( alpha == 7.0 );
          break;
        default:
          CHECK( alpha == 5.0 );
          break;

      }

    }
#else
    for( auto i = 0; i < 100; ++i ) {
      AtomicNumber Z(i);
      auto alpha = 
        default_radial_scaling_factor(RadialQuad::MuraKnowles,Z).get();
      switch(i) {
        case 3:
        case 4:
        case 11:
        case 12:
        case 19:
        case 20:
        case 37:
        case 38:
        case 55:
        case 56:
        case 87:
        case 88:
          CHECK( alpha == 7.0 );
          break;
        default:
          CHECK( alpha == 5.0 );
          break;
      }
    }
#endif

  }


  SECTION("MHL Defaults") {


    for( auto i = 0; i < 100; ++i ) {
      AtomicNumber Z(i);
      auto alpha = 
        default_radial_scaling_factor(RadialQuad::MurrayHandyLaming,Z).get();
      switch(i) {
        case    1: CHECK( alpha == Approx(4.7243153124839748e-01)); break;
        case    2: CHECK( alpha == Approx(2.9290754937400643e-01)); break;
        case    3: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case    4: CHECK( alpha == Approx(9.9210621562163470e-01)); break;
        case    5: CHECK( alpha == Approx(8.0313360312227566e-01)); break;
        case    6: CHECK( alpha == Approx(6.6140414374775647e-01)); break;
        case    7: CHECK( alpha == Approx(6.1416099062291674e-01)); break;
        case    8: CHECK( alpha == Approx(5.6691783749807700e-01)); break;
        case    9: CHECK( alpha == Approx(4.7243153124839748e-01)); break;
        case   10: CHECK( alpha == Approx(3.5904796374878206e-01)); break;
        case   11: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   12: CHECK( alpha == Approx(1.4172945937451924e+00)); break;
        case   13: CHECK( alpha == Approx(1.1810788281209936e+00)); break;
        case   14: CHECK( alpha == Approx(1.0393493687464743e+00)); break;
        case   15: CHECK( alpha == Approx(9.4486306249679497e-01)); break;
        case   16: CHECK( alpha == Approx(9.4486306249679497e-01)); break;
        case   17: CHECK( alpha == Approx(9.4486306249679497e-01)); break;
        case   18: CHECK( alpha == Approx(6.7085277437272439e-01)); break;
        case   19: CHECK( alpha == Approx(2.0786987374929486e+00)); break;
        case   20: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   21: CHECK( alpha == Approx(1.5117808999948719e+00)); break;
        case   22: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   23: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   24: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   25: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   26: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   27: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   28: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   29: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   30: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   31: CHECK( alpha == Approx(1.2283219812458335e+00)); break;
        case   32: CHECK( alpha == Approx(1.1810788281209936e+00)); break;
        case   33: CHECK( alpha == Approx(1.0865925218713142e+00)); break;
        case   34: CHECK( alpha == Approx(1.0865925218713142e+00)); break;
        case   35: CHECK( alpha == Approx(1.0865925218713142e+00)); break;
        case   36: CHECK( alpha == Approx(8.3147949499717955e-01)); break;
        case   37: CHECK( alpha == Approx(2.2204281968674682e+00)); break;
        case   38: CHECK( alpha == Approx(1.8897261249935899e+00)); break;
        case   39: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   40: CHECK( alpha == Approx(1.4645377468700322e+00)); break;
        case   41: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case   42: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case   43: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   44: CHECK( alpha == Approx(1.2283219812458335e+00)); break;
        case   45: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   46: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   47: CHECK( alpha == Approx(1.5117808999948719e+00)); break;
        case   48: CHECK( alpha == Approx(1.4645377468700322e+00)); break;
        case   49: CHECK( alpha == Approx(1.4645377468700322e+00)); break;
        case   50: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case   51: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case   52: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   53: CHECK( alpha == Approx(1.3228082874955129e+00)); break;
        case   54: CHECK( alpha == Approx(1.0204521074965385e+00)); break;
        case   55: CHECK( alpha == Approx(2.5038871156165063e+00)); break;
        case   56: CHECK( alpha == Approx(2.0314555843681092e+00)); break;
        case   57: CHECK( alpha == Approx(1.8424829718687501e+00)); break;
        case   58: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   59: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   60: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   61: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   62: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   63: CHECK( alpha == Approx(1.7479966656190706e+00)); break;
        case   64: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   65: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   66: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   67: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   68: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   69: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   70: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   71: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   72: CHECK( alpha == Approx(1.4645377468700322e+00)); break;
        case   73: CHECK( alpha == Approx(1.3700514406203526e+00)); break;
        case   74: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   75: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   76: CHECK( alpha == Approx(1.2283219812458335e+00)); break;
        case   77: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   78: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   79: CHECK( alpha == Approx(1.2755651343706731e+00)); break;
        case   80: CHECK( alpha == Approx(1.4172945937451924e+00)); break;
        case   81: CHECK( alpha == Approx(1.7952398187439103e+00)); break;
        case   82: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   83: CHECK( alpha == Approx(1.5117808999948719e+00)); break;
        case   84: CHECK( alpha == Approx(1.7952398187439103e+00)); break;
        case   85: CHECK( alpha == Approx(1.1999760893709295e+00)); break;
        case   86: CHECK( alpha == Approx(1.1338356749961540e+00)); break;
        case   88: CHECK( alpha == Approx(2.0314555843681092e+00)); break;
        case   89: CHECK( alpha == Approx(1.8424829718687501e+00)); break;
        case   90: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   91: CHECK( alpha == Approx(1.7007535124942308e+00)); break;
        case   92: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   93: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   94: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   95: CHECK( alpha == Approx(1.6535103593693912e+00)); break;
        case   96: CHECK( alpha == Approx(1.8991750000000001e+00)); break;
        case   97: CHECK( alpha == Approx(1.8991750000000001e+00)); break;
        case   98: CHECK( alpha == Approx(1.8991750000000001e+00)); break;
        case   99: CHECK( alpha == Approx(1.8991750000000001e+00)); break;
        case  100: CHECK( alpha == Approx(1.8991750000000001e+00)); break;
      }
    }

  }


  SECTION("Grid Size") {
    for(auto i = 0; i < 100; ++i) {
      auto [fgr, fga] = 
        default_grid_size( AtomicNumber(i), RadialQuad::MuraKnowles,
          AtomicGridSizeDefault::FineGrid );
      REQUIRE( fgr.get() == 75 );
      REQUIRE( fga.get() == 302 );


      auto [ufgr, ufga] = 
        default_grid_size( AtomicNumber(i), RadialQuad::MuraKnowles,
          AtomicGridSizeDefault::UltraFineGrid );
      REQUIRE( ufgr.get() == 99 );
      REQUIRE( ufga.get() == 590 );
      

      auto [sfgr, sfga] = 
        default_grid_size( AtomicNumber(i), RadialQuad::MuraKnowles,
          AtomicGridSizeDefault::SuperFineGrid );
      REQUIRE( sfga.get() == 974 );
      if( i <= 2 ) REQUIRE( sfgr.get() == 175 );
      else         REQUIRE( sfgr.get() == 250 );

    }
  }

}


TEST_CASE("Grid Specification", "[molgrid]") {

  AtomicNumber Z(6);
  auto rq  = RadialQuad::MuraKnowles;
  auto gsz = AtomicGridSizeDefault::UltraFineGrid;
  RadialSize rsz; AngularSize asz; // Clang doesn't like lambda capture of structured bindings...
  std::tie(rsz, asz) = default_grid_size(Z,rq,gsz);
  auto rscal = default_radial_scaling_factor(rq, Z);

  SECTION("Unpruned") {
    UnprunedAtomicGridSpecification gs;
    SECTION("From Sizes") {
      gs = MolGridFactory::create_default_unpruned_grid_spec(Z,rq,rsz,asz);
    }
    SECTION("From Standard") {
      gs = MolGridFactory::create_default_unpruned_grid_spec(Z,rq,gsz);
    }
    REQUIRE( gs.radial_quad  == rq    );
    REQUIRE( gs.radial_size  == rsz   );
    REQUIRE( gs.radial_scale == rscal );
    REQUIRE( gs.angular_size == asz   );
  }

  SECTION("Pruned") {

    atomic_grid_variant gs;
    std::vector<PruningRegion> ref_pruning_regions;
    UnprunedAtomicGridSpecification unp_gs = 
      MolGridFactory::create_default_unpruned_grid_spec(Z,rq,gsz);
    SECTION("Unpruned") {
      gs = MolGridFactory::create_default_pruned_grid_spec(
        PruningScheme::Unpruned,Z,rq,gsz);
      ref_pruning_regions = {
        {0ul, (size_t)rsz.get(), asz}
      };
    }

    SECTION("Robust") {
      gs = MolGridFactory::create_default_pruned_grid_spec(
        PruningScheme::Robust,Z,rq,gsz);
      size_t rs = rsz.get();
      size_t r4 = rsz.get() / 4ul + 1;
      size_t r2 = rsz.get() / 2ul + 1;
      ref_pruning_regions = {
        {0ul, r4, AngularSize(26)},
        {r4,  r2, AngularSize(434)},
        {r2,  rs, AngularSize(590)}
      };
    }

    SECTION("Treutler") {
      gs = MolGridFactory::create_default_pruned_grid_spec(
        PruningScheme::Treutler,Z,rq,gsz);
      size_t rs = rsz.get();
      size_t r3 = rsz.get() / 3ul + 1;
      size_t r2 = rsz.get() / 2ul + 1;
      ref_pruning_regions = {
        {0ul, r3, AngularSize(26)},
        {r3,  r2, AngularSize(50)},
        {r2,  rs, AngularSize(590)}
      };
    }

    std::visit([=](auto& g) {
      REQUIRE( g.radial_quad  == rq    );
      REQUIRE( g.radial_size  == rsz   );
      REQUIRE( g.radial_scale == rscal );
    }, gs);

    // Pruned Check
    if( const auto* pru = std::get_if<PrunedAtomicGridSpecification>(&gs) ) {
      REQUIRE( pru->pruning_regions == ref_pruning_regions );
    } 

  }

}



#if 0

TEST_CASE("MolGrid", "[molgrid]") {

  // Water
  Molecule mol;
  mol.emplace_back(AtomicNumber(1), 0., 1.579252144093028,  2.174611055780858);
  mol.emplace_back(AtomicNumber(8), 0., 0.000000000000000,  0.000000000000000);
  mol.emplace_back(AtomicNumber(1), 0., 1.579252144093028, -2.174611055780858);

  auto mk_scaling = 
    default_radial_scaling_factors( RadialQuad::MuraKnowles, AtomicNumber(8) );
  auto mhl_scaling = 
    default_radial_scaling_factors( RadialQuad::MurrayHandyLaming, AtomicNumber(8) );

  auto [fg_radial_size, fg_angular_size] =
    default_grid_size( AtomicGridSizeDefault::FineGrid, AtomicNumber(8) );
  auto [ufg_radial_size, ufg_angular_size] =
    default_grid_size( AtomicGridSizeDefault::UltraFineGrid, AtomicNumber(8) );
  auto [sfg_radial_size, sfg_angular_size] =
    default_grid_size( AtomicGridSizeDefault::SuperFineGrid, AtomicNumber(8) );

  SECTION("MK") {

    SECTION("Explicit Construction") {

      MolGrid mg( RadialQuad::MuraKnowles, fg_size, mk_scaling, mol );
      
      CHECK( mg.natoms_uniq() == 2 );

      for( const auto& atom : mol ) {
        auto alpha = mk_scaling[atom.Z];
        auto gsz   = fg_size[atom.Z];

        Grid atom_grid( RadialQuad::MuraKnowles, gsz, alpha );

        //CHECK( mg.get_rscal_factor(atom.Z) == alpha );
        CHECK( mg.get_grid_size( atom. Z ) == gsz   );

        CHECK( atom_grid.batcher().quadrature().points() ==
               mg.get_grid(atom.Z).batcher().quadrature().points() );
        CHECK( atom_grid.batcher().quadrature().weights() ==
               mg.get_grid(atom.Z).batcher().quadrature().weights() );

      }

    }

    SECTION("Default Scaling Factors") {

      MolGrid mg( RadialQuad::MuraKnowles, fg_size, mol );
      
      CHECK( mg.natoms_uniq() == 2 );

      for( const auto& atom : mol ) {
        auto alpha = mk_scaling[atom.Z];
        auto gsz   = fg_size[atom.Z];

        Grid atom_grid( RadialQuad::MuraKnowles, gsz, alpha );

        //CHECK( mg.get_rscal_factor(atom.Z) == alpha );
        CHECK( mg.get_grid_size( atom. Z ) == gsz   );

        CHECK( atom_grid.batcher().quadrature().points() ==
               mg.get_grid(atom.Z).batcher().quadrature().points() );
        CHECK( atom_grid.batcher().quadrature().weights() ==
               mg.get_grid(atom.Z).batcher().quadrature().weights() );

      }

    }

    SECTION("Named Default Grid Size") {

      MolGrid mg( RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid, 
        mk_scaling, mol );
      
      CHECK( mg.natoms_uniq() == 2 );

      for( const auto& atom : mol ) {
        auto alpha = mk_scaling[atom.Z];
        auto gsz   = ufg_size[atom.Z];

        Grid atom_grid( RadialQuad::MuraKnowles, gsz, alpha );

        //CHECK( mg.get_rscal_factor(atom.Z) == alpha );
        CHECK( mg.get_grid_size( atom. Z ) == gsz   );

        CHECK( atom_grid.batcher().quadrature().points() ==
               mg.get_grid(atom.Z).batcher().quadrature().points() );
        CHECK( atom_grid.batcher().quadrature().weights() ==
               mg.get_grid(atom.Z).batcher().quadrature().weights() );

      }

    }

    SECTION("Named Default Grid Size + Default Scaling") {

      MolGrid mg( RadialQuad::MuraKnowles, AtomicGridSizeDefault::SuperFineGrid, 
        mol );
      
      CHECK( mg.natoms_uniq() == 2 );

      for( const auto& atom : mol ) {
        auto alpha = mk_scaling[atom.Z];
        auto gsz   = sfg_size[atom.Z];

        Grid atom_grid( RadialQuad::MuraKnowles, gsz, alpha );

        //CHECK( mg.get_rscal_factor(atom.Z) == alpha );
        CHECK( mg.get_grid_size( atom. Z ) == gsz   );

        CHECK( atom_grid.batcher().quadrature().points() ==
               mg.get_grid(atom.Z).batcher().quadrature().points() );
        CHECK( atom_grid.batcher().quadrature().weights() ==
               mg.get_grid(atom.Z).batcher().quadrature().weights() );

      }

    }

  }

  SECTION("MHL") {

    MolGrid mg( RadialQuad::MurrayHandyLaming, fg_size, mhl_scaling, mol );
    
    CHECK( mg.natoms_uniq() == 2 );

    for( const auto& atom : mol ) {
      auto alpha = mhl_scaling[atom.Z];
      auto gsz   = fg_size[atom.Z];

      Grid atom_grid( RadialQuad::MurrayHandyLaming, gsz, alpha );

      //CHECK( mg.get_rscal_factor(atom.Z) == alpha );
      CHECK( mg.get_grid_size( atom. Z ) == gsz   );

      CHECK( atom_grid.batcher().quadrature().points() ==
             mg.get_grid(atom.Z).batcher().quadrature().points() );
      CHECK( atom_grid.batcher().quadrature().weights() ==
             mg.get_grid(atom.Z).batcher().quadrature().weights() );

    }

  }

#if 0
  SECTION("Default") {
    MolGrid mg( AtomicGridSizeDefault::FineGrid, mol );
    for( const auto& atom: mol )
      CHECK( mg.get_radial_quad(atom.Z) == RadialQuad::MuraKnowles );
  }
#endif

}
#endif




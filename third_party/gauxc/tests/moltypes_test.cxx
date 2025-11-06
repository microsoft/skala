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
#include "ut_common.hpp"
#include "catch2/catch.hpp"
#include <gauxc/molecule.hpp>
#include <gauxc/molmeta.hpp>
#include <gauxc/external/hdf5.hpp>
//#include <filesystem>
#include <fstream>

#include "standards.hpp"

#include <random>

#include <gauxc/gauxc_config.hpp>

#ifdef GAUXC_HAS_MPI
#include <mpi.h>
#endif

using namespace GauXC;

TEST_CASE("Atom", "[moltypes]") {

  double x = 0.2, y = 0.4, z = 6.5;
  int64_t Z = 10;

  Atom atom{ AtomicNumber(Z), x, y, z };

  CHECK( atom.Z.get() == Z );
  CHECK( atom.x       == x );
  CHECK( atom.y       == y );
  CHECK( atom.z       == z );

}

TEST_CASE("Molecule", "[moltypes]") {

  std::default_random_engine gen;
  std::uniform_real_distribution<> real_dist( -10., 10 );
  std::uniform_int_distribution<>  int_dist(1,10);

  size_t natoms_gen = 40;

  SECTION("Default") {

    Molecule mol;

    CHECK(mol.natoms() == 0);
  }

  SECTION("From std::vector<Atom>") {

    std::vector<Atom> atoms;
    
    for( auto i = 0; i < natoms_gen; ++i )
      atoms.push_back( {
        AtomicNumber(int_dist(gen)),
        real_dist(gen),
        real_dist(gen),
        real_dist(gen)
      });

    SECTION("Copy") {

      Molecule mol( atoms );

      CHECK( mol.natoms() == natoms_gen );
      CHECK( atoms.size() == natoms_gen );

      for( auto i = 0; i < natoms_gen; ++i ) 
        CHECK( atoms[i] == mol[i] );
      
    }

    SECTION("Move") {

      std::vector<Atom> atoms_copy(atoms);

      Molecule mol( std::move(atoms) );

      CHECK( mol.natoms() == natoms_gen );
      CHECK( atoms.size() == 0 );

      for( auto i = 0; i < natoms_gen; ++i ) 
        CHECK( atoms_copy[i] == mol[i] );
      
    }

  }

  

  SECTION("Inplace Construction") {

    Molecule mol;
    std::vector<Atom> atoms;

    for(auto i = 0ul; i < natoms_gen; ++i ) {

      AtomicNumber Z( int_dist(gen) );
      double x(real_dist(gen));
      double y(real_dist(gen));
      double z(real_dist(gen));

      atoms.push_back({Z,x,y,z});
      mol.push_back({Z,x,y,z});

    }

    CHECK( mol.natoms() == natoms_gen );
    
    for( auto i = 0; i < natoms_gen; ++i ) 
      CHECK( atoms[i] == mol[i] );

  }

  SECTION("Copy ctor") {

    std::vector<Atom> atoms{Atom(AtomicNumber(1), 0.0, 0.0, 0.0)};
    Molecule mol(atoms);

    Molecule mol_copy(mol);
    CHECK(mol == mol_copy);
  }


  SECTION("Move ctor") {

    std::vector<Atom> atoms{Atom(AtomicNumber(1), 0.0, 0.0, 0.0)};
    Molecule mol(atoms);

    Molecule mol_copy(mol);
    Molecule mol_move(std::move(mol));
    CHECK(mol_move == mol_copy);
  }

}


TEST_CASE( "MolMeta", "[moltypes]" ) {

  // Water
  Molecule mol = make_water();
  MolMeta meta( mol );

  std::vector<double> rab          = meta.rab();
  std::vector<double> dist_nearest = meta.dist_nearest();

  std::vector<double> rab_ref = {
    0.00000000000, 2.68755847909, 4.34922211156,
    2.68755847909, 0.00000000000, 2.68755847909,
    4.34922211156, 2.68755847909, 0.00000000000 
  };

  for( auto i = 0; i < mol.natoms() * mol.natoms(); ++i )
    CHECK( rab[i] == Approx(rab_ref[i]) );

  for( auto i = 0; i < mol.natoms(); ++i )
    CHECK( dist_nearest[i] == Approx(2.68755847909) );
  

}

TEST_CASE("HDF5-MOLECULE", "[moltypes]") {

#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
  if( world_rank ) return; // Only run on root rank
#endif


  Molecule mol = make_water();
  
  // Write file
  const std::string fname = GAUXC_REF_DATA_PATH "/test_mol.hdf5";
  //if( std::filesystem::exists(fname) ) std::filesystem::remove(fname);
  auto file_exists = [](const auto& f ) {
    std::ifstream file(f); return file.good();
  };
  if(file_exists(fname)) std::remove(fname.c_str());

  write_hdf5_record( mol, fname , "/MOL" );

  // Read File
  Molecule mol_read;
  read_hdf5_record( mol_read, fname, "/MOL" );

  // Check that IO was correct
  CHECK( mol == mol_read );

  //std::filesystem::remove(fname); // Delete the test file
  std::remove(fname.c_str());

}

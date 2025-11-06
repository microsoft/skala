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

#include <gauxc/molgrid.hpp>

namespace GauXC {

  double slater_radius_64(AtomicNumber);
  double slater_radius_30(AtomicNumber);
  double clementi_radius_67(AtomicNumber);
  double uff_radius_103(AtomicNumber);
  double default_atomic_radius(AtomicNumber);

  RadialScale default_mk_radial_scaling_factor( AtomicNumber );
  RadialScale default_mhl_radial_scaling_factor( AtomicNumber );
  RadialScale default_ta_radial_scaling_factor( AtomicNumber );
  RadialScale default_radial_scaling_factor( RadialQuad, AtomicNumber );

  std::tuple<RadialSize,AngularSize> 
    default_grid_size(AtomicNumber, RadialQuad, AtomicGridSizeDefault); 

  struct MolGridFactory {

    static UnprunedAtomicGridSpecification create_default_unpruned_grid_spec(
      AtomicNumber, RadialQuad, RadialSize, AngularSize
    );

    static UnprunedAtomicGridSpecification create_default_unpruned_grid_spec(
      AtomicNumber, RadialQuad, AtomicGridSizeDefault
    );

    template <typename... Args>
    inline static atomic_grid_variant 
      create_default_pruned_grid_spec( PruningScheme scheme, Args&&... args ) {
      return create_pruned_spec( scheme, 
        create_default_unpruned_grid_spec(std::forward<Args>(args)...)
      );
    }

    template <typename... Args>
    inline static atomic_grid_spec_map create_default_grid_spec_map( 
      const Molecule& mol, PruningScheme scheme, Args&&... args ) {

      atomic_grid_spec_map molmap;
      for( const auto& atom : mol ) 
      if( !molmap.count(atom.Z) ) {
        molmap.emplace( atom.Z, 
          create_default_pruned_grid_spec(scheme, atom.Z, 
            std::forward<Args>(args)...)
        );
      }

      return molmap;
    }

    inline static atomic_grid_map generate_gridmap(
      const atomic_grid_spec_map& gs_map, BatchSize bsz ) {

      atomic_grid_map molmap;
      for( const auto& [key, val] : gs_map ) {
        molmap.emplace( key, AtomicGridFactory::generate_grid(val, bsz) );
      }
      return molmap;

    }

    template <typename... Args>
    inline static atomic_grid_map create_default_gridmap( 
      const Molecule& mol, PruningScheme scheme, BatchSize bsz,
      Args&&... args ) {

      return generate_gridmap( create_default_grid_spec_map(mol, scheme, 
        std::forward<Args>(args)...), bsz );

    }

    template <typename... Args>
    inline static MolGrid create_default_molgrid( Args&&... args ) {
      return MolGrid( create_default_gridmap(std::forward<Args>(args)...) );
    }

  };

}


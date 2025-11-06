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
#include "molgrid_impl.hpp"
#include <gauxc/exceptions.hpp>

namespace GauXC {
namespace detail {

MolGridImpl::MolGridImpl( const atomic_grid_map& ag ) :
  molgrid_( ag ) { }
  

MolGridImpl::MolGridImpl( const MolGridImpl& )     = default;
MolGridImpl::MolGridImpl( MolGridImpl&& ) noexcept = default;
MolGridImpl::~MolGridImpl() noexcept = default;

size_t MolGridImpl::natoms_uniq() const {
  return molgrid_.size();
}
const Grid& MolGridImpl::get_grid( AtomicNumber Z ) const {
  return molgrid_.at(Z);
}
Grid& MolGridImpl::get_grid( AtomicNumber Z ) {
  return molgrid_.at(Z);
}

size_t MolGridImpl::max_nbatches() const {

  return std::max_element( molgrid_.begin(), molgrid_.end(),
  []( const auto &a, const auto& b ) {
    return a.second.batcher().nbatches() < 
           b.second.batcher().nbatches();
  } )->second.batcher().nbatches();

}





#if 0
void MolGridImpl::generate( RadialQuad rq, const Molecule& mol ) { 

  std::vector<AtomicNumber> Zs; Zs.reserve( mol.natoms() );
  for( const auto& atom : mol ) Zs.emplace_back( atom.Z );

  std::sort(Zs.begin(),Zs.end(),
    [](auto& a, auto& b) { return a.get() < b.get(); }
  );
  auto zuniq_it = std::unique( Zs.begin(), Zs.end() );
  Zs.erase( zuniq_it, Zs.end() );
  Zs.shrink_to_fit();

  molgrid_.clear();
  for( auto Z : Zs ) {

    auto gsz_it = grid_sizes_.find( Z );
    if( gsz_it == grid_sizes_.end() )
      GAUXC_GENERIC_EXCEPTION("Grid Size Map Does Not Contain Z = " + 
        std::to_string( Z.get() )
      );

    auto [Rsz, Asz] = gsz_it->second;

    auto rscl_it = scal_factors_.find( Z );
    if( rscl_it == scal_factors_.end() )
      GAUXC_GENERIC_EXCEPTION("Scaling Factor Map Does Not Contain Z = " + 
        std::to_string( Z.get() )
      );

    auto alpha = rscl_it->second;

    molgrid_.insert_or_assign( Z, Grid( rq, Rsz, Asz, alpha ) );

  }

}
#endif


}
}

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
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/load_balancer.hpp>


using namespace GauXC;

struct ref_weights_data {
  Molecule                  mol;
  std::shared_ptr<MolMeta>  meta;
  std::vector< XCTask > tasks_unm;
  std::vector< XCTask > tasks_mod; // This is only the weights

  template <typename Archive>
  void load( Archive& ar ) {
    ar( mol, tasks_unm, tasks_mod );
    meta = std::make_shared<MolMeta>(mol);
  }
  template <typename Archive>
  void save( Archive& ar ) const {
    ar( mol, tasks_unm, tasks_mod );
  }
};


#ifdef GAUXC_HAS_HOST
#include "host/reference/weights.hpp"

void generate_weights_data( const Molecule& mol, const BasisSet<double>& basis,
                            std::ofstream& out_file, XCWeightAlg weight_alg,
                            size_t ntask_save = 15 ) {


  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
  auto mg = MolGridFactory::create_default_molgrid(mol, PruningScheme::Unpruned,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::FineGrid);

  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);
  auto& tasks = lb.get_tasks();

  ref_weights_data   ref_data;
  ref_data.mol       = mol;

  auto abs_comparator = []( const auto& a, const auto& b ) {
    return std::abs(a) < std::abs(b);
  };

  std::sort( tasks.begin(), tasks.end(),
    [&]( const auto& a, const auto& b ) {
      auto a_max =
        *std::max_element( a.weights.begin(), a.weights.end(),
                           abs_comparator );
      auto b_max =
        *std::max_element( b.weights.begin(), b.weights.end(),
                           abs_comparator );

      return a_max < b_max;
    });

  if( tasks.size() > ntask_save )
    tasks.erase( tasks.begin() + ntask_save, tasks.end() );

  ref_data.tasks_unm = tasks; // Make a copy of un modified tasks


  switch( weight_alg ) {
    case XCWeightAlg::Becke:
      reference_becke_weights_host( 
        mol, lb.molmeta(), tasks.begin(), tasks.end() );
      break;
    case XCWeightAlg::SSF:
      reference_ssf_weights_host( 
        mol, lb.molmeta(), tasks.begin(), tasks.end() );
      break;
    case XCWeightAlg::LKO:
      reference_lko_weights_host( 
        mol, lb.molmeta(), tasks.begin(), tasks.end() );
      break;
  }

  // Clear out unneeded data
  for( auto& task : tasks ) {
    task.points.clear();
    task.bfn_screening.shell_list.clear();
  }
  ref_data.tasks_mod = tasks;

  {
    cereal::BinaryOutputArchive ar( out_file );
    ar( ref_data );
  }

}
#endif

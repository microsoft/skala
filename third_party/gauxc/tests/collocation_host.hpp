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
#ifdef GAUXC_HAS_HOST
#include "collocation_common.hpp"
#include "host/reference/collocation.hpp"

void generate_collocation_data( const Molecule& mol, const BasisSet<double>& basis,
                                std::ofstream& out_file, size_t ntask_save = 10 ) {


  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
  auto mg = MolGridFactory::create_default_molgrid(mol, PruningScheme::Unpruned,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::FineGrid);

  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance( rt, mol, mg, basis);
  auto& tasks = lb.get_tasks();


  std::vector< ref_collocation_data > ref_data;

  for( size_t i = 0; i < ntask_save; ++i ) {
    auto& task = tasks[i];

    auto& pts  = task.points;
    auto& mask = task.bfn_screening.shell_list;

    // Only keep first MAX_NPTS_CHECK points to save on space
    if( task.points.size() > MAX_NPTS_CHECK )
      task.points.erase( task.points.begin() + MAX_NPTS_CHECK, task.points.end() );

    const auto npts = task.points.size();
    const auto nbf  = task.bfn_screening.nbe;

    std::vector<double> eval   ( nbf * npts ),
                        deval_x( nbf * npts ),
                        deval_y( nbf * npts ),
                        deval_z( nbf * npts ),
                        d2eval_xx( nbf * npts ),
                        d2eval_xy( nbf * npts ),
                        d2eval_xz( nbf * npts ),
                        d2eval_yy( nbf * npts ),
                        d2eval_yz( nbf * npts ),
                        d2eval_zz( nbf * npts ),
                        d3eval_xxx( nbf * npts ),
                        d3eval_xxy( nbf * npts ),
                        d3eval_xxz( nbf * npts ),
                        d3eval_xyy( nbf * npts ),
                        d3eval_xyz( nbf * npts ),
                        d3eval_xzz( nbf * npts ),
                        d3eval_yyy( nbf * npts ),
                        d3eval_yyz( nbf * npts ),
                        d3eval_yzz( nbf * npts ),
                        d3eval_zzz( nbf * npts );

    gau2grid_collocation_der3( npts, mask.size(), nbf,
      pts.data()->data(), basis, mask.data(), eval.data(), 
      deval_x.data(), deval_y.data(), deval_z.data(),
      d2eval_xx.data(), d2eval_xy.data(), d2eval_xz.data(),
      d2eval_yy.data(), d2eval_yz.data(), d2eval_zz.data(),
      d3eval_xxx.data(), d3eval_xxy.data(), d3eval_xxz.data(),
      d3eval_xyy.data(), d3eval_xyz.data(), d3eval_xzz.data(),
      d3eval_yyy.data(), d3eval_yyz.data(), d3eval_yzz.data(),
      d3eval_zzz.data());

    std::vector<double> d2eval_lapl(nbf * npts);
    std::vector<double> d3eval_lapl_x(nbf * npts);
    std::vector<double> d3eval_lapl_y(nbf * npts);
    std::vector<double> d3eval_lapl_z(nbf * npts);
    for(auto i = 0; i < nbf*npts; ++i) {
      d2eval_lapl[i] = d2eval_xx[i] + d2eval_yy[i] + d2eval_zz[i];
      d3eval_lapl_x[i] = d3eval_xxx[i] + d3eval_xyy[i] + d3eval_xzz[i];
      d3eval_lapl_y[i] = d3eval_xxy[i] + d3eval_yyy[i] + d3eval_yzz[i];
      d3eval_lapl_z[i] = d3eval_xxz[i] + d3eval_yyz[i] + d3eval_zzz[i];
    }

    

    auto max_abs = *std::max_element( eval.begin(), eval.end(),
                   [](auto a, auto b){ return std::abs(a) < std::abs(b); } );
    if( std::abs(max_abs) < 1e-9 ) continue;

    ref_collocation_data d{ std::move(mask), std::move(pts), std::move(eval),
                            std::move(deval_x), std::move(deval_y), std::move(deval_z),
                            std::move(d2eval_xx), std::move(d2eval_xy), std::move(d2eval_xz),
                            std::move(d2eval_yy), std::move(d2eval_yz), std::move(d2eval_zz),
                            std::move(d2eval_lapl), std::move(d3eval_lapl_x), std::move(d3eval_lapl_y),
                            std::move(d3eval_lapl_z)
                            };

    ref_data.emplace_back( std::move(d) );

  }

  {
    cereal::BinaryOutputArchive ar( out_file );
    ar( ref_data );
  }

}


void test_host_collocation( const BasisSet<double>& basis, std::ifstream& in_file) {



  std::vector<ref_collocation_data> ref_data;

  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  for( auto& d : ref_data ) {

    const auto npts = d.pts.size();
    const auto nbf  = d.eval.size() / npts;

    const auto& mask = d.mask;
    const auto& pts  = d.pts;

    std::vector<double> eval( nbf * npts );


    gau2grid_collocation( npts, mask.size(), nbf,
                          pts.data()->data(), basis,
                          mask.data(),
                          eval.data() );

    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( eval[i] == Approx( d.eval[i] ) );

  }

}

void test_host_collocation_deriv1( const BasisSet<double>& basis, std::ifstream& in_file) {



  std::vector<ref_collocation_data> ref_data;

  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  for( auto& d : ref_data ) {

    const auto npts = d.pts.size();
    const auto nbf  = d.eval.size() / npts;

    const auto& mask = d.mask;
    const auto& pts  = d.pts;

    std::vector<double> eval   ( nbf * npts ),
                        deval_x( nbf * npts ),
                        deval_y( nbf * npts ),
                        deval_z( nbf * npts );


    gau2grid_collocation_gradient( npts, mask.size(), nbf,
                                   pts.data()->data(), basis,
                                   mask.data(),
                                   eval.data(), deval_x.data(),
                                   deval_y.data(), deval_z.data() );

    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( eval[i] == Approx( d.eval[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_x[i] == Approx( d.deval_x[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_y[i] == Approx( d.deval_y[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_z[i] == Approx( d.deval_z[i] ) );
  }

}

void test_host_collocation_deriv2( const BasisSet<double>& basis, std::ifstream& in_file) {



  std::vector<ref_collocation_data> ref_data;

  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  for( auto& d : ref_data ) {

    const auto npts = d.pts.size();
    const auto nbf  = d.eval.size() / npts;

    const auto& mask = d.mask;
    const auto& pts  = d.pts;

    std::vector<double> eval   ( nbf * npts ),
                        deval_x( nbf * npts ),
                        deval_y( nbf * npts ),
                        deval_z( nbf * npts ),
                        d2eval_xx( nbf * npts ),
                        d2eval_xy( nbf * npts ),
                        d2eval_xz( nbf * npts ),
                        d2eval_yy( nbf * npts ),
                        d2eval_yz( nbf * npts ),
                        d2eval_zz( nbf * npts );


    gau2grid_collocation_hessian( npts, mask.size(), nbf,
      pts.data()->data(), basis, mask.data(), eval.data(), 
      deval_x.data(), deval_y.data(), deval_z.data(),
      d2eval_xx.data(), d2eval_xy.data(), d2eval_xz.data(),
      d2eval_yy.data(), d2eval_yz.data(), d2eval_zz.data() );

    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( eval[i] == Approx( d.eval[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_x[i] == Approx( d.deval_x[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_y[i] == Approx( d.deval_y[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( deval_z[i] == Approx( d.deval_z[i] ) );

    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_xx[i] == Approx( d.d2eval_xx[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_xy[i] == Approx( d.d2eval_xy[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_xz[i] == Approx( d.d2eval_xz[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_yy[i] == Approx( d.d2eval_yy[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_yz[i] == Approx( d.d2eval_yz[i] ) );
    for( auto i = 0; i < npts * nbf; ++i )
      CHECK( d2eval_zz[i] == Approx( d.d2eval_zz[i] ) );
  }

}
#endif

// SkalaXC C++ example: evaluate a UKS machine-learning exchange-correlation
// functional on a system + density loaded from an HDF5 file.
//
// Consumption contract: this program includes ONLY <skalaxc/skalaxc.hpp> from
// SkalaXC and has no access to GauXC or LibTorch. Eigen owns the caller-side
// matrices, while HighFive loads the density input.
//
// The program follows the GauXC-style host pipeline mirrored by SkalaXC:
//   RuntimeEnvironment -> Molecule / BasisSet -> MolGrid -> LoadBalancer ->
//   MolecularWeights -> functional_type -> XCIntegratorFactory -> XCIntegrator
//
// Usage: skalaxc_eval_cpp <system.hdf5> [model]
//   <system.hdf5>  HDF5 file with /MOLECULE, /BASIS, /DENSITY_SCALAR,
//                  /DENSITY_Z
//   [model]        "LDA", "PBE" (default), "TPSS", or a path to a .fun model

#include <skalaxc/skalaxc.hpp>

#include <Eigen/Core>
#include <highfive/H5File.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <system.hdf5> [model]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const std::string model = argc > 2 ? argv[2] : "PBE";

#ifdef SKALAXC_HAS_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    return 1;
  }
#endif

  int status = 0;
  try {
    // 1. Runtime environment.
#ifdef SKALAXC_HAS_MPI
    SkalaXC::RuntimeEnvironment rt(MPI_COMM_WORLD);
#else
    SkalaXC::RuntimeEnvironment rt;
#endif

    // 2. Load the molecule and basis set from the file.
    SkalaXC::Molecule mol;
    SkalaXC::read_hdf5_record(mol, path, "/MOLECULE");
    SkalaXC::BasisSet<double> basis;
    SkalaXC::read_hdf5_record(basis, path, "/BASIS");

    // 3. Build the default molecular integration grid.
    auto mg = SkalaXC::MolGridFactory::create_default_molgrid(
        mol, SkalaXC::PruningScheme::Unpruned, SkalaXC::BatchSize(512),
        SkalaXC::RadialQuad::MuraKnowles,
        SkalaXC::AtomicGridSizeDefault::UltraFineGrid);

    // 4. Distribute the quadrature tasks.
    SkalaXC::LoadBalancerFactory lb_factory(SkalaXC::ExecutionSpace::Host);
    auto lb = lb_factory.get_instance(rt, mol, mg, basis);

    // 5. Partition the quadrature weights in place.
    SkalaXC::MolecularWeightsFactory mw_factory(
        SkalaXC::ExecutionSpace::Host, "Default",
        SkalaXC::MolecularWeightsSettings{});
    auto mw = mw_factory.get_instance();
    mw.modify_weights(lb);

    // 6. Select the Skala ML functional model.
    SkalaXC::functional_type func(model);

    // 7. Build the XC integrator over the caller's matrix type.
    SkalaXC::XCIntegratorFactory<Matrix> xc_factory(
        SkalaXC::ExecutionSpace::Host);
    auto integrator = xc_factory.get_instance(func, lb);

    // 8. Load the input spin densities. Here we read them from the same file;
    //    in a real driver they come from your SCF.
    const std::int64_t nbf = basis.nbf();
    HighFive::File file(path, HighFive::File::ReadOnly);
    Matrix Ps(nbf, nbf), Pz(nbf, nbf);
    file.getDataSet("/DENSITY_SCALAR").read(Ps.data());
    file.getDataSet("/DENSITY_Z").read(Pz.data());

    // 9. Evaluate the ML exchange-correlation energy and potential (UKS).
    auto [EXC, VXCs, VXCz] = integrator.eval_exc_vxc(Ps, Pz);

    // 10. Report the energy and a summary of the potential.
    std::printf("model=%s nbf=%lld EXC=%.10f |VXC_scalar|_F=%.6e\n",
                model.c_str(), (long long)nbf, EXC, VXCs.norm());
  } catch (const SkalaXC::Exception& e) {
    std::fprintf(stderr, "SkalaXC error: %s\n", e.what());
    status = 1;
  }

#ifdef SKALAXC_HAS_MPI
  MPI_Finalize();
#endif
  return status;
}

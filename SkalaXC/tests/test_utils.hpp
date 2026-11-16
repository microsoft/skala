#pragma once

#include <skalaxc/skalaxc.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <string>

namespace SkalaXC::test {

struct MolecularSystem {
  Molecule molecule;
  BasisSet<double> basis;
};

struct UksDensity {
  Eigen::MatrixXd scalar;
  Eigen::MatrixXd spin;
};

BasisSet<double> make_sto3g_hydrogen_basis(const Molecule& molecule);

MolecularSystem make_rotated_h2_sto3g_system(double x_displacement = 0.0);

MolGrid make_molgrid(const Molecule& molecule, AtomicGridSizeDefault grid_size,
                     std::int64_t batch_size = 512,
                     PruningScheme pruning = PruningScheme::Unpruned,
                     RadialQuad radial = RadialQuad::MuraKnowles);

#ifdef SKALAXC_HAS_HDF5
MolecularSystem load_molecular_system(const std::string& fixture);

UksDensity load_uks_density(const std::string& fixture,
                            const std::string& scalar_dataset,
                            const std::string& spin_dataset);
#endif

double matrix_error_per_basis(const Eigen::MatrixXd& actual,
                              const Eigen::MatrixXd& reference);

}  // namespace SkalaXC::test
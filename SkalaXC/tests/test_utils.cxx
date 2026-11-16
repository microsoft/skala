#include "test_utils.hpp"

#ifdef SKALAXC_HAS_HDF5
#include <highfive/H5File.hpp>
#endif

#include <stdexcept>
#include <utility>

namespace SkalaXC::test {

#ifdef SKALAXC_HAS_HDF5
namespace {

Eigen::MatrixXd load_square_matrix(HighFive::File& file,
                                   const std::string& dataset) {
  const auto data = file.getDataSet(dataset);
  const auto dimensions = data.getDimensions();
  if (dimensions.size() != 2 || dimensions[0] != dimensions[1])
    throw std::runtime_error("Expected a square matrix in " + dataset);

  Eigen::MatrixXd matrix(static_cast<Eigen::Index>(dimensions[0]),
                         static_cast<Eigen::Index>(dimensions[1]));
  data.read(matrix.data());
  return matrix;
}

}  // namespace
#endif

BasisSet<double> make_sto3g_hydrogen_basis(const Molecule& molecule) {
  Shell<double>::prim_array exponents{};
  Shell<double>::prim_array coefficients{};
  exponents[0] = 3.42525091;
  exponents[1] = 0.62391373;
  exponents[2] = 0.16885540;
  coefficients[0] = 0.15432897;
  coefficients[1] = 0.53532814;
  coefficients[2] = 0.44463454;

  BasisSet<double> basis;
  for (const auto& atom : molecule)
    basis.emplace_back(PrimSize(3), AngularMomentum(0), SphericalType(false),
                       exponents, coefficients,
                       Shell<double>::cart_array{atom.x, atom.y, atom.z});
  return basis;
}

MolecularSystem make_rotated_h2_sto3g_system(double x_displacement) {
  Molecule molecule;
  molecule.emplace_back(AtomicNumber(1), -0.252 - x_displacement, 0.336, -0.56);
  molecule.emplace_back(AtomicNumber(1), 0.252 + x_displacement, -0.336, 0.56);
  auto basis = make_sto3g_hydrogen_basis(molecule);
  return MolecularSystem{std::move(molecule), std::move(basis)};
}

MolGrid make_molgrid(const Molecule& molecule, AtomicGridSizeDefault grid_size,
                     std::int64_t batch_size, PruningScheme pruning,
                     RadialQuad radial) {
  return MolGridFactory::create_default_molgrid(
      molecule, pruning, BatchSize(batch_size), radial, grid_size);
}

#ifdef SKALAXC_HAS_HDF5
MolecularSystem load_molecular_system(const std::string& fixture) {
  MolecularSystem system;
  read_hdf5_record(system.molecule, fixture, "/MOLECULE");
  read_hdf5_record(system.basis, fixture, "/BASIS");
  return system;
}

UksDensity load_uks_density(const std::string& fixture,
                            const std::string& scalar_dataset,
                            const std::string& spin_dataset) {
  HighFive::File file(fixture, HighFive::File::ReadOnly);
  UksDensity density;
  density.scalar = load_square_matrix(file, scalar_dataset);
  density.spin =
      spin_dataset.empty()
          ? Eigen::MatrixXd::Zero(density.scalar.rows(), density.scalar.cols())
          : load_square_matrix(file, spin_dataset);
  if (density.spin.rows() != density.scalar.rows() ||
      density.spin.cols() != density.scalar.cols())
    throw std::runtime_error("Scalar and spin density dimensions differ");
  return density;
}
#endif

double matrix_error_per_basis(const Eigen::MatrixXd& actual,
                              const Eigen::MatrixXd& reference) {
  return (actual - reference).norm() / static_cast<double>(reference.rows());
}

}  // namespace SkalaXC::test
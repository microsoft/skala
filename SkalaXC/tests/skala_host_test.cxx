// SkalaXC host reference-integration harness (white-box: uses GauXC types
// in-build).
//
// Reads the HDF5 reference fixtures produced by the GauXC/skala OneDFT test
// suite (renamed skala_*), drives SkalaHostDriver::eval_exc_vxc_uks, and checks
// EXC + VXC against the stored reference values. This numerically validates the
// host ML port independently of any public SkalaXC API.

#include <gauxc/basisset.hpp>
#include <gauxc/basisset_map.hpp>
#include <gauxc/external/hdf5.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/util/mpi.hpp>

#include <highfive/H5File.hpp>

#include "skala_host_driver.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <skalaxc/skalaxc.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SkalaXC;

namespace {

/** @brief Numerical errors produced by one host EXC/VXC reference evaluation.
 */
struct CaseResult {
  double exc, exc_ref, exc_rel_err;
  double vxcs_err, vxcz_err;  // frobenius(diff) / nbf
  int64_t nbf;
};

/**
 * @brief Evaluate one reference fixture through the standalone host driver.
 * @param fixture HDF5 file containing the system, density, and EXC/VXC values.
 * @param model Skala model identifier.
 * @param rt Runtime environment used by the host driver.
 * @return Computed values and normalized errors against the fixture.
 */
CaseResult run_case(const std::string& fixture, const std::string& model,
                    const GauXC::RuntimeEnvironment& rt,
                    SkalaXC::TimingSettings timing_settings = {}) {

  GauXC::Molecule mol;
  GauXC::BasisSet<double> basis;
  GauXC::read_hdf5_record(mol, fixture, "/MOLECULE");
  GauXC::read_hdf5_record(basis, fixture, "/BASIS");

  HighFive::File file(fixture, HighFive::File::ReadOnly);
  auto dsetP = file.getDataSet("/DENSITY_SCALAR");
  auto dims = dsetP.getDimensions();
  const int64_t nbf = static_cast<int64_t>(dims[0]);
  const size_t n2 = dims[0] * dims[1];

  std::vector<double> Ps(n2), Pz(n2), VXCs_ref(n2), VXCz_ref(n2);
  dsetP.read(Ps.data());
  file.getDataSet("/DENSITY_Z").read(Pz.data());
  file.getDataSet("/VXC_SCALAR").read(VXCs_ref.data());
  file.getDataSet("/VXC_Z").read(VXCz_ref.data());
  double EXC_ref = 0.0;
  file.getDataSet("/EXC").read(&EXC_ref);

  auto mg = GauXC::MolGridFactory::create_default_molgrid(
      mol, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles,
      GauXC::AtomicGridSizeDefault::UltraFineGrid);

  auto driver = SkalaHostDriver::from_system(rt, mol, mg, basis, model,
                                             DomainBatchMode::Conservative,
                                             timing_settings);

  std::vector<double> VXCs(n2, 0.0), VXCz(n2, 0.0);

  const Eigen::Map<const ColMajorMatrix> scalar_density(Ps.data(), nbf, nbf);
  const Eigen::Map<const ColMajorMatrix> spin_density(Pz.data(), nbf, nbf);
  Eigen::Map<ColMajorMatrix> scalar_potential(VXCs.data(), nbf, nbf);
  Eigen::Map<ColMajorMatrix> spin_potential(VXCz.data(), nbf, nbf);
  const double EXC = driver.eval_exc_vxc_uks(scalar_density, spin_density,
                                             scalar_potential, spin_potential);

  auto frob_diff = [nbf](const std::vector<double>& a,
                         const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double d = a[i] - b[i];
      s += d * d;
    }
    return std::sqrt(s) / static_cast<double>(nbf);
  };

  CaseResult r;
  r.nbf = nbf;
  r.exc = EXC;
  r.exc_ref = EXC_ref;
  r.exc_rel_err = std::abs(EXC - EXC_ref) / std::max(1.0, std::abs(EXC_ref));
  r.vxcs_err = frob_diff(VXCs, VXCs_ref);
  r.vxcz_err = frob_diff(VXCz, VXCz_ref);
  return r;
}

/** @brief Ridders derivative estimate and its local extrapolation error. */
struct RiddersResult {
  double derivative;
  double error;
};

/**
 * @brief Estimate a scalar function's derivative at zero with Ridders' method.
 *
 * Each row starts from a centered difference with step
 * `initial_step / ratio^row`. Richardson extrapolation then cancels successive
 * even powers of the step. The returned estimate is the table entry with the
 * smallest change relative to its two direct predecessors.
 *
 * @tparam Function Callable accepting a signed displacement and returning the
 * scalar function value.
 * @param function Function to differentiate.
 * @param initial_step Largest centered-difference displacement.
 * @param levels Number of centered differences and extrapolation rows.
 * @param ratio Factor by which the step decreases between rows.
 * @return Best derivative and conservative local error estimate found.
 */
template <typename Function>
RiddersResult ridders_derivative(Function&& function, double initial_step,
                                 int levels = 3, double ratio = 1.4) {
  std::vector<std::vector<double>> table(levels,
                                         std::vector<double>(levels, 0.0));
  RiddersResult result{0.0, std::numeric_limits<double>::infinity()};

  double step = initial_step;
  for (int row = 0; row < levels; ++row) {
    table[row][0] = (function(step) - function(-step)) / (2.0 * step);
    if (row == 0) result.derivative = table[row][0];

    double factor = ratio * ratio;
    for (int column = 1; column <= row; ++column) {
      table[row][column] =
          (factor * table[row][column - 1] - table[row - 1][column - 1]) /
          (factor - 1.0);
      const double error =
          std::max(std::abs(table[row][column] - table[row][column - 1]),
                   std::abs(table[row][column] - table[row - 1][column - 1]));
      if (error < result.error) {
        result.derivative = table[row][column];
        result.error = error;
      }
      factor *= ratio * ratio;
    }
    step /= ratio;
  }
  return result;
}

/**
 * @brief Evaluate EXC after a collective nuclear displacement.
 *
 * Nuclear coordinates and their atom-centered basis shells move together so
 * the finite difference contains the same basis Pulay contribution as the
 * analytic gradient. The AO density coefficients are intentionally held fixed.
 * A fresh molecular grid and host driver are built for every displacement.
 *
 * @param rt Runtime environment used by the host driver.
 * @param molecule Undisplaced molecular geometry.
 * @param basis Undisplaced atom-centered AO basis.
 * @param shell_centers Map from each shell to its parent atom.
 * @param direction Atom-major Cartesian displacement direction.
 * @param displacement Signed displacement magnitude along `direction`.
 * @param nbf Number of AO basis functions.
 * @param Ps Fixed scalar-spin AO density matrix.
 * @param Pz Fixed z-spin AO density matrix.
 * @param settings Skala model settings.
 * @param grid_size Atomic integration-grid preset.
 * @return Exchange-correlation energy at the displaced geometry.
 */
double displaced_exc(const GauXC::RuntimeEnvironment& rt,
                     const GauXC::Molecule& molecule,
                     const GauXC::BasisSet<double>& basis,
                     const std::vector<int32_t>& shell_centers,
                     const std::vector<double>& direction, double displacement,
                     int64_t nbf, const std::vector<double>& Ps,
                     const std::vector<double>& Pz, const std::string& model,
                     GauXC::AtomicGridSizeDefault grid_size) {
  GauXC::Molecule displaced_molecule = molecule;
  GauXC::BasisSet<double> displaced_basis = basis;
  for (std::size_t atom = 0; atom < displaced_molecule.size(); ++atom) {
    displaced_molecule[atom].x += displacement * direction[3 * atom];
    displaced_molecule[atom].y += displacement * direction[3 * atom + 1];
    displaced_molecule[atom].z += displacement * direction[3 * atom + 2];
  }

  // Moving shells with their parent nuclei is essential for the Pulay term.
  for (std::size_t shell = 0; shell < displaced_basis.size(); ++shell) {
    const int32_t atom = shell_centers[shell];
    if (atom < 0) throw std::runtime_error("Basis shell is not atom-centered");
    for (int xyz = 0; xyz < 3; ++xyz)
      displaced_basis[shell].O()[xyz] +=
          displacement * direction[3 * atom + xyz];
  }

  auto grid = GauXC::MolGridFactory::create_default_molgrid(
      displaced_molecule, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles, grid_size);
  auto driver = SkalaHostDriver::from_system(rt, displaced_molecule, grid,
                                             displaced_basis, model);
  std::vector<double> VXCs(Ps.size()), VXCz(Ps.size());
  const Eigen::Map<const ColMajorMatrix> scalar_density(Ps.data(), nbf, nbf);
  const Eigen::Map<const ColMajorMatrix> spin_density(Pz.data(), nbf, nbf);
  Eigen::Map<ColMajorMatrix> scalar_potential(VXCs.data(), nbf, nbf);
  Eigen::Map<ColMajorMatrix> spin_potential(VXCz.data(), nbf, nbf);
  return driver.eval_exc_vxc_uks(scalar_density, spin_density, scalar_potential,
                                 spin_potential);
}

}  // namespace

TEST_CASE("Skala host reference integration",
          "[skala][host-reference-integration]") {
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_WORLD)};

  const std::string ref_dir = std::string(SKALAXC_TEST_REF_DATA_PATH);

  struct Fixture {
    const char* name;
    const char* file;
    const char* model;
  };
  const std::vector<Fixture> fixtures = {
      {"HE / def2-qzvp / lda", "/skala_he_def2qzvp_lda_uks.hdf5", "LDA"},
      {"HE / def2-qzvp / pbe", "/skala_he_def2qzvp_pbe_uks.hdf5", "PBE"},
      {"HE / def2-qzvp / tpss", "/skala_he_def2qzvp_tpss_uks.hdf5", "TPSS"},
  };

  // Reference thresholds (mirror the GauXC/skala reference test exactly):
  //   EXC == Approx(EXC_ref);  |VXC_SCALAR-ref|/nbf < 1e-7;  |VXC_Z-ref|/nbf <
  //   1e-10.
  const double exc_rel_tol = 1e-5;
  const double vxcs_tol = 1e-7;
  const double vxcz_tol = 1e-10;

  for (const auto& fx : fixtures) {
    SECTION(fx.name) {
      const CaseResult r = run_case(ref_dir + fx.file, fx.model, rt);

      INFO("nbf=" << static_cast<long long>(r.nbf));
      INFO("EXC=" << r.exc << " EXC_ref=" << r.exc_ref);
      INFO("exc_rel_err=" << r.exc_rel_err);
      INFO("vxcs_err=" << r.vxcs_err);
      INFO("vxcz_err=" << r.vxcz_err);

      CHECK(r.exc_rel_err < exc_rel_tol);
      CHECK(r.vxcs_err < vxcs_tol);
      CHECK(r.vxcz_err < vxcz_tol);
    }
  }
}

TEST_CASE("Debug logging preserves host results", "[skala][debug-logging]") {
  GauXC::RuntimeEnvironment runtime{GAUXC_MPI_CODE(MPI_COMM_WORLD)};
  const std::string fixture = std::string(SKALAXC_TEST_REF_DATA_PATH) +
                              "/skala_he_def2qzvp_lda_uks.hdf5";
  const auto quiet = run_case(fixture, "LDA", runtime);
  SkalaXC::TimingSettings settings;
  settings.debug_logging = true;
  const auto logged = run_case(fixture, "LDA", runtime, settings);
  CHECK(logged.exc == Catch::Approx(quiet.exc).epsilon(1e-13));
  CHECK(logged.vxcs_err == Catch::Approx(quiet.vxcs_err).margin(1e-13));
  CHECK(logged.vxcz_err == Catch::Approx(quiet.vxcz_err).margin(1e-13));
}

TEST_CASE("Host domain batching modes are numerically equivalent",
          "[skala][host-batching]") {
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_WORLD)};
  const std::string fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";

  GauXC::Molecule molecule;
  GauXC::BasisSet<double> basis;
  GauXC::read_hdf5_record(molecule, fixture, "/MOLECULE");
  GauXC::read_hdf5_record(basis, fixture, "/BASIS");
  HighFive::File file(fixture, HighFive::File::ReadOnly);
  auto density_data = file.getDataSet("/DENSITY");
  const auto dimensions = density_data.getDimensions();
  const Eigen::Index nbf = static_cast<Eigen::Index>(dimensions[0]);
  std::vector<double> scalar_density_data(dimensions[0] * dimensions[1]);
  std::vector<double> spin_density_data(scalar_density_data.size(), 0.0);
  density_data.read(scalar_density_data.data());

  auto grid = GauXC::MolGridFactory::create_default_molgrid(
      molecule, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles,
      GauXC::AtomicGridSizeDefault::UltraFineGrid);
  auto conservative = SkalaHostDriver::from_system(
      rt, molecule, grid, basis, "PBE", DomainBatchMode::Conservative);
  auto aggressive = SkalaHostDriver::from_system(
      rt, molecule, grid, basis, "PBE", DomainBatchMode::Aggressive);

  const Eigen::Map<const ColMajorMatrix> scalar_density(
      scalar_density_data.data(), nbf, nbf);
  const Eigen::Map<const ColMajorMatrix> spin_density(spin_density_data.data(),
                                                      nbf, nbf);
  ColMajorMatrix conservative_scalar(nbf, nbf), conservative_spin(nbf, nbf);
  ColMajorMatrix aggressive_scalar(nbf, nbf), aggressive_spin(nbf, nbf);
  const double conservative_exc = conservative.eval_exc_vxc_uks(
      scalar_density, spin_density,
      ColMajorMatrixMap(conservative_scalar.data(), nbf, nbf),
      ColMajorMatrixMap(conservative_spin.data(), nbf, nbf));
  const double aggressive_exc = aggressive.eval_exc_vxc_uks(
      scalar_density, spin_density,
      ColMajorMatrixMap(aggressive_scalar.data(), nbf, nbf),
      ColMajorMatrixMap(aggressive_spin.data(), nbf, nbf));

  RowMajorMatrix conservative_gradient(molecule.natoms(), 3);
  RowMajorMatrix aggressive_gradient(molecule.natoms(), 3);
  conservative.eval_exc_grad_uks(
      scalar_density, spin_density,
      RowMajorMatrixMap(conservative_gradient.data(), molecule.natoms(), 3));
  aggressive.eval_exc_grad_uks(
      scalar_density, spin_density,
      RowMajorMatrixMap(aggressive_gradient.data(), molecule.natoms(), 3));

  CHECK(aggressive_exc == Catch::Approx(conservative_exc).epsilon(1e-12));
  CHECK(aggressive_scalar.isApprox(conservative_scalar, 1e-11));
  CHECK(aggressive_spin.isApprox(conservative_spin, 1e-11));
  CHECK(aggressive_gradient.isApprox(conservative_gradient, 1e-10));
  const auto conservative_diagnostics = conservative.diagnostics();
  const auto aggressive_diagnostics = aggressive.diagnostics();
  CHECK(conservative_diagnostics.domains == molecule.natoms() * 2);
  CHECK(conservative_diagnostics.model_batches == molecule.natoms() * 2);
  CHECK(aggressive_diagnostics.domains == molecule.natoms() * 2);
  CHECK(aggressive_diagnostics.model_batches == 2);
  CHECK(conservative_diagnostics.configured_model_batches == molecule.natoms());
  CHECK(conservative_diagnostics.max_domains_per_model_batch == 1);
  CHECK(aggressive_diagnostics.configured_model_batches == 1);
  CHECK(aggressive_diagnostics.max_domains_per_model_batch ==
        molecule.natoms());
  CHECK(conservative_diagnostics.task_points_min > 0);
  CHECK(conservative_diagnostics.task_points_max >=
        conservative_diagnostics.task_points_min);
}

TEST_CASE("Skala host gradient", "[skala][host-gradient]") {
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_WORLD)};
  const std::string fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";

  GauXC::Molecule mol;
  GauXC::BasisSet<double> basis;
  GauXC::read_hdf5_record(mol, fixture, "/MOLECULE");
  GauXC::read_hdf5_record(basis, fixture, "/BASIS");

  HighFive::File file(fixture, HighFive::File::ReadOnly);
  auto density = file.getDataSet("/DENSITY");
  auto dims = density.getDimensions();
  const int64_t nbf = static_cast<int64_t>(dims[0]);
  std::vector<double> Ps(dims[0] * dims[1]);
  std::vector<double> Pz(Ps.size(), 0.0);
  density.read(Ps.data());

  auto mg = GauXC::MolGridFactory::create_default_molgrid(
      mol, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles,
      GauXC::AtomicGridSizeDefault::UltraFineGrid);
  auto driver = SkalaHostDriver::from_system(rt, mol, mg, basis, "TPSS");

  std::vector<double> gradient(3 * mol.size());
  const Eigen::Map<const ColMajorMatrix> scalar_density(Ps.data(), nbf, nbf);
  const Eigen::Map<const ColMajorMatrix> spin_density(Pz.data(), nbf, nbf);
  Eigen::Map<RowMajorMatrix> gradient_matrix(gradient.data(), mol.size(), 3);
  driver.eval_exc_grad_uks(scalar_density, spin_density, gradient_matrix);

  double squared_norm = 0.0;
  double translation[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < gradient.size(); ++i) {
    const double value = gradient[i];
    INFO("gradient component=" << value);
    CHECK(std::isfinite(value));
    squared_norm += value * value;
    translation[i % 3] += value;
  }
  CHECK(std::sqrt(squared_norm) > 1e-3);
  CHECK(std::abs(translation[0]) < 1e-10);
  CHECK(std::abs(translation[1]) < 1e-10);
  CHECK(std::abs(translation[2]) < 1e-10);
}

TEST_CASE("Skala host gradient matches a Ridders derivative",
          "[skala][gradient-numerical][.slow]") {
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_WORLD)};

  // Rotate a 1.4-bohr H2 bond off-axis so x, y, and z gradient components are
  // all exercised by the bond-stretch derivative.
  GauXC::Molecule molecule;
  molecule.push_back(GauXC::Atom{GauXC::AtomicNumber(1), -0.252, 0.336, -0.56});
  molecule.push_back(GauXC::Atom{GauXC::AtomicNumber(1), 0.252, -0.336, 0.56});

  // Build one normalized STO-3G 1s shell on each hydrogen in memory. This
  // keeps the repeated displaced evaluations practical for routine CI.
  GauXC::Shell<double>::prim_array exponents{};
  GauXC::Shell<double>::prim_array coefficients{};
  exponents[0] = 3.42525091;
  exponents[1] = 0.62391373;
  exponents[2] = 0.16885540;
  coefficients[0] = 0.15432897;
  coefficients[1] = 0.53532814;
  coefficients[2] = 0.44463454;
  GauXC::BasisSet<double> basis;
  for (const auto& atom : molecule)
    basis.emplace_back(
        GauXC::PrimSize(3), GauXC::AngularMomentum(0),
        GauXC::SphericalType(false), exponents, coefficients,
        GauXC::Shell<double>::cart_array{atom.x, atom.y, atom.z});
  GauXC::BasisSetMap basis_map(basis, molecule);

  // Ridders differentiates with fixed AO density coefficients, matching the
  // contract of eval_exc_grad_uks. TPSS exercises rho, grad-rho, and tau terms.
  const int64_t nbf = basis.nbf();
  std::vector<double> Ps = {0.5, 0.5, 0.5, 0.5};
  std::vector<double> Pz(Ps.size(), 0.0);

  const std::string model = "TPSS";
  const auto grid_size = GauXC::AtomicGridSizeDefault::FineGrid;
  auto grid = GauXC::MolGridFactory::create_default_molgrid(
      molecule, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles, grid_size);
  auto driver = SkalaHostDriver::from_system(rt, molecule, grid, basis, model);
  std::vector<double> gradient(3 * molecule.size());
  const Eigen::Map<const ColMajorMatrix> scalar_density(Ps.data(), nbf, nbf);
  const Eigen::Map<const ColMajorMatrix> spin_density(Pz.data(), nbf, nbf);
  Eigen::Map<RowMajorMatrix> gradient_matrix(gradient.data(), molecule.size(),
                                             3);
  driver.eval_exc_grad_uks(scalar_density, spin_density, gradient_matrix);

  // Construct the bond stretch from the atomic positions. Moving the atoms
  // equally in opposite directions changes the bond length without translating
  // the molecule. Normalization then gives dE/ds = gradient dot direction.
  REQUIRE(molecule.size() == 2);
  const double dx = molecule[1].x - molecule[0].x;
  const double dy = molecule[1].y - molecule[0].y;
  const double dz = molecule[1].z - molecule[0].z;
  std::vector<double> direction = {-dx, -dy, -dz, dx, dy, dz};
  REQUIRE(direction.size() == gradient.size());
  double direction_norm = 0.0;
  for (double value : direction) direction_norm += value * value;
  direction_norm = std::sqrt(direction_norm);
  REQUIRE(direction_norm > 0.0);
  for (double& value : direction) value /= direction_norm;

  double analytic_derivative = 0.0;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    analytic_derivative += gradient[i] * direction[i];

  // Numerically differentiate the EXC energy along the bond
  // stretch using Ridders' method.
  const auto numerical = ridders_derivative(
      [&](double displacement) {
        return displaced_exc(rt, molecule, basis, basis_map.shell_to_center(),
                             direction, displacement, nbf, Ps, Pz, model,
                             grid_size);
      },
      1e-2);

  INFO("analytic directional derivative=" << analytic_derivative);
  INFO("Ridders directional derivative=" << numerical.derivative);
  INFO("Ridders error estimate=" << numerical.error);
  CHECK(std::abs(analytic_derivative) > 1e-3);
  CHECK(numerical.error < 1e-7);
  CHECK(numerical.derivative ==
        Catch::Approx(analytic_derivative).margin(1e-6));
}

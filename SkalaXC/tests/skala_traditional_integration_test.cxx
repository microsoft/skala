// SkalaXC traditional-functional integration test.
//
// The Skala side is driven entirely through the PUBLIC SkalaXC API (the same
// RuntimeEnvironment -> Molecule/BasisSet -> MolGrid -> LoadBalancer ->
// MolecularWeights -> functional_type -> XCIntegrator pipeline a consumer would
// use); the shared geometry/basis are bridged into SkalaXC's own value types.
// The reference side still uses GauXC + ExchCXX directly as the comparison
// baseline, so this remains a white-box test that pulls GauXC in-build.
//
// SkalaXC ships neural `.fun` reproductions of three traditional
// exchange-correlation functionals: LDA exchange (ldax.fun), PBE (pbe.fun), and
// TPSS (tpss.fun). This test drives each of those bundled baselines through
// the public SkalaXC::XCIntegrator and, on the very same molecular grid,
// evaluates the corresponding ExchCXX functional through GauXC's own reference
// XCIntegrator.
// For several small molecules and random spin-resolved AO densities it checks
// that the two implementations agree on:
//   * the exchange-correlation energy EXC,
//   * the scalar and z exchange-correlation potentials VXC, and
//   * the exchange-correlation energy gradient.
//
// The test is designed to run identically in a serial build and under MPI with
// three or more ranks: every rank builds byte-identical densities (the RNG seed
// is chosen on rank 0 and broadcast), so the replicated GauXC/SkalaXC
// reductions return the same fully-reduced result on every rank.
//
// Reproducibility: by default the density seed comes from std::random_device.
// Set SKALAXC_TEST_SEED=<integer> to force a specific seed. On a mismatch rank
// 0 prints the seed and writes the offending densities as MatrixMarket files;
// set SKALAXC_TEST_DENSITY_DIR=<dir> to replay from previously emitted
// density_<molecule>_{scalar,z}.mtx files instead of generating new ones.

#include <gauxc/basisset.hpp>
#include <gauxc/enums.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/types.hpp>
#include <gauxc/util/mpi.hpp>
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/xc_integrator_settings.hpp>

#include "parse_basis.hpp"

#include <catch2/catch_test_macros.hpp>
#include <skalaxc/skalaxc.hpp>

#include "test_utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SkalaXC;

namespace {

/** @brief One atom of a hard-coded molecule, coordinates in angstrom. */
struct AtomSpec {
  int Z;
  double x, y, z;
};

/** @brief A named molecule together with the SkalaXC/ExchCXX selectors. */
struct MoleculeCase {
  const char* name;
  std::vector<AtomSpec> atoms;
};

/**
 * @brief Build a GauXC molecule (atomic units) from an angstrom specification.
 */
GauXC::Molecule make_molecule(const std::vector<AtomSpec>& atoms) {
  constexpr double angstrom_to_bohr = 1.8897259886;
  GauXC::Molecule mol;
  for (const auto& a : atoms)
    mol.emplace_back(GauXC::AtomicNumber(a.Z), a.x * angstrom_to_bohr,
                     a.y * angstrom_to_bohr, a.z * angstrom_to_bohr);
  return mol;
}

/**
 * @brief Bridge a GauXC molecule into the public SkalaXC::Molecule value type.
 *
 * The reference path is expressed in GauXC types; the Skala side is a pure
 * public-API consumer, so the shared geometry is copied into SkalaXC's own
 * type (coordinates already in bohr).
 */
SkalaXC::Molecule to_skala_molecule(const GauXC::Molecule& mol) {
  SkalaXC::Molecule out;
  out.reserve(mol.size());
  for (const auto& a : mol)
    out.emplace_back(SkalaXC::AtomicNumber(a.Z.get()), a.x, a.y, a.z);
  return out;
}

/**
 * @brief Bridge a parsed GauXC basis into the public SkalaXC::BasisSet type.
 *
 * parse_basis already normalizes the primitive contraction coefficients, so the
 * shells are copied verbatim with normalize=false; the library's internal
 * SkalaXC -> GauXC round-trip then reproduces byte-identical GauXC shells.
 */
SkalaXC::BasisSet<double> to_skala_basis(const GauXC::BasisSet<double>& basis) {
  SkalaXC::BasisSet<double> out;
  out.reserve(basis.size());
  for (const auto& s : basis) {
    const std::int32_t nprim = s.nprim();
    SkalaXC::Shell<double>::prim_array alpha{};
    SkalaXC::Shell<double>::prim_array coeff{};
    for (std::int32_t i = 0; i < nprim; ++i) {
      alpha[i] = s.alpha_data()[i];
      coeff[i] = s.coeff_data()[i];
    }
    const SkalaXC::Shell<double>::cart_array O{s.O_data()[0], s.O_data()[1],
                                               s.O_data()[2]};
    out.emplace_back(SkalaXC::PrimSize(nprim), SkalaXC::AngularMomentum(s.l()),
                     SkalaXC::SphericalType(s.pure()), alpha, coeff, O,
                     /*normalize=*/false);
  }
  return out;
}

/** @brief Traditional functional selector shared by SkalaXC and ExchCXX. */
struct FunctionalCase {
  const char* skala_model;         // SkalaXC .fun selector
  ExchCXX::Functional exchcxx_id;  // matching ExchCXX builtin functional
};

/**
 * @brief Random symmetric, strictly diagonally dominant (hence SPD) matrix.
 *
 * A symmetric AO density matrix that is positive semidefinite guarantees a
 * nonnegative on-grid density everywhere, which keeps the traditional exchange
 * kernels well defined. Strict diagonal dominance with a positive diagonal is a
 * cheap Gershgorin guarantee of positive definiteness; the caller additionally
 * verifies it with a Cholesky factorization.
 */
Eigen::MatrixXd random_spd(int n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const double v = dist(rng);
      M(i, j) = v;
      M(j, i) = v;
    }
  for (int i = 0; i < n; ++i) {
    double off = 0.0;
    for (int j = 0; j < n; ++j)
      if (j != i) off += std::abs(M(i, j));
    M(i, i) = off + 0.5;  // strictly diagonally dominant, positive diagonal
  }
  // Keep the on-grid density in a physically reasonable magnitude range.
  M *= 1.0 / static_cast<double>(n);
  return M;
}

/**
 * @brief Random symmetric spin density scaled small relative to the scalar one.
 *
 * The z (spin) density is left indefinite but scaled so that
 * max|Pz| * 10 < max|Ps|, keeping |rho_z(r)| well below rho_s(r) so the
 * per-spin densities stay nonnegative.
 */
Eigen::MatrixXd random_small_symmetric(const Eigen::MatrixXd& Ps, int n,
                                       std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const double v = dist(rng);
      M(i, j) = v;
      M(j, i) = v;
    }
  const double max_s = Ps.cwiseAbs().maxCoeff();
  const double max_z = M.cwiseAbs().maxCoeff();
  if (max_z > 0.0) M *= 0.09 * max_s / max_z;  // => max|Pz| * 10 < max|Ps|
  return M;
}

/** @brief A scalar/z AO density pair for one molecule. */
struct DensityPair {
  Eigen::MatrixXd Ps, Pz;
};

/**
 * @brief Generate (or replay from disk) the density pair for one molecule.
 * @param name Molecule label used for replay/dump file names.
 * @param nbf Number of AO basis functions.
 * @param rng Seeded generator (advanced only when generating).
 * @param replay_dir If non-null, load densities from this directory instead.
 */
DensityPair make_density(const std::string& name, int nbf, std::mt19937_64& rng,
                         const char* replay_dir) {
  DensityPair d;
  if (replay_dir) {
    const std::string base = std::string(replay_dir) + "/density_" + name;
    if (!Eigen::loadMarketDense(d.Ps, base + "_scalar.mtx") ||
        !Eigen::loadMarketDense(d.Pz, base + "_z.mtx"))
      throw std::runtime_error("Failed to load replay densities from " + base +
                               "_{scalar,z}.mtx");
    return d;
  }
  d.Ps = random_spd(nbf, rng);
  d.Pz = random_small_symmetric(d.Ps, nbf, rng);
  return d;
}

/** @brief Relative error of two scalars, floored so tiny values stay stable. */
double rel_err(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::abs(b));
}

/** @brief Max-norm relative error of two matrices. */
double matrix_rel_err(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  const double denom = std::max(1.0, b.cwiseAbs().maxCoeff());
  return (a - b).cwiseAbs().maxCoeff() / denom;
}

/**
 * @brief Evaluate one traditional functional through GauXC's reference path.
 *
 * Mirrors the public Skala path's grid/load-balancer/molecular-weight setup so
 * the quadrature is identical, then integrates the requested ExchCXX builtin
 * functional. Returns the polarized EXC, scalar/z potentials, and the
 * weight-derivative-inclusive XC gradient.
 */
struct ReferenceResult {
  double exc;
  Eigen::MatrixXd vxc_scalar, vxc_z;
  std::vector<double> gradient;  // 3 * natoms, atom-major xyz
};

ReferenceResult evaluate_reference(const GauXC::RuntimeEnvironment& rt,
                                   const GauXC::Molecule& mol,
                                   const GauXC::MolGrid& mg,
                                   const GauXC::BasisSet<double>& basis,
                                   ExchCXX::Functional functional_id,
                                   const Eigen::MatrixXd& Ps,
                                   const Eigen::MatrixXd& Pz) {
  using matrix_type = Eigen::MatrixXd;

  GauXC::LoadBalancerFactory lb_factory(GauXC::ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);

  GauXC::MolecularWeightsFactory mw_factory(GauXC::ExecutionSpace::Host,
                                            "Default",
                                            GauXC::MolecularWeightsSettings{});
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  GauXC::functional_type func(ExchCXX::Backend::builtin, functional_id,
                              ExchCXX::Spin::Polarized);

  GauXC::XCIntegratorFactory<matrix_type> integrator_factory(
      GauXC::ExecutionSpace::Host, "Replicated", "Default", "Default",
      "Default");
  auto integrator = integrator_factory.get_instance(func, lb);

  ReferenceResult r;
  auto exc_vxc = integrator.eval_exc_vxc(Ps, Pz);
  r.exc = std::get<0>(exc_vxc);
  r.vxc_scalar = std::get<1>(exc_vxc);
  r.vxc_z = std::get<2>(exc_vxc);

  GauXC::IntegratorSettingsEXC_GRAD exc_grad_settings;
  exc_grad_settings.include_weight_derivatives = true;
  r.gradient = integrator.eval_exc_grad(Ps, Pz, exc_grad_settings);
  return r;
}

/** @brief SkalaXC baseline EXC/VXC/gradient for one .fun model. */
struct SkalaResult {
  double exc;
  Eigen::MatrixXd vxc_scalar, vxc_z;
  std::vector<double> gradient;  // 3 * natoms, atom-major xyz
};

SkalaResult evaluate_skala(const SkalaXC::RuntimeEnvironment& rt,
                           const GauXC::Molecule& mol,
                           const GauXC::BasisSet<double>& basis,
                           const std::string& model, const Eigen::MatrixXd& Ps,
                           const Eigen::MatrixXd& Pz) {
  const SkalaXC::Molecule skala_mol = to_skala_molecule(mol);
  const SkalaXC::BasisSet<double> skala_basis = to_skala_basis(basis);

  // Mirror evaluate_reference's grid preset exactly so both paths integrate on
  // an identical quadrature.
  auto mg = SkalaXC::test::make_molgrid(
      skala_mol, SkalaXC::AtomicGridSizeDefault::FineGrid);

  SkalaXC::LoadBalancerFactory lb_factory(SkalaXC::ExecutionSpace::Host,
                                          "Default");
  auto lb = lb_factory.get_instance(rt, skala_mol, mg, skala_basis);

  SkalaXC::MolecularWeightsFactory mw_factory(
      SkalaXC::ExecutionSpace::Host, "Default",
      SkalaXC::MolecularWeightsSettings{});
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  SkalaXC::functional_type func(model);
  SkalaXC::XCIntegratorFactory<Eigen::MatrixXd> integrator_factory(
      SkalaXC::ExecutionSpace::Host);
  auto integrator = integrator_factory.get_instance(func, lb);

  SkalaResult r;
  auto exc_vxc = integrator.eval_exc_vxc(Ps, Pz);
  r.exc = std::get<0>(exc_vxc);
  r.vxc_scalar = std::get<1>(exc_vxc);
  r.vxc_z = std::get<2>(exc_vxc);

  // include_weight_derivatives defaults to true (the only mode SkalaXC
  // supports), matching evaluate_reference's gradient settings.
  r.gradient = integrator.eval_exc_grad(Ps, Pz);
  return r;
}

/** @brief Pick the density RNG seed on rank 0 and broadcast it to all ranks. */
std::uint64_t choose_seed(const GauXC::RuntimeEnvironment& rt) {
  std::uint64_t seed = 0;
  if (rt.comm_rank() == 0) {
    if (const char* env = std::getenv("SKALAXC_TEST_SEED"))
      seed = std::strtoull(env, nullptr, 10);
    else
      seed = std::random_device{}();
  }
#ifdef GAUXC_HAS_MPI
  MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#endif
  return seed;
}

}  // namespace

TEST_CASE("SkalaXC baselines reproduce GauXC traditional functionals",
          "[skala][traditional-integration]") {
  // The reference path uses GauXC's runtime; the Skala path uses SkalaXC's own
  // (public) runtime. Both wrap the same MPI communicator.
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_WORLD)};
  SkalaXC::RuntimeEnvironment skala_rt{SKALAXC_MPI_CODE(MPI_COMM_WORLD)};
  const bool is_root = rt.comm_rank() == 0;

  const std::uint64_t seed = choose_seed(rt);
  std::mt19937_64 rng(seed);
  const char* replay_dir = std::getenv("SKALAXC_TEST_DENSITY_DIR");
  const bool verbose = std::getenv("SKALAXC_TEST_VERBOSE") != nullptr;

  const std::string basis_path =
      std::string(SKALAXC_TEST_BASIS_PATH) + "/cc-pvdz.g94";

  // Six small (3-4 atom) molecules; every element appears in cc-pVDZ.
  const std::vector<MoleculeCase> molecules = {
      {"h2o",
       {{8, 0.0000, 0.0000, 0.1173},
        {1, 0.0000, 0.7572, -0.4692},
        {1, 0.0000, -0.7572, -0.4692}}},
      {"nh3",
       {{7, 0.0000, 0.0000, 0.1128},
        {1, 0.0000, 0.9377, -0.2633},
        {1, 0.8121, -0.4689, -0.2633},
        {1, -0.8121, -0.4689, -0.2633}}},
      {"hcn",
       {{1, 0.0000, 0.0000, -1.0640},
        {6, 0.0000, 0.0000, 0.0000},
        {7, 0.0000, 0.0000, 1.1560}}},
      {"co2",
       {{6, 0.0000, 0.0000, 0.0000},
        {8, 0.0000, 0.0000, 1.1620},
        {8, 0.0000, 0.0000, -1.1620}}},
      {"h2o2",
       {{8, 0.0000, 0.7375, -0.0528},
        {8, 0.0000, -0.7375, -0.0528},
        {1, 0.8190, 0.8170, 0.4220},
        {1, -0.8190, -0.8170, 0.4220}}},
      {"ch2o",
       {{6, 0.0000, 0.0000, -0.5296},
        {8, 0.0000, 0.0000, 0.6742},
        {1, 0.0000, 0.9337, -1.1109},
        {1, 0.0000, -0.9337, -1.1109}}},
  };

  const std::vector<FunctionalCase> functionals = {
      {"LDA", ExchCXX::Functional::LDA},
      {"PBE", ExchCXX::Functional::PBE},
      {"TPSS", ExchCXX::Functional::TPSS},
  };

  // The neural .fun baselines reproduce the traditional functionals closely but
  // not to machine precision: measured worst-case relative errors are ~1e-6
  // (LDA exchange is exact to ~1e-15). These tolerances leave headroom for
  // random-density variation while still catching any real regression, which
  // would degrade agreement by several orders of magnitude.
  const double exc_tol = 1e-5;
  const double vxc_tol = 2e-5;
  const double grad_tol = 1e-4;

  for (const auto& mol_case : molecules) {
    const GauXC::Molecule mol = make_molecule(mol_case.atoms);
    GauXC::BasisSet<double> basis =
        GauXC::parse_basis(mol, basis_path, GauXC::SphericalType(true));
    const int nbf = static_cast<int>(basis.nbf());

    const DensityPair density =
        make_density(mol_case.name, nbf, rng, replay_dir);

    // A positive semidefinite scalar density guarantees a nonnegative on-grid
    // density; verify it explicitly rather than trusting diagonal dominance.
    Eigen::LLT<Eigen::MatrixXd> llt(density.Ps);
    REQUIRE(llt.info() == Eigen::Success);
    REQUIRE(density.Pz.cwiseAbs().maxCoeff() * 10.0 <
            density.Ps.cwiseAbs().maxCoeff());

    const auto mg = GauXC::MolGridFactory::create_default_molgrid(
        mol, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
        GauXC::RadialQuad::MuraKnowles, GauXC::AtomicGridSizeDefault::FineGrid);

    for (const auto& fun_case : functionals) {
      const SkalaResult skala = evaluate_skala(
          skala_rt, mol, basis, fun_case.skala_model, density.Ps, density.Pz);
      const ReferenceResult ref = evaluate_reference(
          rt, mol, mg, basis, fun_case.exchcxx_id, density.Ps, density.Pz);

      const double exc_err = rel_err(skala.exc, ref.exc);
      const double vxcs_err = matrix_rel_err(skala.vxc_scalar, ref.vxc_scalar);
      const double vxcz_err = matrix_rel_err(skala.vxc_z, ref.vxc_z);

      double grad_abs_err = 0.0, grad_max = 0.0;
      REQUIRE(skala.gradient.size() == ref.gradient.size());
      for (std::size_t i = 0; i < ref.gradient.size(); ++i) {
        grad_abs_err = std::max(grad_abs_err,
                                std::abs(skala.gradient[i] - ref.gradient[i]));
        grad_max = std::max(grad_max, std::abs(ref.gradient[i]));
      }
      const double grad_err = grad_abs_err / std::max(1.0, grad_max);

      if (verbose && is_root)
        std::cerr << "[traditional-integration] " << mol_case.name << " / "
                  << fun_case.skala_model << " nbf=" << nbf
                  << " EXC_skala=" << skala.exc << " EXC_ref=" << ref.exc
                  << " exc_rel=" << exc_err << " vxcs_rel=" << vxcs_err
                  << " vxcz_rel=" << vxcz_err << " grad_rel=" << grad_err
                  << "\n";

      const bool ok = exc_err < exc_tol && vxcs_err < vxc_tol &&
                      vxcz_err < vxc_tol && grad_err < grad_tol;
      if (!ok && is_root && !replay_dir) {
        const std::string base =
            std::string("skalaxc_integration_fail_density_") + mol_case.name;
        Eigen::saveMarketDense(density.Ps, base + "_scalar.mtx");
        Eigen::saveMarketDense(density.Pz, base + "_z.mtx");
        std::cerr << "[skala][traditional-integration] MISMATCH for "
                  << mol_case.name << " / " << fun_case.skala_model
                  << ". Reproduce with SKALAXC_TEST_SEED=" << seed
                  << " or replay via SKALAXC_TEST_DENSITY_DIR containing "
                  << base << "_{scalar,z}.mtx\n";
      }

      INFO("molecule=" << mol_case.name
                       << " functional=" << fun_case.skala_model
                       << " nbf=" << nbf << " seed=" << seed);
      INFO("EXC_skala=" << skala.exc << " EXC_ref=" << ref.exc);
      INFO("exc_rel_err=" << exc_err << " vxcs_rel_err=" << vxcs_err
                          << " vxcz_rel_err=" << vxcz_err
                          << " grad_rel_err=" << grad_err);
      CHECK(exc_err < exc_tol);
      CHECK(vxcs_err < vxc_tol);
      CHECK(vxcz_err < vxc_tol);
      CHECK(grad_err < grad_tol);
    }
  }
}

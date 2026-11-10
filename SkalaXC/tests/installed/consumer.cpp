#include <skalaxc/skalaxc.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class Matrix {
 public:
  using value_type = double;

  Matrix(std::int64_t rows, std::int64_t cols)
      : rows_(rows), cols_(cols), values_(rows * cols) {}

  std::int64_t rows() const { return rows_; }
  std::int64_t cols() const { return cols_; }
  double* data() { return values_.data(); }
  const double* data() const { return values_.data(); }

 private:
  std::int64_t rows_;
  std::int64_t cols_;
  std::vector<double> values_;
};

int main(int argc, char** argv) {
#ifdef SKALAXC_HAS_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
#else
  (void)argc;
  (void)argv;
#endif

  int result = 1;
  try {
    if (SkalaXC::version().empty())
      throw std::runtime_error("SkalaXC version is unavailable");
    {
#ifdef SKALAXC_HAS_MPI
      SkalaXC::RuntimeEnvironment runtime(MPI_COMM_WORLD);
#else
      SkalaXC::RuntimeEnvironment runtime;
#endif
      SkalaXC::Molecule molecule{{SkalaXC::AtomicNumber(1), -0.7, 0.0, 0.0},
                                 {SkalaXC::AtomicNumber(1), 0.7, 0.0, 0.0}};
      SkalaXC::BasisSet<double> basis;
      SkalaXC::Shell<double>::prim_array exponents{};
      SkalaXC::Shell<double>::prim_array coefficients{};
      exponents[0] = 3.42525091;
      exponents[1] = 0.62391373;
      exponents[2] = 0.16885540;
      coefficients[0] = 0.15432897;
      coefficients[1] = 0.53532814;
      coefficients[2] = 0.44463454;
      for (const auto& atom : molecule) {
        basis.emplace_back(
            SkalaXC::PrimSize(3), SkalaXC::AngularMomentum(0),
            SkalaXC::SphericalType(0), exponents, coefficients,
            SkalaXC::Shell<double>::cart_array{atom.x, atom.y, atom.z});
      }

      auto grid = SkalaXC::MolGridFactory::create_default_molgrid(
          molecule, SkalaXC::PruningScheme::Unpruned, SkalaXC::BatchSize(128),
          SkalaXC::RadialQuad::MuraKnowles,
          SkalaXC::AtomicGridSizeDefault::FineGrid);
      SkalaXC::LoadBalancerFactory load_balancer_factory(
          SkalaXC::ExecutionSpace::Host);
      auto load_balancer =
          load_balancer_factory.get_instance(runtime, molecule, grid, basis);
      SkalaXC::MolecularWeightsFactory weights_factory(
          SkalaXC::ExecutionSpace::Host, "Default");
      auto weights = weights_factory.get_instance();
      weights.modify_weights(load_balancer);

      SkalaXC::XCIntegratorFactory<Matrix> integrator_factory(
          SkalaXC::ExecutionSpace::Host);
      auto integrator = integrator_factory.get_instance(
          SkalaXC::functional_type("LDA"), load_balancer);
      Matrix scalar_density(2, 2);
      Matrix spin_density(2, 2);
      for (std::size_t index = 0; index < 4; ++index) {
        scalar_density.data()[index] = 0.5;
        spin_density.data()[index] = 0.0;
      }
      auto [energy, scalar_potential, spin_potential] =
          integrator.eval_exc_vxc(scalar_density, spin_density);
      result = std::isfinite(energy) &&
                       std::isfinite(scalar_potential.data()[0]) &&
                       std::isfinite(spin_potential.data()[0])
                   ? 0
                   : 1;
    }
  } catch (...) {
    result = 1;
  }

#ifdef SKALAXC_HAS_MPI
  if (MPI_Finalize() != MPI_SUCCESS) return 1;
#endif
  return result;
}

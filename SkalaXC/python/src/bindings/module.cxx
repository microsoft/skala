#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <skalaxc/skalaxc.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SKALAXC_HAS_MPI
#include <mpi4py/mpi4py.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Shell = SkalaXC::Shell<double>;

/**
 * @brief Matrix adapter for the templated public SkalaXC integrator API.
 *
 * Integrator outputs use the two-argument constructor and own writable,
 * column-major storage. Python inputs use the three-argument constructor as
 * read-only, non-owning views of nanobind's DLPack-backed array storage. The
 * input arrays (including any dtype or layout conversion temporaries) remain
 * alive for the complete synchronous integrator call, so no additional input
 * copy is required.
 */
class Matrix {
 public:
  using value_type = double;

  Matrix(std::int64_t rows, std::int64_t cols)
      : rows_(rows),
        cols_(cols),
        values_(static_cast<std::size_t>(rows * cols)),
        view_data_(nullptr),
        owns_storage_(true) {}

  Matrix(std::int64_t rows, std::int64_t cols, const double* data)
      : rows_(rows), cols_(cols), view_data_(data), owns_storage_(false) {}

  std::int64_t rows() const { return rows_; }
  std::int64_t cols() const { return cols_; }
  double* data() {
    if (!owns_storage_)
      throw std::logic_error("cannot write to a matrix input view");
    return values_.data();
  }
  const double* data() const {
    return owns_storage_ ? values_.data() : view_data_;
  }

 private:
  std::int64_t rows_;
  std::int64_t cols_;
  std::vector<double> values_;
  const double* view_data_;
  bool owns_storage_;
};

// Keep conversions enabled so nanobind materializes float64, Fortran-contiguous
// CPU temporaries for other NumPy dtypes and layouts.
using InputMatrix = nb::ndarray<nb::numpy, const double, nb::ndim<2>,
                                nb::f_contig, nb::device::cpu>;
using OutputMatrix =
    nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::f_contig, nb::device::cpu>;
using OutputGradient =
    nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Integrator = SkalaXC::XCIntegrator<Matrix>;
using IntegratorFactory = SkalaXC::XCIntegratorFactory<Matrix>;

struct PythonRuntimeEnvironment {
#ifndef SKALAXC_HAS_MPI
  PythonRuntimeEnvironment() : value() {}
  explicit PythonRuntimeEnvironment(SkalaXC::DeviceRuntimeSettings settings)
      : value(settings) {}
#else
  PythonRuntimeEnvironment(nb::object owner, MPI_Comm communicator, bool device,
                           SkalaXC::DeviceRuntimeSettings settings = {})
      : communicator_owner(std::move(owner)),
        value(device ? SkalaXC::RuntimeEnvironment(communicator, settings)
                     : SkalaXC::RuntimeEnvironment(communicator)) {}
#endif

  nb::object communicator_owner;
  SkalaXC::RuntimeEnvironment value;
};

struct PythonLoadBalancer {
  PythonLoadBalancer(SkalaXC::LoadBalancer&& load_balancer,
                     std::int64_t basis_size, std::int64_t atom_count,
                     nb::object communicator)
      : communicator_owner(std::move(communicator)),
        value(std::move(load_balancer)),
        nbf(basis_size),
        natoms(atom_count) {}

  nb::object communicator_owner;
  SkalaXC::LoadBalancer value;
  std::int64_t nbf;
  std::int64_t natoms;
};

struct PythonIntegrator {
  PythonIntegrator(Integrator&& integrator, std::int64_t basis_size,
                   std::int64_t atom_count, nb::object communicator)
      : communicator_owner(std::move(communicator)),
        value(std::move(integrator)),
        nbf(basis_size),
        natoms(atom_count) {}

  nb::object communicator_owner;
  Integrator value;
  std::int64_t nbf;
  std::int64_t natoms;
};

Matrix matrix_view(const InputMatrix& array, std::int64_t expected_size) {
  if (array.shape(0) != static_cast<std::size_t>(expected_size) ||
      array.shape(1) != static_cast<std::size_t>(expected_size))
    throw nb::value_error("density matrices must have shape (nbf, nbf)");
  return Matrix(expected_size, expected_size, array.data());
}

OutputMatrix move_matrix_to_numpy(Matrix&& matrix) {
  auto storage = std::make_unique<Matrix>(std::move(matrix));
  const auto rows = static_cast<std::size_t>(storage->rows());
  const auto cols = static_cast<std::size_t>(storage->cols());
  double* data = storage->data();
  nb::capsule owner(storage.get(), [](void* pointer) noexcept {
    delete static_cast<Matrix*>(pointer);
  });
  storage.release();
  return OutputMatrix(data, {rows, cols}, owner);
}

OutputGradient move_gradient_to_numpy(std::vector<double>&& gradient,
                                      std::int64_t atom_count) {
  if (atom_count < 0 ||
      gradient.size() != static_cast<std::size_t>(atom_count) * 3)
    throw std::runtime_error("gradient has an unexpected size");
  auto storage = std::make_unique<std::vector<double>>(std::move(gradient));
  double* data = storage->data();
  nb::capsule owner(storage.get(), [](void* pointer) noexcept {
    delete static_cast<std::vector<double>*>(pointer);
  });
  storage.release();
  return OutputGradient(
      data, {static_cast<std::size_t>(atom_count), std::size_t{3}}, owner);
}

#ifdef SKALAXC_HAS_MPI
void check_mpi_call(int error, const char* operation) {
  if (error == MPI_SUCCESS) return;
  char message[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(error, message, &length);
  throw std::runtime_error(
      std::string(operation) +
      " failed: " + std::string(message, static_cast<std::size_t>(length)));
}

MPI_Comm mpi4py_communicator(const nb::object& communicator) {
  int initialized = 0;
  int finalized = 0;
  check_mpi_call(MPI_Initialized(&initialized), "MPI_Initialized");
  check_mpi_call(MPI_Finalized(&finalized), "MPI_Finalized");
  if (!initialized)
    throw nb::value_error("MPI must be initialized before creating a runtime");
  if (finalized) throw nb::value_error("MPI has already been finalized");
  if (communicator.is_none() ||
      !PyObject_TypeCheck(communicator.ptr(), &PyMPIComm_Type))
    throw nb::type_error(
        "communicator must be an explicit mpi4py.MPI.Comm instance");
  MPI_Comm* native_communicator = PyMPIComm_Get(communicator.ptr());
  if (native_communicator == nullptr) throw nb::python_error();
  if (*native_communicator == MPI_COMM_NULL)
    throw nb::value_error("communicator must not be MPI.COMM_NULL or freed");
  int is_intercommunicator = 0;
  check_mpi_call(
      MPI_Comm_test_inter(*native_communicator, &is_intercommunicator),
      "MPI_Comm_test_inter");
  if (is_intercommunicator)
    throw nb::value_error("MPI intercommunicators are not supported");
  return *native_communicator;
}

std::string trim_mpi_version(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  return value;
}

void verify_mpi_implementation() {
  char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
  int length = 0;
  check_mpi_call(MPI_Get_library_version(version, &length),
                 "MPI_Get_library_version");
  const std::string native_version =
      trim_mpi_version(std::string(version, static_cast<std::size_t>(length)));
  const std::string mpi4py_version = trim_mpi_version(nb::cast<std::string>(
      nb::module_::import_("mpi4py.MPI").attr("Get_library_version")()));
  if (native_version != mpi4py_version) {
    PyErr_Format(PyExc_ImportError,
                 "SkalaXC and mpi4py use different MPI implementations: "
                 "SkalaXC='%s', mpi4py='%s'",
                 native_version.c_str(), mpi4py_version.c_str());
    throw nb::python_error();
  }
}
#endif

}  // namespace

NB_MODULE(_skalaxc, module) {
  module.doc() = "Direct bindings for the public SkalaXC C++ API";

  nb::exception<SkalaXC::Exception>(module, "SkalaXCError");
  module.def("native_version", [] { return std::string(SkalaXC::version()); });

  auto execution_space =
      nb::enum_<SkalaXC::ExecutionSpace>(module, "ExecutionSpace")
          .value("HOST", SkalaXC::ExecutionSpace::Host);
#ifdef SKALAXC_HAS_CUDA
  execution_space.value("DEVICE", SkalaXC::ExecutionSpace::Device);
#endif
  nb::enum_<SkalaXC::DomainBatchMode>(module, "DomainBatchMode")
      .value("CONSERVATIVE", SkalaXC::DomainBatchMode::Conservative)
      .value("AGGRESSIVE", SkalaXC::DomainBatchMode::Aggressive);
  nb::enum_<SkalaXC::RadialQuad>(module, "RadialQuad")
      .value("BECKE", SkalaXC::RadialQuad::Becke)
      .value("MURA_KNOWLES", SkalaXC::RadialQuad::MuraKnowles)
      .value("MURRAY_HANDY_LAMING", SkalaXC::RadialQuad::MurrayHandyLaming)
      .value("TREUTLER_AHLRICHS", SkalaXC::RadialQuad::TreutlerAhlrichs);
  nb::enum_<SkalaXC::AtomicGridSizeDefault>(module, "AtomicGridSize")
      .value("FINE", SkalaXC::AtomicGridSizeDefault::FineGrid)
      .value("ULTRA_FINE", SkalaXC::AtomicGridSizeDefault::UltraFineGrid)
      .value("SUPER_FINE", SkalaXC::AtomicGridSizeDefault::SuperFineGrid)
      .value("GM3", SkalaXC::AtomicGridSizeDefault::GM3)
      .value("GM5", SkalaXC::AtomicGridSizeDefault::GM5);
  nb::enum_<SkalaXC::PruningScheme>(module, "PruningScheme")
      .value("UNPRUNED", SkalaXC::PruningScheme::Unpruned)
      .value("ROBUST", SkalaXC::PruningScheme::Robust)
      .value("TREUTLER", SkalaXC::PruningScheme::Treutler);
  nb::enum_<SkalaXC::XCWeightAlg>(module, "XCWeightAlgorithm")
      .value("NOT_PARTITIONED", SkalaXC::XCWeightAlg::NOTPARTITIONED)
      .value("BECKE", SkalaXC::XCWeightAlg::Becke)
      .value("SSF", SkalaXC::XCWeightAlg::SSF)
      .value("LKO", SkalaXC::XCWeightAlg::LKO);
  nb::enum_<SkalaXC::TimingStatus>(module, "TimingStatus")
      .value("UNAVAILABLE", SkalaXC::TimingStatus::Unavailable)
      .value("PENDING", SkalaXC::TimingStatus::Pending)
      .value("COMPLETE", SkalaXC::TimingStatus::Complete);
  nb::enum_<SkalaXC::TimingMetric>(module, "TimingMetric")
      .value("MODEL_LOAD", SkalaXC::TimingMetric::ModelLoad)
      .value("FEATURE_CONSTRUCTION", SkalaXC::TimingMetric::FeatureConstruction)
      .value("MODEL_BATCH_PACKING", SkalaXC::TimingMetric::ModelBatchPacking)
      .value("MODEL_FORWARD", SkalaXC::TimingMetric::ModelForward)
      .value("MODEL_BACKWARD", SkalaXC::TimingMetric::ModelBackward)
      .value("POTENTIAL_MAPPING", SkalaXC::TimingMetric::PotentialMapping)
      .value("AO_ASSEMBLY", SkalaXC::TimingMetric::AOAssembly)
      .value("GRADIENT_ASSEMBLY", SkalaXC::TimingMetric::GradientAssembly)
      .value("MPI_REDUCTION", SkalaXC::TimingMetric::MPIReduction)
      .value("TOTAL_EXC_VXC", SkalaXC::TimingMetric::TotalEXCVXC)
      .value("TOTAL_EXC_GRADIENT", SkalaXC::TimingMetric::TotalEXCGradient);

#ifdef SKALAXC_HAS_CUDA
  nb::class_<SkalaXC::DeviceRuntimeSettings>(module, "DeviceRuntimeSettings")
      .def(nb::init<>())
      .def_rw("device_id", &SkalaXC::DeviceRuntimeSettings::device_id)
      .def_rw("memory_fraction",
              &SkalaXC::DeviceRuntimeSettings::memory_fraction);
#endif
  nb::class_<SkalaXC::TimingSettings>(module, "TimingSettings")
      .def(nb::init<>())
      .def_rw("verbose", &SkalaXC::TimingSettings::verbose)
      .def_rw("debug_logging", &SkalaXC::TimingSettings::debug_logging);
  nb::class_<SkalaXC::MolecularWeightsSettings>(module,
                                                "MolecularWeightsSettings")
      .def(nb::init<>())
      .def_rw("weight_algorithm",
              &SkalaXC::MolecularWeightsSettings::weight_alg);
  nb::class_<SkalaXC::IntegratorSettingsEXC_GRAD>(module, "GradientSettings")
      .def(nb::init<>())
      .def_rw("include_weight_derivatives",
              &SkalaXC::IntegratorSettingsEXC_GRAD::include_weight_derivatives);

  nb::class_<SkalaXC::TimingValue>(module, "TimingValue")
      .def_ro("last_nanoseconds", &SkalaXC::TimingValue::last_nanoseconds)
      .def_ro("total_nanoseconds", &SkalaXC::TimingValue::total_nanoseconds)
      .def_ro("call_count", &SkalaXC::TimingValue::call_count)
      .def_ro("status", &SkalaXC::TimingValue::status);
  nb::class_<SkalaXC::DiagnosticsSnapshot>(module, "DiagnosticsSnapshot")
      .def_ro("backend", &SkalaXC::DiagnosticsSnapshot::backend)
      .def_ro("rank", &SkalaXC::DiagnosticsSnapshot::rank)
      .def_ro("communicator_size",
              &SkalaXC::DiagnosticsSnapshot::communicator_size)
      .def_ro("device_id", &SkalaXC::DiagnosticsSnapshot::device_id)
      .def_ro("openmp_threads", &SkalaXC::DiagnosticsSnapshot::openmp_threads)
      .def_ro("device_memory_fraction",
              &SkalaXC::DiagnosticsSnapshot::device_memory_fraction)
      .def_ro("domain_batch_mode",
              &SkalaXC::DiagnosticsSnapshot::domain_batch_mode)
      .def_ro("exc_vxc_calls", &SkalaXC::DiagnosticsSnapshot::exc_vxc_calls)
      .def_ro("exc_gradient_calls",
              &SkalaXC::DiagnosticsSnapshot::exc_gradient_calls)
      .def_ro("model_batches", &SkalaXC::DiagnosticsSnapshot::model_batches)
      .def_ro("domains", &SkalaXC::DiagnosticsSnapshot::domains)
      .def_ro("tasks", &SkalaXC::DiagnosticsSnapshot::tasks)
      .def_ro("points", &SkalaXC::DiagnosticsSnapshot::points)
      .def_ro("local_atoms", &SkalaXC::DiagnosticsSnapshot::local_atoms)
      .def("timing", &SkalaXC::DiagnosticsSnapshot::timing,
           nb::rv_policy::reference_internal, "metric"_a);

  nb::class_<SkalaXC::Atom>(module, "Atom")
      .def(
          "__init__",
          [](SkalaXC::Atom* atom, std::int64_t atomic_number, double x,
             double y, double z) {
            new (atom)
                SkalaXC::Atom(SkalaXC::AtomicNumber(atomic_number), x, y, z);
          },
          "atomic_number"_a, "x"_a, "y"_a, "z"_a)
      .def_prop_rw(
          "atomic_number",
          [](const SkalaXC::Atom& atom) { return atom.Z.raw(); },
          [](SkalaXC::Atom& atom, std::int64_t value) {
            atom.Z = SkalaXC::AtomicNumber(value);
          })
      .def_rw("x", &SkalaXC::Atom::x)
      .def_rw("y", &SkalaXC::Atom::y)
      .def_rw("z", &SkalaXC::Atom::z);

  nb::class_<SkalaXC::Molecule>(module, "Molecule")
      .def(nb::init<>())
#ifdef SKALAXC_HAS_HDF5
      .def_static(
          "from_hdf5",
          [](const std::string& path, const std::string& dataset) {
            SkalaXC::Molecule molecule;
            SkalaXC::read_hdf5_record(molecule, path, dataset);
            return molecule;
          },
          "path"_a, "dataset"_a = "/MOLECULE")
#endif
      .def("append",
           [](SkalaXC::Molecule& molecule, const SkalaXC::Atom& atom) {
             molecule.push_back(atom);
           })
      .def("__len__", &SkalaXC::Molecule::natoms)
      .def(
          "__getitem__",
          [](SkalaXC::Molecule& molecule, std::size_t index) -> SkalaXC::Atom& {
            if (index >= molecule.size()) throw nb::index_error();
            return molecule[index];
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("natoms", &SkalaXC::Molecule::natoms);

  nb::class_<Shell>(module, "Shell")
      .def(
          "__init__",
          [](Shell* shell, std::int32_t angular_momentum, bool pure,
             const std::vector<double>& exponents,
             const std::vector<double>& coefficients,
             const Shell::cart_array& center, bool normalize) {
            if (exponents.empty() || exponents.size() > 32 ||
                coefficients.size() != exponents.size())
              throw nb::value_error(
                  "exponents and coefficients must have equal lengths in "
                  "the range [1, 32]");
            Shell::prim_array alpha{};
            Shell::prim_array coeff{};
            std::copy(exponents.begin(), exponents.end(), alpha.begin());
            std::copy(coefficients.begin(), coefficients.end(), coeff.begin());
            new (shell) Shell(
                SkalaXC::PrimSize(static_cast<std::int32_t>(exponents.size())),
                SkalaXC::AngularMomentum(angular_momentum),
                SkalaXC::SphericalType(pure ? 1 : 0), alpha, coeff, center,
                normalize);
          },
          "angular_momentum"_a, "pure"_a, "exponents"_a, "coefficients"_a,
          "center"_a, "normalize"_a = true)
      .def_prop_ro("nprim", &Shell::nprim)
      .def_prop_ro("angular_momentum", &Shell::l)
      .def_prop_ro("pure", [](const Shell& shell) { return shell.pure() != 0; })
      .def_prop_ro("normalized", &Shell::normalized)
      .def_prop_ro("size", &Shell::size)
      .def_prop_ro("exponents",
                   [](const Shell& shell) {
                     return std::vector<double>(
                         shell.alpha_data(),
                         shell.alpha_data() + shell.nprim());
                   })
      .def_prop_ro("coefficients",
                   [](const Shell& shell) {
                     return std::vector<double>(
                         shell.coeff_data(),
                         shell.coeff_data() + shell.nprim());
                   })
      .def_prop_ro("center", [](const Shell& shell) {
        return Shell::cart_array{shell.O_data()[0], shell.O_data()[1],
                                 shell.O_data()[2]};
      });

  nb::class_<SkalaXC::BasisSet<double>>(module, "BasisSet")
      .def(nb::init<>())
#ifdef SKALAXC_HAS_HDF5
      .def_static(
          "from_hdf5",
          [](const std::string& path, const std::string& dataset) {
            SkalaXC::BasisSet<double> basis;
            SkalaXC::read_hdf5_record(basis, path, dataset);
            return basis;
          },
          "path"_a, "dataset"_a = "/BASIS")
#endif
      .def("append", [](SkalaXC::BasisSet<double>& basis,
                        const Shell& shell) { basis.push_back(shell); })
      .def("__len__",
           [](const SkalaXC::BasisSet<double>& basis) { return basis.size(); })
      .def(
          "__getitem__",
          [](SkalaXC::BasisSet<double>& basis, std::size_t index) -> Shell& {
            if (index >= basis.size()) throw nb::index_error();
            return basis[index];
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("nshells", &SkalaXC::BasisSet<double>::nshells)
      .def_prop_ro("nbf", &SkalaXC::BasisSet<double>::nbf)
      .def_prop_ro("nbf_cart", &SkalaXC::BasisSet<double>::nbf_cart)
      .def_prop_ro("max_angular_momentum", &SkalaXC::BasisSet<double>::max_l);

  nb::class_<SkalaXC::functional_type>(module, "Functional")
      .def(nb::init<std::string>(), "model"_a)
      .def_prop_ro("model", &SkalaXC::functional_type::model)
      .def_prop_ro("empty", &SkalaXC::functional_type::empty);

  auto runtime =
      nb::class_<PythonRuntimeEnvironment>(module, "RuntimeEnvironment");
#ifdef SKALAXC_HAS_MPI
  if (import_mpi4py() < 0) throw nb::python_error();
  verify_mpi_implementation();
  runtime
      .def(
          "__init__",
          [](PythonRuntimeEnvironment* environment, nb::object communicator) {
            const MPI_Comm native = mpi4py_communicator(communicator);
            new (environment) PythonRuntimeEnvironment(std::move(communicator),
                                                       native, false);
          },
          "communicator"_a.none())
#ifdef SKALAXC_HAS_CUDA
      .def(
          "__init__",
          [](PythonRuntimeEnvironment* environment, nb::object communicator,
             SkalaXC::DeviceRuntimeSettings settings) {
            const MPI_Comm native = mpi4py_communicator(communicator);
            new (environment) PythonRuntimeEnvironment(std::move(communicator),
                                                       native, true, settings);
          },
          "communicator"_a.none(), "settings"_a)
#endif
      ;
#else
  runtime.def(nb::init<>());
#ifdef SKALAXC_HAS_CUDA
  runtime.def(nb::init<SkalaXC::DeviceRuntimeSettings>(), "settings"_a);
#endif
#endif
  runtime
      .def_prop_ro("rank",
                   [](const PythonRuntimeEnvironment& environment) {
                     return environment.value.comm_rank();
                   })
      .def_prop_ro("size", [](const PythonRuntimeEnvironment& environment) {
        return environment.value.comm_size();
      });

  nb::class_<SkalaXC::MolGrid>(module, "MolGrid");
  nb::class_<SkalaXC::MolGridFactory>(module, "MolGridFactory")
      .def_static(
          "create_default",
          [](const SkalaXC::Molecule& molecule,
             SkalaXC::PruningScheme pruning_scheme, std::int64_t batch_size,
             SkalaXC::RadialQuad radial_quad,
             SkalaXC::AtomicGridSizeDefault grid_size) {
            if (batch_size <= 0)
              throw nb::value_error("batch_size must be positive");
            return SkalaXC::MolGridFactory::create_default_molgrid(
                molecule, pruning_scheme, SkalaXC::BatchSize(batch_size),
                radial_quad, grid_size);
          },
          "molecule"_a, "pruning_scheme"_a = SkalaXC::PruningScheme::Unpruned,
          "batch_size"_a = 512,
          "radial_quad"_a = SkalaXC::RadialQuad::MuraKnowles,
          "grid_size"_a = SkalaXC::AtomicGridSizeDefault::UltraFineGrid);

  nb::class_<PythonLoadBalancer>(module, "LoadBalancer")
      .def_prop_ro("nbf",
                   [](const PythonLoadBalancer& load_balancer) {
                     return load_balancer.nbf;
                   })
      .def_prop_ro("natoms", [](const PythonLoadBalancer& load_balancer) {
        return load_balancer.natoms;
      });
  nb::class_<SkalaXC::LoadBalancerFactory>(module, "LoadBalancerFactory")
      .def(nb::init<SkalaXC::ExecutionSpace, std::string>(),
           "execution_space"_a, "kernel"_a = "Default")
      .def(
          "get_instance",
          [](SkalaXC::LoadBalancerFactory& factory,
             const PythonRuntimeEnvironment& runtime,
             const SkalaXC::Molecule& molecule, const SkalaXC::MolGrid& grid,
             const SkalaXC::BasisSet<double>& basis) {
            return PythonLoadBalancer(
                factory.get_instance(runtime.value, molecule, grid, basis),
                basis.nbf(), static_cast<std::int64_t>(molecule.natoms()),
                runtime.communicator_owner);
          },
          "runtime"_a, "molecule"_a, "grid"_a, "basis"_a);

  nb::class_<SkalaXC::MolecularWeights>(module, "MolecularWeights")
      .def("modify_weights", [](const SkalaXC::MolecularWeights& weights,
                                PythonLoadBalancer& load_balancer) {
        weights.modify_weights(load_balancer.value);
      });
  nb::class_<SkalaXC::MolecularWeightsFactory>(module,
                                               "MolecularWeightsFactory")
      .def(nb::init<SkalaXC::ExecutionSpace, std::string,
                    SkalaXC::MolecularWeightsSettings>(),
           "execution_space"_a, "kernel"_a = "Default",
           "settings"_a = SkalaXC::MolecularWeightsSettings{})
      .def("get_instance", &SkalaXC::MolecularWeightsFactory::get_instance);

  nb::class_<PythonIntegrator>(
      module, "XCIntegrator",
      "ML XC integrator. Instances are not safe for concurrent calls; "
      "serialize shared access or use one integrator per thread.")
      .def(
          "eval_exc_vxc",
          [](PythonIntegrator& integrator, const InputMatrix& scalar_density,
             const InputMatrix& spin_density) {
            Matrix scalar = matrix_view(scalar_density, integrator.nbf);
            Matrix spin = matrix_view(spin_density, integrator.nbf);
            auto result = [&] {
              nb::gil_scoped_release release;
              return integrator.value.eval_exc_vxc(scalar, spin);
            }();

            return nb::make_tuple(
                std::get<0>(result),
                move_matrix_to_numpy(std::move(std::get<1>(result))),
                move_matrix_to_numpy(std::move(std::get<2>(result))));
          },
          "scalar_density"_a, "spin_density"_a,
          "Evaluate UKS XC energy and potential. Releases the Python GIL; "
          "do not call concurrently on the same instance.")
      .def(
          "eval_exc_grad",
          [](PythonIntegrator& integrator, const InputMatrix& scalar_density,
             const InputMatrix& spin_density,
             const SkalaXC::IntegratorSettingsEXC_GRAD* settings) {
            Matrix scalar = matrix_view(scalar_density, integrator.nbf);
            Matrix spin = matrix_view(spin_density, integrator.nbf);
            const SkalaXC::IntegratorSettingsEXC_GRAD default_settings{};

            auto result = [&] {
              nb::gil_scoped_release release;
              return integrator.value.eval_exc_grad(
                  scalar, spin, settings ? *settings : default_settings);
            }();

            return move_gradient_to_numpy(std::move(result), integrator.natoms);
          },
          "scalar_density"_a, "spin_density"_a,
          "settings"_a.none() = nb::none(),
          "Evaluate the UKS XC nuclear gradient. Releases the Python GIL; "
          "do not call concurrently on the same instance.")
      .def("diagnostics",
           [](const PythonIntegrator& integrator) {
             return integrator.value.diagnostics();
           })
      .def("reset_diagnostics",
           [](PythonIntegrator& integrator) {
             integrator.value.reset_diagnostics();
           })
      .def_prop_ro(
          "nbf",
          [](const PythonIntegrator& integrator) { return integrator.nbf; })
      .def_prop_ro("natoms", [](const PythonIntegrator& integrator) {
        return integrator.natoms;
      });
  nb::class_<IntegratorFactory>(module, "XCIntegratorFactory")
      .def(nb::init<SkalaXC::ExecutionSpace, SkalaXC::TimingSettings,
                    SkalaXC::DomainBatchMode>(),
           "execution_space"_a, "timing_settings"_a = SkalaXC::TimingSettings{},
           "domain_batch_mode"_a = SkalaXC::DomainBatchMode::Conservative)
      .def(
          "get_instance",
          [](IntegratorFactory& factory, const SkalaXC::functional_type& func,
             const PythonLoadBalancer& load_balancer) {
            return PythonIntegrator(
                factory.get_instance(func, load_balancer.value),
                load_balancer.nbf, load_balancer.natoms,
                load_balancer.communicator_owner);
          },
          "functional"_a, "load_balancer"_a);
}

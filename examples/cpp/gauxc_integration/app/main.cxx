// For GauXC core functionality
#include <gauxc/exceptions.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>

// For loading data from HDF5 files
#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>

// For providing matrix implementation
#define EIGEN_DONT_VECTORIZE
#define EIGEN_NO_CUDA
#include <Eigen/Core>
using matrix = Eigen::MatrixXd;

// For command line interface
#include <CLI/CLI.hpp>

GauXC::RuntimeEnvironment
get_runtime()
{
#ifdef GAUXC_HAS_DEVICE
  auto rt = GauXC::DeviceRuntimeEnvironment( GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9 );
  // Calculate GauXC Device buffer size
  size_t available_mem, total_mem;
  cudaMemGetInfo(&available_mem, &total_mem);
  int device_id;
  cudaGetDevice(&device_id);
  size_t sz = 0.9 * available_mem;  
  void* p;
  cudaMallocAsync(&p, sz, 0);
  cudaStreamSynchronize(0);
  rt.set_buffer(p, sz);
#else
  auto rt = GauXC::RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif
  return rt;
}

// Load molecule from HDF5 dataset
GauXC::Molecule
read_molecule(std::string ref_file)
{
  GauXC::Molecule mol;
  GauXC::read_hdf5_record(mol, ref_file, "/MOLECULE");
  return mol;
}

// Load basis from HDF5 dataset
GauXC::BasisSet<double>
read_basis(std::string ref_file, double basis_tol)
{
  GauXC::BasisSet<double> basis;
  GauXC::read_hdf5_record(basis, ref_file, "/BASIS");
  for(auto& shell : basis){
    shell.set_shell_tolerance(basis_tol);
  }
  return basis;
}

GauXC::AtomicGridSizeDefault
read_atomic_grid_size(std::string spec)
{
    std::map<std::string, GauXC::AtomicGridSizeDefault> mg_map = {
      {"fine",      GauXC::AtomicGridSizeDefault::FineGrid},
      {"ultrafine", GauXC::AtomicGridSizeDefault::UltraFineGrid},
      {"superfine", GauXC::AtomicGridSizeDefault::SuperFineGrid},
      {"gm3",       GauXC::AtomicGridSizeDefault::GM3},
      {"gm5",       GauXC::AtomicGridSizeDefault::GM5}
    };
    return mg_map.at(spec);
}

GauXC::PruningScheme
read_pruning_scheme(std::string spec)
{
    std::map<std::string, GauXC::PruningScheme> prune_map = {
      {"unpruned", GauXC::PruningScheme::Unpruned},
      {"robust",   GauXC::PruningScheme::Robust},
      {"treutler", GauXC::PruningScheme::Treutler}
    };
    return prune_map.at(spec);
}

GauXC::RadialQuad
read_radial_quad(std::string spec)
{
    std::map<std::string, GauXC::RadialQuad> rad_quad_map = {
      {"becke",             GauXC::RadialQuad::Becke},
      {"muraknowles",       GauXC::RadialQuad::MuraKnowles},
      {"treutlerahlrichs",  GauXC::RadialQuad::TreutlerAhlrichs},
      {"murrayhandylaming", GauXC::RadialQuad::MurrayHandyLaming},
    };
    return rad_quad_map.at(spec);
}

std::pair<matrix, matrix>
read_density_matrix(std::string ref_file)
{
  HighFive::File h5file(ref_file, HighFive::File::ReadOnly);

  auto dset = h5file.getDataSet("/DENSITY_SCALAR");
  auto dims = dset.getDimensions();
  auto P_s = matrix(dims[0], dims[1]);
  auto P_z = matrix(dims[0], dims[1]);

  dset.read(P_s.data());

  dset = h5file.getDataSet("/DENSITY_Z");
  dset.read(P_z.data());

  return std::make_pair(P_s, P_z);
}

int
main(int argc, char** argv)
{
#ifdef GAUXC_HAS_MPI
  MPI_Init(NULL, NULL);
#endif
  {
    std::string input_file;
    std::string model;
    std::string grid_spec = "fine";
    std::string rad_quad_spec = "muraknowles";
    std::string prune_spec = "robust";
    std::string lb_exec_space_str = "host";
    std::string int_exec_space_str = "host";
    int batch_size = 512;
    double basis_tol = 1e-10;
    {
      auto string_to_lower = CLI::Validator(
        [](auto& str){
          std::transform(str.begin(), str.end(), str.begin(), ::tolower);
          return "";
        }, std::string(""), std::string("argument is case-insensitive"));
      CLI::App app{"Skala GauXC driver"};
      app.option_defaults()->always_capture_default();
      app.add_option("input", input_file, "Input file in HDF5 format")->required()->check(CLI::ExistingFile);
      app.add_option("--model", model, "Model checkpoint to evaluate")->required();
      app.add_option("--grid-size", grid_spec, "Grid specification (fine|ultrafine|superfine|gm3|gm5)")->transform(string_to_lower);
      app.add_option("--radial-quad", rad_quad_spec, "Radial quadrature specification (becke|muraknowles|treutlerahlrichs|murrayhandylaming)")->transform(string_to_lower);
      app.add_option("--prune-scheme", prune_spec, "Pruning scheme (unpruned|robust|treutler)")->transform(string_to_lower);
      app.add_option("--lb-exec-space", lb_exec_space_str, "Load balancing execution space")->transform(string_to_lower);
      app.add_option("--int-exec-space", int_exec_space_str, "Integration execution space")->transform(string_to_lower);
      app.add_option("--batch-size", batch_size, "");
      app.add_option("--basis-tol", basis_tol, "");
      CLI11_PARSE(app, argc, argv);
    }
    // Create runtime
    auto rt = get_runtime();
    auto world_rank = rt.comm_rank();
    auto world_size = rt.comm_size();

    if (!world_rank) {
      std::cout << std::boolalpha;
      std::cout << "Configuration" << std::endl
                << "-> Input file        : " << input_file << std::endl
                << "-> Model             : " << model << std::endl
                << "-> Grid              : " << grid_spec << std::endl
                << "-> Radial quadrature : " << rad_quad_spec << std::endl
                << "-> Pruning scheme    : " << prune_spec << std::endl
                << std::endl;
    }

    // Get molecule (atomic numbers and cartesian coordinates)
    auto mol = read_molecule(input_file);

    // Get basis set
    auto basis = read_basis(input_file, basis_tol);

    // Define molecular grid from grid size, radial quadrature and pruning scheme
    auto grid = GauXC::MolGridFactory::create_default_molgrid(
      mol,
      read_pruning_scheme(prune_spec),
      GauXC::BatchSize(batch_size),
      read_radial_quad(rad_quad_spec),
      read_atomic_grid_size(grid_spec));

    // Choose whether we run on host or device
  #ifdef GAUXC_HAS_DEVICE
    std::map<std::string, GauXC::ExecutionSpace> exec_space_map = {
      { "host",   GauXC::ExecutionSpace::Host },
      { "device", GauXC::ExecutionSpace::Device }
    };

    auto lb_exec_space = exec_space_map.at(lb_exec_space_str);
    auto int_exec_space = exec_space_map.at(int_exec_space_str);
  #else
    auto lb_exec_space  = GauXC::ExecutionSpace::Host;
    auto int_exec_space = GauXC::ExecutionSpace::Host;
  #endif

    // Setup load balancer based on molecule, grid and basis set
    GauXC::LoadBalancerFactory lb_factory(lb_exec_space, "Replicated");
    auto lb = lb_factory.get_shared_instance(rt, mol, grid, basis);

    // Apply partitioning weights to the molecule grid
    GauXC::MolecularWeightsFactory mw_factory(int_exec_space, "Default", 
      GauXC::MolecularWeightsSettings{} );
    auto mw = mw_factory.get_instance();
    mw.modify_weights(*lb);

    // Setup exchange-correlation integrator
    GauXC::functional_type func;
    GauXC::XCIntegratorFactory<matrix> integrator_factory(int_exec_space, "Replicated", "Default", "Default", "Default");
    auto integrator = integrator_factory.get_instance(func, lb);

    // Configure model checkpoint
    GauXC::OneDFTSettings onedft_settings;
    onedft_settings.model = model;

    // Load density matrix from input
    matrix P_s, P_z;
    std::tie(P_s, P_z) = read_density_matrix(input_file);

#ifdef GAUXC_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto xc_int_start = std::chrono::high_resolution_clock::now();

    // Integrate exchange correlation energy
    double EXC;
    matrix VXC_s, VXC_z;
    std::tie(EXC, VXC_s, VXC_z) = integrator.eval_exc_vxc_onedft(P_s, P_z, onedft_settings);

#ifdef GAUXC_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto xc_int_end = std::chrono::high_resolution_clock::now();
    double xc_int_dur = std::chrono::duration<double>(xc_int_end - xc_int_start).count();

    std::cout << std::scientific << std::setprecision(12);
    if(!world_rank) {
      std::cout << "EXC          = " << EXC << " Eh" << std::endl
                << "|VXC(a+b)|_F = " << VXC_s.norm() << std::endl
                << "|VXC(a-b)|_F = " << VXC_z.norm() << std::endl
                << "Runtime XC   = " << xc_int_dur << " s" << std::endl
                << std::endl;
    }
  }
#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif
}

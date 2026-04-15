// Standard C libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

// For GauXC core functionality
#include <gauxc/c/types.h>
#include <gauxc/c/status.h>
#include <gauxc/c/molecule.h>
#include <gauxc/c/basisset.h>
#include <gauxc/c/molgrid.h>
#include <gauxc/c/runtime_environment.h>
#include <gauxc/c/load_balancer.h>
#include <gauxc/c/molecular_weights.h>
#include <gauxc/c/functional.h>
#include <gauxc/c/xc_integrator.h>

// For HDF5 I/O
#include <hdf5.h>
#include <gauxc/c/hdf5.h>

// For command line interface
#include <argtable3.h>

enum GauXC_ExecutionSpace
read_execution_space(GauXCStatus* status, const char* exec_space_str)
{
  status->code = 0;
  if(strcmp(exec_space_str, "host") == 0)
    return GauXC_ExecutionSpace_Host;
  if(strcmp(exec_space_str, "device") == 0)
    return GauXC_ExecutionSpace_Device;
  status->message = "Invalid execution space specification";
  status->code = 1;
}

enum GauXC_RadialQuad
read_radial_quad(GauXCStatus* status, const char* rad_quad_spec)
{
  status->code = 0;
  if(strcmp(rad_quad_spec, "becke") == 0)
    return GauXC_RadialQuad_Becke;
  if(strcmp(rad_quad_spec, "muraknowles") == 0)
    return GauXC_RadialQuad_MuraKnowles;
  if(strcmp(rad_quad_spec, "treutlerahlrichs") == 0)
    return GauXC_RadialQuad_TreutlerAhlrichs;
  if(strcmp(rad_quad_spec, "murrayhandylaming") == 0)
    return GauXC_RadialQuad_MurrayHandyLaming;
  status->message = "Invalid radial quadrature specification";
  status->code = 1;
}

enum GauXC_AtomicGridSizeDefault
read_atomic_grid_size(GauXCStatus* status, const char* spec)
{
    status->code = 0;
    if(strcmp(spec, "fine") == 0)
      return GauXC_AtomicGridSizeDefault_FineGrid;
    if(strcmp(spec, "ultrafine") == 0)
      return GauXC_AtomicGridSizeDefault_UltraFineGrid;
    if(strcmp(spec, "superfine") == 0)
      return GauXC_AtomicGridSizeDefault_SuperFineGrid;
    if(strcmp(spec, "gm3") == 0)
      return GauXC_AtomicGridSizeDefault_GM3;
    if(strcmp(spec, "gm5") == 0)
      return GauXC_AtomicGridSizeDefault_GM5;
    status->message = "Invalid atomic grid size specification";
    status->code = 1;
}

enum GauXC_PruningScheme
read_pruning_scheme(GauXCStatus* status, const char* spec)
{
  status->code = 0;
  if(strcmp(spec, "unpruned") == 0)
    return GauXC_PruningScheme_Unpruned;
  if(strcmp(spec, "robust") == 0)
    return GauXC_PruningScheme_Robust;
  if(strcmp(spec, "treutler") == 0)
    return GauXC_PruningScheme_Treutler;
  status->message = "Invalid pruning scheme specification";
  status->code = 1;
}

void
read_matrix_from_hdf5_record(GauXCStatus* status, double** mat, int64_t* n, const char* file, const char* dataset)
{
  status->code = 0;
  *mat = NULL;
  bool file_opened = false, dataset_opened = false, dataspace_opened = false;
  hid_t hdf5_file = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (hdf5_file < 0) {
    status->message = "Failed to open HDF5 file"; status->code = 1;
    goto err;
  }
  file_opened = true;

  hid_t hdf5_dataset = H5Dopen2(hdf5_file, dataset, H5P_DEFAULT);
  if (hdf5_dataset < 0) {
    status->message = "Failed to open HDF5 dataset";
    status->code = 1;
    goto err;
  }
  dataset_opened = true;

  hid_t hdf5_dataspace = H5Dget_space(hdf5_dataset);
  if (hdf5_dataspace < 0) {
    status->message = "Failed to get HDF5 dataspace";
    status->code = 1;
    goto err;
  }
  dataspace_opened = true;

  int ndims = H5Sget_simple_extent_ndims(hdf5_dataspace);
  if (ndims != 2) {
    status->message = "Expected 2D dataset in HDF5 file";
    status->code = 1;
    goto err;
  }

  hsize_t dims[2];
  H5Sget_simple_extent_dims(hdf5_dataspace, dims, NULL);
  *n = dims[0];
  if (dims[1] != dims[0]) {
    status->message = "Expected square matrix dataset in HDF5 file";
    status->code = 1;
    goto err;
  }

  *mat = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  if (*mat == NULL) {
    status->message = "Failed to allocate memory for matrix";
    status->code = 1;
    goto err;
  }

  herr_t err = H5Dread(hdf5_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *mat);
  if (err < 0) {
    status->message = "Failed to read matrix data from HDF5 dataset";
    status->code = 1;
    free(*mat);
    *mat = NULL;
    goto err;
  }  

err:
  if (dataspace_opened) H5Sclose(hdf5_dataspace);
  if (dataset_opened) H5Dclose(hdf5_dataset);
  if (file_opened) H5Fclose(hdf5_file);
}

inline static double
matrix_norm(const int64_t m, const int64_t n, const double* mat, const int64_t ld)
{
  double norm = 0.0;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double val = mat[i * ld + j];
      norm += val * val;
    }
  }
  return sqrt(norm);
}

inline static char * arg_strcpy(struct arg_str * arg, char * default_str)
{
  if(arg->count == 0 && default_str != NULL) {
    return strcpy(malloc(strlen(default_str) + 1), default_str);
  }
  if(arg->count > 0) {
    const char* sval = arg->sval[arg->count - 1];
    return strcpy(malloc(strlen(sval) + 1), sval);
  }
  return NULL;
}

inline static char * lowercase(char * str)
{
  for(char * p = str; *p; ++p) *p = tolower(*p);
  return str;
}

int
main(int argc, char** argv)
{
#ifdef GAUXC_HAS_MPI
  MPI_Init(NULL, NULL);
#endif
  struct arg_lit *help; 
  struct arg_file *input_file_;
  struct arg_int *batch_size_;
  struct arg_dbl *basis_tol_;
  struct arg_str *model_, *grid_spec_, *rad_quad_spec_, *prune_spec_;
  struct arg_str *lb_exec_space_, *int_exec_space_;
  struct arg_end *end;

  void* argtable[] = {
    input_file_    = arg_filen(NULL, NULL, "<file>", 1, 1, "Input file containing molecular geometry and density matrix"),
    model_         = arg_strn(NULL, "model", "<str>", 1, 1, "OneDFT model to use, can be a path to a checkpoint"),
    grid_spec_     = arg_strn(NULL, "grid-spec", "<str>", 0, 1, "Atomic grid size specification (default: Fine)"),
                     arg_rem(NULL, "Possible values are: Fine, UltraFine, SuperFine, GM3, GM5"),
    rad_quad_spec_ = arg_strn(NULL, "radial-quad", "<str>", 0, 1, "Radial quadrature scheme (default: MuraKnowles)"),
                     arg_rem(NULL, "Possible values are: Becke, MuraKnowles, TreutlerAhlrichs, MurrayHandyLaming"),
    prune_spec_    = arg_strn(NULL, "prune-scheme", "<str>", 0, 1, "Pruning scheme (default: Robust)"),
                     arg_rem(NULL, "Possible values are: Unpruned, Robust, Treutler"),
    lb_exec_space_ = arg_strn(NULL, "lb-exec-space", "<str>", 0, 1, "Load balancer execution space"),
                     arg_rem(NULL, "Possible values are: Host, Device"),
    int_exec_space_= arg_strn(NULL, "int-exec-space", "<str>", 0, 1, "Integrator execution space"),
                     arg_rem(NULL, "Possible values are: Host, Device"),
    batch_size_    = arg_intn(NULL, "batch-size", "<int>", 0, 1, "Batch size for grid point processing (default: 512)"),
    basis_tol_     = arg_dbln(NULL, "basis-tol", "<double>", 0, 1, "Basis function evaluation tolerance (default: 1e-10)"),
    help           = arg_litn(NULL, "help", 0, 1, "Print this help and exit"),
    end = arg_end(20)
  };

  int nerrors = arg_parse(argc, argv, argtable);
  if(help->count > 0) {
    printf("Usage: %s", argv[0]);
    arg_print_syntax(stdout, argtable, "\n\n");
    printf("Options:\n");
    arg_print_glossary(stdout, argtable, "  %-25s %s\n");
#ifdef GAUXC_HAS_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
  }

  if (nerrors > 0 || input_file_->count == 0 || model_->count == 0) {
    printf("Usage: %s", argv[0]);
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_errors(stderr, end, argv[0]);
#ifdef GAUXC_HAS_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return EXIT_FAILURE;
  }

  char* input_file = strcpy(malloc(strlen(input_file_->filename[0]) + 1), input_file_->filename[0]);
  char* model = arg_strcpy(model_, NULL);
  char* grid_spec = lowercase(arg_strcpy(grid_spec_, "Fine"));
  char* rad_quad_spec = lowercase(arg_strcpy(rad_quad_spec_, "MuraKnowles"));
  char* prune_spec = lowercase(arg_strcpy(prune_spec_, "Robust"));
  char* lb_exec_space_str = lowercase(arg_strcpy(lb_exec_space_, "Host"));
  char* int_exec_space_str = lowercase(arg_strcpy(int_exec_space_, "Host"));
  int batch_size = batch_size_->count > 0 ? batch_size_->ival[0] : 512;
  double basis_tol = basis_tol_->count > 0 ? basis_tol_->dval[0] : 1e-10;

  // Clean up argtable memory
  arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

  // Memory management for GauXC
  void* objs[16];
  size_t nobj = 0;

  // Add handler for events like exceptions
  GauXCStatus status = {0, NULL};

  // Create runtime
  GauXCRuntimeEnvironment rt = gauxc_runtime_environment_new(&status GAUXC_MPI_CODE(, MPI_COMM_WORLD));
  objs[nobj++] = &rt;
  if (status.code) goto err;
  int world_rank = gauxc_runtime_environment_comm_rank(&status, rt);
  if (status.code) goto err;
  int world_size = gauxc_runtime_environment_comm_size(&status, rt);
  if (status.code) goto err;

  if(!world_rank && !status.code) {
    printf("Configuration\n");
    printf("-> Input file        : %s\n", input_file);
    printf("-> Model             : %s\n", model);
    printf("-> Grid              : %s\n", grid_spec);
    printf("-> Radial quadrature : %s\n", rad_quad_spec);
    printf("-> Pruning scheme    : %s\n", prune_spec);
    printf("\n");
  }

  // Get molecule (atomic numbers and cartesian coordinates)
  GauXCMolecule mol = gauxc_molecule_new(&status);
  objs[nobj++] = &mol;
  if (status.code) goto err;
  // Load molecule from HDF5 dataset
  gauxc_molecule_read_hdf5_record(&status, mol, input_file, "/MOLECULE");
  if (status.code) goto err;

  // Get basis set
  GauXCBasisSet basis = gauxc_basisset_new(&status);
  objs[nobj++] = &basis;
  if (status.code) goto err;
  // Load basis set from HDF5 dataset
  gauxc_basisset_read_hdf5_record(&status, basis, input_file, "/BASIS");
  if (status.code) goto err;

  // Define molecular grid from grid size, radial quadrature and pruning scheme
  enum GauXC_AtomicGridSizeDefault grid_type = read_atomic_grid_size(&status, grid_spec);
  if (status.code) goto err;
  enum GauXC_RadialQuad radial_quad = read_radial_quad(&status, rad_quad_spec);
  if (status.code) goto err;
  enum GauXC_PruningScheme pruning_scheme = read_pruning_scheme(&status, prune_spec);
  if (status.code) goto err;
  GauXCMolGrid grid = gauxc_molgrid_new_default(
    &status,
    mol,
    pruning_scheme,
    batch_size,
    radial_quad,
    grid_type);
  objs[nobj++] = &grid;
  if (status.code) goto err;

  // Choose whether we run on host or device
  enum GauXC_ExecutionSpace lb_exec_space, int_exec_space;
  lb_exec_space = read_execution_space(&status, lb_exec_space_str);
  if (status.code) goto err;
  int_exec_space = read_execution_space(&status, int_exec_space_str);
  if (status.code) goto err;

  // Setup load balancer based on molecule, grid and basis set
  GauXCLoadBalancerFactory lb_factory = gauxc_load_balancer_factory_new(&status, lb_exec_space, "Replicated");
  objs[nobj++] = &lb_factory;
  if (status.code) goto err;
  GauXCLoadBalancer lb = gauxc_load_balancer_factory_get_instance(&status, lb_factory, rt, mol, grid, basis);
  objs[nobj++] = &lb;
  if (status.code) goto err;

  // Apply partitioning weights to the molecule grid
  GauXCMolecularWeightsSettings settings = {GauXC_XCWeightAlg_SSF, false};
  GauXCMolecularWeightsFactory mw_factory = gauxc_molecular_weights_factory_new(&status, int_exec_space,
    "Default", settings);
  objs[nobj++] = &mw_factory;
  if (status.code) goto err;
  GauXCMolecularWeights mw = gauxc_molecular_weights_factory_get_instance(&status, mw_factory);
  if (status.code) goto err;
  gauxc_molecular_weights_modify_weights(&status, mw, lb);
  objs[nobj++] = &mw;
  if (status.code) goto err;

  // Setup exchange-correlation integrator
  GauXCFunctional func = gauxc_functional_from_string(&status, "PBE", true);
  objs[nobj++] = &func;
  if (status.code) goto err;
  GauXCIntegrator integrator = gauxc_integrator_new(&status, func, lb, int_exec_space,
    "Replicated", "Default", "Default", "Default");
  objs[nobj++] = &integrator;
  if (status.code) goto err;

  // Load density matrix from input
  int64_t nbf;
  double* P_s = NULL;
  double* P_z = NULL;
  read_matrix_from_hdf5_record(&status, &P_s, &nbf, input_file, "/DENSITY_SCALAR");
  if (status.code) goto err;
  read_matrix_from_hdf5_record(&status, &P_z, &nbf, input_file, "/DENSITY_Z");
  if (status.code) goto err;

#ifdef GAUXC_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Integrate exchange correlation energy
  double EXC;
  double* VXC_s = (double*)malloc(nbf * nbf * sizeof(double));
  double* VXC_z = (double*)malloc(nbf * nbf * sizeof(double));
  gauxc_integrator_eval_exc_vxc_onedft_uks(&status, integrator,
    nbf, nbf, P_s, nbf, P_z, nbf, model, &EXC, VXC_s, nbf, VXC_z, nbf);
  if (status.code) goto err;

#ifdef GAUXC_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if(!world_rank && !status.code) {
    printf("Results\n");
    printf("-> EXC          : %.10f\n", EXC);
    printf("-> |VXC(a+b)|_F : %.10f\n", matrix_norm(nbf, nbf, VXC_s, nbf));
    printf("-> |VXC(a-b)|_F : %.10f\n", matrix_norm(nbf, nbf, VXC_z, nbf));
    printf("\n");
  }

err:
  int error_code = status.code;
  if (error_code) {
    fprintf(stderr, "Error (code %d)", error_code);
    if (status.message) fprintf(stderr, ": %s", status.message);
    fprintf(stderr, "\n");
  }
  // Clean up memory
  free(input_file);
  free(model);
  free(grid_spec);
  free(rad_quad_spec);
  free(prune_spec);
  free(lb_exec_space_str);
  free(int_exec_space_str);
  free(P_s);
  free(P_z);
  free(VXC_s);
  free(VXC_z);
  gauxc_objects_delete(&status, objs, nobj);
  if (status.code) {
    fprintf(stderr, "Error during cleanup (code %d)", status.code);
    if (status.message) fprintf(stderr, ": %s", status.message);
    fprintf(stderr, "\n");
    error_code = status.code;
  }

#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif
  return error_code ? EXIT_FAILURE : EXIT_SUCCESS;
}

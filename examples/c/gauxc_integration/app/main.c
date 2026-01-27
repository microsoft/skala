// Standard C libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// For GauXC core functionality
#include <gauxc/status.h>
#include <gauxc/molecule.h>
#include <gauxc/basisset.h>
#include <gauxc/molgrid.h>
#include <gauxc/runtime_environment.h>
#include <gauxc/load_balancer.h>
#include <gauxc/molecular_weights.h>
#include <gauxc/functional.h>
#include <gauxc/xc_integrator.h>
#include <gauxc/matrix.h>

// For HDF5 I/O
#include <gauxc/external/hdf5.h>

// For command line interface
#include <argtable3.h>

// Generic delete macro
#define gauxc_delete(status, ptr) _Generic( (ptr), \
                 GauXCMolecule : gauxc_molecule_delete, \
                 GauXCBasisSet : gauxc_basisset_delete, \
                  GauXCMolGrid : gauxc_molgrid_delete, \
       GauXCRuntimeEnvironment : gauxc_runtime_environment_delete, \
             GauXCLoadBalancer : gauxc_load_balancer_delete, \
      GauXCLoadBalancerFactory : gauxc_load_balancer_factory_delete, \
         GauXCMolecularWeights : gauxc_molecular_weights_delete, \
  GauXCMolecularWeightsFactory : gauxc_molecular_weights_factory_delete, \
               GauXCFunctional : gauxc_functional_delete, \
               GauXCIntegrator : gauxc_integrator_delete, \
        GauXCIntegratorFactory : gauxc_integrator_factory_delete, \
                   GauXCMatrix : gauxc_matrix_delete \
                               )(status, &(ptr))

enum GauXC_ExecutionSpace
read_execution_space(GauXCStatus* status, const char* exec_space_str)
{
  status->code = 0;
  if(strcmp(exec_space_str, "host") == 0)
    return GauXC_ExecutionSpace_Host;
  if(strcmp(exec_space_str, "device") == 0)
    return GauXC_ExecutionSpace_Device;
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
    status->code = 1;
}

enum GauXC_PruningScheme
read_pruning_scheme(GauXCStatus* status, const char* spec)
{
  status->code = 0;
  if(strcmp(spec, "Unpruned") == 0)
    return GauXC_PruningScheme_Unpruned;
  if(strcmp(spec, "Robust") == 0)
    return GauXC_PruningScheme_Robust;
  if(strcmp(spec, "Treutler") == 0)
    return GauXC_PruningScheme_Treutler;
  status->code = 1;
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
    return EXIT_SUCCESS;
  }

  if (nerrors > 0 || input_file_->count == 0 || model_->count == 0) {
    printf("Usage: %s", argv[0]);
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_errors(stderr, end, argv[0]);
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

  // Add handler for events like exceptions
  GauXCStatus status;

  // Create runtime
  GauXCRuntimeEnvironment rt = gauxc_runtime_environment_new(&status GAUXC_MPI_CODE(, MPI_COMM_WORLD));
  int world_rank = gauxc_runtime_environment_comm_rank(&status, rt);
  int world_size = gauxc_runtime_environment_comm_size(&status, rt);

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
  // Load molecule from HDF5 dataset
  gauxc_molecule_read_hdf5_record(&status, mol, input_file, "/MOLECULE");

  // Get basis set
  GauXCBasisSet basis = gauxc_basisset_new(&status);
  // Load basis set from HDF5 dataset
  gauxc_basisset_read_hdf5_record(&status, basis, input_file, "/BASIS");

  // Define molecular grid from grid size, radial quadrature and pruning scheme
  enum GauXC_AtomicGridSizeDefault grid_type = read_atomic_grid_size(&status, grid_spec);
  enum GauXC_RadialQuad radial_quad = read_radial_quad(&status, rad_quad_spec);
  enum GauXC_PruningScheme pruning_scheme = read_pruning_scheme(&status, prune_spec);
  GauXCMolGrid grid = gauxc_molgrid_new_default(
    &status,
    mol,
    pruning_scheme,
    batch_size,
    radial_quad,
    grid_type);

  // Choose whether we run on host or device
  enum GauXC_ExecutionSpace lb_exec_space, int_exec_space;
  lb_exec_space = read_execution_space(&status, lb_exec_space_str);
  int_exec_space = read_execution_space(&status, int_exec_space_str);

  // Setup load balancer based on molecule, grid and basis set
  GauXCLoadBalancerFactory lb_factory = gauxc_load_balancer_factory_new(&status, lb_exec_space, "Replicated");
  GauXCLoadBalancer lb = gauxc_load_balancer_factory_get_shared_instance(&status, lb_factory, rt, mol, grid, basis);

  // Apply partitioning weights to the molecule grid
  GauXCMolecularWeightsSettings settings = {GauXC_XCWeightAlg_SSF, false};
  GauXCMolecularWeightsFactory mw_factory = gauxc_molecular_weights_factory_new(&status, int_exec_space,
    "Default", settings);
  GauXCMolecularWeights mw = gauxc_molecular_weights_factory_get_instance(&status, mw_factory);
  gauxc_molecular_weights_modify_weights(&status, mw, lb);

  // Setup exchange-correlation integrator
  GauXCFunctional func = gauxc_functional_from_string(&status, "PBE", true);
  GauXCIntegratorFactory integrator_factory = gauxc_integrator_factory_new(&status, int_exec_space,
    "Replicated", "Default", "Default", "Default");
  GauXCIntegrator integrator = gauxc_integrator_factory_get_instance(&status, integrator_factory, func, lb);

  // Configure model checkpoint
  // GauXCOneDFTSettings onedft_settings;
  // onedft_settings.model = model;

  // Load density matrix from input
  GauXCMatrix P_s = gauxc_matrix_empty(&status);
  GauXCMatrix P_z = gauxc_matrix_empty(&status);
  gauxc_matrix_read_hdf5_record(&status, P_s, input_file, "/DENSITY_SCALAR");
  gauxc_matrix_read_hdf5_record(&status, P_z, input_file, "/DENSITY_Z");

#ifdef GAUXC_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Integrate exchange correlation energy
  double EXC;
  GauXCMatrix VXC_s = gauxc_matrix_empty(&status);
  GauXCMatrix VXC_z = gauxc_matrix_empty(&status);

  gauxc_integrator_eval_exc_vxc_onedft_uks(&status, integrator, P_s, P_z, model, &EXC, &VXC_s, &VXC_z);

#ifdef GAUXC_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if(!world_rank && !status.code) {
    printf("Results\n");
    printf("-> EXC : %.10f\n", EXC);
    printf("\n");
  }

  // Clean up memory
  free(input_file);
  free(model);
  free(grid_spec);
  free(rad_quad_spec);
  free(prune_spec);
  free(lb_exec_space_str);
  free(int_exec_space_str);
  gauxc_delete(&status, mol);
  gauxc_delete(&status, basis);
  gauxc_delete(&status, grid);
  gauxc_delete(&status, rt);
  gauxc_delete(&status, lb_factory);
  gauxc_delete(&status, lb);
  gauxc_delete(&status, mw_factory);
  gauxc_delete(&status, mw);
  gauxc_delete(&status, func);
  gauxc_delete(&status, integrator_factory);
  gauxc_delete(&status, integrator);
  gauxc_delete(&status, P_s);
  gauxc_delete(&status, P_z);
  gauxc_delete(&status, VXC_s);
  gauxc_delete(&status, VXC_z);

#ifdef GAUXC_HAS_MPI
  MPI_Finalize();
#endif
}

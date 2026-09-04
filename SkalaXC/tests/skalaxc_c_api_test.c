/* SkalaXC black-box C test.
 *
 * A pure C consumer: it includes ONLY the public SkalaXC C header and links
 * ONLY libskalaxc (plus the C HDF5 library to load its own density input). It
 * has NO access to GauXC, LibTorch, or C++ symbols. Successful compilation and
 * link -- as C, not C++ -- proves the C API is self-contained and ABI-isolated.
 *
 * It drives the per-stage pipeline (runtime -> molecule/basis -> molgrid ->
 * load balancer -> molecular weights -> functional -> integrator), mirroring
 * the C++ and Fortran surfaces.
 */

#include <skalaxc/skalaxc.h>

#include <hdf5.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_doubles(hid_t file, const char* dset, double* buf) {
  hid_t d = H5Dopen2(file, dset, H5P_DEFAULT);
  herr_t st;
  if (d < 0) return -1;
  st = H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  H5Dclose(d);
  return st < 0 ? -1 : 0;
}

typedef struct error_call_result {
  skalaxc_status_t status;
  int output_reset;
} error_call_result_t;

typedef error_call_result_t (*error_call_t)(void);

static error_call_result_t null_runtime_output(void) {
  error_call_result_t result;
  result.status = skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
      MPI_COMM_WORLD,
#endif
      NULL);
  result.output_reset = 1;
  return result;
}

static error_call_result_t invalid_molecule_arrays(void) {
  double xyz[3] = {0.0, 0.0, 0.0};
  char sentinel;
  skalaxc_molecule_t output = (skalaxc_molecule_t)&sentinel;
  error_call_result_t result;
  result.status = skalaxc_molecule_from_arrays(1, NULL, xyz, &output);
  result.output_reset = output == NULL;
  return result;
}

static error_call_result_t null_functional_model(void) {
  char sentinel;
  skalaxc_functional_t output = (skalaxc_functional_t)&sentinel;
  error_call_result_t result;
  result.status = skalaxc_functional_create(NULL, &output);
  result.output_reset = output == NULL;
  return result;
}

static error_call_result_t null_exc_vxc_buffers(void) {
  double value = 0.0;
  error_call_result_t result;
  result.status = skalaxc_xc_integrator_eval_exc_vxc_uks(
      NULL, &value, &value, &value, &value, &value);
  result.output_reset = 1;
  return result;
}

static error_call_result_t null_gradient_buffer(void) {
  double value = 0.0;
  error_call_result_t result;
  result.status =
      skalaxc_xc_integrator_eval_exc_grad_uks(NULL, &value, &value, NULL);
  result.output_reset = 1;
  return result;
}

static error_call_result_t null_diagnostics_integrator(void) {
  skalaxc_diagnostics_snapshot_t diagnostics;
  error_call_result_t result;
  result.status = skalaxc_xc_integrator_get_diagnostics(NULL, &diagnostics);
  result.output_reset = 1;
  return result;
}

static error_call_result_t null_diagnostics_reset(void) {
  error_call_result_t result;
  result.status = skalaxc_xc_integrator_reset_diagnostics(NULL);
  result.output_reset = 1;
  return result;
}

static int run_error_contracts(int* total) {
  struct error_case {
    const char* name;
    error_call_t call;
    int require_output_reset;
  };
  static const struct error_case cases[] = {
      {"null runtime output", null_runtime_output, 0},
      {"invalid molecule arrays", invalid_molecule_arrays, 1},
      {"null functional model", null_functional_model, 1},
      {"null EXC/VXC buffers", null_exc_vxc_buffers, 0},
      {"null gradient buffer", null_gradient_buffer, 0},
      {"null diagnostics integrator", null_diagnostics_integrator, 0},
      {"null diagnostics reset", null_diagnostics_reset, 0},
  };
  int failures = 0;
  size_t index;

  for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    const error_call_result_t result = cases[index].call();
    const char* message = skalaxc_last_error_message();
    const int passed =
        result.status == SKALAXC_INVALID_ARGUMENT &&
        (!cases[index].require_output_reset || result.output_reset) &&
        message != NULL && strstr(message, "null argument") != NULL;
    printf("[%s] C error contract: %s\n", passed ? "PASS" : "FAIL",
           cases[index].name);
    if (!passed) ++failures;
    ++*total;
  }

  skalaxc_xc_integrator_destroy(NULL);
  skalaxc_functional_destroy(NULL);
  skalaxc_molecular_weights_destroy(NULL);
  skalaxc_load_balancer_destroy(NULL);
  skalaxc_molgrid_destroy(NULL);
  skalaxc_basisset_destroy(NULL);
  skalaxc_molecule_destroy(NULL);
  skalaxc_runtime_environment_destroy(NULL);
  skalaxc_device_runtime_settings_default(NULL);
  skalaxc_timing_settings_default(NULL);
  skalaxc_grid_settings_default(NULL);
  printf("[PASS] C null destruction/default initialization\n");
  ++*total;

  {
    const int queries_failed =
        skalaxc_runtime_environment_comm_rank(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL &&
        skalaxc_runtime_environment_comm_size(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL &&
        skalaxc_molecule_natoms(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL &&
        skalaxc_basisset_nbf(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL &&
        skalaxc_xc_integrator_nbf(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL &&
        skalaxc_xc_integrator_natoms(NULL) == -1 &&
        strstr(skalaxc_last_error_message(), "null argument") != NULL;
    printf("[%s] C null query sentinels and errors\n",
           queries_failed ? "PASS" : "FAIL");
    if (!queries_failed) ++failures;
    ++*total;
  }
  return failures;
}

static int run_hdf5_failure_contracts(const char* path, int* total) {
  char sentinel;
  skalaxc_molecule_t molecule = (skalaxc_molecule_t)&sentinel;
  skalaxc_basisset_t basis = (skalaxc_basisset_t)&sentinel;
  skalaxc_status_t status;
  int failures = 0;
  int passed;

  status = skalaxc_molecule_from_hdf5(path, "/MISSING_MOLECULE", &molecule);
  passed = status == SKALAXC_ERROR && molecule == NULL &&
           skalaxc_last_error_message() != NULL &&
           skalaxc_last_error_message()[0] != '\0';
  printf("[%s] C HDF5 molecule failure is atomic\n", passed ? "PASS" : "FAIL");
  if (!passed) ++failures;
  ++*total;

  status = skalaxc_basisset_from_hdf5(path, "/MISSING_BASIS", &basis);
  passed = status == SKALAXC_ERROR && basis == NULL &&
           skalaxc_last_error_message() != NULL &&
           skalaxc_last_error_message()[0] != '\0';
  printf("[%s] C HDF5 basis failure is atomic\n", passed ? "PASS" : "FAIL");
  if (!passed) ++failures;
  ++*total;

  skalaxc_molecule_destroy(molecule);
  skalaxc_basisset_destroy(basis);
  return failures;
}

static int check_invalid_enum(const char* name, skalaxc_status_t status,
                              int output_reset, const char* expected_message,
                              int* total) {
  const char* message = skalaxc_last_error_message();
  const int passed = status == SKALAXC_INVALID_ARGUMENT && output_reset &&
                     message != NULL &&
                     strstr(message, expected_message) != NULL;
  printf("[%s] C invalid enum: %s\n", passed ? "PASS" : "FAIL", name);
  ++*total;
  return passed ? 0 : 1;
}

static int run_invalid_enum_contracts(const char* path, int* total) {
  skalaxc_runtime_environment_t rt = NULL;
  skalaxc_molecule_t mol = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_molgrid_t mg = NULL;
  skalaxc_load_balancer_t lb = NULL;
  skalaxc_molecular_weights_t mw = NULL;
  skalaxc_functional_t func = NULL;
  skalaxc_grid_settings_t grid;
  skalaxc_integrator_settings_t integrator_settings;
  skalaxc_status_t status;
  char sentinel;
  int failures = 0;

  status = skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
      MPI_COMM_WORLD,
#endif
      &rt);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_molecule_from_hdf5(path, "/MOLECULE", &mol);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_basisset_from_hdf5(path, "/BASIS", &basis);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_molgrid_create_default(mol, NULL, &mg);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_load_balancer_create(SkalaXC_ExecutionSpace_Host, rt, mol,
                                        mg, basis, &lb);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_molecular_weights_create(SkalaXC_ExecutionSpace_Host,
                                            SkalaXC_XCWeightAlg_SSF, &mw);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_molecular_weights_modify_weights(mw, lb);
  if (status != SKALAXC_SUCCESS) goto setup_failed;
  status = skalaxc_functional_create("LDA", &func);
  if (status != SKALAXC_SUCCESS) goto setup_failed;

  skalaxc_grid_settings_default(&grid);
  {
    skalaxc_molgrid_t output = (skalaxc_molgrid_t)&sentinel;
    grid.pruning = (enum SkalaXC_PruningScheme)99;
    status = skalaxc_molgrid_create_default(mol, &grid, &output);
    failures += check_invalid_enum("pruning scheme", status, output == NULL,
                                   "invalid pruning scheme", total);
  }
  skalaxc_grid_settings_default(&grid);
  {
    skalaxc_molgrid_t output = (skalaxc_molgrid_t)&sentinel;
    grid.radial_quad = (enum SkalaXC_RadialQuad)99;
    status = skalaxc_molgrid_create_default(mol, &grid, &output);
    failures += check_invalid_enum("radial quadrature", status, output == NULL,
                                   "invalid radial quadrature", total);
  }
  skalaxc_grid_settings_default(&grid);
  {
    skalaxc_molgrid_t output = (skalaxc_molgrid_t)&sentinel;
    grid.atomic_grid = (enum SkalaXC_AtomicGridSizeDefault)99;
    status = skalaxc_molgrid_create_default(mol, &grid, &output);
    failures += check_invalid_enum("atomic grid size", status, output == NULL,
                                   "invalid atomic grid size", total);
  }
  {
    skalaxc_load_balancer_t output = (skalaxc_load_balancer_t)&sentinel;
    status = skalaxc_load_balancer_create((enum SkalaXC_ExecutionSpace)99, rt,
                                          mol, mg, basis, &output);
    failures +=
        check_invalid_enum("load-balancer execution space", status,
                           output == NULL, "invalid execution space", total);
  }
  {
    skalaxc_molecular_weights_t output = (skalaxc_molecular_weights_t)&sentinel;
    status = skalaxc_molecular_weights_create((enum SkalaXC_ExecutionSpace)99,
                                              SkalaXC_XCWeightAlg_SSF, &output);
    failures +=
        check_invalid_enum("weight execution space", status, output == NULL,
                           "invalid execution space", total);
  }
  {
    skalaxc_molecular_weights_t output = (skalaxc_molecular_weights_t)&sentinel;
    status = skalaxc_molecular_weights_create(
        SkalaXC_ExecutionSpace_Host, (enum SkalaXC_XCWeightAlg)99, &output);
    failures += check_invalid_enum("weight algorithm", status, output == NULL,
                                   "invalid XC weight algorithm", total);
  }
  {
    skalaxc_xc_integrator_t output = (skalaxc_xc_integrator_t)&sentinel;
    status = skalaxc_xc_integrator_create((enum SkalaXC_ExecutionSpace)99, func,
                                          lb, &output);
    failures +=
        check_invalid_enum("integrator execution space", status, output == NULL,
                           "invalid execution space", total);
  }
  skalaxc_integrator_settings_default(&integrator_settings);
  integrator_settings.domain_batch_mode = (enum SkalaXC_DomainBatchMode)99;
  {
    skalaxc_xc_integrator_t output = (skalaxc_xc_integrator_t)&sentinel;
    status = skalaxc_xc_integrator_create_with_settings(
        SkalaXC_ExecutionSpace_Host, func, lb, &integrator_settings, &output);
    failures += check_invalid_enum("domain batch mode", status, output == NULL,
                                   "invalid domain batch mode", total);
  }
  goto cleanup;

setup_failed:
  printf("[FAIL] C invalid enum setup: %s\n", skalaxc_last_error_message());
  ++failures;
  ++*total;

cleanup:
  skalaxc_functional_destroy(func);
  skalaxc_molecular_weights_destroy(mw);
  skalaxc_load_balancer_destroy(lb);
  skalaxc_molgrid_destroy(mg);
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(mol);
  skalaxc_runtime_environment_destroy(rt);
  return failures;
}

static int evaluate_native_system(skalaxc_molecule_t mol,
                                  skalaxc_basisset_t basis, const char* name) {
  skalaxc_runtime_environment_t rt = NULL;
  skalaxc_molgrid_t mg = NULL;
  skalaxc_load_balancer_t lb = NULL;
  skalaxc_molecular_weights_t mw = NULL;
  skalaxc_functional_t func = NULL;
  skalaxc_xc_integrator_t xc = NULL;
  skalaxc_timing_settings_t timing_settings;
  skalaxc_diagnostics_snapshot_t diagnostics;
  const double Ps[4] = {0.5, 0.5, 0.5, 0.5};
  const double Pz[4] = {0.0, 0.0, 0.0, 0.0};
  double VXCs[4] = {0.0, 0.0, 0.0, 0.0};
  double VXCz[4] = {0.0, 0.0, 0.0, 0.0};
  double exc = 0.0;
  int rc = 1;
  int index;
  skalaxc_status_t status;

  status = skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
      MPI_COMM_WORLD,
#endif
      &rt);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_molgrid_create_default(mol, NULL, &mg);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_load_balancer_create(SkalaXC_ExecutionSpace_Host, rt, mol,
                                        mg, basis, &lb);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_molecular_weights_create(SkalaXC_ExecutionSpace_Host,
                                            SkalaXC_XCWeightAlg_SSF, &mw);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_molecular_weights_modify_weights(mw, lb);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_functional_create("LDA", &func);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  skalaxc_timing_settings_default(&timing_settings);
  {
    skalaxc_integrator_settings_t integrator_settings;
    skalaxc_integrator_settings_default(&integrator_settings);
    if (integrator_settings.timing.verbose != 0 ||
        integrator_settings.timing.debug_logging != 0 ||
        integrator_settings.domain_batch_mode !=
            SkalaXC_DomainBatchMode_Conservative)
      goto cleanup;
  }
  status = skalaxc_xc_integrator_create_with_timing(
      SkalaXC_ExecutionSpace_Host, func, lb, &timing_settings, &xc);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_xc_integrator_eval_exc_vxc_uks(xc, Ps, Pz, VXCs, VXCz, &exc);
  if (status != SKALAXC_SUCCESS || skalaxc_molecule_natoms(mol) != 2 ||
      skalaxc_basisset_nbf(basis) != 2 || skalaxc_xc_integrator_nbf(xc) != 2 ||
      skalaxc_xc_integrator_natoms(xc) != 2 || !isfinite(exc) ||
      fabs(VXCs[1] - VXCs[2]) >= 1e-10 || fabs(VXCz[1] - VXCz[2]) >= 1e-10)
    goto cleanup;
  for (index = 0; index < 4; ++index)
    if (!isfinite(VXCs[index]) || !isfinite(VXCz[index])) goto cleanup;

  status = skalaxc_xc_integrator_get_diagnostics(xc, &diagnostics);
  if (status != SKALAXC_SUCCESS ||
      diagnostics.backend != SkalaXC_ExecutionSpace_Host ||
      diagnostics.communicator_size < 1 || diagnostics.device_id != -1 ||
      diagnostics.openmp_threads < 1 || diagnostics.exc_vxc_calls != 1 ||
      diagnostics.tasks == 0 || diagnostics.points == 0 ||
      diagnostics.model_batches != 2 || diagnostics.domains != 2 ||
      diagnostics.local_atoms != 2 ||
      diagnostics.configured_model_batches != 2 ||
      diagnostics.task_points_min <= 0 || diagnostics.task_points_max <= 0 ||
      diagnostics.timings[SkalaXC_TimingMetric_ModelForward].status !=
          SkalaXC_TimingStatus_Complete ||
      diagnostics.timings[SkalaXC_TimingMetric_TotalEXCVXC].call_count != 1)
    goto cleanup;

  status = skalaxc_xc_integrator_reset_diagnostics(xc);
  if (status != SKALAXC_SUCCESS) goto cleanup;
  status = skalaxc_xc_integrator_get_diagnostics(xc, &diagnostics);
  if (status != SKALAXC_SUCCESS || diagnostics.exc_vxc_calls != 0 ||
      diagnostics.model_batches != 0 || diagnostics.tasks == 0 ||
      diagnostics.points == 0 || diagnostics.configured_model_batches != 2 ||
      diagnostics.timings[SkalaXC_TimingMetric_ModelLoad].call_count != 1 ||
      diagnostics.timings[SkalaXC_TimingMetric_ModelForward].status !=
          SkalaXC_TimingStatus_Unavailable)
    goto cleanup;

  rc = 0;

cleanup:
  printf("[%s] C native construction: %s%s%s\n", rc == 0 ? "PASS" : "FAIL",
         name, rc == 0 ? "" : " : ",
         rc == 0 ? "" : skalaxc_last_error_message());
  skalaxc_xc_integrator_destroy(xc);
  skalaxc_functional_destroy(func);
  skalaxc_molecular_weights_destroy(mw);
  skalaxc_load_balancer_destroy(lb);
  skalaxc_molgrid_destroy(mg);
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(mol);
  skalaxc_runtime_environment_destroy(rt);
  return rc;
}

static int run_native_construction_case(int use_arrays) {
  static const double exponents[3] = {3.42525091, 0.62391373, 0.16885540};
  static const double coefficients[3] = {0.15432897, 0.53532814, 0.44463454};
  static const double atom_xyz[6] = {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0};
  skalaxc_molecule_t mol = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_status_t status;

  if (use_arrays) {
    const int64_t atomic_numbers[2] = {1, 1};
    const int32_t shell_l[2] = {0, 0};
    const int32_t shell_pure[2] = {0, 0};
    const int32_t shell_nprim[2] = {3, 3};
    double primitive_exponents[6];
    double primitive_coefficients[6];
    int index;
    for (index = 0; index < 3; ++index) {
      primitive_exponents[index] = primitive_exponents[index + 3] =
          exponents[index];
      primitive_coefficients[index] = primitive_coefficients[index + 3] =
          coefficients[index];
    }
    status = skalaxc_molecule_from_arrays(2, atomic_numbers, atom_xyz, &mol);
    if (status == SKALAXC_SUCCESS)
      status = skalaxc_basisset_from_arrays(2, shell_l, shell_pure, atom_xyz,
                                            shell_nprim, primitive_exponents,
                                            primitive_coefficients, &basis);
  } else {
    status = skalaxc_molecule_create(&mol);
    if (status == SKALAXC_SUCCESS)
      status = skalaxc_molecule_add_atom(mol, 1, atom_xyz[0], atom_xyz[1],
                                         atom_xyz[2]);
    if (status == SKALAXC_SUCCESS)
      status = skalaxc_molecule_add_atom(mol, 1, atom_xyz[3], atom_xyz[4],
                                         atom_xyz[5]);
    if (status == SKALAXC_SUCCESS) status = skalaxc_basisset_create(&basis);
    if (status == SKALAXC_SUCCESS)
      status = skalaxc_basisset_add_shell(basis, 0, 0, atom_xyz, 3, exponents,
                                          coefficients, 1);
    if (status == SKALAXC_SUCCESS)
      status = skalaxc_basisset_add_shell(basis, 0, 0, atom_xyz + 3, 3,
                                          exponents, coefficients, 1);
  }

  if (status == SKALAXC_SUCCESS)
    return evaluate_native_system(mol, basis,
                                  use_arrays ? "arrays" : "incremental");

  printf("[FAIL] C native construction: %s : %s\n",
         use_arrays ? "arrays" : "incremental", skalaxc_last_error_message());
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(mol);
  return 1;
}

/* Build the full pipeline for one fixture. On success all output handles are
 * set and SKALAXC_SUCCESS is returned; on failure everything is torn down. */
static skalaxc_status_t build(
    const char* path, const char* model, const skalaxc_grid_settings_t* grid,
    enum SkalaXC_ExecutionSpace execution_space,
    skalaxc_runtime_environment_t* rt, skalaxc_molecule_t* mol,
    skalaxc_basisset_t* basis, skalaxc_molgrid_t* mg,
    skalaxc_load_balancer_t* lb, skalaxc_molecular_weights_t* mw,
    skalaxc_functional_t* func, skalaxc_xc_integrator_t* xc) {
  skalaxc_status_t st;

  if (execution_space == SkalaXC_ExecutionSpace_Device) {
    skalaxc_device_runtime_settings_t device_settings;
    skalaxc_device_runtime_settings_default(&device_settings);
    st = skalaxc_device_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
        MPI_COMM_WORLD,
#endif
        &device_settings, rt);
  } else {
    st = skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
        MPI_COMM_WORLD,
#endif
        rt);
  }
  if (st != SKALAXC_SUCCESS) return st;

  st = skalaxc_molecule_from_hdf5(path, "/MOLECULE", mol);
  if (st != SKALAXC_SUCCESS) return st;
  st = skalaxc_basisset_from_hdf5(path, "/BASIS", basis);
  if (st != SKALAXC_SUCCESS) return st;

  st = skalaxc_molgrid_create_default(*mol, grid, mg);
  if (st != SKALAXC_SUCCESS) return st;

  st =
      skalaxc_load_balancer_create(execution_space, *rt, *mol, *mg, *basis, lb);
  if (st != SKALAXC_SUCCESS) return st;

  st = skalaxc_molecular_weights_create(execution_space,
                                        SkalaXC_XCWeightAlg_SSF, mw);
  if (st != SKALAXC_SUCCESS) return st;
  st = skalaxc_molecular_weights_modify_weights(*mw, *lb);
  if (st != SKALAXC_SUCCESS) return st;

  st = skalaxc_functional_create(model, func);
  if (st != SKALAXC_SUCCESS) return st;

  return skalaxc_xc_integrator_create(execution_space, *func, *lb, xc);
}

static int run_case(const char* path, const char* model, const char* name,
                    const skalaxc_grid_settings_t* grid,
                    enum SkalaXC_ExecutionSpace execution_space) {
  skalaxc_runtime_environment_t rt = NULL;
  skalaxc_molecule_t mol = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_molgrid_t mg = NULL;
  skalaxc_load_balancer_t lb = NULL;
  skalaxc_molecular_weights_t mw = NULL;
  skalaxc_functional_t func = NULL;
  skalaxc_xc_integrator_t xc = NULL;
  double *Ps = NULL, *Pz = NULL, *VXCs = NULL, *VXCz = NULL;
  double exc_ref = 0.0, exc = 0.0, sym_err = 0.0, denom, rel_err;
  int64_t nbf = 0, i, j;
  size_t n2 = 0;
  hid_t file = -1;
  int rc = 1;
  skalaxc_status_t st;

  st = build(path, model, grid, execution_space, &rt, &mol, &basis, &mg, &lb,
             &mw, &func, &xc);
  if (st != SKALAXC_SUCCESS) {
    printf("[FAIL] %s : build failed: %s\n", name,
           skalaxc_last_error_message());
    goto cleanup;
  }

  nbf = skalaxc_xc_integrator_nbf(xc);
  if (nbf <= 0) {
    printf("[FAIL] %s : nbf <= 0\n", name);
    goto cleanup;
  }
  n2 = (size_t)nbf * (size_t)nbf;
  Ps = (double*)calloc(n2, sizeof(double));
  Pz = (double*)calloc(n2, sizeof(double));
  VXCs = (double*)calloc(n2, sizeof(double));
  VXCz = (double*)calloc(n2, sizeof(double));
  if (!Ps || !Pz || !VXCs || !VXCz) {
    printf("[FAIL] %s : out of memory\n", name);
    goto cleanup;
  }

  file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0 || read_doubles(file, "/DENSITY_SCALAR", Ps) ||
      read_doubles(file, "/DENSITY_Z", Pz) ||
      read_doubles(file, "/EXC", &exc_ref)) {
    printf("[FAIL] %s : HDF5 read failed\n", name);
    goto cleanup;
  }
  H5Fclose(file);
  file = -1;

  st = skalaxc_xc_integrator_eval_exc_vxc_uks(xc, Ps, Pz, VXCs, VXCz, &exc);
  if (st != SKALAXC_SUCCESS) {
    printf("[FAIL] %s : eval failed: %s\n", name, skalaxc_last_error_message());
    goto cleanup;
  }

  for (i = 0; i < nbf; ++i)
    for (j = 0; j < nbf; ++j) {
      double d = fabs(VXCs[i * nbf + j] - VXCs[j * nbf + i]);
      if (d > sym_err) sym_err = d;
    }

  denom = fabs(exc_ref) > 1.0 ? fabs(exc_ref) : 1.0;
  rel_err = fabs(exc - exc_ref) / denom;

  if (rel_err < 1e-5 && sym_err < 1e-10) {
    printf("[PASS] %s : nbf=%lld EXC=%.10f (ref %.10f, rel %.2e) sym=%.2e\n",
           name, (long long)nbf, exc, exc_ref, rel_err, sym_err);
    rc = 0;
  } else {
    printf("[FAIL] %s : EXC=%.10f ref=%.10f rel=%.2e sym=%.2e\n", name, exc,
           exc_ref, rel_err, sym_err);
    rc = 1;
  }

cleanup:
  if (file >= 0) H5Fclose(file);
  free(Ps);
  free(Pz);
  free(VXCs);
  free(VXCz);
  skalaxc_xc_integrator_destroy(xc);
  skalaxc_functional_destroy(func);
  skalaxc_molecular_weights_destroy(mw);
  skalaxc_load_balancer_destroy(lb);
  skalaxc_molgrid_destroy(mg);
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(mol);
  skalaxc_runtime_environment_destroy(rt);
  return rc;
}

static int run_gradient_case(const char* path, const char* model,
                             const char* name,
                             enum SkalaXC_ExecutionSpace execution_space) {
  skalaxc_runtime_environment_t rt = NULL;
  skalaxc_molecule_t mol = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_molgrid_t mg = NULL;
  skalaxc_load_balancer_t lb = NULL;
  skalaxc_molecular_weights_t mw = NULL;
  skalaxc_functional_t func = NULL;
  skalaxc_xc_integrator_t xc = NULL;
  double *Ps = NULL, *Pz = NULL, *gradient = NULL;
  double squared_norm = 0.0, translation[3];
  int64_t nbf = 0, natoms = 0, i;
  size_t n2 = 0;
  hid_t file = -1;
  int rc = 1;
  skalaxc_status_t st;

  translation[0] = translation[1] = translation[2] = 0.0;

  st = build(path, model, NULL, execution_space, &rt, &mol, &basis, &mg, &lb,
             &mw, &func, &xc);
  if (st != SKALAXC_SUCCESS) {
    printf("[FAIL] %s : build failed: %s\n", name,
           skalaxc_last_error_message());
    goto cleanup;
  }

  nbf = skalaxc_xc_integrator_nbf(xc);
  natoms = skalaxc_xc_integrator_natoms(xc);
  n2 = (size_t)nbf * (size_t)nbf;
  Ps = (double*)calloc(n2, sizeof(double));
  Pz = (double*)calloc(n2, sizeof(double));
  gradient = (double*)calloc((size_t)(3 * natoms), sizeof(double));
  if (!Ps || !Pz || !gradient) {
    printf("[FAIL] %s : out of memory\n", name);
    goto cleanup;
  }

  file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0 || read_doubles(file, "/DENSITY", Ps)) {
    printf("[FAIL] %s : HDF5 read failed\n", name);
    goto cleanup;
  }
  H5Fclose(file);
  file = -1;

  st = skalaxc_xc_integrator_eval_exc_grad_uks(xc, Ps, Pz, gradient);
  if (st != SKALAXC_SUCCESS) {
    printf("[FAIL] %s : eval failed: %s\n", name, skalaxc_last_error_message());
    goto cleanup;
  }

  for (i = 0; i < 3 * natoms; ++i) {
    if (!isfinite(gradient[i])) goto cleanup;
    squared_norm += gradient[i] * gradient[i];
    translation[i % 3] += gradient[i];
  }
  if (squared_norm > 1e-6 && fabs(translation[0]) < 1e-10 &&
      fabs(translation[1]) < 1e-10 && fabs(translation[2]) < 1e-10) {
    printf("[PASS] %s : natoms=%lld squared_norm=%.10f\n", name,
           (long long)natoms, squared_norm);
    rc = 0;
  }

cleanup:
  if (file >= 0) H5Fclose(file);
  free(Ps);
  free(Pz);
  free(gradient);
  skalaxc_xc_integrator_destroy(xc);
  skalaxc_functional_destroy(func);
  skalaxc_molecular_weights_destroy(mw);
  skalaxc_load_balancer_destroy(lb);
  skalaxc_molgrid_destroy(mg);
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(mol);
  skalaxc_runtime_environment_destroy(rt);
  return rc;
}

int main(void) {
  const char* ref = SKALAXC_TEST_REF_DATA_PATH;
  const char* names[] = {"HE/def2-qzvp/lda", "HE/def2-qzvp/pbe",
                         "HE/def2-qzvp/tpss"};
  const char* files[] = {"skala_he_def2qzvp_lda_uks.hdf5",
                         "skala_he_def2qzvp_pbe_uks.hdf5",
                         "skala_he_def2qzvp_tpss_uks.hdf5"};
  const char* models[] = {"LDA", "PBE", "TPSS"};
  int total = 0, failures = 0, i;
  char path[1024];
  skalaxc_grid_settings_t grid;
  skalaxc_device_runtime_settings_t device_settings;

  if (skalaxc_version() != NULL &&
      strcmp(skalaxc_version(), SKALAXC_EXPECTED_VERSION) == 0) {
    printf("[PASS] SkalaXC version %s\n", skalaxc_version());
  } else {
    printf("[FAIL] SkalaXC version: expected %s, got %s\n",
           SKALAXC_EXPECTED_VERSION,
           skalaxc_version() == NULL ? "(null)" : skalaxc_version());
    ++failures;
  }
  ++total;

#ifdef SKALAXC_HAS_MPI
  {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) MPI_Init(NULL, NULL);
  }
#endif

  failures += run_error_contracts(&total);
  snprintf(path, sizeof(path), "%s/%s", ref, files[0]);
  failures += run_hdf5_failure_contracts(path, &total);
  failures += run_invalid_enum_contracts(path, &total);
  failures += run_native_construction_case(0);
  ++total;
  failures += run_native_construction_case(1);
  ++total;

  skalaxc_device_runtime_settings_default(&device_settings);
  if (device_settings.device_id == 0 &&
      fabs(device_settings.memory_fraction - 0.75) < 1e-15) {
    printf("[PASS] device runtime defaults\n");
  } else {
    printf("[FAIL] device runtime defaults\n");
    ++failures;
  }
  ++total;

#ifndef SKALAXC_HAS_CUDA
  {
    char sentinel;
    skalaxc_runtime_environment_t device_rt =
        (skalaxc_runtime_environment_t)&sentinel;
    const skalaxc_status_t status = skalaxc_device_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
        MPI_COMM_WORLD,
#endif
        &device_settings, &device_rt);
    if (status == SKALAXC_ERROR && device_rt == NULL &&
        strstr(skalaxc_last_error_message(), "without CUDA support") != NULL) {
      printf("[PASS] device runtime rejected by host build\n");
    } else {
      printf("[FAIL] device runtime accepted by host build\n");
      ++failures;
      if (device_rt != (skalaxc_runtime_environment_t)&sentinel)
        skalaxc_runtime_environment_destroy(device_rt);
    }
    ++total;
  }
#endif

  for (i = 0; i < 3; ++i) {
    snprintf(path, sizeof(path), "%s/%s", ref, files[i]);
    failures +=
        run_case(path, models[i], names[i], NULL, SkalaXC_ExecutionSpace_Host);
    ++total;
  }

  /* Grid API: an explicit default-filled struct and a NULL pointer must both
   * reproduce the built-in preset (identical energy to the plain path). */
  snprintf(path, sizeof(path), "%s/%s", ref, files[1]); /* PBE fixture */
  skalaxc_grid_settings_default(&grid);
  failures += run_case(path, models[1], "HE/def2-qzvp/pbe [grid:default]",
                       &grid, SkalaXC_ExecutionSpace_Host);
  ++total;
  failures += run_case(path, models[1], "HE/def2-qzvp/pbe [grid:null]", NULL,
                       SkalaXC_ExecutionSpace_Host);
  ++total;

  snprintf(path, sizeof(path), "%s/%s", SKALAXC_GAUXC_REF_DATA_PATH,
           "h2o2_def2-tzvp.hdf5");
  failures += run_gradient_case(path, "TPSS", "H2O2 gradient",
                                SkalaXC_ExecutionSpace_Host);
  ++total;

#ifdef SKALAXC_HAS_CUDA
  snprintf(path, sizeof(path), "%s/%s", ref, files[2]);
  failures += run_case(path, models[2], "HE/def2-qzvp/tpss [cuda]", NULL,
                       SkalaXC_ExecutionSpace_Device);
  ++total;

  snprintf(path, sizeof(path), "%s/%s", SKALAXC_GAUXC_REF_DATA_PATH,
           "h2o2_def2-tzvp.hdf5");
  /* The current TPSS trace can exceed sm_120 resources during backward.
   * PBE keeps this language-binding test focused on the public gradient API
   * until TPSS is retraced with a smaller TensorExpr kernel. */
  failures += run_gradient_case(path, "PBE", "H2O2 gradient [cuda]",
                                SkalaXC_ExecutionSpace_Device);
  ++total;
#endif

  printf("\n%d / %d C public-API cases passed\n", total - failures, total);

#ifdef SKALAXC_HAS_MPI
  {
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) MPI_Finalize();
  }
#endif

  return failures == 0 ? 0 : 1;
}

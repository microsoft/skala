/* SkalaXC C example: evaluate a UKS machine-learning exchange-correlation
 * functional on a system + density loaded from an HDF5 file.
 *
 * Consumption contract: this program includes ONLY <skalaxc/skalaxc.h> and
 * links ONLY libskalaxc (plus the C HDF5 library to load its own density
 * input). No GauXC, LibTorch, or C++ symbols appear in the link line.
 *
 * The program follows the SkalaXC C pipeline (one opaque handle per stage):
 *   runtime environment -> molecule / basis set -> molecular grid ->
 *   load balancer -> molecular weights -> functional -> XC integrator
 *
 * Usage: skalaxc_eval_c <system.hdf5> [model]
 *   <system.hdf5>  HDF5 file with /MOLECULE, /BASIS, /DENSITY_SCALAR,
 *                  /DENSITY_Z
 *   [model]        "LDA", "PBE" (default), "TPSS", or a path to a .fun model
 */

#include <skalaxc/skalaxc.h>

#include <hdf5.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int read_doubles(hid_t file, const char* dset, double* buf) {
  hid_t d = H5Dopen2(file, dset, H5P_DEFAULT);
  herr_t st;
  if (d < 0) return -1;
  st = H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  H5Dclose(d);
  return st < 0 ? -1 : 0;
}

int main(int argc, char** argv) {
  const char* path;
  const char* model;
  skalaxc_runtime_environment_t rt = NULL;
  skalaxc_molecule_t mol = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_molgrid_t mg = NULL;
  skalaxc_load_balancer_t lb = NULL;
  skalaxc_molecular_weights_t mw = NULL;
  skalaxc_functional_t func = NULL;
  skalaxc_xc_integrator_t xc = NULL;
  double* Ps = NULL;
  double* Pz = NULL;
  double* VXCs = NULL;
  double* VXCz = NULL;
  int64_t nbf = 0;
  size_t n2 = 0, i;
  double exc = 0.0, vs = 0.0;
  hid_t file;
  int rc = 1;

  if (argc < 2) {
    fprintf(stderr, "usage: %s <system.hdf5> [model]\n", argv[0]);
    return 2;
  }
  path = argv[1];
  model = argc > 2 ? argv[2] : "PBE";

#ifdef SKALAXC_HAS_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    return 1;
  }
#endif

  /* 1. Runtime environment. */
  if (skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
          MPI_COMM_WORLD,
#endif
          &rt) != SKALAXC_SUCCESS)
    goto fail;

  /* 2. Load the molecule and basis set from the file. */
  if (skalaxc_molecule_from_hdf5(path, "/MOLECULE", &mol) != SKALAXC_SUCCESS)
    goto fail;
  if (skalaxc_basisset_from_hdf5(path, "/BASIS", &basis) != SKALAXC_SUCCESS)
    goto fail;

  /* 3. Build the default molecular integration grid (NULL = built-in preset).
   */
  if (skalaxc_molgrid_create_default(mol, NULL, &mg) != SKALAXC_SUCCESS)
    goto fail;

  /* 4. Distribute the quadrature tasks. */
  if (skalaxc_load_balancer_create(SkalaXC_ExecutionSpace_Host, rt, mol, mg,
                                   basis, &lb) != SKALAXC_SUCCESS)
    goto fail;

  /* 5. Partition the quadrature weights in place. */
  if (skalaxc_molecular_weights_create(SkalaXC_ExecutionSpace_Host,
                                       SkalaXC_XCWeightAlg_SSF,
                                       &mw) != SKALAXC_SUCCESS)
    goto fail;
  if (skalaxc_molecular_weights_modify_weights(mw, lb) != SKALAXC_SUCCESS)
    goto fail;

  /* 6. Select the Skala ML functional model. */
  if (skalaxc_functional_create(model, &func) != SKALAXC_SUCCESS) goto fail;

  /* 7. Build the XC integrator. */
  if (skalaxc_xc_integrator_create(SkalaXC_ExecutionSpace_Host, func, lb,
                                   &xc) != SKALAXC_SUCCESS)
    goto fail;

  nbf = skalaxc_xc_integrator_nbf(xc);
  if (nbf <= 0) {
    fprintf(stderr, "SkalaXC error: invalid nbf\n");
    goto cleanup;
  }
  n2 = (size_t)nbf * (size_t)nbf;
  Ps = (double*)calloc(n2, sizeof(double));
  Pz = (double*)calloc(n2, sizeof(double));
  VXCs = (double*)calloc(n2, sizeof(double));
  VXCz = (double*)calloc(n2, sizeof(double));
  if (!Ps || !Pz || !VXCs || !VXCz) {
    fprintf(stderr, "allocation failed\n");
    goto cleanup;
  }

  /* 8. Load the input spin densities (your data; here from the same file). */
  file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    fprintf(stderr, "H5Fopen failed\n");
    goto cleanup;
  }
  if (read_doubles(file, "/DENSITY_SCALAR", Ps) ||
      read_doubles(file, "/DENSITY_Z", Pz)) {
    fprintf(stderr, "HDF5 read failed\n");
    H5Fclose(file);
    goto cleanup;
  }
  H5Fclose(file);

  /* 9. Evaluate the ML exchange-correlation energy and potential (UKS). */
  if (skalaxc_xc_integrator_eval_exc_vxc_uks(xc, Ps, Pz, VXCs, VXCz, &exc) !=
      SKALAXC_SUCCESS)
    goto fail;

  /* 10. Report the energy and a summary of the potential. */
  for (i = 0; i < n2; ++i) vs += VXCs[i] * VXCs[i];
  printf("model=%s nbf=%lld EXC=%.10f |VXC_scalar|_F=%.6e\n", model,
         (long long)nbf, exc, sqrt(vs));
  rc = 0;
  goto cleanup;

fail:
  fprintf(stderr, "SkalaXC error: %s\n", skalaxc_last_error_message());

cleanup:
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
#ifdef SKALAXC_HAS_MPI
  MPI_Finalize();
#endif
  return rc;
}

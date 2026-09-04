#include <skalaxc/skalaxc.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

int main(int argc, char** argv) {
  static const int64_t atomic_numbers[2] = {1, 1};
  static const double atom_xyz[6] = {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0};
  static const int32_t shell_l[2] = {0, 0};
  static const int32_t shell_pure[2] = {0, 0};
  static const int32_t shell_nprim[2] = {3, 3};
  static const double exponents[6] = {3.42525091, 0.62391373, 0.16885540,
                                      3.42525091, 0.62391373, 0.16885540};
  static const double coefficients[6] = {0.15432897, 0.53532814, 0.44463454,
                                         0.15432897, 0.53532814, 0.44463454};
  skalaxc_runtime_environment_t runtime = NULL;
  skalaxc_molecule_t molecule = NULL;
  skalaxc_basisset_t basis = NULL;
  skalaxc_molgrid_t grid = NULL;
  skalaxc_load_balancer_t load_balancer = NULL;
  skalaxc_molecular_weights_t weights = NULL;
  skalaxc_functional_t functional = NULL;
  skalaxc_xc_integrator_t integrator = NULL;
  skalaxc_grid_settings_t grid_settings;
  double scalar_density[4] = {0.5, 0.5, 0.5, 0.5};
  double spin_density[4] = {0.0, 0.0, 0.0, 0.0};
  double scalar_potential[4], spin_potential[4], energy;
  int result = 1;

#ifdef SKALAXC_HAS_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
#else
  (void)argc;
  (void)argv;
#endif

  if (skalaxc_version() == NULL || skalaxc_version()[0] == '\0') goto cleanup;
  if (skalaxc_runtime_environment_create(
#ifdef SKALAXC_HAS_MPI
          MPI_COMM_WORLD,
#endif
          &runtime) != SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_molecule_from_arrays(2, atomic_numbers, atom_xyz, &molecule) !=
      SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_basisset_from_arrays(2, shell_l, shell_pure, atom_xyz,
                                   shell_nprim, exponents, coefficients,
                                   &basis) != SKALAXC_SUCCESS)
    goto cleanup;
  skalaxc_grid_settings_default(&grid_settings);
  grid_settings.batch_size = 128;
  if (skalaxc_molgrid_create_default(molecule, &grid_settings, &grid) !=
      SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_load_balancer_create(SkalaXC_ExecutionSpace_Host, runtime,
                                   molecule, grid, basis,
                                   &load_balancer) != SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_molecular_weights_create(SkalaXC_ExecutionSpace_Host,
                                       SkalaXC_XCWeightAlg_SSF,
                                       &weights) != SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_molecular_weights_modify_weights(weights, load_balancer) !=
      SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_functional_create("LDA", &functional) != SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_xc_integrator_create(SkalaXC_ExecutionSpace_Host, functional,
                                   load_balancer,
                                   &integrator) != SKALAXC_SUCCESS)
    goto cleanup;
  if (skalaxc_xc_integrator_eval_exc_vxc_uks(
          integrator, scalar_density, spin_density, scalar_potential,
          spin_potential, &energy) != SKALAXC_SUCCESS)
    goto cleanup;
  if (isfinite(energy) && isfinite(scalar_potential[0]) &&
      isfinite(spin_potential[0]))
    result = 0;

cleanup:
  skalaxc_xc_integrator_destroy(integrator);
  skalaxc_functional_destroy(functional);
  skalaxc_molecular_weights_destroy(weights);
  skalaxc_load_balancer_destroy(load_balancer);
  skalaxc_molgrid_destroy(grid);
  skalaxc_basisset_destroy(basis);
  skalaxc_molecule_destroy(molecule);
  skalaxc_runtime_environment_destroy(runtime);
#ifdef SKALAXC_HAS_MPI
  if (MPI_Finalize() != MPI_SUCCESS) return 1;
#endif
  return result;
}

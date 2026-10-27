/**
 * @file
 * @brief SkalaXC C API: runtime environment handle.
 *
 * Mirrors SkalaXC::RuntimeEnvironment. Part of the public C API; include
 * <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <stdint.h>

#include <skalaxc/c/mpi.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque runtime-environment handle (mirrors
 * SkalaXC::RuntimeEnvironment).
 */
typedef struct skalaxc_runtime_environment* skalaxc_runtime_environment_t;

/** @brief CUDA device selection and GauXC memory-pool settings. */
typedef struct {
  int32_t device_id;       ///< CUDA device ordinal
  double memory_fraction;  ///< Fraction of available memory in (0, 1]
} skalaxc_device_runtime_settings_t;

/**
 * @brief Populate device settings with device 0 and memory fraction 0.75.
 * @param settings Output settings; NULL is tolerated.
 */
void skalaxc_device_runtime_settings_default(
    skalaxc_device_runtime_settings_t* settings) SKALAXC_EXPORT;

/**
 * @brief Create a runtime environment.
 * @param comm MPI communicator (MPI builds only) defining the cooperating
 *             ranks.
 * @param out Output owning handle on success (free with
 *            skalaxc_runtime_environment_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_runtime_environment_create(SKALAXC_MPI_CODE(
    MPI_Comm comm, ) skalaxc_runtime_environment_t* out) SKALAXC_EXPORT;

/**
 * @brief Create a CUDA runtime environment.
 * @param comm MPI communicator (MPI builds only).
 * @param settings Device settings, or NULL for the built-in defaults.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_device_runtime_environment_create(
    SKALAXC_MPI_CODE(MPI_Comm comm, )
        const skalaxc_device_runtime_settings_t* settings,
    skalaxc_runtime_environment_t* out) SKALAXC_EXPORT;

#ifdef SKALAXC_HAS_MPI
/**
 * @brief Create a runtime environment from a Fortran MPI handle (MPI builds
 * only).
 * @param comm Fortran MPI communicator handle.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_runtime_environment_create_f(
    MPI_Fint comm, skalaxc_runtime_environment_t* out) SKALAXC_EXPORT;

/**
 * @brief Create a CUDA runtime from a Fortran MPI handle.
 * @param comm Fortran MPI communicator handle.
 * @param settings Device settings, or NULL for the built-in defaults.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_device_runtime_environment_create_f(
    MPI_Fint comm, const skalaxc_device_runtime_settings_t* settings,
    skalaxc_runtime_environment_t* out) SKALAXC_EXPORT;
#endif

/**
 * @brief Return the calling rank within the environment (0 in serial builds),
 *        or -1 if rt is null.
 * @param rt Runtime environment handle.
 * @return MPI rank of the caller.
 */
int skalaxc_runtime_environment_comm_rank(skalaxc_runtime_environment_t rt)
    SKALAXC_EXPORT;

/**
 * @brief Return the number of cooperating ranks (1 in serial builds), or -1 if
 *        rt is null.
 * @param rt Runtime environment handle.
 * @return MPI size of the environment.
 */
int skalaxc_runtime_environment_comm_size(skalaxc_runtime_environment_t rt)
    SKALAXC_EXPORT;

/**
 * @brief Destroy a runtime-environment handle (NULL is tolerated).
 * @param rt Runtime environment handle to destroy.
 */
void skalaxc_runtime_environment_destroy(skalaxc_runtime_environment_t rt)
    SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif

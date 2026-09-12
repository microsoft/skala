/**
 * @file
 * @brief SkalaXC public MPI interop helper macros.
 *
 * Mirrors GauXC's <gauxc/c/mpi.h>: SKALAXC_MPI_CODE(...) expands to its
 * arguments when SkalaXC was built with MPI support and to nothing otherwise.
 * This lets the public calculator constructors take a native MPI_Comm only in
 * MPI builds, leaving the serial ABI byte-for-byte unchanged.
 *
 * MPI_Comm is a standard MPI C type -- not a GauXC or LibTorch type -- so it is
 * the sole third-party type permitted across the SkalaXC public boundary, and
 * only when SkalaXC (and hence the host) is built with MPI.
 */
#pragma once

#include <skalaxc/c/config.h>

#ifdef SKALAXC_HAS_MPI
/** @brief Retain optional public-API tokens in MPI-enabled builds. */
#define SKALAXC_MPI_CODE(...) __VA_ARGS__
#else
/** @brief Remove optional public-API tokens in serial builds. */
#define SKALAXC_MPI_CODE(...)
#endif

#ifdef SKALAXC_HAS_MPI
#include <mpi.h>
#endif

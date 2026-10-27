/**
 * @file
 * @brief SkalaXC public C API umbrella header.
 *
 * ABI isolation contract: this API exposes only opaque handles and C POD types.
 * No GauXC or LibTorch type crosses this boundary. Densities and potentials are
 * passed as raw double arrays (column-major, nbf x nbf).
 * No C++ exception crosses this API. Failures are reported through status
 * codes, documented sentinel return values, and skalaxc_last_error_message().
 *
 * The C API mirrors the SkalaXC C++ pipeline
 * (runtime environment -> molecule / basis set -> molecular grid ->
 * load balancer -> molecular weights -> functional -> XC integrator) with one
 * opaque handle per stage. The API is organized into the skalaxc/c/ headers;
 * this umbrella includes them all. Consumers may include the individual headers
 * instead if preferred.
 */
#pragma once

#include <skalaxc/c/basisset.h>
#include <skalaxc/c/diagnostics.h>
#include <skalaxc/c/enums.h>
#include <skalaxc/c/functional.h>
#include <skalaxc/c/grid.h>
#include <skalaxc/c/integrator.h>
#include <skalaxc/c/load_balancer.h>
#include <skalaxc/c/molecular_weights.h>
#include <skalaxc/c/molecule.h>
#include <skalaxc/c/molgrid.h>
#include <skalaxc/c/runtime.h>
#include <skalaxc/c/status.h>
#include <skalaxc/c/version.h>

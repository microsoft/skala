/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once

namespace GauXC {

/**
 *  @brief GauXC specific enums for the specification of radial quadratures
 *
 *  Generally mapped to equivalent enums in IntegratorXX
 */
enum class RadialQuad {
  Becke,             ///< Becke radial quadrature
  MuraKnowles,       ///< Mura-Knowles radial quadrature
  MurrayHandyLaming, ///< Murray-Handy-Laming radial quadrature
  TreutlerAhlrichs   ///< Treutler-Ahlrichs radial quadrature
};

/**
 *  @brief Specifications of grid defaults for atomic integration
 *
 *  See https://gaussian.com/integral for specification
 */
enum class AtomicGridSizeDefault {
  FineGrid,       ///< Fine grid      (least accurate)
  UltraFineGrid,  ///< Ultrafine grid (appropriate accuracy)
  SuperFineGrid,  ///< Superfine grid (most accurate)
  GM3,            ///< Treutler-Ahlrichs GM3
  GM5             ///< Treutlet-Ahlrichs GM5
};

/**
 *  @brief Specifications of atomic partitioning scheme for the
 *  molecular integration
 */
enum class XCWeightAlg {
  NOTPARTITIONED, ///< Not partitioned
  Becke, ///< The original Becke weighting scheme
  SSF,   ///< The Stratmann-Scuseria-Frisch weighting scheme
  LKO    ///< The Lauqua-Kuessman-Ochsenfeld weighting scheme
};

/**
 *  @brief Specification of the execution space for various operations
 */
enum class ExecutionSpace {
  Host,  ///< Execute task on the host
  Device ///< Execute task on the device (e.g. GPU)
};

/// Supported Algorithms / Integrands
enum class SupportedAlg {
  XC,
  DEN,
  SNLINK
};

} // namespace GauXC

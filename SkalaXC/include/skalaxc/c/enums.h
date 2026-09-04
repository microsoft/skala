/**
 * @file
 * @brief SkalaXC C API: pipeline enumerations.
 *
 * These mirror the SkalaXC:: C++ enums (which in turn mirror GauXC). The
 * integer values are pinned to match the C++ enums, so they may be cast across
 * the boundary.
 *
 * Part of the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Radial quadrature scheme (mirrors SkalaXC::RadialQuad).
 */
enum SkalaXC_RadialQuad {
  SkalaXC_RadialQuad_Becke,        ///< Becke radial quadrature
  SkalaXC_RadialQuad_MuraKnowles,  ///< Mura-Knowles radial quadrature (default)
  SkalaXC_RadialQuad_MurrayHandyLaming,  ///< Murray-Handy-Laming radial
                                         ///< quadrature
  SkalaXC_RadialQuad_TreutlerAhlrichs  ///< Treutler-Ahlrichs radial quadrature
};

/**
 * @brief Atomic grid size preset, in ascending accuracy (mirrors
 * SkalaXC::AtomicGridSizeDefault).
 */
enum SkalaXC_AtomicGridSizeDefault {
  SkalaXC_AtomicGridSizeDefault_FineGrid,       ///< Fine grid (least accurate)
  SkalaXC_AtomicGridSizeDefault_UltraFineGrid,  ///< Ultrafine grid (default)
  SkalaXC_AtomicGridSizeDefault_SuperFineGrid,  ///< Superfine grid (most
                                                ///< accurate)
  SkalaXC_AtomicGridSizeDefault_GM3,            ///< Treutler-Ahlrichs GM3
  SkalaXC_AtomicGridSizeDefault_GM5             ///< Treutler-Ahlrichs GM5
};

/**
 * @brief Atomic quadrature pruning scheme (mirrors SkalaXC::PruningScheme).
 */
enum SkalaXC_PruningScheme {
  SkalaXC_PruningScheme_Unpruned,  ///< Unpruned atomic quadrature (default)
  SkalaXC_PruningScheme_Robust,    ///< The "Robust" scheme of Psi4
  SkalaXC_PruningScheme_Treutler   ///< The Treutler-Ahlrichs scheme
};

/**
 * @brief Execution space selector (mirrors SkalaXC::ExecutionSpace).
 *
 * Device execution is available when SkalaXC is built with CUDA.
 */
enum SkalaXC_ExecutionSpace {
  SkalaXC_ExecutionSpace_Host,   ///< Host (CPU) evaluation
  SkalaXC_ExecutionSpace_Device  ///< CUDA device evaluation
};

/**
 * @brief Complete atomic-domain model batching policy.
 */
enum SkalaXC_DomainBatchMode {
  SkalaXC_DomainBatchMode_Conservative,  ///< One domain per model call
  SkalaXC_DomainBatchMode_Aggressive     ///< All exact-size local domains
};

/**
 * @brief XC weight partitioning scheme (mirrors SkalaXC::XCWeightAlg).
 */
enum SkalaXC_XCWeightAlg {
  SkalaXC_XCWeightAlg_NOTPARTITIONED,  ///< Weights are not partitioned
  SkalaXC_XCWeightAlg_Becke,           ///< Becke partitioning
  SkalaXC_XCWeightAlg_SSF,             ///< Stratmann-Scuseria-Frisch (default)
  SkalaXC_XCWeightAlg_LKO              ///< Laqua-Kussmann-Ochsenfeld
};

#ifdef __cplusplus
}  // extern "C"
#endif

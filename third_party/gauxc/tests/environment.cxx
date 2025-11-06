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
#include "ut_common.hpp"
#include <gauxc/util/environment.hpp>

using namespace GauXC;
TEST_CASE("Environment", "[env]") {

  SECTION("Host") {
    auto xc  = gauxc_max_am(ExecutionSpace::Host, SupportedAlg::XC    );
    auto den = gauxc_max_am(ExecutionSpace::Host, SupportedAlg::DEN   );
    auto snk = gauxc_max_am(ExecutionSpace::Host, SupportedAlg::SNLINK);

#ifdef GAUXC_HAS_HOST
    REQUIRE(xc  == GAUXC_CPU_XC_MAX_AM);
    REQUIRE(den == GAUXC_CPU_XC_MAX_AM);
    REQUIRE(snk == GAUXC_CPU_SNLINK_MAX_AM);
#else
    REQUIRE(xc  == -1);
    REQUIRE(den == -1);
    REQUIRE(snk == -1);
#endif
  }

  SECTION("Device") {
    auto xc  = gauxc_max_am(ExecutionSpace::Device, SupportedAlg::XC    );
    auto den = gauxc_max_am(ExecutionSpace::Device, SupportedAlg::DEN   );
    auto snk = gauxc_max_am(ExecutionSpace::Device, SupportedAlg::SNLINK);

#ifdef GAUXC_HAS_DEVICE
    REQUIRE(xc  == GAUXC_GPU_XC_MAX_AM);
    REQUIRE(den == GAUXC_GPU_XC_MAX_AM);
    REQUIRE(snk == GAUXC_GPU_SNLINK_MAX_AM);
#else
    REQUIRE(xc  == -1);
    REQUIRE(den == -1);
    REQUIRE(snk == -1);
#endif
  }

}

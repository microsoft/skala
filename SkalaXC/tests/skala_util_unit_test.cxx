#include <catch2/catch.hpp>

#include "task_data.hpp"
#include <skalaxc/skalaxc.hpp>

TEST_CASE("functional_type carries a model selector", "[skala][settings]") {
  SkalaXC::functional_type empty;
  REQUIRE(empty.empty());
  REQUIRE(empty.model().empty());

  SkalaXC::functional_type pbe("PBE");
  REQUIRE_FALSE(pbe.empty());
  REQUIRE(pbe.model() == "PBE");

  SkalaXC::functional_type custom("/tmp/custom_model.fun");
  REQUIRE(custom.model() == "/tmp/custom_model.fun");
}

TEST_CASE("Task model data starts empty", "[skala][features]") {
  SkalaXC::TaskFeatureData features;
  SkalaXC::TaskPotentialData potentials;

  REQUIRE(features.density.size() == 0);
  REQUIRE(features.density_gradient.points() == 0);
  REQUIRE(features.kinetic.size() == 0);

  REQUIRE(potentials.density.size() == 0);
  REQUIRE(potentials.density_gradient.points() == 0);
  REQUIRE(potentials.kinetic.size() == 0);
  REQUIRE(potentials.dE_dw.size() == 0);
}

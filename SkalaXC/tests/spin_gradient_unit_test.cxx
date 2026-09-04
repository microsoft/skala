#include <catch2/catch_test_macros.hpp>

#include "component_matrix_map.hpp"
#include "spin_gradient.hpp"

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

double value(SkalaXC::Direction direction, Eigen::Index point,
             SkalaXC::SpinChannel channel) {
  return 100.0 * static_cast<double>(direction) +
         10.0 * static_cast<double>(point) + static_cast<double>(channel);
}

SkalaXC::SpinGradient make_gradient(Eigen::Index points) {
  SkalaXC::SpinGradient gradient(points);
  for (Eigen::Index point = 0; point < points; ++point)
    for (Eigen::Index direction = 0; direction < SkalaXC::direction_dimension;
         ++direction)
      for (Eigen::Index spin = 0; spin < SkalaXC::spin_dimension; ++spin)
        gradient(static_cast<SkalaXC::Direction>(direction), point,
                 static_cast<SkalaXC::SpinChannel>(spin)) =
            value(static_cast<SkalaXC::Direction>(direction), point,
                  static_cast<SkalaXC::SpinChannel>(spin));
  return gradient;
}

std::vector<SkalaXC::types::PermutationIndex> permutation(
    std::initializer_list<std::int64_t> values) {
  std::vector<SkalaXC::types::PermutationIndex> result;
  result.reserve(values.size());
  for (const auto value : values)
    result.push_back(SkalaXC::types::PermutationIndex{value});
  return result;
}

}  // namespace

TEST_CASE("SpinGradient exposes GauXC direction-major storage",
          "[eigen][spin-gradient]") {
  const auto gradient = make_gradient(3);

  CHECK(gradient.points() == 3);
  CHECK(gradient.direction_data(SkalaXC::Y) -
            gradient.direction_data(SkalaXC::X) ==
        6);
  CHECK(gradient.direction_data(SkalaXC::Z) -
            gradient.direction_data(SkalaXC::Y) ==
        6);

  const auto x_direction = gradient.direction(SkalaXC::X);
  CHECK(x_direction.rows() == gradient.points());
  CHECK(x_direction.cols() == SkalaXC::spin_dimension);
  for (Eigen::Index point = 0; point < gradient.points(); ++point) {
    CHECK(x_direction(point, SkalaXC::SpinChannel::Alpha) ==
          value(SkalaXC::X, point, SkalaXC::SpinChannel::Alpha));
    CHECK(x_direction(point, SkalaXC::SpinChannel::Beta) ==
          value(SkalaXC::X, point, SkalaXC::SpinChannel::Beta));
  }
}

TEST_CASE("SpinGradient reuses storage for an unchanged point count",
          "[eigen][spin-gradient]") {
  SkalaXC::SpinGradient gradient(3);
  auto* storage = gradient.direction_data(SkalaXC::X);

  gradient.resize(3);

  CHECK(gradient.points() == 3);
  CHECK(gradient.direction_data(SkalaXC::X) == storage);
  CHECK_THROWS_AS(gradient.resize(-1), std::invalid_argument);
}

TEST_CASE("Scalar-z gradients convert once to alpha-beta gradients",
          "[eigen][spin-gradient][representation]") {
  SkalaXC::ScalarZGradient scalar_z(1);
  scalar_z(SkalaXC::X, 0, SkalaXC::PauliChannel::Scalar) = 6.0;
  scalar_z(SkalaXC::Y, 0, SkalaXC::PauliChannel::Scalar) = 8.0;
  scalar_z(SkalaXC::Z, 0, SkalaXC::PauliChannel::Scalar) = 10.0;
  scalar_z(SkalaXC::X, 0, SkalaXC::PauliChannel::SpinZ) = 2.0;
  scalar_z(SkalaXC::Y, 0, SkalaXC::PauliChannel::SpinZ) = -2.0;
  scalar_z(SkalaXC::Z, 0, SkalaXC::PauliChannel::SpinZ) = 4.0;

  SkalaXC::SpinGradient alpha_beta;
  SkalaXC::convert_scalar_z_to_alpha_beta(scalar_z, alpha_beta);

  REQUIRE(alpha_beta(SkalaXC::X, 0, SkalaXC::SpinChannel::Alpha) == 4.0);
  REQUIRE(alpha_beta(SkalaXC::Y, 0, SkalaXC::SpinChannel::Alpha) == 3.0);
  REQUIRE(alpha_beta(SkalaXC::Z, 0, SkalaXC::SpinChannel::Alpha) == 7.0);
  REQUIRE(alpha_beta(SkalaXC::X, 0, SkalaXC::SpinChannel::Beta) == 2.0);
  REQUIRE(alpha_beta(SkalaXC::Y, 0, SkalaXC::SpinChannel::Beta) == 5.0);
  REQUIRE(alpha_beta(SkalaXC::Z, 0, SkalaXC::SpinChannel::Beta) == 3.0);

  SkalaXC::AlphaBetaMatrix model_potential(1, SkalaXC::spin_dimension);
  model_potential << 7.0, 3.0;
  const auto potential_scalar_z =
      SkalaXC::alpha_beta_to_scalar_z(model_potential);
  STATIC_REQUIRE(SkalaXC::ScalarZChannels::ColsAtCompileTime ==
                 SkalaXC::spin_dimension);
  REQUIRE(potential_scalar_z(0, SkalaXC::PauliChannel::Scalar) == 5.0);
  REQUIRE(potential_scalar_z(0, SkalaXC::PauliChannel::SpinZ) == 2.0);
}

TEST_CASE("SpinGradient converts Torch tensors by semantic index",
          "[eigen][spin-gradient]") {
  const auto original = make_gradient(4);
  const auto tensor = SkalaXC::spin_gradient_to_torch(original, true);

  REQUIRE(tensor.dim() == 3);
  CHECK(tensor.size(0) == 2);
  CHECK(tensor.size(1) == 3);
  CHECK(tensor.size(2) == 4);
  CHECK(tensor.scalar_type() == torch::kFloat64);
  CHECK(tensor.requires_grad());

  const auto round_trip = SkalaXC::spin_gradient_from_torch(tensor);
  for (Eigen::Index point = 0; point < original.points(); ++point)
    for (Eigen::Index direction = 0; direction < SkalaXC::direction_dimension;
         ++direction)
      for (Eigen::Index spin = 0; spin < SkalaXC::spin_dimension; ++spin)
        CHECK(round_trip(static_cast<SkalaXC::Direction>(direction), point,
                         static_cast<SkalaXC::SpinChannel>(spin)) ==
              original(static_cast<SkalaXC::Direction>(direction), point,
                       static_cast<SkalaXC::SpinChannel>(spin)));

  CHECK_THROWS_AS(SkalaXC::spin_gradient_from_torch(
                      torch::zeros({2, 4, 3}, torch::kFloat64)),
                  std::invalid_argument);
  CHECK_THROWS_AS(SkalaXC::spin_gradient_from_torch(
                      torch::zeros({2, 3, 3}, torch::kFloat32)),
                  std::invalid_argument);
}

TEST_CASE("SpinGradient permutes whole point records",
          "[eigen][spin-gradient]") {
  auto gradient = make_gradient(3);
  gradient.permute_points(permutation({2, 0, 1}));

  for (Eigen::Index direction = 0; direction < SkalaXC::direction_dimension;
       ++direction)
    for (Eigen::Index spin = 0; spin < SkalaXC::spin_dimension; ++spin) {
      const auto direction_value = static_cast<SkalaXC::Direction>(direction);
      const auto spin_value = static_cast<SkalaXC::SpinChannel>(spin);
      CHECK(gradient(direction_value, 2, spin_value) ==
            value(direction_value, 0, spin_value));
      CHECK(gradient(direction_value, 0, spin_value) ==
            value(direction_value, 1, spin_value));
      CHECK(gradient(direction_value, 1, spin_value) ==
            value(direction_value, 2, spin_value));
    }

  CHECK_THROWS_AS(gradient.permute_points(permutation({0, 1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(gradient.permute_points(permutation({0, 0, 2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(gradient.permute_points(permutation({0, 1, 3})),
                  std::invalid_argument);
}

TEST_CASE("ComponentMatrixMap flattens component matrices without copying",
          "[eigen][component-map]") {
  constexpr Eigen::Index components = 3;
  constexpr Eigen::Index rows = 2;
  constexpr Eigen::Index points = 4;
  std::vector<double> storage(components * rows * points, -1.0);
  SkalaXC::ComponentMatrixMap values(storage.data(), components, rows, points);

  CHECK(values.components() == components);
  CHECK(values.rows() == rows);
  CHECK(values.points() == points);
  for (Eigen::Index component = 0; component < components; ++component) {
    CHECK(values.component_data(component) ==
          storage.data() + component * rows * points);
    for (Eigen::Index point = 0; point < points; ++point)
      for (Eigen::Index row = 0; row < rows; ++row) {
        const auto expected_offset =
            component * rows * points + point * rows + row;
        values(component, row, point) = static_cast<double>(expected_offset);
        CHECK(&values(component, row, point) ==
              storage.data() + expected_offset);
        CHECK(values.component(component)(row, point) ==
              static_cast<double>(expected_offset));
      }
  }

  const auto& const_values = values;
  CHECK(const_values.component(2)(1, 3) == storage.back());
  CHECK_THROWS_AS(values.component(components), std::out_of_range);
  CHECK_THROWS_AS(values(0, rows, 0), std::out_of_range);
  CHECK_THROWS_AS(values(0, 0, points), std::out_of_range);
}
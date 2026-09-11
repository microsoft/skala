#include <catch2/catch_test_macros.hpp>

#include "component_matrix_map.hpp"
#include "derivative_component_map.hpp"
#include "spin_gradient.hpp"

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
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

TEST_CASE("SpinGradient copies point views with independent parent strides",
          "[eigen][spin-gradient][point-view]") {
  const auto source = make_gradient(5);
  auto destination = make_gradient(8);
  const auto original = destination;
  auto* storage = destination.direction_data(SkalaXC::X);

  const auto source_view = source.point_slice(1, 3);
  auto destination_view = destination.point_slice(4, 3);
  REQUIRE(source_view.rows() == 3);
  REQUIRE(source_view.cols() == 6);
  REQUIRE(destination_view.cols() == 6);
  SkalaXC::copy_points(source_view, destination_view);

  REQUIRE(destination.direction_data(SkalaXC::X) == storage);
  for (const auto direction : {SkalaXC::X, SkalaXC::Y, SkalaXC::Z}) {
    REQUIRE(source_view.row(direction).data() ==
            source.direction_data(direction) + 2);
    REQUIRE(destination_view.row(direction).data() ==
            destination.direction_data(direction) + 8);
    for (Eigen::Index point = 0; point < destination.points(); ++point)
      for (const auto spin : {SkalaXC::Alpha, SkalaXC::Beta}) {
        const auto expected = point >= 4 && point < 7
                                  ? source(direction, point - 3, spin)
                                  : original(direction, point, spin);
        REQUIRE(destination(direction, point, spin) == expected);
      }
  }

  SkalaXC::SpinGradient extracted(3);
  SkalaXC::copy_points(destination_view, extracted);
  REQUIRE(extracted.point_slice(0, 3) == source_view);
  SkalaXC::copy_points(extracted, destination.point_slice(0, 3));
  REQUIRE(destination.point_slice(0, 3) == source_view);
}

TEST_CASE("SpinGradient point blocks preserve mutability and constness",
          "[eigen][spin-gradient][point-view]") {
  using Gradient = SkalaXC::SpinGradient;
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<Gradient&>().point_slice(0, 0)),
                     Gradient::PointSlice>);
  STATIC_REQUIRE(std::is_same_v<
                 decltype(std::declval<const Gradient&>().point_slice(0, 0)),
                 Gradient::ConstPointSlice>);
  STATIC_REQUIRE(
      !std::is_assignable_v<
          decltype(std::declval<Gradient::ConstPointSlice&>()(0, 0)), double>);

  auto gradient = make_gradient(5);
  auto view = gradient.point_slice(2, 2);
  view.row(SkalaXC::Y).setConstant(-7.0);
  REQUIRE(gradient(SkalaXC::Y, 2, SkalaXC::Alpha) == -7.0);
  REQUIRE(gradient(SkalaXC::Y, 3, SkalaXC::Beta) == -7.0);
  REQUIRE(gradient(SkalaXC::X, 2, SkalaXC::Alpha) ==
          value(SkalaXC::X, 2, SkalaXC::Alpha));
  REQUIRE(gradient(SkalaXC::Y, 4, SkalaXC::Beta) ==
          value(SkalaXC::Y, 4, SkalaXC::Beta));
}

TEST_CASE(
    "SpinGradient rejects overlapping point copies without modifying values",
    "[eigen][spin-gradient][point-view]") {
  for (const auto offsets : {std::array<Eigen::Index, 3>{0, 1, 4},
                             std::array<Eigen::Index, 3>{1, 0, 4},
                             std::array<Eigen::Index, 3>{1, 1, 3}}) {
    auto gradient = make_gradient(5);
    const auto original = gradient;
    const auto source_offset = offsets[0];
    const auto destination_offset = offsets[1];
    const auto count = offsets[2];
    REQUIRE_THROWS_AS(
        SkalaXC::copy_points(gradient.point_slice(source_offset, count),
                             gradient.point_slice(destination_offset, count)),
        std::invalid_argument);
    REQUIRE(gradient.point_slice(0, 5) == original.point_slice(0, 5));
  }
}

TEST_CASE("SpinGradient copies disjoint slices within the same owner",
          "[eigen][spin-gradient][point-view]") {
  auto gradient = make_gradient(5);
  const auto original = gradient;
  SkalaXC::copy_points(gradient.point_slice(0, 2), gradient.point_slice(2, 2));
  REQUIRE(gradient.point_slice(2, 2) == original.point_slice(0, 2));
  REQUIRE(gradient.point_slice(0, 2) == original.point_slice(0, 2));
  REQUIRE(gradient.point_slice(4, 1) == original.point_slice(4, 1));
}

TEST_CASE("SpinGradient validates point ranges and accepts empty copies",
          "[eigen][spin-gradient][point-view]") {
  auto gradient = make_gradient(5);
  const auto& const_gradient = gradient;
  SkalaXC::SpinGradient empty;
  REQUIRE_NOTHROW(SkalaXC::copy_points(empty, gradient.point_slice(5, 0)));
  REQUIRE_NOTHROW(
      SkalaXC::copy_points(const_gradient.point_slice(2, 0), empty));
  REQUIRE_NOTHROW(
      SkalaXC::copy_points(empty.point_slice(0, 0), empty.point_slice(0, 0)));
  REQUIRE_THROWS_AS(gradient.point_slice(-1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(gradient.point_slice(0, -1), std::out_of_range);
  REQUIRE_THROWS_AS(gradient.point_slice(6, 0), std::out_of_range);
  REQUIRE_THROWS_AS(gradient.point_slice(4, 2), std::out_of_range);
  REQUIRE_THROWS_AS(
      const_gradient.point_slice(1, std::numeric_limits<Eigen::Index>::max()),
      std::out_of_range);
  REQUIRE_THROWS_AS(SkalaXC::copy_points(gradient.point_slice(0, 2),
                                         gradient.point_slice(2, 3)),
                    std::invalid_argument);
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

TEST_CASE("Basis derivative accessors preserve GauXC component ordering",
          "[eigen][component-map][derivative-components]") {
  std::vector<double> storage(10 * 2 * 4, 0.0);
  SkalaXC::ComponentMatrixMap components(storage.data(), 10, 2, 4);
  SkalaXC::BasisComponentView basis(components);
  const SkalaXC::ComponentMatrixMap& read_only = components;
  const SkalaXC::BasisComponentView const_basis(read_only);
  STATIC_REQUIRE(
      !std::is_assignable_v<decltype(const_basis.value()(0, 0)), double>);
  REQUIRE(basis.value().data() == components.component_data(0));
  constexpr std::array<std::array<Eigen::Index, 3>, 3> expected_hessian{
      {{{4, 5, 6}}, {{5, 7, 8}}, {{6, 8, 9}}}};
  for (const auto direction : {SkalaXC::X, SkalaXC::Y, SkalaXC::Z}) {
    REQUIRE(basis.first_derivative(direction).data() ==
            components.component_data(direction + 1));
    for (const auto response : {SkalaXC::X, SkalaXC::Y, SkalaXC::Z}) {
      REQUIRE(const_basis.hessian(direction, response).data() ==
              components.component_data(expected_hessian[direction][response]));
      REQUIRE(basis.hessian(direction, response).rows() == 2);
      REQUIRE(basis.hessian(direction, response).cols() == 4);
    }
  }
  basis.hessian(SkalaXC::Y, SkalaXC::X).setConstant(7.0);
  REQUIRE(components.component(5).isConstant(7.0));
  REQUIRE(const_basis.hessian(SkalaXC::X, SkalaXC::Y).isConstant(7.0));
  REQUIRE_THROWS_AS(basis.first_derivative(static_cast<SkalaXC::Direction>(3)),
                    std::out_of_range);
  REQUIRE_THROWS_AS(
      basis.hessian(SkalaXC::X, static_cast<SkalaXC::Direction>(3)),
      std::out_of_range);

  for (const Eigen::Index count : {1, 4}) {
    SkalaXC::ComponentMatrixMap limited(storage.data(), count, 2, 4);
    SkalaXC::BasisComponentView limited_basis(limited);
    REQUIRE(limited_basis.value().data() == storage.data());
    REQUIRE_THROWS_AS(limited_basis.hessian(SkalaXC::X, SkalaXC::X),
                      std::out_of_range);
    if (count == 1)
      REQUIRE_THROWS_AS(limited_basis.first_derivative(SkalaXC::X),
                        std::out_of_range);
    else
      REQUIRE(limited_basis.first_derivative(SkalaXC::Z).data() ==
              limited.component_data(3));
  }
  SkalaXC::ComponentMatrixMap invalid(storage.data(), 5, 2, 4);
  REQUIRE_THROWS_AS(SkalaXC::BasisComponentView(invalid),
                    std::invalid_argument);
}

TEST_CASE("Pauli derivative accessors preserve both channel layouts",
          "[eigen][component-map][derivative-components]") {
  for (const Eigen::Index per_channel : {1, 4}) {
    std::vector<double> storage(2 * per_channel * 3 * 5, 0.0);
    SkalaXC::ComponentMatrixMap components(storage.data(), 2 * per_channel, 3,
                                           5);
    SkalaXC::PauliComponentView channels(components);
    const SkalaXC::ComponentMatrixMap& read_only = components;
    const SkalaXC::PauliComponentView const_channels(read_only);
    STATIC_REQUIRE(
        !std::is_assignable_v<
            decltype(const_channels.value(SkalaXC::Scalar)(0, 0)), double>);
    for (const auto channel : {SkalaXC::Scalar, SkalaXC::SpinZ}) {
      REQUIRE(channels.value(channel).data() ==
              components.component_data(channel * per_channel));
      for (const auto direction : {SkalaXC::X, SkalaXC::Y, SkalaXC::Z}) {
        if (per_channel == 1) {
          REQUIRE_THROWS_AS(channels.first_derivative(channel, direction),
                            std::out_of_range);
        } else {
          REQUIRE(
              const_channels.first_derivative(channel, direction).data() ==
              components.component_data(channel * per_channel + direction + 1));
          channels.first_derivative(channel, direction).setConstant(9.0);
          REQUIRE(components.component(channel * per_channel + direction + 1)
                      .isConstant(9.0));
        }
      }
    }
    REQUIRE_THROWS_AS(channels.value(static_cast<SkalaXC::PauliChannel>(2)),
                      std::out_of_range);
    REQUIRE_THROWS_AS(channels.first_derivative(
                          SkalaXC::Scalar, static_cast<SkalaXC::Direction>(3)),
                      std::out_of_range);
  }
  std::vector<double> storage(6, 0.0);
  SkalaXC::ComponentMatrixMap invalid(storage.data(), 6, 1, 1);
  REQUIRE_THROWS_AS(SkalaXC::PauliComponentView(invalid),
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
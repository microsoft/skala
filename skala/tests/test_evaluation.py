from dataclasses import FrozenInstanceError

import pytest
import torch
from skala.features import AOFeatureSpec, Feature, ao_derivative_order
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.feature_math import AODirection, MGGAFeatureFunction, PackedAO


def test_feature_name_parses_model_metadata_string() -> None:
    assert Feature("density") is Feature.DENSITY


def test_packed_ao_views_preserve_trailing_dimensions() -> None:
    components = torch.arange(60).reshape(10, 2, 3)

    for tensor in (components, components.transpose(-1, -2)):
        packed_ao = PackedAO(tensor)

        torch.testing.assert_close(packed_ao.value, tensor[0])
        torch.testing.assert_close(packed_ao.gradient, tensor[1:4])
        for actual, component in zip(
            packed_ao.diagonal_hessian, (4, 7, 9), strict=True
        ):
            torch.testing.assert_close(actual, tensor[component])


def test_packed_ao_normalizes_unpacked_values() -> None:
    values = torch.arange(6).reshape(2, 3)

    packed_ao = PackedAO(values)

    assert packed_ao._tensor.shape == (1, 2, 3)
    torch.testing.assert_close(packed_ao.value, values)


def test_packed_ao_rejects_unsupported_component_count() -> None:
    with pytest.raises(ValueError, match="1, 4, or 10"):
        PackedAO(torch.zeros((3, 2, 3)))


def test_packed_ao_hessian_uses_pyscf_component_order() -> None:
    packed_ao = PackedAO(torch.arange(10).reshape(10, 1, 1))

    components = [
        (ao_direction, feature_direction, int(value.item()))
        for ao_direction, feature_direction, value in packed_ao.hessian()
    ]

    assert components == [
        (AODirection.X, AODirection.X, 4),
        (AODirection.X, AODirection.Y, 5),
        (AODirection.X, AODirection.Z, 6),
        (AODirection.Y, AODirection.X, 5),
        (AODirection.Y, AODirection.Y, 7),
        (AODirection.Y, AODirection.Z, 8),
        (AODirection.Z, AODirection.X, 6),
        (AODirection.Z, AODirection.Y, 8),
        (AODirection.Z, AODirection.Z, 9),
    ]


def test_ao_feature_spec_normalizes_and_derives_requirements() -> None:
    spec = AOFeatureSpec([Feature.DENSITY, Feature.DENSITY, Feature.GRAD, Feature.LAPL])

    assert list(spec) == [
        (Feature.DENSITY, slice(0, 1)),
        (Feature.GRAD, slice(1, 4)),
        (Feature.LAPL, slice(4, 5)),
    ]
    assert spec.nderiv == 2
    assert spec.nfeats == 5


def test_ao_feature_spec_rejects_grid_features() -> None:
    with pytest.raises(ValueError, match="grid_weights"):
        AOFeatureSpec([Feature.DENSITY, Feature.GRID_WEIGHTS])


def test_ao_feature_spec_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="At least one"):
        AOFeatureSpec([])


def test_feature_spec_normalizes_model_contract() -> None:
    spec = FeatureSpec([Feature.DENSITY, Feature.DENSITY])

    assert set(spec) == {Feature.DENSITY}
    assert spec.requests(Feature.DENSITY)
    assert not spec.requests(Feature.GRAD)


@pytest.mark.parametrize(
    ("feature", "supports_screened_evaluation"),
    [
        (Feature.ATOMIC_GRID_WEIGHTS, False),
        (Feature.ATOMIC_GRID_SIZES, True),
        (Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE, False),
    ],
)
def test_feature_spec_derives_spatial_decomposition_support(
    feature: Feature, supports_screened_evaluation: bool
) -> None:
    spec = FeatureSpec([feature, feature])

    assert set(spec) == {feature}
    assert spec.supports_spatial_decomposition is supports_screened_evaluation


def test_feature_spec_derives_evaluation_needs() -> None:
    feature_spec = FeatureSpec(
        [Feature.DENSITY, Feature.DENSITY, Feature.ATOMIC_GRID_WEIGHTS]
    )

    assert set(feature_spec) == {Feature.DENSITY, Feature.ATOMIC_GRID_WEIGHTS}
    assert feature_spec.ao_features == AOFeatureSpec([Feature.DENSITY])
    assert feature_spec.requires_ao_evaluation
    assert feature_spec.requires_atomic_layout

    grid_feature_spec = FeatureSpec([Feature.GRID_WEIGHTS])
    assert grid_feature_spec.ao_features is None
    assert not grid_feature_spec.requires_ao_evaluation


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ([], 0),
        ([Feature.DENSITY], 0),
        ([Feature.GRAD], 1),
        ([Feature.KIN], 1),
        ([Feature.LAPL], 2),
    ],
)
def test_ao_derivative_order(features: list[Feature], expected: int) -> None:
    assert ao_derivative_order(features) == expected


@pytest.mark.parametrize(
    ("features", "expected_derivative_order", "expected_feature_count"),
    [
        ([Feature.DENSITY], 0, 1),
        ([Feature.GRAD], 1, 3),
        ([Feature.KIN], 1, 1),
        ([Feature.LAPL], 2, 1),
        ([Feature.DENSITY, Feature.GRAD, Feature.KIN, Feature.LAPL], 2, 6),
    ],
)
def test_mgga_feature_function_derives_channel_requirements(
    features: list[Feature],
    expected_derivative_order: int,
    expected_feature_count: int,
) -> None:
    feature_function = MGGAFeatureFunction(AOFeatureSpec(features))

    assert feature_function.deriv == expected_derivative_order
    assert feature_function.nfeats == expected_feature_count


def test_mgga_feature_function_to_dict_preserves_public_feature_shapes() -> None:
    feature_function = MGGAFeatureFunction(
        AOFeatureSpec([Feature.DENSITY, Feature.GRAD, Feature.KIN])
    )
    packed_features = torch.zeros((2, 5, 7))

    features = feature_function.to_dict(packed_features)

    assert features[Feature.DENSITY].shape == (2, 7)
    assert features[Feature.GRAD].shape == (2, 3, 7)
    assert features[Feature.KIN].shape == (2, 7)


def test_evaluation_policy_defaults_and_is_immutable() -> None:
    policy = EvaluationPolicy()

    assert policy.ao_block_size is None
    assert policy.safety_fraction == 0.8
    with pytest.raises(FrozenInstanceError):
        policy.safety_fraction = 0.5  # type: ignore[misc]

from dataclasses import FrozenInstanceError

import pytest
from skala.features import Feature, ao_derivative_order
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.feature_math import MGGAFeatureFunction


def test_feature_name_parses_model_metadata_string() -> None:
    assert Feature("density") is Feature.DENSITY


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
    assert feature_spec.ao_features == frozenset({Feature.DENSITY})
    assert feature_spec.requires_ao_evaluation
    assert feature_spec.requires_atomic_layout


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
    feature_function = MGGAFeatureFunction(features)

    assert feature_function.deriv == expected_derivative_order
    assert feature_function.nfeats == expected_feature_count


def test_mgga_feature_function_rejects_grid_features() -> None:
    with pytest.raises(ValueError, match="grid_weights"):
        MGGAFeatureFunction([Feature.DENSITY, Feature.GRID_WEIGHTS])


def test_evaluation_policy_defaults_and_is_immutable() -> None:
    policy = EvaluationPolicy()

    assert policy.ao_block_size is None
    assert policy.safety_fraction == 0.8
    with pytest.raises(FrozenInstanceError):
        policy.safety_fraction = 0.5  # type: ignore[misc]

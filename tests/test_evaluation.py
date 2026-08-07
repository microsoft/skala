from dataclasses import FrozenInstanceError

import pytest

from skala.features import Feature
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec


def test_feature_name_parses_model_metadata_string() -> None:
    assert Feature("density") is Feature.DENSITY


@pytest.mark.parametrize(
    ("features", "expected_order"),
    [
        ([], 0),
        ([Feature.DENSITY], 0),
        ([Feature.GRAD], 1),
        ([Feature.KIN], 1),
        ([Feature.LAPL], 2),
        (
            [
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.LAPL,
            ],
            2,
        ),
    ],
)
def test_feature_spec_derives_mgga_requirements(
    features: list[Feature], expected_order: int
) -> None:
    spec = FeatureSpec(features)

    assert spec.requires_ao_evaluation is bool(features)
    assert spec.ao_derivative_order == expected_order
    assert spec.with_density is (Feature.DENSITY in features)
    assert spec.with_grad is (Feature.GRAD in features)
    assert spec.with_kin is (Feature.KIN in features)
    assert spec.with_lapl is (Feature.LAPL in features)


@pytest.mark.parametrize(
    ("feature", "supports_screened_evaluation"),
    [
        (Feature.ATOMIC_GRID_WEIGHTS, False),
        (Feature.ATOMIC_GRID_SIZES, True),
        (Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE, False),
    ],
)
def test_feature_spec_derives_atomic_layout_requirements(
    feature: Feature, supports_screened_evaluation: bool
) -> None:
    spec = FeatureSpec([feature, feature])

    assert spec.names == frozenset({feature})
    assert spec.requires_atomic_layout
    assert spec.supports_screened_evaluation is supports_screened_evaluation


def test_evaluation_policy_defaults_and_is_immutable() -> None:
    policy = EvaluationPolicy()

    assert policy.ao_block_size is None
    assert policy.safety_fraction == 0.8
    with pytest.raises(FrozenInstanceError):
        policy.safety_fraction = 0.5  # type: ignore[misc]

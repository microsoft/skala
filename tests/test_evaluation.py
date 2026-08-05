from dataclasses import FrozenInstanceError

import pytest
import torch

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.features import generate_features
from skala.pyscf.numint import SkalaNumInt


@pytest.mark.parametrize(
    ("features", "expected_order"),
    [
        ([], 0),
        (["density"], 0),
        (["grad"], 1),
        (["kin"], 1),
        (["lapl"], 2),
        (["density", "grad", "kin", "lapl"], 2),
    ],
)
def test_feature_spec_derives_mgga_requirements(
    features: list[str], expected_order: int
) -> None:
    spec = FeatureSpec(features)

    assert spec.requires_mgga is bool(features)
    assert spec.ao_derivative_order == expected_order
    assert spec.with_density is ("density" in features)
    assert spec.with_grad is ("grad" in features)
    assert spec.with_kin is ("kin" in features)
    assert spec.with_lapl is ("lapl" in features)


@pytest.mark.parametrize(
    ("feature", "supports_screened_evaluation"),
    [
        ("atomic_grid_weights", False),
        ("atomic_grid_sizes", True),
        ("atomic_grid_size_bound_shape", False),
    ],
)
def test_feature_spec_derives_atomic_layout_requirements(
    feature: str, supports_screened_evaluation: bool
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


def test_explicit_empty_feature_set_stays_empty() -> None:
    features = generate_features(
        mol=object(),
        dm=torch.eye(1),
        grids=object(),
        features=set(),
    )

    assert features == {}


class DensityFunctional(ExcFunctionalBase):
    def __init__(self) -> None:
        super().__init__()
        self.features = ["density"]


def test_numint_translates_chunk_size_into_evaluation_policy() -> None:
    numint = SkalaNumInt(DensityFunctional(), chunk_size=96)

    assert numint.feature_spec == FeatureSpec(["density"])
    assert numint.evaluation_policy == EvaluationPolicy(ao_block_size=96)

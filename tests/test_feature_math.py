from itertools import combinations

import pytest
import torch

from skala.features import Feature
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.feature_math import MGGAFeatureFunction

_MGGA_FEATURES = (Feature.DENSITY, Feature.GRAD, Feature.KIN, Feature.LAPL)
_MGGA_FEATURE_COMBINATIONS = [
    combination
    for size in range(1, len(_MGGA_FEATURES) + 1)
    for combination in combinations(_MGGA_FEATURES, size)
]


@pytest.mark.parametrize(
    ("feature_names", "expected_deriv", "expected_nfeats"),
    [
        ({Feature.DENSITY}, 0, 1),
        ({Feature.GRAD}, 1, 3),
        ({Feature.KIN}, 1, 1),
        ({Feature.LAPL}, 2, 1),
        (
            {
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.LAPL,
            },
            2,
            6,
        ),
    ],
)
def test_mgga_supported_features_are_linear_in_density_matrix(
    feature_names: set[Feature],
    expected_deriv: int,
    expected_nfeats: int,
) -> None:
    feature_spec = FeatureSpec(feature_names)
    feature_function = MGGAFeatureFunction(feature_spec)
    ncomp = (expected_deriv + 1) * (expected_deriv + 2) * (expected_deriv + 3) // 6
    ao = torch.arange(1, ncomp * 2 * 3 + 1, dtype=torch.float64).reshape(ncomp, 2, 3)
    if expected_deriv == 0:
        ao = ao[0]
    dm = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    tangent = torch.tensor([[0.2, -0.1], [-0.1, 0.3]], dtype=torch.float64)

    features = feature_function(dm, ao)
    _, feature_jvp = torch.func.jvp(
        lambda value: feature_function(value, ao),
        (dm,),
        (tangent,),
    )

    def first_jvp(value: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(
            lambda inner: feature_function(inner, ao),
            (value,),
            (tangent,),
        )[1]

    _, second_jvp = torch.func.jvp(
        first_jvp,
        (dm,),
        (torch.ones_like(dm),),
    )

    assert feature_function.deriv == expected_deriv
    assert feature_function.nfeats == expected_nfeats
    assert feature_function.feature_spec is feature_spec
    assert features.shape == (expected_nfeats, 3)
    assert set(feature_function.to_dict(features)) == feature_names
    torch.testing.assert_close(feature_jvp, feature_function(tangent, ao))
    torch.testing.assert_close(second_jvp, torch.zeros_like(second_jvp))


@pytest.mark.parametrize("feature_names", _MGGA_FEATURE_COMBINATIONS)
@pytest.mark.parametrize("spin_channels", [None, 2])
def test_mgga_analytic_vjp_matches_autograd(
    feature_names: tuple[Feature, ...], spin_channels: int | None
) -> None:
    feature_function = MGGAFeatureFunction(FeatureSpec(feature_names))
    ncomp = (
        (feature_function.deriv + 1)
        * (feature_function.deriv + 2)
        * (feature_function.deriv + 3)
        // 6
    )
    generator = torch.Generator().manual_seed(0)
    ao = torch.randn((ncomp, 3, 5), dtype=torch.float64, generator=generator)
    if feature_function.deriv == 0:
        ao = ao[0]
    dm_shape = (3, 3) if spin_channels is None else (spin_channels, 3, 3)
    dm = torch.randn(dm_shape, dtype=torch.float64, generator=generator)

    features, pullback = torch.func.vjp(lambda value: feature_function(value, ao), dm)
    cotangent = torch.randn(features.shape, dtype=features.dtype, generator=generator)

    expected = pullback(cotangent)[0]
    actual = feature_function.vjp(ao, cotangent)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("feature_names", [[], [Feature.GRID_WEIGHTS]])
def test_mgga_requires_at_least_one_ao_derived_feature(
    feature_names: list[Feature],
) -> None:
    with pytest.raises(
        ValueError, match="At least one AO-derived feature must be selected"
    ):
        MGGAFeatureFunction(FeatureSpec(feature_names))

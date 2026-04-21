# SPDX-License-Identifier: MIT

"""
Traditional exchange-correlation functionals.

This module implements standard DFT exchange-correlation functionals
including LDA, PBE, and TPSS using exact spin scaling for exchange.
"""

import math
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from skala.functional import density
from skala.functional.base import ExcFunctionalBase


class SpinScaledXCFunctional(ExcFunctionalBase):
    """
    Base class for XC functionals using exact spin scaling of exchange.

    This class implements the exact spin scaling relation for exchange:
    E_x[ρ_α, ρ_β] = 1/2 * (E_x[2ρ_α] + E_x[2ρ_β])
    """

    def get_d3_settings(self) -> str:
        return self.__class__.__name__.lower()

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        """
        Compute the exchange energy density.

        Parameters
        ----------
        mol_features : dict[str, Tensor]
            Dictionary containing molecular features.

        Returns
        -------
        Tensor
            Exchange energy density.
        """
        raise NotImplementedError()

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        """
        Compute the correlation energy density.

        Parameters
        ----------
        mol_features : dict[str, Tensor]
            Dictionary containing molecular features.

        Returns
        -------
        Tensor
            Correlation energy density.
        """
        raise NotImplementedError()

    def correlation(self, mol_features: dict[str, Tensor]) -> Tensor:
        """
        Compute the correlation energy.

        Parameters
        ----------
        mol_features : dict[str, Tensor]
            Dictionary containing molecular features.

        Returns
        -------
        Tensor
            Correlation energy.
        """
        rho_total = mol_features["density"].sum(0)
        return rho_total * self.correlation_density(mol_features)

    def get_exc_density(self, mol: dict[str, Tensor]) -> Tensor:
        exch = self.exchange(density.scale_by(mol, 2)).sum(0) / 2
        corr = self.correlation(mol)
        return exch + corr


class LDA(SpinScaledXCFunctional):
    """
    Local Density Approximation (LDA) functional.

    Implements LDA exchange with no correlation.
    Exchange: E_x[ρ] = -3/4 * (3/π)^(1/3) * ρ^(4/3)
    """

    features = ["density", "grid_weights"]

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        return (
            -3 / 4 * (3 / math.pi) ** (1 / 3) * mol_features["density"].abs() ** (4 / 3)
        )

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        return mol_features["density"].new_zeros((1,))


class SPW92(SpinScaledXCFunctional):
    """
    SPW92 functional: LDA exchange + Perdew-Wang 92 correlation.

    This is LDA exchange with the PW92 parameterization of the
    correlation energy of the uniform electron gas.
    """

    features = ["density", "grid_weights"]

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        return (
            -3 / 4 * (3 / math.pi) ** (1 / 3) * mol_features["density"].abs() ** (4 / 3)
        )

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        def Gamma(
            rs: Tensor, A: float, a1: float, b1: float, b2: float, b3: float, b4: float
        ) -> Tensor:
            rs_sq = rs.sqrt()
            poly = (b1 + (b2 + (b3 + b4 * rs_sq) * rs_sq) * rs_sq) * rs_sq
            return -2 * A * (1 + a1 * rs) * torch.log(1 + 0.5 / (A * poly))

        rho = mol_features["density"]
        zeta, rho_total = density.zeta(rho), rho.sum(0)
        ff0 = 1.709921
        ff = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
        rs = (3 / torch.clamp(4 * math.pi * rho_total, density.EPS)) ** (1 / 3)
        eps_c0 = Gamma(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        eps_c1 = Gamma(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        alpha_c = -Gamma(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
        return (
            eps_c0
            + alpha_c * ff / ff0 * (1 - zeta**4)
            + (eps_c1 - eps_c0) * ff * zeta**4
        )


class PBE(SpinScaledXCFunctional):
    """
    Perdew-Burke-Ernzerhof (PBE) generalized gradient approximation.

    PBE is a widely-used GGA functional that includes both exchange
    and correlation gradient corrections to the local density approximation.
    """

    features = ["density", "grad", "grid_weights"]

    def __init__(self) -> None:
        super().__init__()
        self.lda = SPW92()
        self.beta = nn.Parameter(torch.tensor(0.066725), requires_grad=False)
        self.kappa = nn.Parameter(torch.tensor(0.804), requires_grad=False)
        self.mu = self.beta * (math.pi**2 / 3)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from super().parameters(recurse)
        yield from [self.beta, self.kappa]

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        rho = mol_features["density"]
        grad = mol_features["grad"]
        FX = (
            1
            + self.kappa
            - self.kappa
            / (1 + self.mu * density.reduced_gradient(rho, grad) ** 2 / self.kappa)
        )
        return self.lda.exchange(mol_features) * FX

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        eps_c_unif = self.lda.correlation_density(mol_features)
        rho = mol_features["density"]
        grad = mol_features["grad"]
        rho_total, grad_total = rho.sum(0), grad.sum(0)
        zeta = density.zeta(rho)
        ks = torch.sqrt(4 * density.kF(rho_total) / math.pi)
        phi = (
            torch.clamp(1 + zeta, density.EPS) ** (2 / 3)
            + torch.clamp(1 - zeta, density.EPS) ** (2 / 3)
        ) / 2
        t = density.grad_norm(grad_total) / torch.clamp(
            2 * phi * ks * rho_total, density.EPS
        )
        gamma = (1 - math.log(2)) / math.pi**2
        Ainv = (
            torch.expm1(-eps_c_unif / (gamma * phi**3)) * gamma / self.beta
        )  # numerically much better behaved than A
        t2 = t**2
        poly = t2 * Ainv * (Ainv + t2) / (Ainv**2 + (Ainv + t2) * t2)
        H = gamma * phi**3 * torch.log(1 + self.beta / gamma * poly)
        return eps_c_unif + H


class TPSS(SpinScaledXCFunctional):
    """
    Tao-Perdew-Staroverov-Scuseria (TPSS) meta-GGA functional.

    TPSS is a meta-GGA that depends on the kinetic energy density
    in addition to the density and its gradient. It satisfies many
    exact constraints of density functional theory.
    """

    features = ["density", "kin", "grad", "grid_weights"]

    def __init__(self) -> None:
        super().__init__()
        self.lda = SPW92()
        self.pbe = PBE()
        self.c = nn.Parameter(torch.tensor(1.59096), requires_grad=False)
        self.e = nn.Parameter(torch.tensor(1.537), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(0.40), requires_grad=False)
        self.d = nn.Parameter(torch.tensor(2.8), requires_grad=False)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from super().parameters(recurse)
        yield from [self.c, self.e, self.b, self.d]

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        rho = mol_features["density"]
        grad = mol_features["grad"]
        kin = mol_features["kin"]
        # p is the reduced gradient squared, z is the zeta value
        p, z = density.reduced_gradient(rho, grad) ** 2, density.z(rho, grad, kin)
        alpha = (5 * p / 3) * (1 / torch.clamp(z, density.EPS) - 1)
        q_b = (9 / 20) * (alpha - 1) / torch.sqrt(
            1 + self.b * alpha * (alpha - 1)
        ) + 2 * p / 3
        kappa = self.pbe.kappa
        x = (
            (10 / 81 + self.c * z**2 / (1 + z**2) ** 2) * p
            + 146 / 2025 * q_b**2
            - 73 / 405 * q_b * torch.sqrt(1 / 2 * (3 / 5 * z) ** 2 + 1 / 2 * p**2)
            + 1 / kappa * (10 / 81) ** 2 * p**2
            + 2 * torch.sqrt(self.e) * 10 / 81 * (3 / 5 * z) ** 2
            + self.e * self.pbe.mu * p**3
        ) / (1 + torch.sqrt(self.e) * p) ** 2
        FX = 1 + kappa - kappa / (1 + x / kappa)
        return self.lda.exchange(mol_features) * FX

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        rho = mol_features["density"]
        grad = mol_features["grad"]
        kin = mol_features["kin"]
        rho_total, grad_total, kin_total = rho.sum(0), grad.sum(0), kin.sum(0)
        zeta, grad_zeta = density.zeta(rho), density.grad_zeta(rho, grad).norm(dim=-2)

        xi = grad_zeta / torch.clamp(
            2 * (3 * math.pi**2 * rho_total.abs()) ** (1 / 3), density.EPS
        )

        CC0 = 0.53 + 0.87 * zeta**2 + 0.50 * zeta**4 + 2.26 * zeta**6
        Czetaxi = (
            CC0
            / (1 + xi**2 * ((1 + zeta) ** (-4 / 3) + (1 - zeta) ** (-4 / 3)) / 2) ** 4
        )
        eps_c_pbe = self.pbe.correlation_density(mol_features)
        z = density.z(rho_total, grad_total, kin_total)
        mols = density.separate(mol_features)
        eps_c_revpkzb = eps_c_pbe * (1 + Czetaxi * z**2) - (1 + Czetaxi) * z**2 * sum(
            (mols[spin]["density"][spin] / rho_total)
            * torch.max(eps_c_pbe, self.pbe.correlation_density(mols[spin]))
            for spin in range(2)
        )
        return eps_c_revpkzb * (1 + self.d * eps_c_revpkzb * z**3)


XC_FUNCTIONAL_MAP: dict[str, type[ExcFunctionalBase]] = {
    "lda": LDA,
    "spw92": SPW92,
    "pbe": PBE,
    "tpss": TPSS,
}


def get_traditional_functional(xc: str) -> type[ExcFunctionalBase]:
    """
    Get a traditional functional class by name.

    Parameters
    ----------
    xc : str
        Name of the functional ("lda", "spw92", "pbe", or "tpss").

    Returns
    -------
    type[ExcFunctionalBase]
        The functional class.

    Raises
    ------
    KeyError
        If the functional name is not supported.
    """
    if xc not in XC_FUNCTIONAL_MAP:
        raise KeyError(f"Unsupported traditional functional '{xc}' requested")
    return XC_FUNCTIONAL_MAP[xc]

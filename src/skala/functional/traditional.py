# SPDX-License-Identifier: MIT

"""
Traditional exchange-correlation functionals.

This module implements standard DFT exchange-correlation functionals
including LDA, PBE, and TPSS using exact spin scaling for exchange.
"""

import math

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


class _SCANLikeFunctional(SpinScaledXCFunctional):
    features = ["density", "kin", "grad", "grid_weights"]

    def __init__(
        self, alpha_mode: int, interpolation_mode: int, gradient_correction_mode: int
    ) -> None:
        super().__init__()

        self.alpha_mode = alpha_mode
        self.interpolation_mode = interpolation_mode
        self.gradient_correction_mode = gradient_correction_mode

        x_params = (
            -0.023185843322,
            0.234528941479,
            -0.887998041597,
            1.451297044490,
            -0.663086601049,
            -0.4445555,
            -0.667,
            1.0,
        )
        ie_params_c = (
            -0.64,
            -0.4352,
            -1.535685604549,
            3.061560252175,
            -1.915710236206,
            0.516884468372,
            -0.051848879792,
        )

        self.cp = 4 * (3 * math.pi**2) ** (2 / 3)
        self.tueg_con = 3 / 10 * (3 * math.pi**2) ** (2 / 3)
        self.rs_prefac = (0.75 / math.pi) ** (1 / 3)
        self.zeta_limit = 0.99999999999990
        self.x_powers = (7, 6, 5, 4, 3, 2, 1, 0)
        self.ie_params_c_powers = tuple(range(len(ie_params_c), 0, -1))
        self.afix_t = math.sqrt(math.pi / 4) * (9 * math.pi / 4) ** (1 / 6)
        self.gam = 0.5198420997897463
        self.fzz = 8 / (9 * self.gam)
        self.ax = -3 / (4 * math.pi) * (3 * math.pi**2) ** (1 / 3)
        self.b2 = math.sqrt(5913 / 405000)
        self.b1 = 511 / 13500 / (2 * self.b2)
        self.b3 = 0.5

        def parameter(value: float | tuple[float, ...]) -> nn.Parameter:
            return nn.Parameter(torch.tensor(value), requires_grad=False)

        self.eta = parameter(1.0e-3)
        self.tau_r = parameter(1.0e-4)
        self.a_reg = parameter(1.0e-3)
        self.a1 = parameter(4.9479)
        self.k0 = parameter(0.174)
        self.mu = parameter(10 / 81)
        self.k1 = parameter(0.065)
        self.cfx1 = parameter(0.667)
        self.cfx2 = parameter(0.800)
        self.cfdx1 = parameter(1.24)
        self.d_damp2 = parameter(0.361)
        self.x_params = parameter(x_params)
        self.cfdc1 = parameter(0.7)
        self.cfc1 = parameter(0.640)
        self.cfc2 = parameter(1.5)
        self.ie_params_c = parameter(tuple(reversed(ie_params_c)))
        self.b1c = parameter(0.0285764)
        self.chi_ld = parameter(0.12802585262625815)
        self.gamma = parameter(0.031090690869655)
        self.beta_mb = parameter(0.066725)
        self.afactor = parameter(0.1)
        self.bfactor = parameter(0.1778)

    @property
    def alpha_ge(self) -> Tensor:
        return 20 / 27 + self.eta * 5 / 3

    @property
    def x_del_f2(self) -> Tensor:
        powers = self.x_params.new_tensor(self.x_powers[:7])
        return (powers * self.x_params[:7]).sum()

    @property
    def c_del_f2(self) -> Tensor:
        powers = self.ie_params_c.new_tensor(self.ie_params_c_powers)
        return (powers * self.ie_params_c).sum()

    def _scan_bounded_exp(self, x: Tensor) -> Tensor:
        return torch.exp(torch.clamp(x, min=-80.0, max=80.0))

    def _scan_bounded_expm1(self, x: Tensor) -> Tensor:
        return torch.expm1(torch.clamp(x, min=-80.0, max=80.0))

    def _scan_signed_clamp(self, x: Tensor) -> Tensor:
        eps = torch.finfo(x.dtype).eps
        return torch.where(x >= 0, torch.clamp(x, min=eps), torch.clamp(x, max=-eps))

    def _scan_horner(self, alpha: Tensor, coeffs: Tensor) -> Tensor:
        result = torch.ones_like(alpha) * coeffs[0]
        for coeff in coeffs[1:]:
            result = result * alpha + coeff
        return result

    def _scan_interp_cutoffs(
        self, c1: Tensor, c2: Tensor, d: Tensor, dtype: torch.dtype
    ) -> tuple[float, float]:
        """libxc-style cutoffs where the SCAN f(alpha) exponentials underflow to 0."""
        neg_log_eps = -math.log(torch.finfo(dtype).eps)
        left_cut = neg_log_eps / (neg_log_eps + float(c1))
        neg_log_eps_d = -math.log(torch.finfo(dtype).eps / float(d.abs()))
        right_cut = (neg_log_eps_d + float(c2)) / neg_log_eps_d
        return left_cut, right_cut

    def _scan_f_alpha_branches(
        self, alpha: Tensor, c1: Tensor, c2: Tensor, d: Tensor
    ) -> tuple[Tensor, Tensor]:
        left_cut, right_cut = self._scan_interp_cutoffs(c1, c2, d, alpha.dtype)
        # Clamp the *argument* first so 1 - a is never 0; then mask outside cutoffs to zero.
        a_left = torch.clamp(alpha, max=left_cut)
        a_right = torch.clamp(alpha, min=right_cut)
        scan_interp = torch.where(
            alpha > left_cut,
            torch.zeros_like(alpha),
            self._scan_bounded_exp(-c1 * a_left / (1 - a_left)),
        )
        tail_interp = torch.where(
            alpha < right_cut,
            torch.zeros_like(alpha),
            -d * self._scan_bounded_exp(c2 / (1 - a_right)),
        )
        return scan_interp, tail_interp

    def _scan_exchange_interpolation(self, alpha: Tensor) -> Tensor:
        scan_interp, tail_interp = self._scan_f_alpha_branches(
            alpha, self.cfx1, self.cfx2, self.cfdx1
        )
        if self.interpolation_mode == 0:
            return torch.where(alpha <= 1.0, scan_interp, tail_interp)
        if self.interpolation_mode == 1:
            poly_interp = self._scan_horner(alpha, self.x_params)
            return torch.where(
                alpha <= 0.0,
                scan_interp,
                torch.where(alpha <= 2.5, poly_interp, tail_interp),
            )
        raise ValueError(
            f"Unsupported SCAN exchange interpolation mode {self.interpolation_mode}"
        )

    def _scan_exchange_enhancement(self, p: Tensor, alpha: Tensor) -> Tensor:
        h0x = 1 + self.k0
        ief = self._scan_exchange_interpolation(alpha)
        oma = 1 - alpha

        if self.gradient_correction_mode == 0:
            b4 = self.mu.square() / self.k1 - (1606 / 18225 + self.b1**2)
            wfac = b4 * p.square() * self._scan_bounded_exp(-b4 * p / self.mu)
            vfac = self.b1 * p + self.b2 * oma * self._scan_bounded_exp(
                -self.b3 * oma.square()
            )
            yfac = self.mu * p + wfac + vfac.square()
            h1x = 1 + self.k1 - self.k1 / (1 + yfac / self.k1)
        elif self.gradient_correction_mode in (1, 2):
            c2 = -self.x_del_f2 * (1 - h0x)
            damp = self._scan_bounded_exp(-(p.square()) / self.d_damp2**4)
            h1x = (
                1
                + self.k1
                - self.k1 / (1 + p * (self.mu + self.alpha_ge * c2 * damp) / self.k1)
            )
        else:
            raise ValueError(
                "Unsupported SCAN exchange gradient correction mode "
                f"{self.gradient_correction_mode}"
            )

        safe_p = torch.clamp(p, min=density.EPS)
        gx = 1 - self._scan_bounded_exp(-self.a1 / safe_p.pow(1 / 4))
        return (h1x + ief * (h0x - h1x)) * gx

    def _scan_exchange_density(
        self,
        density_spin: Tensor,
        grad_spin: Tensor,
        kin_spin: Tensor,
    ) -> Tensor:
        safe_density = torch.clamp(density_spin, min=density.EPS)
        p = grad_spin.square() / (self.cp * safe_density.pow(8 / 3))
        tueg = self.tueg_con * safe_density.pow(5 / 3)
        if self.alpha_mode == 1:
            tueg = tueg + self.tau_r

        tauw = grad_spin.square() / (8 * safe_density)
        if self.alpha_mode in (0, 1):
            alpha = (kin_spin - tauw) / tueg
            if self.alpha_mode == 1:
                alpha = alpha.clamp(min=0.0)
                alpha = alpha.pow(3) / (alpha.square() + self.a_reg)
        elif self.alpha_mode == 2:
            alpha = (kin_spin - tauw) / (tueg + self.eta * tauw)
        else:
            raise ValueError(f"Unsupported SCAN exchange alpha mode {self.alpha_mode}")

        exlda = self.ax * density_spin.abs().pow(4 / 3)
        fx = self._scan_exchange_enhancement(p, alpha)
        return torch.where(density_spin > 0, exlda * fx, torch.zeros_like(density_spin))

    def _scan_grcor2(
        self,
        rs: Tensor,
        A: float,
        A1: float,
        B1: float,
        B2: float,
        B3: float,
        B4: float,
    ) -> tuple[Tensor, Tensor]:
        sqrt_rs = torch.sqrt(torch.clamp(rs, min=density.EPS))

        q0 = -2 * A * (1 + A1 * rs)
        q0_rs = -2 * A * A1
        q1 = 2 * A * sqrt_rs * (B1 + sqrt_rs * (B2 + sqrt_rs * (B3 + B4 * sqrt_rs)))
        q1_rs = A * (2 * B2 + B1 / sqrt_rs + 3 * B3 * sqrt_rs + 4 * B4 * rs)
        q2 = torch.log1p(1 / q1)

        gg = q0 * q2
        ggrs = q0 * (-q1_rs / ((q1 + 1) * q1)) + q2 * q0_rs
        return gg, ggrs

    def _scan_lda0(self, rs: Tensor) -> tuple[Tensor, Tensor]:
        safe_rs = torch.clamp(rs, min=density.EPS)
        sqrt_rs = torch.sqrt(safe_rs)

        b2c = 0.0889
        b3c = 0.125541
        elda0 = -self.b1c / (1 + b2c * sqrt_rs + b3c * safe_rs)
        dlda0_drs = (b3c + b2c / (2 * sqrt_rs)) * elda0.square() / self.b1c
        return elda0, dlda0_drs

    def _scan_lsda1(self, rs: Tensor, zeta: Tensor) -> tuple[Tensor, Tensor]:
        plus = 1 + zeta
        minus = 1 - zeta
        eu, deudrs = self._scan_grcor2(
            rs, 0.03109070, 0.213700, 7.59570, 3.58760, 1.63820, 0.492940
        )
        ep, depdrs = self._scan_grcor2(
            rs, 0.015545350, 0.205480, 14.11890, 6.19770, 3.36620, 0.625170
        )
        alfm, dalfmdrs = self._scan_grcor2(
            rs, 0.01688690, 0.111250, 10.3570, 3.62310, 0.880260, 0.496710
        )

        z3 = zeta.pow(3)
        z4 = zeta * z3
        f_zeta = (plus.pow(4 / 3) + minus.pow(4 / 3) - 2) / self.gam

        eclda1 = (
            eu * (1 - f_zeta * z4)
            + ep * f_zeta * z4
            - alfm * f_zeta * (1 - z4) / self.fzz
        )
        declda1_drs = (
            (1 - z4 * f_zeta) * deudrs
            + z4 * f_zeta * depdrs
            - ((1 - z4) * f_zeta * dalfmdrs / self.fzz)
        )
        return eclda1, declda1_drs

    def _scan_ec0(self, rs: Tensor, s: Tensor, zeta: Tensor) -> Tensor:
        plus = 1 + zeta
        minus = 1 - zeta
        eclda0, _ = self._scan_lda0(rs)
        dx_z = (plus.pow(4 / 3) + minus.pow(4 / 3)) / 2
        # 2.363 in reference implementation and libxc, 2.3631 in paper
        gc_z = (1 - 2.363 * (dx_z - 1)) * (1 - zeta.pow(12))
        w0 = self._scan_bounded_expm1(-eclda0 / self.b1c)
        ginf = (1 + 4 * self.chi_ld * s.square()).pow(-1 / 4)
        h0 = self.b1c * torch.log1p(w0 * (1 - ginf))
        return (eclda0 + h0) * gc_z

    def _scan_get_y(self, rs: Tensor, t: Tensor, w1: Tensor) -> Tensor:
        beta = self.beta_mb * (1 + self.afactor * rs) / (1 + self.bfactor * rs)
        return beta * t.square() / (self.gamma * torch.clamp(w1, min=density.EPS))

    def _scan_get_del_y(
        self,
        rs: Tensor,
        s: Tensor,
        zeta: Tensor,
        lda0: Tensor,
        dlda0_drs: Tensor,
        lsda1: Tensor,
        dlsda1_drs: Tensor,
        gc_z: Tensor,
        phi3: Tensor,
        w1: Tensor,
    ) -> Tensor:
        plus = 1 + zeta
        minus = 1 - zeta
        p = s.square()
        ds_z = (plus.pow(5 / 3) + minus.pow(5 / 3)) / 2
        lsda0 = lda0 * gc_z
        dlsda0_drs = dlda0_drs * gc_z
        t1 = self.c_del_f2 / (
            27 * self.gamma * ds_z * phi3 * torch.clamp(w1, min=density.EPS)
        )
        t2 = 20 * rs * (dlsda0_drs - dlsda1_drs)
        t3 = 45 * self.eta * (lsda0 - lsda1)
        k_factor = t1 * (t2 - t3)
        damp = self._scan_bounded_exp(-(p.square()) / self.d_damp2**4)
        return k_factor * p * damp

    def _scan_ec1(self, rs: Tensor, s: Tensor, zeta: Tensor) -> Tensor:
        plus = 1 + zeta
        minus = 1 - zeta
        dx_z = (plus.pow(4 / 3) + minus.pow(4 / 3)) / 2
        # 2.363 in libxc, 2.3631 in reference implementation and paper
        gc_z = (1 - 2.363 * (dx_z - 1)) * (1 - zeta.pow(12))
        phi = (plus.pow(2 / 3) + minus.pow(2 / 3)) / 2
        phi3 = phi.pow(3)
        lda0, dlda0_drs = self._scan_lda0(rs)
        lsda1, dlsda1_drs = self._scan_lsda1(rs, zeta)

        t = self.afix_t * s / (torch.sqrt(torch.clamp(rs, min=density.EPS)) * phi)
        w1 = self._scan_bounded_expm1(-lsda1 / (self.gamma * phi3))
        y = self._scan_get_y(rs, t, w1)

        if self.gradient_correction_mode == 0:
            del_y = torch.zeros_like(y)
        elif self.gradient_correction_mode in (1, 2):
            del_y = self._scan_get_del_y(
                rs,
                s,
                zeta,
                lda0,
                dlda0_drs,
                lsda1,
                dlsda1_drs,
                gc_z,
                phi3,
                w1,
            )
        else:
            raise ValueError(
                "Unsupported SCAN correlation gradient correction mode "
                f"{self.gradient_correction_mode}"
            )

        g_y = (1 + 4 * (y - del_y)).pow(-1 / 4)
        h1 = self.gamma * phi3 * torch.log1p(w1 * (1 - g_y))
        return lsda1 + h1

    def _scan_correlation_interpolation(self, alpha: Tensor) -> Tensor:
        scan_interp, tail_interp = self._scan_f_alpha_branches(
            alpha, self.cfc1, self.cfc2, self.cfdc1
        )
        if self.interpolation_mode == 0:
            return torch.where(alpha <= 1.0, scan_interp, tail_interp)
        if self.interpolation_mode == 1:
            poly_interp = self._scan_horner(
                alpha,
                torch.cat((self.ie_params_c, self.ie_params_c.new_ones(1))),
            )
            return torch.where(
                alpha <= 0.0,
                scan_interp,
                torch.where(alpha <= 2.5, poly_interp, tail_interp),
            )
        raise ValueError(
            f"Unsupported SCAN correlation interpolation mode {self.interpolation_mode}"
        )

    def _scan_correlation_per_particle(
        self,
        rho: Tensor,
        grad: Tensor,
        kin: Tensor,
    ) -> Tensor:
        safe_rho = torch.clamp(rho, min=0.0)
        total_density = safe_rho.sum(0)
        safe_total_density = torch.clamp(total_density, min=density.EPS)
        zeta = torch.clamp(
            (safe_rho[0] - safe_rho[1]) / safe_total_density,
            min=-self.zeta_limit,
            max=self.zeta_limit,
        )
        grad_total = grad.sum(0)
        grad_norm = density.grad_norm(grad_total)
        s = grad_norm / (
            2 * (3 * math.pi**2) ** (1 / 3) * safe_total_density.pow(4 / 3)
        )
        ds_z = ((1 + zeta).pow(5 / 3) + (1 - zeta).pow(5 / 3)) / 2
        kinetic_total = torch.clamp(kin, min=0.0).sum(0)

        if self.alpha_mode == 1:
            tueg = (self.tueg_con * safe_total_density.pow(5 / 3) + self.tau_r) * ds_z
        else:
            tueg = self.tueg_con * safe_total_density.pow(5 / 3) * ds_z

        tauw = grad_norm.square() / (8 * safe_total_density)
        if self.alpha_mode in (0, 1):
            alpha = (kinetic_total - tauw) / tueg
            if self.alpha_mode == 1:
                alpha = alpha.pow(3) / (alpha.square() + self.a_reg)
        elif self.alpha_mode == 2:
            alpha = (kinetic_total - tauw) / (tueg + self.eta * tauw)
        else:
            raise ValueError(
                f"Unsupported SCAN correlation alpha mode {self.alpha_mode}"
            )

        rs = self.rs_prefac / safe_total_density.pow(1 / 3)
        ec0 = self._scan_ec0(rs, s, zeta)
        ec1 = self._scan_ec1(rs, s, zeta)
        ief = self._scan_correlation_interpolation(alpha)
        energy = ec1 + ief * (ec0 - ec1)
        return torch.where(total_density > 0, energy, torch.zeros_like(energy))

    def exchange(self, mol_features: dict[str, Tensor]) -> Tensor:
        rho = torch.clamp(mol_features["density"], min=0.0)
        grad_norm = density.grad_norm(mol_features["grad"])
        kin = torch.clamp(mol_features["kin"], min=0.0)
        return self._scan_exchange_density(rho, grad_norm, kin)

    def correlation_density(self, mol_features: dict[str, Tensor]) -> Tensor:
        rho = torch.clamp(mol_features["density"], min=0.0)
        grad = mol_features["grad"]
        kin = torch.clamp(mol_features["kin"], min=0.0)
        return self._scan_correlation_per_particle(rho, grad, kin)


class SCAN(_SCANLikeFunctional):
    """
    Strongly Constrained and Appropriately Normed (SCAN) meta-GGA functional.

    SCAN is a meta-GGA that satisfies all 17 known exact constraints for
    semilocal functionals and is appropriately normed to the uniform
    electron gas and the hydrogen atom.
    """

    def __init__(self) -> None:
        super().__init__(
            alpha_mode=0,
            interpolation_mode=0,
            gradient_correction_mode=0,
        )


class RSCAN(_SCANLikeFunctional):
    """
    Regularized SCAN (rSCAN) meta-GGA functional.

    rSCAN is a regularized version of SCAN that improves numerical stability
    however breaking some of the exact constraints of the original SCAN functional.
    """

    def __init__(self) -> None:
        super().__init__(
            alpha_mode=1,
            interpolation_mode=1,
            gradient_correction_mode=0,
        )


class R2SCAN(_SCANLikeFunctional):
    """
    Regularized and restored SCAN (r2SCAN) meta-GGA functional.

    r2SCAN is a regularized version of SCAN that improves numerical stability
    while preserving the exact constraints and norms of the original SCAN functional.
    """

    def __init__(self) -> None:
        super().__init__(
            alpha_mode=2,
            interpolation_mode=1,
            gradient_correction_mode=1,
        )


XC_FUNCTIONAL_MAP: dict[str, type[ExcFunctionalBase]] = {
    "lda": LDA,
    "spw92": SPW92,
    "pbe": PBE,
    "tpss": TPSS,
    "scan": SCAN,
    "rscan": RSCAN,
    "r2scan": R2SCAN,
}


def get_traditional_functional(xc: str) -> type[ExcFunctionalBase]:
    """
    Get a traditional functional class by name.

    Parameters
    ----------
    xc : str
        Name of the functional ("lda", "spw92", "pbe", "tpss", "scan", "rscan", or "r2scan").

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

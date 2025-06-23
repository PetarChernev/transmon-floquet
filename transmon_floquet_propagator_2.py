#!/usr/bin/env python3
# floquet_multilevel_refactored.py  (July 2025)

from __future__ import annotations
import math, typing
from typing import Sequence, Union
import numpy as np
import torch
from torch import Tensor

from transmon_core import TransmonCore


# ============================================================================
#  Fourier coefficients  of Gaussian × cos  (period 2π)
# ============================================================================

def cos_gauss_coeffs(N, sigma, floquet_period, carrier_period, dtype, device):
    k = torch.arange(-N, N + 1, dtype=torch.float64, device=device)
    omega_floquet = 2 * math.pi / floquet_period
    omega_carrier = 2 * math.pi / carrier_period

    # Correct Fourier coefficients:
    # Integral: e^(-t²/(2σ²)) cos(ω_carrier t) exp(-i k ω_floquet t)
    # => Two Gaussians centered at ± ω_carrier
    arg1 = sigma * (k * omega_floquet - omega_carrier)
    arg2 = sigma * (k * omega_floquet + omega_carrier)

    g1 = torch.exp(-arg1**2 / 2)
    g2 = torch.exp(-arg2**2 / 2)

    scale = math.sqrt(2*math.pi)*sigma / floquet_period
    return scale * 0.5 * (g1 + g2).to(dtype)           # shape: (2 N+1,)



# ============================================================================
#  Single-pulse Floquet operator
# ============================================================================

class FloquetPulse:
    def __init__(
        self,
        *,
        levels: int = 6,
        n_side: int = 8,
        sigma_frac: float = 1 / 6,
        alpha: float = 0.043,
        drive_freq_ratio: float = 1.0,
        total_time: float = 1.0,
        subpulse_count: int = 8,
        cr_weight: float = 1.0,
        EJ_EC_ratio= 50.0,
        numeric_lambda: bool = False,
        device: str = "cpu",
        dtype=torch.complex128,
    ):
        self.L, self.N = levels, n_side
        self.alpha = alpha
        self.device, self.dtype = torch.device(device), dtype
        self.cr_weight = cr_weight

        # ---- time scaling ------------------------------------------------
        self.floquet_period = total_time / subpulse_count  # Floquet period (long repetition period)
        self.carrier_period = 2 * math.pi / drive_freq_ratio  # Short carrier oscillation period
        self.sigma = sigma_frac * self.floquet_period
        self.omega_d = drive_freq_ratio

        # ---- Duffing energies -------------------------------------------
        E = TransmonCore.duffing_energies(
            levels, alpha, dtype=dtype, device=self.device
        )

        dim = levels * (2 * n_side + 1)
        self._static = torch.zeros((dim, dim), dtype=dtype, device=self.device)
        n = torch.arange(levels, device=self.device, dtype=torch.float64)
        for m in range(-n_side, n_side + 1):
            block = (E - n) - m * self.omega_d
            i = (m + n_side) * levels
            self._static[i : i + levels, i : i + levels] = block

        # ---- drive templates --------------------------------------------
        if numeric_lambda:
            # pull numerical λ from charge-basis once
            _, lam_full = TransmonCore.compute_transmon_parameters(
                n_levels=levels, EJ_EC_ratio=EJ_EC_ratio
            )
            lam_full = torch.tensor(lam_full, dtype=dtype, device=self.device)
            X_drive = lam_full + lam_full.T.conj()
        else:
            X_drive = sum(TransmonCore.ladder(levels, dtype=dtype, device=self.device))

        fourier_coeff_limit = 2 * self.floquet_period / (2 * math.pi * self.omega_d)
        C = cos_gauss_coeffs(
            fourier_coeff_limit, self.sigma, self.floquet_period, self.carrier_period, dtype=dtype, device=self.device
        )
        drv_rwa = torch.zeros_like(self._static)
        drv_cr  = torch.zeros_like(self._static)

        for d, c in enumerate(C, start=-n_side):
            if c.abs() < 1e-16:
                continue
            tgt = drv_rwa if d == 0 else drv_cr
            for m in range(-n_side, n_side + 1):
                n_val = m - d
                if -n_side <= n_val <= n_side:
                    i = (m + n_side) * levels
                    j = (n_val + n_side) * levels
                    tgt[i : i + levels, j : j + levels] += 0.5 * c * X_drive
        self._drv_rwa, self._drv_cr = drv_rwa, drv_cr

    # ............ propagator for one Gaussian pulse .....................
    def unitary(self, omega_ratio: Union[float, Tensor], phi: Union[float, Tensor]):
        om = omega_ratio if torch.is_tensor(omega_ratio) else torch.tensor(
            float(omega_ratio), dtype=torch.float64, device=self.device
        )
        ph = phi if torch.is_tensor(phi) else torch.tensor(
            float(phi), dtype=torch.float64, device=self.device
        )
        H = self._static + om * (self._drv_rwa + self.cr_weight * self._drv_cr)
        U_big = torch.linalg.matrix_exp(-1j * H * self.floquet_period)

        col0 = slice(self.N * self.L, (self.N + 1) * self.L)
        U = torch.zeros((self.L, self.L), dtype=self.dtype, device=self.device)
        for m in range(-self.N, self.N + 1):
            row = slice((m + self.N) * self.L, (m + self.N + 1) * self.L)
            U += U_big[row, col0]

        if ph.item() != 0.0:
            R = torch.diag(torch.exp(-1j * ph * torch.arange(self.L, device=self.device))).to(self.dtype)
            U = R @ U @ R.conj().T
        return U


# ============================================================================
#  Composite sequence wrapper
# ============================================================================

class GaussianPulseSequence:
    def __init__(self, *, subpulse_count: int = 8, **pulse_kw):
        self.N = subpulse_count
        self.pulse = FloquetPulse(subpulse_count=subpulse_count, **pulse_kw)
        self.device, self.dtype = self.pulse.device, self.pulse.dtype
        self.L = self.pulse.L

    # -------- helpers / checks ------------------------------------------
    def _broadcast_amps(self, amps):
        if isinstance(amps, (int, float, Tensor)) and not (
            torch.is_tensor(amps) and amps.dim() > 0
        ):
            om = amps if torch.is_tensor(amps) else torch.tensor(
                float(amps), dtype=torch.float64, device=self.device
            )
            return [om] * self.N
        if len(amps) != self.N:
            raise ValueError(f"Need {self.N} amplitudes (got {len(amps)}).")
        return [
            a if torch.is_tensor(a) else torch.tensor(float(a), dtype=torch.float64, device=self.device)
            for a in amps
        ]

    def _check_phases(self, phases):
        if len(phases) != self.N:
            raise ValueError(f"Need {self.N} phases (got {len(phases)}).")
        return [
            p if torch.is_tensor(p) else torch.tensor(float(p), dtype=torch.float64, device=self.device)
            for p in phases
        ]

    # -------- full propagator -------------------------------------------
    def full_unitary(self, amps, phases):
        om_list = self._broadcast_amps(amps)
        ph_list = self._check_phases(phases) + [0.0]
        U = torch.eye(self.L, dtype=self.dtype, device=self.device)
        for om, ph in zip(om_list, ph_list):
            U = self.pulse.unitary(om, ph) @ U
        return U

    # -------- project fidelity ------------------------------------------
    def project_fidelity(self, U, target):
        if target.shape == (2, 2):
            P = torch.zeros((2, self.L), dtype=self.dtype, device=self.device)
            P[0, 0] = P[1, 1] = 1
            U = P @ U @ P.conj().T
        return float(torch.abs(torch.trace(target.conj().T @ U)) / 2.0)


# ============================================================================
#  Example run: 8 Gaussian slices in 20 ns  (paper Table I sequence)
# ============================================================================

if __name__ == "__main__":
    dev, dtype = "cuda", torch.complex128
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)

    seq = GaussianPulseSequence(
        subpulse_count=7,
        levels=6,
        n_side=14,
        sigma_frac=1 / 6,
        alpha=0.043,
        numeric_lambda=True,          # <- use exact λ from charge basis
        drive_freq_ratio=1.0,         # resonant
        total_time=880,
        cr_weight=0.0,                
        EJ_EC_ratio=EJ_EC_ratio,
        device=dev,
        dtype=dtype,
    )

    # Table I data (MHz)  ->  Ω/ω0  (ω0/2π = 7 GHz)
    amps  = np.array(
        [42.497, 69.996, 69.996, 69.761, 63.782, 69.996, 58.263]
    ) / 7000.0
    phases = np.array(
        [-0.3875, 0.0188, 0.0191, 0.1258, 0.2469, 0.3139, 0.2516]
    ) * math.pi

    U = seq.full_unitary(amps, phases)

    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=dev)

    F = seq.project_fidelity(U, X)
    print(f"Propagator:\n{U}")
    print(f"Projected fidelity to X: {F:.6f}")

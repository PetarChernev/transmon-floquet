#!/usr/bin/env python3
"""
floquet_multilevel_paper.py  (rev-1, June 2025)
===============================================
Multi-level Floquet simulator tailored to the model in
“Robust, fast and high-fidelity composite single-qubit gates for superconducting
transmon qubits” (Tonchev et al., 2025).

*  Duffing spectrum  E_n = n − (α/2) n(n−1),  α = |δ| / ω0.
*  Drive H_d(t) = Ω(t) cos(ω_d t) ( a + a† ),  with √n matrix elements or
   numeric λ_n obtained from the charge basis.
*  Floquet period T = 2π / ω_d  (dimension-less ω0 = 1).
*  Homotopy dial  cr_weight ∈ [0,1]  multiplies the |d|≥1 Fourier blocks.
*  Fidelity helper projects the full D×D unitary to the qubit sub-space when
   the target is 2×2.
"""

from __future__ import annotations
import math, cmath, pathlib
from typing import Sequence, List, Tuple, Union

import torch
from torch import Tensor

# ==============================================================
# Duffing ladder, numeric λ_n (optional)
# ==============================================================

def duffing_energies(levels: int, alpha: float, *, dtype, device):
    n = torch.arange(levels, dtype=torch.float64, device=device)        # real
    e = n - 0.5 * alpha * n * (n - 1)
    return torch.diag(e).to(dtype)                                      # cast once


def ladder(levels: int, *, dtype, device, numeric_lambda: bool = False):
    a_real = torch.zeros((levels, levels), dtype=torch.float64, device=device)
    for k in range(1, levels):
        a_real[k - 1, k] = math.sqrt(k)
    a = a_real.to(dtype)
    return a, a.conj().T

# ==============================================================
# Fourier coeffs of Gaussian × cos  (period 2π)
# ==============================================================

def cos_gauss_coeffs(N: int, sigma: float, *, dtype, device):
    k = torch.arange(-N, N + 1, dtype=torch.float64, device=device)
    g = torch.exp(-2 * (math.pi * sigma * k) ** 2)
    return 0.5 * (torch.roll(g, -1) + torch.roll(g, 1)).to(dtype)


# ==============================================================
# Single-pulse Floquet operator
# ==============================================================

class FloquetPulse:
    def __init__(
        self,
        *,
        levels: int = 6,
        n_side: int = 6,
        sigma: float = 1 / 6,
        alpha: float = 0.043,              # |δ| / ω0   (dimension-less)
        drive_freq_ratio: float = 1.0,     # ω_d / ω0
        cr_weight: float = 1.0,
        numeric_lambda: bool = False,
        device: str = "cpu",
        dtype=torch.complex128,
    ):
        self.L = levels
        self.N = n_side
        self.sigma = sigma
        self.alpha = alpha
        self.omega_d = drive_freq_ratio
        self.cr_weight = cr_weight

        self.device = torch.device(device)
        self.dtype = dtype

        self.period = 2 * math.pi / self.omega_d          # dimension-less
        self.dim = self.L * (2 * self.N + 1)

        # Static (diagonal) block -------------------------------
        E = duffing_energies(self.L, self.alpha, dtype=self.dtype, device=self.device)
        static = torch.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        for m in range(-self.N, self.N + 1):
            μ = E - m * self.omega_d * torch.eye(self.L, dtype=self.dtype, device=self.device)
            i = (m + self.N) * self.L
            static[i : i + self.L, i : i + self.L] = μ
        self._static = static

        # Drive templates --------------------------------------
        a, adag = ladder(self.L, dtype=self.dtype, device=self.device, numeric_lambda=numeric_lambda)
        X_drive = a + adag
        coeffs = cos_gauss_coeffs(self.N, self.sigma, dtype=self.dtype, device=self.device)

        drv_rwa = torch.zeros_like(static)
        drv_cr  = torch.zeros_like(static)
        for d, c in enumerate(coeffs, start=-self.N):
            if c.abs() < 1e-16:
                continue
            target = drv_rwa if d == 0 else drv_cr
            for m in range(-self.N, self.N + 1):
                n_val = m - d
                if -self.N <= n_val <= self.N:
                    i = (m + self.N) * self.L
                    j = (n_val + self.N) * self.L
                    target[i:i+self.L, j:j+self.L] += 0.5 * c * X_drive
        self._drv_rwa, self._drv_cr = drv_rwa, drv_cr

    # .........................................................
    def unitary(self, omega_ratio: Union[float, Tensor], phi: Union[float, Tensor]) -> Tensor:
        om = omega_ratio if torch.is_tensor(omega_ratio) else torch.tensor(
            float(omega_ratio), dtype=torch.float64, device=self.device)
        ph = phi if torch.is_tensor(phi) else torch.tensor(
            float(phi), dtype=torch.float64, device=self.device)

        HF = self._static + om * (self._drv_rwa + self.cr_weight * self._drv_cr)
        U_big = torch.matrix_exp(-1j * HF * self.period)

        # project m=0 column, sum all rows
        col0 = slice(self.N * self.L, (self.N + 1) * self.L)
        U_phys = torch.zeros((self.L, self.L), dtype=self.dtype, device=self.device)
        for m in range(-self.N, self.N + 1):
            row = slice((m + self.N) * self.L, (m + self.N + 1) * self.L)
            U_phys += U_big[row, col0]

        if ph.item() != 0.0:
            phase_vec = torch.exp(-1j * ph * torch.arange(self.L, device=self.device))
            Rphi = torch.diag(phase_vec).to(self.dtype)
            U_phys = Rphi @ U_phys @ Rphi.conj().T
        return U_phys


# ==============================================================
# Composite-sequence wrapper
# ==============================================================

class GaussianPulseSequence:
    def __init__(self, **pulse_kwargs):
        self.pulse = FloquetPulse(**pulse_kwargs)
        self.device = self.pulse.device
        self.dtype = self.pulse.dtype
        self.L = self.pulse.L

    # helper to repeat / broadcast Ω
    def _broadcast_omegas(self, omegas, n_phi):
        if isinstance(omegas, (int, float)) or (
            torch.is_tensor(omegas) and omegas.dim() == 0
        ):
            om = omegas if torch.is_tensor(omegas) else torch.tensor(
                float(omegas), dtype=torch.float64, device=self.device)
            return [om] * (n_phi + 1)
        if len(omegas) not in {n_phi, n_phi + 1}:
            raise ValueError("Amp list length mismatch")
        out = [
            o if torch.is_tensor(o) else torch.tensor(float(o), dtype=torch.float64, device=self.device)
            for o in omegas
        ]
        if len(out) == n_phi:
            out.append(out[-1].clone())
        return out

    # full D×D propagator
    def full_unitary(self, omegas, phis):
        U = torch.eye(self.L, dtype=self.dtype, device=self.device)
        om_list = self._broadcast_omegas(omegas, len(phis))
        for om, ph in zip(om_list[:-1], phis):
            U = self.pulse.unitary(om, ph) @ U
        U = self.pulse.unitary(om_list[-1], 0.0) @ U
        return U

    # fidelity (projects if target is 2×2)
    def fidelity_and_derivs(
        self,
        phis: Sequence[Union[float, Tensor]],
        amp: Union[Sequence[Union[float, Tensor]], float, Tensor],
        *,
        target: Tensor,
        max_deriv: int = 0,
        area_scales: Sequence[float] | None = None,
    ):
        vec_amp = not (isinstance(amp, (int, float)) or (
            torch.is_tensor(amp) and amp.dim() == 0))
        if max_deriv > 0 and vec_amp:
            raise NotImplementedError

        area_scales = area_scales or [1.0]
        phi_t = [p if torch.is_tensor(p) else torch.tensor(
            float(p), dtype=torch.float64, device=self.device) for p in phis]

        if vec_amp:
            amp_list = self._broadcast_omegas(amp, len(phis))
        else:
            amp_base = (
                amp if torch.is_tensor(amp) else torch.tensor(
                    float(amp), dtype=torch.float64, device=self.device,
                    requires_grad=max_deriv > 0)
            )

        # projector onto |0>,|1>
        P01 = torch.zeros((2, self.L), dtype=self.dtype, device=self.device); P01[0,0]=P01[1,1]=1

        fids, derivs = [], []
        for s in area_scales:
            if vec_amp:
                U = self.full_unitary([a*s for a in amp_list], phi_t)
            else:
                U = self.full_unitary(amp_base * s, phi_t)

            # project if needed
            U_use = U
            if target.shape == (2,2):
                U_use = P01 @ U @ P01.conj().T

            F = torch.abs(torch.trace(target.conj().T @ U_use)) / 2.0
            fids.append(float(F))

            if max_deriv and not vec_amp:
                cur, d_here = F, []
                for _ in range(max_deriv):
                    (grad,) = torch.autograd.grad(cur, amp_base, retain_graph=True, create_graph=True)
                    d_here.append(float(grad)); cur = grad
                derivs.append(d_here)
        return fids, derivs


# ==============================================================
# Sanity check  (20 ns, N=10, α=0.043)
# ==============================================================

if __name__ == "__main__":
    dev, dtype = "cuda", torch.complex128

    levels, n_side, sigma = 6, 12, 1/6
    alpha = 0.043
    drive_ratio = 1.0        # ω_d = ω_0 (resonant)
    seq = GaussianPulseSequence(levels=levels, n_side=n_side, sigma=sigma,
                                alpha=alpha, drive_freq_ratio=drive_ratio,
                                cr_weight=0, device=dev, dtype=dtype)

    # π-area for Gaussian
    omega_pi = 2 * math.pi / (sigma * math.sqrt(math.pi/2))   # ~30.09
    # full 6×6 unitary for single Gaussian, then project
    U_full = seq.full_unitary(omega_pi, [])
    P = torch.zeros((2,levels), dtype=dtype, device=dev); P[0,0]=P[1,1]=1
    Uq = P @ U_full @ P.conj().T

    X = torch.tensor([[0,1],[1,0]], dtype=dtype, device=dev)
    fid = torch.abs(torch.trace(X.conj().T @ Uq)) / 2
    print("Projected fidelity of π-Gaussian (RWA) to X:", fid.item())

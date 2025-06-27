#!/usr/bin/env python3
"""
CMA-ES search for pulse parameters that maximise (unitarity + fidelity)
of the stroboscopic Floquet propagator for the first two levels.

Author: <you>
Date  : 2025-06-26
"""

from __future__ import annotations
from dataclasses import dataclass
import torch.multiprocessing as mp
from typing import Optional, Tuple, Sequence, List

import numpy as np
import torch
import cma

from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence_stroboscopic 

DURATION_PENALTY_WEIGHT = 1e3         # tune: big enough to dominate the loss

device        = torch.device("cuda")
dtype_real = torch.float32
dtype_complex = torch.complex64

n_levels = 2
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, lambdas_full = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

omega_d       = 1.0                                 # driving frequency (rad s⁻¹)
floquet_cutoff: int = 25

U_TARGET = torch.tensor([[0, 1], [1, 0]], dtype=dtype_complex, device=device)    # example: identity in the {|0>,|1>} sub-space

max_periods = 500          # 20ns
energies = torch.tensor(energies, dtype=dtype_real, device=device)
lambdas_full = torch.tensor(lambdas_full, dtype=dtype_complex, device=device)
robustness_detunings = [0.98, 0.99, 1.0, 1.01, 1.02]
detuning_weights = [0.5, 0.75, 10.0, 0.75, 0.5]
loss_offset = -sum(detuning_weights) * (1 + 2 * (n_levels - 2))
loss_tolerance = 1e-2
# ────────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def unitarity_score(U2: torch.Tensor) -> float:
    """
    1 ↔ perfect unitarity, 0 ↔ totally non-unitary.
    Metric: 1 − ||U†U − I||₁ / (2×2) .
    """
    diff = U2.conj().T @ U2 - torch.eye(2, dtype=U2.dtype, device=device)
    err  = torch.linalg.norm(diff, ord='fro').real      # Frobenius
    return float(max(0.0, 1.0 - err / 2.0))


def fidelity_score(U2: torch.Tensor, target: torch.Tensor = U_TARGET) -> float:
    """
    Standard trace fidelity F = |Tr(U† U_target)| / 2 .
    1 = perfect.
    """
    F = torch.trace(target.conj().T @ U2)
    return float(abs(F).real / 2.0)


# ────────────────────────────────────────────────────────────────────────────────
#  PARAMETER EN/DECODERS
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class PulseEncoding:
    """Maps between flat CMA vectors and (complex pulses, int durations)."""
    n_pulses: int
    rabi_bounds     : Tuple[float, float]  = (0.0, 1.0)      # in whichever units
    phase_bounds    : Tuple[float, float]  = (0.0, 2 * np.pi)
    duration_bounds : Tuple[int,   int]    = (0, 100)         # in drive periods

    def encode(self,
               pulses : np.ndarray,          # shape (n_pulses,) complex
               periods: np.ndarray           # shape (n_pulses,) int
               ) -> np.ndarray:
        """complex→flat real vector for CMA-ES (2n + n ints)."""
        return np.concatenate([pulses.real, pulses.imag, periods.astype(float)])

    def decode(self,
               genome: Sequence[float]
               ) -> Tuple[np.ndarray, np.ndarray]:
        """CMA genome→(complex pulses, integer periods)."""
        g = np.asarray(genome, dtype=float)
        if g.size != 3 * self.n_pulses:
            raise ValueError("Genome length mismatch")

        rab  = np.clip(g[:self.n_pulses],  *self.rabi_bounds)
        ph   = np.mod(g[self.n_pulses:2*self.n_pulses],
                      self.phase_bounds[1])          # wrap in [0,2π)

        pulses   = rab + 1j * ph
        periods  = np.rint(                          # force integer
                    np.clip(g[2*self.n_pulses:], *self.duration_bounds)
                  ).astype(int)

        return pulses, periods
    

def _loss_worker_func(detuning, detuning_weight, rabi_t, phase_t, dur_t, floquet_args, device):
    # shift the omega_d, which is the 3rd arg - not very clean, could use a dict ot explicit args
    floquet_args_detuned = [arg if j != 2 else arg * detuning for j, arg in enumerate(floquet_args)]
    U = floquet_propagator_square_sequence_stroboscopic(
            rabi_t, phase_t, dur_t, *floquet_args_detuned, device=device
    )   
    loss = detuning_weight * fidelity_score(U[:2, :2])
    if U.size(0) > 2:
        higher_state_leakage = torch.sum(torch.abs(U[2:, :2]).sum()).cpu().detach()
        loss -= detuning_weight * higher_state_leakage
    return loss


# ────────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FOR CMA-ES
# ────────────────────────────────────────────────────────────────────────────────
def make_loss(
    enc: PulseEncoding,
    floquet_args: tuple
):
    """Return function f(x) to be minimised by CMA-ES."""
    def _loss(genome: Sequence[float]) -> float:
        pulses_c, periods_i = enc.decode(genome)

        
        # to torch
        rabi_t  = torch.tensor(pulses_c.real, dtype=dtype_real, device=device)
        phase_t = torch.tensor(pulses_c.imag, dtype=dtype_real, device=device)
        dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device)
        
        tasks = [(detuning, detuning_weights[i], rabi_t, phase_t, dur_t, floquet_args, device) for i, detuning in enumerate(detuning_weights)]
        losses = pool.starmap(_loss_worker_func, tasks)
        loss = sum(losses)
        loss += max(0, sum(periods_i) - max_periods)
        return loss + loss_offset
    return _loss


# ────────────────────────────────────────────────────────────────────────────────
#  RUN OPTIMISATION
# ────────────────────────────────────────────────────────────────────────────────
def optimise_pulses(
    n_pulses       : int,
    sigma0         : float = 0.3,               # initial step size
    seed           : Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    enc = PulseEncoding(n_pulses)

    # Initial centre: mid-range rabies, random phases, mid-range periods
    rab0  = np.full(n_pulses, sum(enc.rabi_bounds) / 2.0)
    ph0   = np.random.default_rng(seed).uniform(*enc.phase_bounds, size=n_pulses)
    per0  = np.full(n_pulses, round(sum(enc.duration_bounds) / 2))

    x0    = enc.encode(rab0 + 1j*ph0, per0)

    # CMA options
    int_indices = list(range(2*n_pulses, 3*n_pulses))        # durations
    cma_opts    = {
        "integer_variables": int_indices,
        "tolx":   1e-10,
        "verb_time": 0,
        "seed": seed,
        "ftarget": loss_tolerance
    }

    # Build loss closure with fixed system tensors
    loss = make_loss(enc,
        floquet_args=(energies, lambdas_full, omega_d, floquet_cutoff)
    )

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
    es.optimize(loss)

    best_genome, best_f = es.result.xbest, es.result.fbest
    pulses_opt, periods_opt = enc.decode(best_genome)

    print(f"\n⇨  Finished after {es.result.evaluations} evaluations")
    print(f"⇨  Best (-unitarity - fidelity)  = {best_f:+.3e}")
    print(f"⇨  Expected unitary + fidelity   = {-best_f:+.6f}\n")

    return pulses_opt, periods_opt


# ────────────────────────────────────────────────────────────────────────────────
#  MAIN  (example)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method('spawn')
    pool = mp.Pool(len(detuning_weights))

    N_PULSES = 8
    pulses, periods = optimise_pulses(N_PULSES, seed=42)

    print("Optimal Rabi frequencies [Hz] and phases [rad]:")
    for k, p in enumerate(pulses):
        print(f"  pulse {k:2d}:  Ω = {p.real:+.6f},  φ = {p.imag:+.6f}")

    print("\nOptimal integer pulse durations (drive periods):")
    print(" ", periods)

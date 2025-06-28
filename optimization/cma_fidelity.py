#!/usr/bin/env python3
"""
CMA-ES search for pulse parameters that maximise (unitarity + fidelity)
of the stroboscopic Floquet propagator for the first two levels.

Author: <you>
Date  : 2025-06-26
"""

from __future__ import annotations
from copy import copy
from dataclasses import dataclass
import torch.multiprocessing as mp
from typing import Optional, Tuple, Sequence, List
import torch.multiprocessing as mp
from cma import CMAEvolutionStrategy

import numpy as np
import torch
import cma
from tqdm import tqdm

from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence_stroboscopic 


# technical params
device        = torch.device("cuda")
dtype_real = torch.float32
dtype_complex = torch.complex64

# target unitary on first 2 levels
U_TARGET = torch.tensor([[0, 1], [1, 0]], dtype=dtype_complex, device=device)  
U_TARGEET_DAGGER = U_TARGET.conj().T  
# system params
n_pulses = 9
n_levels = 2
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)
energies = torch.tensor(energies, dtype=dtype_real, device=device)
couplings = torch.tensor(couplings, dtype=dtype_complex, device=device)

omega_d = 1.0                                
floquet_cutoff: int = 25

# loss params
robustness_detunings = [0.98, 0.99, 1.0, 1.01, 1.02]
detuning_weights = [0.5, 0.75, 10.0, 0.75, 0.5]
loss_target = 1e-3
duration_penality = 1e3        
max_rabi = 3.0
max_pulse_periods = 100
max_total_periods = 2000         

propagator_static_args = dict(
    energies=energies,
    couplings=couplings,
    omega_d=omega_d,
    floquet_cutoff=floquet_cutoff
)


def detune(args, detuning):
    args = copy(args)
    args['omega_d'] = args['omega_d'] * detuning
    return args


# ────────────────────────────────────────────────────────────────────────────────
#  PARAMETER EN/DECODERS
# ────────────────────────────────────────────────────────────────────────────────
class PulseEncoding:
    rabi_bounds     : Tuple[float, float]  = (0.0, max_rabi)      # in whichever units
    phase_bounds    : Tuple[float, float]  = (0.0, 2 * np.pi)
       
    @staticmethod
    def encode(
        rabis : np.ndarray,         # shape (n_pulses,) float
        phases: np.ndarray,         # shape (n_pulses,) float
        periods: np.ndarray         # shape (n_pulses,) float
    ) -> np.ndarray:
        return np.concatenate([rabis, phases, periods])

    @staticmethod
    def decode(genome: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pulses = len(genome) // 3
        g = np.asarray(genome, dtype=float)
        if g.size != 3 * n_pulses:
            raise ValueError("Genome length mismatch")

        rab  = np.clip(g[:n_pulses], *PulseEncoding.rabi_bounds)
        ph   = np.mod(g[n_pulses:2*n_pulses], PulseEncoding.phase_bounds[1])

        periods  = (g[2*n_pulses:3*n_pulses] * max_pulse_periods).astype(int)

        return rab, ph, periods

# ────────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FOR CMA-ES
# ────────────────────────────────────────────────────────────────────────────────

def cma_objective_precise(genome: Sequence) -> float:
    rabi, phase, periods_i = PulseEncoding.decode(genome)
    # to torch
    rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
    phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
    dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device) 
    U = floquet_propagator_square_sequence_stroboscopic(
        rabi_t, phase_t, dur_t, **propagator_static_args, device=device
    )
    loss = _propagator_loss(U)
    # add penalty for going over max duration
    loss += max(0, sum(periods_i) - max_total_periods)
    return loss


def cma_objective_robust(genome: Sequence) -> float:
    rabi, phase, periods_i = PulseEncoding.decode(genome)
    # to torch
    rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
    phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
    dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device)
    
    tasks = [(detuning, detuning_weights[i], rabi_t, phase_t, dur_t, propagator_static_args, device) for i, detuning in enumerate(detuning_weights)]
    losses = pool.starmap(_loss_worker_func, tasks)
    loss = sum(losses)
    # add penalty for going over max duration
    loss += max(0, sum(periods_i) - max_total_periods) * len(detuning_weights)
    return loss

def _loss_worker_func(detuning, detuning_weight, rabi_t, phase_t, dur_t, floquet_args, device):
    floquet_args_detuned = detune(floquet_args, detuning)
    U = floquet_propagator_square_sequence_stroboscopic(
            rabi_t, phase_t, dur_t, *floquet_args_detuned, device=device
    )   
    loss = detuning_weight * _propagator_loss(U)
    return loss


def _propagator_loss(U):
    # project onto the first two levels
    M = U_TARGEET_DAGGER.conj().T @ U[:2, :2]

    tr_MMdag = torch.trace(M @ M.conj().T)
    tr_M     = torch.trace(M)
    fidelity = (tr_MMdag + torch.abs(tr_M) ** 2) / 6.0  

    return fidelity.real.item()


# ────────────────────────────────────────────────────────────────────────────────
#  RUN OPTIMISATION
# ────────────────────────────────────────────────────────────────────────────────
def optimise_pulses(
    n_pulses,
    loss_target
) -> Tuple[np.ndarray, List[float]]:
    n_cand = 36
    tqdm.write(f"Running CMA-ES to generate {n_cand} candidates...")
    cma_sols, attempts = [], 0
    
    with mp.Pool(12) as pool:
        while len(cma_sols) < n_cand and attempts < 3*n_cand:
            attempts +=1
            p0 = np.concatenate([
                np.random.random(n_pulses),
                np.random.random(n_pulses) * 2 * np.pi,
                np.random.random(n_pulses)
            ])
            es = CMAEvolutionStrategy(
                p0,
                1,
                {
                    'popsize':50,
                    'maxiter':1000,
                    'tolfun':1e-10,
                }
            )
            for _ in range(500):
                print(f"  CMA iteration {es.countiter} (attempt {attempts})")
                sols = es.ask(); 
                losses = pool.map(cma_objective_precise, sols)
                print(f"  Evaluated {len(sols)} candidates, min loss: {min(losses):.6f}")
                es.tell(sols, losses)
                if min(losses) < loss_target: break
            opt = es.result.xbest; 
            F_c = es.result.fbest
            tqdm.write(f"  Candidate {len(cma_sols)+1}: F={F_c:.6f}")
            cma_sols.append((len(cma_sols)+1, opt))
        if not cma_sols:
            tqdm.write("No CMA candidates found."); return None


# ────────────────────────────────────────────────────────────────────────────────
#  MAIN  (example)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method('spawn')

    result = optimise_pulses(n_pulses, loss_target=loss_target)
    if result:
        rabi, phases, periods = result
        print("Optimal Rabi frequencies:")
        print(list(rabi))
        print("Optimal phases:")
        print(list(phases))
        print("Optimal phases:")
        print(list(periods))
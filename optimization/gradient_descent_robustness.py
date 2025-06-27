# Robust pulse optimisation with analytic omega derivatives

from cmath import sqrt
import torch
from torch import einsum
from typing import List, Optional

from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import (
    floquet_propagator_square_sequence_stroboscopic,
)

##############################################################################
# fixed chip parameters (define once)
##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_levels = 6
# find EJ/EC ratio that gives target anharmonicity
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, lambdas_full = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

energies = torch.tensor(energies, dtype=torch.float64, device=device)
lambdas_full = torch.tensor(lambdas_full, dtype=torch.float64, device=device)

omega_d = 1.0  # nominal drive frequency, rad s^-1
floquet_cutoff: int = 25

U_TARGET = torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble, device=device) / sqrt(2)

##############################################################################
# learnable pulse parameters
##############################################################################

pulse_durations_periods = torch.tensor([7, 1, 7, 1, 7, 1, 7, 1,], dtype=torch.int64)
n_pulses = pulse_durations_periods.numel()

rabi = torch.nn.Parameter(
    torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64, device=device)
)

phases = torch.nn.Parameter(
    torch.tensor(
        [
            1.510568,
            4.856720,
            3.922234,
            1.510568,
            4.856720,
            3.922234,
            1.510568,
            4.856720,           
        ],
        dtype=torch.float64,
        device=device,
    )
)

TWO_PI = 2.0 * torch.pi


##############################################################################
# helper functions
##############################################################################

def fidelity(U: torch.Tensor, U_target: torch.Tensor):
    """Gate fidelity |Tr(U^dagger U_target)|^2 / d^2."""
    d = U.shape[0]
    fid = torch.abs(torch.trace(U.conj().T @ U_target)) / U.shape[0]
    return fid.real


def U_pulse_sequence(rabi, phases, omega):
    return floquet_propagator_square_sequence_stroboscopic(
        rabi,
        phases,
        pulse_durations_periods,
        energies,
        lambdas_full,
        omega,
        floquet_cutoff,
    )

##############################################################################
# loss with analytic omega derivatives
##############################################################################

omegas_nom = torch.tensor([0.98, 0.99, 1.00, 1.01, 1.02], dtype=torch.float64, device=device)
weights = torch.tensor([10, 100, 10000.0, 100, 10], dtype=torch.float64, device=device)
deriv_weights = torch.tensor([10000, 1], dtype=torch.float64, device=device)
n_derivs = 0


def one_point(rabi: torch.Tensor, phases: torch.Tensor, omega_scalar: torch.Tensor):
    """Return F, |dF/dw|, |d2F/dw2| for one detuning."""
    w = omega_scalar.clone().requires_grad_(True)
    U = U_pulse_sequence(rabi, phases, w)
    F = fidelity(U[:2, :2], U_TARGET).real
    higher_state_losses = torch.sum(torch.abs(U[2:, :2]))
    result = [higher_state_losses, (1 - F)**2]

    if n_derivs >= 1:
        # first derivative dF/dw
        (dF_dw,) = torch.autograd.grad(F, w, create_graph=True)
        result.append(dF_dw)

    if n_derivs >= 2:
        # second derivative d2F/dw2
        (d2F_dw2,) = torch.autograd.grad(dF_dw, w, create_graph=True)
        result.append(d2F_dw2)    

    return result

def loss_fn():
    total = 0.0
    results = []

    for idx, (w_i, wgt) in enumerate(zip(omegas_nom, weights)):
        result = one_point(rabi, phases, w_i)
        results.append([r.item() for r in result])
        total += result[0]
        for i in range(n_derivs + 1):
            total += wgt * deriv_weights[i] * result[i + 1]
    return total, results

##############################################################################
# optimisation loop
##############################################################################

opt = torch.optim.Adam([rabi, phases], lr=1.0e-2)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=25, cooldown=10
)

best_loss = float("inf")
best_rabi = rabi.detach().clone()
best_phases = phases.detach().clone()
step = 0

try:
    while True:
        step += 1
        opt.zero_grad()
        loss, fid_now = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([rabi, phases], 5.0)
        opt.step()
        sched.step(loss)

        # update best if improved
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_rabi = rabi.detach().clone()
            best_phases = phases.detach().clone()

        # progress printout
        print(f"{step:5d}  loss={loss.item():.6f}")
        print(f"infidelities: {', '.join(f'{r[1]:3e}' for r in fid_now)}")
        print(f"higher state losses: {', '.join(f'{r[0]:.3e}' for r in fid_now)}")

except KeyboardInterrupt:
    print(f"\nBest loss achieved: {best_loss:.6f}")
    print("Optimised Rabi amplitudes :", best_rabi.cpu().numpy())
    print("Optimised phases (rad)    :", best_phases.cpu().numpy())


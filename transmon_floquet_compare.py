"""
Compares the propoagator computed from the Floquet Hamiltonian
with the unitary estimated from a simulation of a transmon.

This is for a single period of constant Rabi cosine drive.
"""

import math
import numpy as np
import torch

from transmon_core import TransmonCore
from transmon_dynamics import simulate_transmon_dynamics

from transmon_floquet_propagator import compute_fourier_coeffs, floquet_hamiltonian_const_rabi_period


def estimate_unitary(simulate, dim):
    """
    Reconstructs the unitary (up to a global phase) implemented by
    the black-box `simulate(initial_state)`.

    Assumes:
      * `simulate` takes a size-d 1-D complex array (|ψ_in⟩)
      * returns the size-d 1-D complex array (|ψ_out⟩)
      * Dynamics is closed and therefore unitary.
    """
    # Work out Hilbert-space dimension from a single call
    U_est = np.zeros((dim, dim), dtype=complex)

    # Standard basis {|0⟩, |1⟩, …}
    for j in range(dim):
        e_j = np.zeros(dim, dtype=complex)
        e_j[j] = 1.0
        ψ_out = simulate(e_j)
        U_est[:, j] = ψ_out      # each output column is U|j⟩

    # Remove a global phase (optional but usually convenient)
    phase = np.exp(-1j * np.angle(U_est[0, 0]))
    return phase * U_est


def estimate_transmon_unitary(*args, **kwargs):
    def _f(initial_state):
        final_state, _, _ = simulate_transmon_dynamics(
            initial_state,
            *args,
            **kwargs
        )
        return final_state
    return estimate_unitary(_f, n_levels)


def unitary_fidelity(U_1, U_2):
    if torch.is_tensor(U_1):
        U_1 = U_1.detach().cpu().numpy()
    if torch.is_tensor(U_2):
        U_2 = U_2.detach().cpu().numpy()
    d = U_1.shape[0]
    return abs(np.trace(U_2.conj().T @ U_1)) / d

if __name__ == "__main__":
    dev, dtype = "cuda", torch.complex128
    n_levels = 6

    # From Table I - complete population transfer
    rabi_frequencies = np.array([1])

    phases = np.array([1.5]) * np.pi

    total_time = 2 * np.pi 
    delta = -0.0429
    
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(delta)
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    unitary_est = estimate_transmon_unitary(
        rabi_frequencies,
        phases,
        energies,
        lambdas_full,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=50000,
        use_rwa=False,
    )
    
    
    rabi_frequencies = torch.tensor(rabi_frequencies, dtype=torch.float64, requires_grad=True)
    phases = torch.tensor(phases, dtype=torch.float64, requires_grad=True)
    
    
    energies = torch.tensor(energies, dtype=torch.float64)
    lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128)   
    # Compute Fourier coefficients
    fourier_coeffs = compute_fourier_coeffs(rabi_frequencies[0], phases[0], lambdas_full, 100)


    # Compute Floquet propagator for one periodz
    U = floquet_hamiltonian_const_rabi_period(fourier_coeffs, energies, 1, 100)
    
    print(f"Fidelity: {unitary_fidelity(unitary_est, U)}")
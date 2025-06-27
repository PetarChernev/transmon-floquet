"""
Tests the floquet theory proagator for non stroboscopic times.
Currently not implemented.
"""
import torch
import numpy as np

from transmon.transmon_dynamics_qutip import pulse_sequence_qutip
from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence, floquet_propagator_square_sequence_stroboscopic


test_cases = [
    {
        "desc": "Two-level system, weak drive",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [0.01],
        "phases": [0.0],
        "pulse_durations": [10.0],       # 10 s
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Two-level system, strong drive",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [10.0],
        "phases": [0.0],
        "pulse_durations": [10.0],
        "omega_d": 1.5,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Three-level system, detuned drive",
        "energies": [0.0, 1.0, 2.1],
        "lambdas_full": [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        "rabi_frequencies": [0.5, 0.2],
        "phases": [0.0, np.pi/2],
        "pulse_durations": [5.0, 5.0],
        "omega_d": 1.5,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Three-level system, very short pulse",
        "energies": [0.0, 1.0, 2.0],
        "lambdas_full": [
            [0.0, 0.8, 0.0],
            [0.8, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        "rabi_frequencies": [0.3],
        "phases": [np.pi],
        "pulse_durations": [0.1],        # much shorter than drive period (2π)
        "omega_d": 2.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Six-level ladder, multi-pulse",
        "energies": [0., 1., 2., 3., 4., 5.],
        "lambdas_full": [
            [0., 1., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 0.],
            [0., 1., 0., 1., 0., 0.],
            [0., 0., 1., 0., 1., 0.],
            [0., 0., 0., 1., 0., 1.],
            [0., 0., 0., 0., 1., 0.],
        ],
        "rabi_frequencies": [0.05, 0.10, 0.05],
        "phases": [0.0, np.pi/2, np.pi],
        "pulse_durations": [4.0, 5.0, 6.0],
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Three-level degenerate energies",
        "energies": [0.0, 0.0, 0.0],
        "lambdas_full": [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        "rabi_frequencies": [0.20],
        "phases": [0.0],
        "pulse_durations": [8.0],
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Four-level system, non-uniform durations",
        "energies": [0.0, 1.0, 1.8, 3.0],
        "lambdas_full": [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.9, 0.0],
            [0.0, 0.9, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ],
        "rabi_frequencies": [0.30, 0.60, 0.30],
        "phases": [0.0, np.pi/4, np.pi/2],
        "pulse_durations": [2.0, 7.0, 3.0],
        "omega_d": 1.3,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Zero-duration pulse (identity check)",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [0.5],
        "phases": [0.0],
        "pulse_durations": [0.0],        # should produce identity
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Uncoupled three-level system",
        "energies": [0.0, 1.0, 2.0],
        "lambdas_full": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "rabi_frequencies": [0.2, 0.2],
        "phases": [0.0, 0.0],
        "pulse_durations": [3.0, 3.0],   # no coupling → diagonal evolution
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    {
        "desc": "Very long pulse (aliasing edge)",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [0.1],
        "phases": [0.0],
        "pulse_durations": [100.0],      # long compared to cutoff
        "omega_d": 1.0,
        "floquet_cutoff": 50,
    },
    
]

def compare_propagators(
    rabi_frequencies,
    phases,
    pulse_durations,
    energies,
    lambdas_full,
    *,
    omega_d=1.0,
    floquet_cutoff=50,
):
    # Set device and precision
    device = torch.device('cuda')
    dtype = torch.float64

    # Convert inputs to double precision CUDA tensors
    rabi_frequencies = torch.tensor(rabi_frequencies, dtype=dtype, device=device)
    phases = torch.tensor(phases, dtype=dtype, device=device)
    energies = torch.tensor(energies, dtype=dtype, device=device)
    lambdas_full = torch.tensor(lambdas_full, dtype=dtype, device=device)
    omega_d = torch.tensor(omega_d, dtype=dtype, device=device)

    if all(isinstance(t, int) for t in pulse_durations):
        # Compute Floquet propagator on CUDA
        U_floquet = floquet_propagator_square_sequence_stroboscopic(
            rabi_frequencies,
            phases,
            pulse_durations,
            energies,
            lambdas_full,
            omega_d,
            floquet_cutoff
        )
    else:
        U_floquet = floquet_propagator_square_sequence(
            rabi_frequencies,
            phases,
            pulse_durations,
            energies,
            lambdas_full,
            omega_d,
            floquet_cutoff
        )

    # Compute numerical propagator with QuTiP
    U_numerical = pulse_sequence_qutip(
        rabi_frequencies.cpu().numpy(),
        phases.cpu().numpy(),
        pulse_durations,
        energies.cpu().numpy(),
        lambdas_full.cpu().numpy(),
        omega_d=omega_d.cpu().numpy(),
        options={
            "atol": 1e-12,
            "rtol": 1e-12,
            "nsteps": 20000
        }
    )

    # Move numerical result to CUDA double tensor
    U_numerical = torch.from_numpy(U_numerical.full()).to(dtype=torch.complex128, device=device)

    U_floquet_aligned = U_floquet

    # Compute differences and fidelity
    diff = U_numerical - U_floquet_aligned
    diff_norm = torch.linalg.norm(diff)
    fidelity = torch.abs(torch.trace(U_numerical.conj().T @ U_floquet_aligned)) / U_numerical.shape[0]

    return {
        "diff_norm": diff_norm.item(),
        "fidelity": fidelity.item(),
        "U_floquet": U_floquet_aligned,
        "U_numerical": U_numerical,
    }

def test_propagator_accuracy(case):
    # Run comparison
    result = compare_propagators(
        rabi_frequencies=case['rabi_frequencies'],
        phases=case['phases'],
        pulse_durations=case['pulse_durations'],
        energies=case['energies'],
        lambdas_full=case['lambdas_full'],
        omega_d=case['omega_d'],
        floquet_cutoff=200
    )

    U_num = result['U_numerical']
    U_floq = result['U_floquet']

    # Compute unitarity errors
    identity = torch.eye(U_num.shape[0], dtype=U_num.dtype, device=U_num.device)
    unit_err_num = torch.linalg.norm(U_num.conj().T @ U_num - identity).item()
    unit_err_floq = torch.linalg.norm(U_floq.conj().T @ U_floq - identity).item()

    # Check Floquet unitarity error below threshold
    assert unit_err_floq < 1e-11, f"Floquet unitarity error too large: {unit_err_floq}"

    # Check fidelity close to 1, tolerance based on max unitarity error
    max_err = max(unit_err_num, unit_err_floq)
    fidelity = result['fidelity']
    assert abs(1.0 - fidelity) <= max_err, \
        f"Fidelity deviation {abs(1.0 - fidelity)} exceeds tolerance {max_err}"


if __name__ == "__main__":
    raise NotImplementedError(
        "The current implementation of the Floquet propagator only computes it at integer numbers of periods of the drive."
    )

    for case in test_cases:
        test_propagator_accuracy(case)
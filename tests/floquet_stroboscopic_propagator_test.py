import pytest
import torch
import numpy as np

from tests.floquet_propagator_test import compare_propagators

# Define test cases as raw Python lists
test_cases = [
    {
        "desc": "Two-level system, weak drive",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [0.01],
        "phases": [0.0],
        "pulse_durations": [10],
        "omega_d": 1.0,
    },
    {
        "desc": "Two-level system, strong drive",
        "energies": [0.0, 1.0],
        "lambdas_full": [[0.0, 1.0], [1.0, 0.0]],
        "rabi_frequencies": [1.0],
        "phases": [0.0],
        "pulse_durations": [10],
        "omega_d": 1.0,
    },
    {
        "desc": "Three-level system, detuned drive",
        "energies": [0.0, 1.0, 2.1],
        "lambdas_full": [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        "rabi_frequencies": [0.5, 0.2],
        "phases": [0.0, np.pi/2],
        "pulse_durations": [5, 5],
        "omega_d": 1.5,
    },
    {
        "desc": "Three-level system, short pulse",
        "energies": [0.0, 1.0, 2.0],
        "lambdas_full": [[0.0, 0.8, 0.0], [0.8, 0.0, 1.0], [0.0, 1.0, 0.0]],
        "rabi_frequencies": [0.3],
        "phases": [np.pi],
        "pulse_durations": [1],
        "omega_d": 1.2,
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
        "pulse_durations": [4, 5, 6],
        "omega_d": 1.0,
    },
    {
        "desc": "Three-level degenerate energies",
        "energies": [0.0, 0.0, 0.0],
        "lambdas_full": [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        "rabi_frequencies": [0.20],
        "phases": [0.0],
        "pulse_durations": [8],
        "omega_d": 1.0,
    },
    {
        "desc": "Four-level system, varying durations",
        "energies": [0.0, 1.0, 1.8, 3.0],
        "lambdas_full": [[0.0, 0.9, 0.0, 0.0], [0.9, 0.0, 0.9, 0.0], [0.0, 0.9, 0.0, 0.9], [0.0, 0.0, 0.9, 0.0]],
        "rabi_frequencies": [0.30, 0.60, 0.30],
        "phases": [0.0, np.pi/4, np.pi/2],
        "pulse_durations": [2, 7, 3],
        "omega_d": 1.3,
    },
]


def test_propagator_accuracy(case):
    # Run comparison
    result = compare_propagators(
        rabi_frequencies=case['rabi_frequencies'],
        phases=case['phases'],
        pulse_durations=case['pulse_durations'],
        energies=case['energies'],
        lambdas_full=case['lambdas_full'],
        omega_d=case['omega_d'],
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
    print(case['desc'])
    print('\tFidelity: ', fidelity)
    print('\tUnitarity error:', unit_err_floq)
    assert abs(1.0 - fidelity) <= max_err, \
        f"Fidelity deviation {abs(1.0 - fidelity)} exceeds tolerance {max_err}"


if __name__ == "__main__":
    for case in test_cases:
        test_propagator_accuracy(case)
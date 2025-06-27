import numpy as np
import torch

from optimization.cma_fidelity import PulseEncoding
from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence_stroboscopic

# technical params
device        = torch.device("cuda")
dtype_real = torch.float64
dtype_complex = torch.complex128

U_target = torch.tensor([[0, 1], [1, 0]], dtype=dtype_complex, device=device)  # target unitary on first 2 levels

def get_system_params(n_levels):
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    energies = torch.tensor(energies, dtype=dtype_real, device=device)
    lambdas_full = torch.tensor(lambdas_full, dtype=dtype_complex, device=device)

    omega_d = 1.0                                
    floquet_cutoff: int = 50

    return dict(
        energies=energies,
        lambdas_full=lambdas_full,
        omega_d=omega_d,
        floquet_cutoff=floquet_cutoff
    )

def get_random_propagator(p0, system_params):

    rabi, phase, periods_i = PulseEncoding.decode(p0)
    # to torch
    rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
    phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
    dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device) 
    return floquet_propagator_square_sequence_stroboscopic(
                rabi_t, phase_t, dur_t, **system_params, device=device
        )  


def get_unitarity(U):
    return torch.linalg.norm(U.conj().T @ U - torch.eye(U.shape[0], dtype= U.dtype, device= U.device)).item()


def qubit_fidelity(
    U_actual: torch.Tensor
) -> torch.Tensor:
    """
    Eq. (8) with n_rel=2 and P = |0><0|+|1><1|:

        M = U_target^† · U_actual[:2,:2]
        F = [Tr(M M†) + |Tr(M)|^2] / [2·(2+1)].

    U_target must be 2x2; U_actual can be any NxN with N≥2.
    """

    if U_actual.ndim != 2 or U_actual.shape[0] < 2 or U_actual.shape[1] < 2:
        raise ValueError("U_actual must be at least 2x2")

    # project onto the first two levels
    M = U_target.conj().T @ U_actual[:2, :2]

    tr_MMdag = torch.trace(M @ M.conj().T)
    tr_M     = torch.trace(M)
    fidelity = (tr_MMdag + torch.abs(tr_M) ** 2) / 6.0  

    return fidelity.real


def test_unitarity():
    for n_levels in range(2, 7):
        system_params = get_system_params(n_levels)
        for n_pulses in range(1, 21):
            for i in range(10):
                p0 = np.concatenate([
                    np.random.random(n_pulses),
                    np.random.random(n_pulses) * 2 * np.pi,
                    np.random.random(n_pulses)
                ])
                U = get_random_propagator(p0, system_params)
                unitarity = get_unitarity(U)
                print(f"Test run: n_levels={n_levels}, n_pulses={n_pulses}, i={i}")
                print("Params:")
                print(f"  Rabi: {p0[:n_pulses]}")
                print(f"  Phase: {p0[n_pulses:2*n_pulses]}")
                print(f"  Periods: {p0[2*n_pulses:]}")
                print(f"Unitarity: {unitarity}")
                assert unitarity < 1e-3, f"Unitarity check failed for n_levels={n_levels}, n_pulses={n_pulses}, i={i}"
                print("Propagator U:")
                print(U)
                print("Fidelity:", qubit_fidelity(U))
                
                
if __name__ == "__main__":
    test_unitarity()
    print("All tests passed successfully!")
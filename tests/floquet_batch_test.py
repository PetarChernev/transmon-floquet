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


def test_oneshot_function():
    n_levels = 2
    n_pulses = 5
    system_params = get_system_params(n_levels)
    p0 = np.concatenate([
        np.random.random(n_pulses),
        np.random.random(n_pulses) * 2 * np.pi,
        np.random.random(n_pulses)
    ])
    U = get_random_propagator(p0, system_params)
   
                
                
if __name__ == "__main__":
    test_oneshot_function()
    print("All tests passed successfully!")
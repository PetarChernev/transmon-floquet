from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

from transmon.transmon_core import TransmonCore
from transmon.transmon_dynamics_qutip import pulse_sequence_qutip

# Assuming rabi_frequencies, phases, pulse_durations, epsilon, lambda_matrix, and options 
# are already defined in your environment

# Sweep drive frequency ω_d from 0.95 to 1.05
omega_values = np.linspace(0.95, 1.05, 100)
fidelities = []

target = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

n_levels = 6
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

rabi_frequencies = np.array([0.95107027, 0.79220659, 1.0358085,  1.05042276, 1.0133318,  0.87797412, 1.02144238, 0.64034026])
phases = np.array([2.15531313, 4.82939254, 3.9763329,  1.28857258, 4.64092377, 3.82731991, 1.45256058, 4.74845489])
pulse_durations = [7, 1, 7, 1, 7, 1, 7, 1,]

target = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def get_fidelity(omega):
    # Compute propagator for this drive frequency
    U = pulse_sequence_qutip(
        rabi_frequencies, 
        phases, 
        pulse_durations, 
        energies, 
        couplings, 
        omega, 
        options={
            "atol": 1e-8,
            "rtol": 1e-8,
            "nsteps": 20000
        },
    )
    # Convert to NumPy array if it's a Qobj
    U_mat = np.array(U)[:2, :2]  # Take only the first 2x2 block for qubit fidelity
    fidelity = np.abs(np.trace(U_mat.conj().T @ target)) / U_mat.shape[0]
    return fidelity.real

with Pool(20) as pool:
    # Compute fidelities in parallel
    fidelities = pool.map(get_fidelity, omega_values)
    
# Plot the transfer probability versus drive frequency
plt.figure()
plt.plot(omega_values, fidelities)
plt.xlabel('Drive frequency ω_d')
plt.ylabel('Transfer probability |⟨1|U|0⟩|^2')
plt.title('Transfer probability vs. drive frequency')
plt.show()

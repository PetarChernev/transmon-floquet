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

target = np.array([[0, 1], [1, 0]])

n_levels = 6
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

rabi_frequencies = np.array([0.40005119, -0.25700129,  0.17662924,  0.29273975,  0.08082814,
        0.42413851,  0.43772231, -0.37279788  ]) / 7000
phases = np.array([0.74597256,  1.58      ,
       -0.91019485, -1.06866579, -0.18841594,  0.69103507,  0.50755041,
        1.58])
pulse_durations = np.array([140.0 / 8] * 8)

target = np.array([[0, 1], [1, 0]])


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
            "atol": 1e-6,
            "rtol": 1e-6,
            "nsteps": 200000
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
plt.title('Fidelity vs. drive frequency')
plt.show()

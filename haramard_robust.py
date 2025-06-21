#!/usr/bin/env python3
"""
Sweep pulse-area deviations and plot fidelity to Hadamard target.
Minimal changes to compare against actual Hadamard gate action.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Pauli matrices and Hadamard gate
data_type = complex
sigma_x = np.array([[0, 1], [1, 0]], dtype=data_type)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=data_type)
# Hadamard unitary
H_gate = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=data_type)

# Hamiltonian in RWA
def H_rwa(omega, phi):
    return 0.5 * omega * (sigma_x * np.cos(phi) + sigma_y * np.sin(phi))

# Schrödinger RHS
def schrodinger_rhs(t, psi, H):
    return -1j * H.dot(psi)

# Fixed pulse phases from optimizer
phi_list = [-0.3947719931602478, 2.1863861083984375, 6.735899925231934, 4.6223368644714355, 0.13510499894618988, 3.6400818824768066, 6.658060073852539, 3.377084970474243, 5.965268135070801, 0.5913599729537964, 5.973409175872803, 3.4744880199432373, -1.9250119924545288]

# Sweep epsilon values (fractional deviation)
epsilons = np.linspace(-0.2, 0.2, 41)
fidelities = []

# Initial state |0>
psi0 = np.array([1.0, 0.0], dtype=data_type)
# Target state = H |0>
psi_target = H_gate.dot(psi0)
# explicit normalization (unitary should ensure norm=1)
psi_target /= np.linalg.norm(psi_target)

total = len(epsilons)
for idx, eps in enumerate(epsilons):
    print(f"Progress: {idx+1}/{total}")
    # reset to ground state
    psi = psi0.copy()

    # apply each pulse with area = (π/2)*(1+ε)
    for ph in phi_list:
        omega = (np.pi/2) * (1 + eps)
        H = H_rwa(omega, ph)
        sol = solve_ivp(
            schrodinger_rhs,
            (0.0, 1.0),
            psi,
            args=(H,),
            t_eval=[1.0],
            atol=1e-9,
            rtol=1e-7
        )
        psi = sol.y[:, -1]

    # renormalize psi to guard against drift
    psi /= np.linalg.norm(psi)

    # fidelity against Hadamard outcome
    fid = np.abs(np.vdot(psi_target, psi))**2
    print(f"Fidelity for psi={psi}: {fid:.4f}")
    fidelities.append(fid)
print(psi_target)

# Plot fidelity vs epsilon
plt.figure(figsize=(6, 4))
plt.plot(epsilons, fidelities, '-o')
plt.axhline(1.0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Fractional pulse-area deviation ε')
plt.ylabel('Fidelity to H |0>')
plt.title('Composite Pulse Fidelity vs. Area Error')
plt.grid(True)
plt.tight_layout()
plt.show()

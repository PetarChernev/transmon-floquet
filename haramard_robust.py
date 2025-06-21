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
phi = [3.864738173689102, 0.2565983147674805, 1.7094730858086657, 4.925178395210474, 0.7460716091347271, 2.0142139997185433, 2.324948223685373, 1.9518937578324786, 5.2101190190926685, 0.08421968698429469, 5.823308477226751, 0.2609984202122077, 0]
omega_0 = 1.425738
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
    for ph in phi:
        omega = omega_0 * (1 + eps)
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
plt.ylim(0.95, 1)
plt.title('Composite Pulse Fidelity vs. Area Error')
plt.grid(True)
plt.tight_layout()
plt.show()

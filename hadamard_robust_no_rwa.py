#!/usr/bin/env python3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Pauli matrices and Hadamard gate ---
data_type = complex
sigma_x = np.array([[0, 1], [1, 0]], dtype=data_type)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=data_type)
H_gate = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=data_type)

# --- Lab-frame Hamiltonian for a single resonant pulse ---
def H_lab(t, omega, phase, omega_drive):
    """
    Lab-frame Hamiltonian for a resonant drive of instantaneous Rabi rate `omega`
    and phase `phase`, oscillating at `omega_drive`.
    """
    return 0.5 * omega * (
        sigma_x * np.cos(omega_drive * t + phase)
      + sigma_y * np.sin(omega_drive * t + phase)
    )

def schrodinger_rhs(t, psi, omega, phase, omega_drive):
    H = H_lab(t, omega, phase, omega_drive)
    return -1j * H.dot(psi)

# --- Composite-pulse parameters from your optimizer ---
phi = [1.2612, 4.9806, 1.7797, 5.6662, 2.0883, 5.2595, 0.8755, 1.1791, 5.1672, 4.1891, 1.2348, 4.9342, 0]
phi_sequence = np.array(phi, dtype=data_type)
omega_0 = 1.0001
omega_drive = 1  # drive frequency (≫ Rabi) to expose counter-rotating terms

# --- Fidelity sweep over fractional pulse-area error ε ---
epsilons = np.linspace(-0.2, 0.2, 41)
fidelities = []

# initial state |0⟩ and target H|0⟩
psi0 = np.array([1.0, 0.0], dtype=data_type)
psi_target = H_gate.dot(psi0)
psi_target /= np.linalg.norm(psi_target)

for eps in epsilons:
    psi = psi0.copy()
    # each pulse is applied for duration T=1 (so ideal RWA rotation angle = omega*1)
    for phi in phi_sequence:
        omega = omega_0 * (1 + eps)
        sol = solve_ivp(
            lambda t, y: schrodinger_rhs(t, y, omega, phi, omega_drive),
            t_span=(0.0, 1.0),
            y0=psi,
            t_eval=[1.0],
            atol=1e-9,
            rtol=1e-7
        )
        psi = sol.y[:, -1]
    # renormalize and compute fidelity
    psi /= np.linalg.norm(psi)
    fidelities.append(np.abs(np.vdot(psi_target, psi))**2)

# --- Plot the result ---
plt.figure(figsize=(6, 4))
plt.plot(epsilons, fidelities, '-o', label='exact (lab-frame)')
plt.xlabel('Fractional pulse-area deviation ε')
plt.ylabel('Fidelity to H |0⟩')
plt.ylim(0, 1.0005)
plt.title('Composite-Pulse Fidelity vs. Area Error (no RWA)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

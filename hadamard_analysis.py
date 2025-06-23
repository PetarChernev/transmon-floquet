#!/usr/bin/env python3
"""
Additional verification tests for the composite pulse sequence
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Hadamard gate
H_gate = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def H_rwa(omega, phi):
    return 0.5 * omega * (sigma_x * np.cos(phi) + sigma_y * np.sin(phi))

def evolve_unitary(omega, phi, t=1.0):
    """Calculate the unitary evolution operator"""
    H = H_rwa(omega, phi)
    # For time-independent H: U = exp(-i H t)
    return scipy.linalg.expm(-1j * H * t)

def schrodinger_rhs(t, psi, H):
    return -1j * H.dot(psi)

# Your pulse sequence
phi = [1.1314, 3.6023, 2.6789, 
       4.9733, 3.3520, 4.7573, 
       2.6406, 2.1545, 4.4087, 
       5.9988, 3.2227, 5.5234, 0]
omega_0 = 1.415336

print(f"Number of pulses: {len(phi)}")
print(f"omega_0 = {omega_0:.6f}")
print(f"π/2 = {np.pi/2:.6f}")
print(f"Ratio omega_0/(π/2) = {omega_0/(np.pi/2):.6f}")

# Test 1: Check the composite operation at ε=0
print("\n=== Test 1: Composite operation at ε=0 ===")
psi0 = np.array([1.0, 0.0], dtype=complex)
psi = psi0.copy()

# Import scipy for matrix exponential
import scipy.linalg

# Build composite unitary
U_total = I.copy()
for ph in phi:
    U_k = evolve_unitary(omega_0, ph, t=1.0)
    U_total = U_k @ U_total

print("\nComposite unitary U_total (raw):")
print(U_total)

# Find the global phase by comparing to Hadamard
# Better method: use trace of U H†
trace_product = np.trace(U_total @ H_gate.conj().T)
global_phase = np.angle(trace_product / 2)  # Divide by 2 since trace of 2×2 unitary is 2e^(iθ)

print(f"\nGlobal phase offset: {global_phase:.4f} rad = {global_phase/np.pi:.4f}π")

# Alternative check using determinant (should be e^(i·2θ) for SU(2))
det_ratio = np.linalg.det(U_total) / np.linalg.det(H_gate)
global_phase_from_det = np.angle(det_ratio) / 2
print(f"Global phase from determinant: {global_phase_from_det:.4f} rad = {global_phase_from_det/np.pi:.4f}π")

# Apply global phase correction
U_corrected = U_total * np.exp(1j * global_phase)
print(f"\nPhase correction factor: exp(i·{global_phase/np.pi:.4f}π)")

print("\nComposite unitary U_corrected (phase-adjusted):")
print(U_corrected)
print("\nHadamard gate:")
print(H_gate)
print(f"\nFrobenius norm ||U_corrected - H||_F = {np.linalg.norm(U_corrected - H_gate, 'fro'):.6f}")
print(f"Original norm ||U_total - H||_F = {np.linalg.norm(U_total - H_gate, 'fro'):.6f}")

# Check element-wise agreement
print("\nElement-wise comparison (U_corrected vs H):")
for i in range(2):
    for j in range(2):
        diff = U_corrected[i,j] - H_gate[i,j]
        print(f"  [{i},{j}]: {U_corrected[i,j]:.6f} vs {H_gate[i,j]:.6f}, diff = {abs(diff):.6f}")

# Test 2: Check with different initial states
print("\n=== Test 2: Fidelity for different initial states ===")
test_states = {
    '|0⟩': np.array([1, 0], dtype=complex),
    '|1⟩': np.array([0, 1], dtype=complex),
    '|+⟩': np.array([1, 1], dtype=complex) / np.sqrt(2),
    '|-⟩': np.array([1, -1], dtype=complex) / np.sqrt(2),
}

print("Using raw U_total (with global phase):")
for name, psi_init in test_states.items():
    psi_target = H_gate @ psi_init
    psi_actual = U_total @ psi_init
    fidelity = np.abs(np.vdot(psi_target, psi_actual))**2
    print(f"  {name}: fidelity = {fidelity:.6f}")

print("\nUsing phase-corrected U_corrected:")
for name, psi_init in test_states.items():
    psi_target = H_gate @ psi_init
    psi_actual = U_corrected @ psi_init
    fidelity = np.abs(np.vdot(psi_target, psi_actual))**2
    print(f"  {name}: fidelity = {fidelity:.6f}")
    
# Check unitarity
print("\nUnitarity check:")
print(f"||U_total† U_total - I||_F = {np.linalg.norm(U_total.conj().T @ U_total - I, 'fro'):.1e}")
print(f"||U_corrected† U_corrected - I||_F = {np.linalg.norm(U_corrected.conj().T @ U_corrected - I, 'fro'):.1e}")

# Test 3: Visualize the unitary operation
print("\n=== Test 3: Process tomography ===")
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# Real and imaginary parts of U_total, U_corrected, and H_gate
titles = ['Re(U_raw)', 'Im(U_raw)', 
          'Re(U_corrected)', 'Im(U_corrected)',
          'Re(H)', 'Im(H)']
matrices = [U_total.real, U_total.imag, 
            U_corrected.real, U_corrected.imag,
            H_gate.real, H_gate.imag]

for ax, title, matrix in zip(axes.flat, titles, matrices):
    im = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['|0⟩', '|1⟩'])
    ax.set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# Additional visualization: Difference matrix
print("\n=== Difference Matrix ===")
diff_matrix = U_corrected - H_gate
print("U_corrected - H_gate:")
print(f"Real part max deviation: {np.max(np.abs(diff_matrix.real)):.6f}")
print(f"Imag part max deviation: {np.max(np.abs(diff_matrix.imag)):.6f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im1 = axes[0].imshow(diff_matrix.real, cmap='RdBu', vmin=-0.01, vmax=0.01)
axes[0].set_title('Re(U_corrected - H)')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(diff_matrix.imag, cmap='RdBu', vmin=-0.01, vmax=0.01)
axes[1].set_title('Im(U_corrected - H)')
plt.colorbar(im2, ax=axes[1])

for ax in axes:
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['|0⟩', '|1⟩'])
    ax.set_yticklabels(['⟨0|', '⟨1|'])

plt.tight_layout()
plt.show()

# Test 4: Check if this could be a known composite pulse
print("\n=== Test 4: Pulse sequence analysis ===")
print("Phase sequence (in units of π):")
for i, ph in enumerate(phi):
    print(f"  Pulse {i+1}: φ = {ph/np.pi:.3f}π")

# Check total rotation
total_rotation = len(phi) * omega_0
print(f"\nTotal rotation (at ε=0): {total_rotation:.3f} rad = {total_rotation/np.pi:.3f}π")

# For BB1-type sequences, we expect specific patterns
# BB1 typically has 5 pulses, but there are extended versions
print(f"\nThis appears to be a {len(phi)}-pulse composite sequence")
print("The high robustness to ±20% amplitude errors suggests a sophisticated design")

# Test 5: Show how to incorporate global phase in original code
print("\n=== Test 5: Global phase correction in practice ===")
print("The global phase doesn't affect physical observables (measurement probabilities),")
print("which is why your original fidelity calculations are correct.")
print("\nIf you want to match the exact matrix elements, you could:")
print(f"1. Multiply final state by exp(i·{global_phase/np.pi:.4f}π)")
print(f"2. Or add a virtual Z-rotation of {-global_phase/np.pi:.4f}π at the end")
print("\nFor robust pulse sequences, global phases often accumulate and are typically ignored.")
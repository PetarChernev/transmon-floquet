#!/usr/bin/env python3
"""
Global optimization of a composite pulse sequence to approximate the Hadamard gate
with robustness to pulse-area errors, using CMA-ES.
"""
import numpy as np
from cma import CMAEvolutionStrategy

# ----------------------------------------------------------------------------
# 1) Single-pulse unitary (numpy version)
# ----------------------------------------------------------------------------
def pulse_unitary(A, phi, eps=0.0):
    half = (A * (1.0 + eps)) / 2.0
    c = np.cos(half)
    s = np.sin(half)
    exp_ip = np.exp(-1j * phi)
    exp_im = np.exp( 1j * phi)
    return np.array([[    c,    -1j * exp_ip * s],
                     [-1j * exp_im * s,        c     ]], dtype=complex)

# ----------------------------------------------------------------------------
# 2) Composite sequence propagator at given eps
# ----------------------------------------------------------------------------
def composite_unitary(phi_list, eps, A=np.pi/2):
    U = np.eye(2, dtype=complex)
    for phi in phi_list:
        U = pulse_unitary(A, phi, eps) @ U
    return U

# ----------------------------------------------------------------------------
# 3) Fidelity & robustness objective
# ----------------------------------------------------------------------------
# target Hadamard
H = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
# finite-difference grid
delta = 1e-3
B = 33
eps_grid = np.linspace(-delta, +delta, B)
lambda_deriv = 1.0

def objective(phi_flat):
    # phi_flat: length-N array of phases
    U0 = composite_unitary(phi_flat, eps=0.0)
    # trace-based fidelity
    inner = np.trace(np.conj(H.T) @ U0)
    F = np.abs(inner) / 2.0
    loss_fid = 1.0 - F**2
    # robustness penalty via finite differences
    penalty = 0.0
    for eps in eps_grid:
        U_eps = composite_unitary(phi_flat, eps)
        diff = U_eps - U0
        penalty += np.sum(np.abs(diff)**2)
    penalty /= (delta**2)
    return loss_fid + lambda_deriv * penalty

# ----------------------------------------------------------------------------
# 4) CMA-ES setup
# ----------------------------------------------------------------------------
N = 13  # number of pulses
# initial guess: random phases in [0,2π]
p0 = np.random.rand(N) * 2*np.pi
sigma0 = 1.0  # initial step-size
es = CMAEvolutionStrategy(p0, sigma0, {'popsize': 20})

# run CMA indefinitely (or until high fidelity reached)
# you can interrupt with Ctrl+C once satisfied
threshold_F = 0.995  # target fidelity
generation = 0
while True:
    solutions = es.ask()
    losses = [objective(sol) for sol in solutions]
    es.tell(solutions, losses)
    phi_opt = es.result.xbest
    best_loss = es.result.fbest
    # compute current fidelity
    U0 = composite_unitary(phi_opt, eps=0.0)
    inner = np.trace(np.conj(H.T) @ U0)
    F_current = np.abs(inner) / 2.0
    if generation % 10 == 0:
        print(f"Gen {generation}: best loss = {best_loss:.4e}, fidelity = {F_current:.6f}")
    if F_current >= threshold_F:
        print(f"Reached fidelity {F_current:.6f} at generation {generation}")
        break
    generation += 1

# CMA-ES result
phi_opt = es.result.xbest
best_loss = es.result.fbest
U_opt = composite_unitary(phi_opt, eps=0.0)
inner = np.trace(np.conj(H.T) @ U_opt)
F_opt = np.abs(inner) / 2.0
print("CMA-ES optimization complete.")
print(f"Best loss = {best_loss:.6f}, fidelity = {F_opt:.6f}")
print("Optimized phases (rad):", np.round(phi_opt, 6).tolist())
print("Resulting unitary:")
print(U_opt)

# ----------------------------------------------------------------------------
# 7) Gradient-descent refinement starting from CMA-ES result
# ----------------------------------------------------------------------------
import torch
# prepare phases as torch parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.complex64
phi_t = torch.nn.Parameter(torch.tensor(phi_opt, dtype=torch.float32))
optimizer2 = torch.optim.Adam([phi_t], lr=1e-2)
lambda_deriv_td = lambda_deriv
# no eps grid needed for pure Hadamard
for it in range(1, 2001):
    optimizer2.zero_grad()
    # build U_seq at eps=0
    U = torch.eye(2, dtype=dtype, device=device)
    for ph in phi_t:
        half = ( (torch.tensor(np.pi/2, device=device) ) / 2 )
        c = torch.cos(half)
        s = torch.sin(half)
        exp_ip = torch.exp(-1j * ph)
        exp_im = torch.exp( 1j * ph)
        Ui = torch.stack([
            torch.stack([     c,    -1j * exp_ip * s], dim=-1),
            torch.stack([-1j * exp_im * s,           c ], dim=-1)
        ], dim=-2)
        U = Ui @ U
    # compute phase-insensitive fidelity to Hadamard
    targ = torch.tensor(H, dtype=dtype, device=device)
    inner_t = torch.trace(targ.conj().T @ U)
    F_t = torch.abs(inner_t) / 2.0
    loss_t = 1.0 - F_t**2
    loss_t.backward()
    optimizer2.step()
    if it % 100 == 0:
        print(f"  Refinement iter {it}: F={F_t.item():.6f}")

# final refined result
phi_refined = phi_t.detach().cpu().numpy()
U_final = composite_unitary(phi_refined, eps=0.0)
inner_f = np.trace(np.conj(H.T) @ U_final)
F_final = np.abs(inner_f) / 2.0
print("After gradient refinement:")
print(f"Fidelity = {F_final:.6f}")
print("Refined phases (rad):", np.round(phi_refined, 6).tolist())
print("Resulting unitary:")
print(U_final)

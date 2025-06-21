#!/usr/bin/env python3
"""
Hybrid optimization of a composite pulse sequence to approximate the Hadamard gate
with minimized derivatives with respect to pulse area A.
Uses CMA-ES for finding high-fidelity solutions, then gradient descent to minimize derivatives.
Last phase fixed to 0; phases and area constrained mod 2π.
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
from cma import CMAEvolutionStrategy
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1) PyTorch pulse unitary (for GPU acceleration)
# ----------------------------------------------------------------------------
def pulse_unitary_torch(A, phi, eps=0.0, device='cpu', dtype=torch.complex128):
    """Single pulse unitary in PyTorch"""
    half = (A * (1.0 + eps)) / 2.0
    c = torch.cos(half)
    s = torch.sin(half)
    exp_ip = torch.exp(-1j * phi)
    exp_im = torch.exp(1j * phi)
    U = torch.zeros((2, 2), dtype=dtype, device=device)
    U[0, 0] = c
    U[0, 1] = -1j * exp_ip * s
    U[1, 0] = -1j * exp_im * s
    U[1, 1] = c
    return U

# ----------------------------------------------------------------------------
# 2) Composite sequence propagator (PyTorch version)
# ----------------------------------------------------------------------------
def composite_unitary_torch(phi_list, A, eps=0.0, device='cpu', dtype=torch.complex128):
    """Composite unitary in PyTorch, with last phase fixed to 0"""
    U = torch.eye(2, dtype=dtype, device=device)
    # apply each free-phase pulse
    for phi in phi_list:
        U = pulse_unitary_torch(A, phi, eps, device, dtype) @ U
    # apply final pulse with phase 0, ensuring phi is a tensor
    zero_phi = torch.zeros((), dtype=phi_list.dtype, device=device)
    U = pulse_unitary_torch(A, zero_phi, eps, device, dtype) @ U
    return U

# ----------------------------------------------------------------------------
# 3) Compute N derivatives of fidelity w.r.t. A
# ----------------------------------------------------------------------------
def compute_fidelity_derivatives(phi_list, A_val, N_derivs, device='cpu'):
    """Compute fidelity and its first N derivatives w.r.t. A using PyTorch autograd"""
    dtype = torch.complex128
    # Convert to torch tensors
    phi_tensor = torch.tensor(phi_list, dtype=torch.float64, device=device)
    A = torch.tensor(A_val, dtype=torch.float64, device=device, requires_grad=True)
    # Target Hadamard
    H = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device) / np.sqrt(2)
    # Compute composite unitary
    U = composite_unitary_torch(phi_tensor, A, device=device, dtype=dtype)
    # Compute fidelity
    inner = torch.trace(H.conj().T @ U)
    F = torch.abs(inner) / 2.0
    # Compute derivatives
    derivatives = []
    current = F
    for i in range(N_derivs):
        grad = torch.autograd.grad(current, A, create_graph=True, retain_graph=True)[0]
        derivatives.append(grad.item())
        current = grad
    return F.item(), derivatives

# ----------------------------------------------------------------------------
# 4) CMA-ES objective (fidelity only, no derivative consideration)
# ----------------------------------------------------------------------------
def cmaes_objective(params, target_A=np.pi/2):
    """CMA-ES objective: maximize fidelity (minimize 1-F^2), with periodic constraints"""
    # Wrap parameters to [0, 2π)
    params_mod = np.mod(params, 2 * np.pi)
    phi_list = params_mod[:-1]
    A = params_mod[-1]
    # Target Hadamard
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    U = np.eye(2, dtype=complex)
    # Apply free-phase pulses
    for phi in phi_list:
        half = A / 2.0
        c = np.cos(half)
        s = np.sin(half)
        exp_ip = np.exp(-1j * phi)
        exp_im = np.exp(1j * phi)
        Ui = np.array([[c, -1j * exp_ip * s],
                       [-1j * exp_im * s, c]], dtype=complex)
        U = Ui @ U
    # Apply final pulse with phase 0
    half = A / 2.0
    c = np.cos(half)
    s = np.sin(half)
    Ui = np.array([[c, -1j * s],
                   [-1j * s, c]], dtype=complex)
    U = Ui @ U
    # Fidelity
    inner = np.trace(np.conj(H.T) @ U)
    F = np.abs(inner) / 2.0
    return 1.0 - F**2

# ----------------------------------------------------------------------------
# 5) Gradient descent for derivative minimization
# ----------------------------------------------------------------------------
def minimize_derivatives(phi_init, A_init, N_derivs, max_iters=5000, 
                        fidelity_threshold=0.99, device='cpu'):
    """
    Minimize derivatives while maintaining high fidelity using gradient descent.
    Returns success flag, final parameters, final fidelity, and derivatives.
    """
    dtype_real = torch.float64
    dtype_complex = torch.complex128
    # Initialize parameters
    phi_param = torch.nn.Parameter(torch.tensor(phi_init, dtype=dtype_real, device=device))
    A_param = torch.nn.Parameter(torch.tensor(A_init, dtype=dtype_real, device=device))
    optimizer = torch.optim.Adam([phi_param, A_param], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    # Target Hadamard
    H = torch.tensor([[1, 1], [1, -1]], dtype=dtype_complex, device=device) / np.sqrt(2)
    # Track best
    best_deriv_norm = float('inf')
    best_fidelity = 0.0
    best_deriv_vals = None
    best_params = None
    for it in range(max_iters):
        optimizer.zero_grad()
        # Compute unitary
        U = composite_unitary_torch(phi_param, A_param, device=device, dtype=dtype_complex)
        # Fidelity
        inner = torch.trace(H.conj().T @ U)
        F = torch.abs(inner) / 2.0
        # Derivatives
        derivatives = []
        current = F
        for i in range(N_derivs):
            grad = torch.autograd.grad(current, A_param, create_graph=True, retain_graph=True)[0]
            derivatives.append(grad)
            current = grad
        deriv_vals = [d.item() for d in derivatives]
        deriv_norm = np.linalg.norm(deriv_vals)
        # Update best if above threshold
        F_val = F.item()
        if F_val >= fidelity_threshold and deriv_norm < best_deriv_norm:
            best_deriv_norm = deriv_norm
            best_fidelity = F_val
            best_deriv_vals = deriv_vals.copy()
            best_params = (phi_param.detach().cpu().numpy().copy(), A_param.detach().cpu().numpy().copy())
        # Loss: prioritize fidelity more strongly
        deriv_loss = sum(torch.abs(d) for d in derivatives) / N_derivs
        fidelity_loss = torch.relu(fidelity_threshold - F) * 1e5  # enforce hard fidelity constraint  # increased penalty
        loss = deriv_loss + fidelity_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([phi_param, A_param], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        # Enforce periodic constraints
        with torch.no_grad():
            phi_param.data %= (2 * math.pi)
            A_param.data %= (2 * math.pi)
        # Progress printing
        if it % 500 == 0:
            print(f"  GD iter {it}: F={F_val:.6f}, deriv_norm={deriv_norm:.6e}, loss={loss.item():.6e}")
            print(f"    Derivatives: {[f'{d:.3e}' for d in deriv_vals]}")
            if best_params is not None:
                print(f"    Best so far: F={best_fidelity:.6f}, deriv_norm={best_deriv_norm:.6e}, derivatives={[f'{d:.3e}' for d in best_deriv_vals]}")
            else:
                print("    No acceptable solution found so far.")
        # Early stopping
        if F_val >= fidelity_threshold and deriv_norm < 1e-6:
            print(f"  Early stopping: achieved small derivatives at iter {it}")
            return True, best_params, best_fidelity, best_deriv_vals
    # After loop, return best found
    if best_params is not None:
        return True, best_params, best_fidelity, best_deriv_vals
    else:
        return False, None, 0.0, None

# ----------------------------------------------------------------------------
# 6) Main hybrid optimization loop
# ----------------------------------------------------------------------------
def hybrid_optimization(N_pulses=13, N_derivs=3, max_cmaes_attempts=10, 
                       fidelity_target=0.995, device='cpu'):
    """
    Hybrid optimization: CMA-ES for high fidelity, then gradient descent for derivatives.
    Last phase fixed to 0.
    """
    print(f"Starting hybrid optimization with {N_pulses} pulses, {N_derivs} derivatives")
    print(f"Device: {device}")
    print("-" * 80)
    attempt = 0
    best_solution = None
    best_deriv_norm = float('inf')
    # CMA-ES dimension = (N_pulses-1) free phases + area
    dim = N_pulses
    while attempt < max_cmaes_attempts:
        attempt += 1
        print(f"\n=== CMA-ES Attempt {attempt} ===")
        # Initial guess: random phases (N-1) and A near π/2
        p0 = np.concatenate([
            np.random.rand(N_pulses-1) * 2 * np.pi,
            [np.pi/2 + np.random.randn() * 0.1]
        ])
        sigma0 = 0.5
        es = CMAEvolutionStrategy(p0, sigma0, {
            'popsize': 50,
            'maxiter': 1000,
            'tolfun': 1e-10,
            'verb_disp': 0
        })
        # Run CMA-ES
        generation = 0
        while not es.stop() and generation < 500:
            solutions = es.ask()
            losses = [cmaes_objective(sol) for sol in solutions]
            es.tell(solutions, losses)
            if generation % 50 == 0:
                best_loss = min(losses)
                F_approx = np.sqrt(1 - best_loss)
                print(f"  Gen {generation}: best F ≈ {F_approx:.6f}")
            if min(losses) < (1 - fidelity_target**2):
                break
            generation += 1
        # Evaluate best from CMA-ES
        params_opt = es.result.xbest
        F_cmaes, derivs_cmaes = compute_fidelity_derivatives(params_opt[:-1], params_opt[-1], N_derivs, device)
        deriv_norm_cmaes = np.linalg.norm(derivs_cmaes)
        print(f"\nCMA-ES result: F={F_cmaes:.6f}, A={params_opt[-1]:.6f}")
        print(f"Initial derivatives: {[f'{d:.3e}' for d in derivs_cmaes]}, norm={deriv_norm_cmaes:.3e}")
        if F_cmaes < fidelity_target:
            print("Fidelity too low, trying again...")
            continue
        # Gradient descent phase
        print("\n--- Starting gradient descent for derivative minimization ---")
        success, gd_params, F_gd, derivs_gd = minimize_derivatives(
            params_opt[:-1], params_opt[-1], N_derivs,
            max_iters=5000,
            fidelity_threshold=fidelity_target * 0.99,
            device=device
        )
        if success and gd_params is not None:
            phi_gd, A_gd = gd_params
            deriv_norm_gd = np.linalg.norm(derivs_gd)
            print(f"\nGD result: F={F_gd:.6f}, A={A_gd:.6f}")
            print(f"Final derivatives: {[f'{d:.3e}' for d in derivs_gd]}, norm={deriv_norm_gd:.3e}")
            print(f"Derivative reduction: {deriv_norm_cmaes/deriv_norm_gd:.2f}x")
            if deriv_norm_gd < best_deriv_norm:
                best_deriv_norm = deriv_norm_gd
                best_solution = {
                    'phi': phi_gd,
                    'A': A_gd,
                    'fidelity': F_gd,
                    'derivatives': derivs_gd,
                    'deriv_norm': deriv_norm_gd
                }
                if deriv_norm_gd < 1e-5:
                    print("\nExcellent solution found! Derivatives are very small.")
                    break
        else:
            print("Gradient descent failed to maintain fidelity while reducing derivatives.")
    return best_solution

# ----------------------------------------------------------------------------
# 7) Main execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    N_pulses = 13
    N_derivs = 3
    solution = hybrid_optimization(
        N_pulses=N_pulses,
        N_derivs=N_derivs,
        max_cmaes_attempts=10,
        fidelity_target=0.995,
        device=device
    )
    if solution:
        print("\n" + "="*80)
        print("FINAL BEST SOLUTION:")
        print("="*80)
        print(f"Fidelity: {solution['fidelity']:.8f}")
        print(f"Pulse area A: {solution['A']:.8f} rad")
        print(f"Phases (rad, last fixed to 0): {np.round(solution['phi'], 6).tolist()} + [0]")
        print(f"Derivatives w.r.t. A: {[f'{d:.6e}' for d in solution['derivatives']]}")
        print(f"Derivative norm: {solution['deriv_norm']:.6e}")
        # Verification
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        U = np.eye(2, dtype=complex)
        A = solution['A']
        for phi in solution['phi']:
            half = A / 2.0
            c = np.cos(half)
            s = np.sin(half)
            exp_ip = np.exp(-1j * phi)
            exp_im = np.exp(1j * phi)
            Ui = np.array([[c, -1j * exp_ip * s],
                           [-1j * exp_im * s, c]], dtype=complex)
            U = Ui @ U
        # final pulse
        half = A / 2.0
        c = np.cos(half)
        s = np.sin(half)
        Ui = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        U = Ui @ U
        print("\nFinal unitary:")
        print(U)
        print(f"\nUnitary error: ||U - H||_F = {np.linalg.norm(U - H):.6e}")
    else:
        print("\nNo satisfactory solution found within the given attempts.")

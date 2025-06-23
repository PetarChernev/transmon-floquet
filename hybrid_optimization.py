#!/usr/bin/env python3
"""
Hybrid optimization of a composite pulse sequence to approximate the Hadamard gate
with minimized derivatives with respect to pulse area A, using parallel gradient descent.
Last phase fixed to 0; phases and area constrained mod 2π.
Progress tracking: one overall bar plus per-process GD iteration bars using tqdm positions.
Replaced prints during progress with tqdm.write to avoid bar displacement.


"""
import os
import pickle
import numpy as np
import math
import torch
import torch.nn.functional as F
from cma import CMAEvolutionStrategy
import warnings
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from transmon_core import TransmonCore
from transmon_dynamics import simulate_transmon_propagator
from transmon_dynamics_pytorch import transmon_propagator_pytorch

RUN_NAME = "H_5x3_L2_5000+10000 fidelity weight"
# Ensure CUDA-friendly multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

warnings.filterwarnings('ignore')


n_levels = 6
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)

energies, lambdas_full = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)
energies = torch.tensor(energies, dtype=torch.float64)
lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128)



def make_transmon_composite_wrapper(
    energies,
    lambdas_full,
    *,
    n_levels=6,
    total_time=20.0,
    n_time_steps=2000,
    pulse_type="square",
    use_rwa=True,
    use_pytorch=True
):
    """
    Returns a function with the same interface as composite_unitary_torch(phi_list, A, eps, device, dtype),
    but internally calls transmon_propagator_pytorch using fixed transmon parameters.
    """
    def wrapped(rabi_list, phi_list, eps=0.0, device="cpu", dtype=torch.complex128):

        # Apply scaling
        rabi_list = rabi_list * (1.0 + eps)
       
        if use_pytorch:
            # Ensure input types are torch tensors
            phi_tensor = torch.tensor(phi_list, dtype=torch.float64, device=device)
            rabi_tensor = torch.tensor(rabi_list, dtype=torch.float64, device=device)
        
            return transmon_propagator_pytorch(
                rabi_frequencies=rabi_tensor,
                phases=phi_tensor,
                energies=energies,
                lambdas_full=lambdas_full,
                n_levels=n_levels,
                total_time=total_time,
                n_time_steps=n_time_steps,
                pulse_type=pulse_type,
                use_rwa=use_rwa,
                device=device
            ).to(dtype=dtype)
        else:
            return simulate_transmon_propagator(
                rabi_frequencies=rabi_list,
                phases=phi_list,
                energies=energies,
                lambdas_full=lambdas_full,
                n_levels=n_levels,
                total_time=total_time,
                n_time_steps=n_time_steps,
                pulse_type=pulse_type,
                use_rwa=use_rwa,
            )
    return wrapped


unitary_function = make_transmon_composite_wrapper(
    energies=energies,
    lambdas_full=lambdas_full,
    n_levels=6,
    total_time=20.0 * 2 * np.pi * 7,
    n_time_steps=5000,
    pulse_type="square",
    use_rwa=True
)

unitary_function_np = make_transmon_composite_wrapper(
    energies=energies,
    lambdas_full=lambdas_full,
    n_levels=6,
    total_time=20.0 * 2 * np.pi * 7,
    n_time_steps=5000,
    pulse_type="square",
    use_rwa=True,
    use_pytorch=False
)

# ----------------------------------------------------------------------------
# 1) PyTorch pulse unitary (for GPU acceleration)
# ----------------------------------------------------------------------------
def pulse_unitary_torch(A, phi, eps=0.0, device='cpu', dtype=torch.complex128):
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
    U = torch.eye(2, dtype=dtype, device=device)
    for phi in phi_list:
        U = pulse_unitary_torch(A, phi, eps, device, dtype) @ U
    zero_phi = torch.zeros((), dtype=phi_list.dtype, device=device)
    U = unitary_function(A, zero_phi, eps, device, dtype) @ U
    return U

# ----------------------------------------------------------------------------
# 3) Compute N derivatives of fidelity w.r.t. A
# ----------------------------------------------------------------------------
def compute_fidelity_derivatives(phi_list, A_val, N_derivs, device='cpu'):
    dtype = torch.complex128
    phi_tensor = torch.tensor(phi_list, dtype=torch.float64, device=device, requires_grad=True)
    A = torch.tensor(A_val, dtype=torch.float64, device=device, requires_grad=True)
    H = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device) / np.sqrt(2)
    U = unitary_function(phi_tensor, A, device=device, dtype=dtype)
    inner = torch.trace(H.conj().T @ U[:2, :2])
    F = torch.abs(inner) / 2.0
    derivatives = []
    current = F
    for _ in range(N_derivs):
        grad = torch.autograd.grad(current, A, create_graph=True, retain_graph=True)[0]
        derivatives.append(grad)
        current = grad
    deriv_vals = [d.item() for d in derivatives]
    return F.item(), deriv_vals

# ----------------------------------------------------------------------------
# 4) Compute fidelity and derivatives at multiple area scaling points
# ----------------------------------------------------------------------------
def compute_multi_point_derivatives(phi_param, A_param, N_derivs, area_scales, H, device='cpu', dtype_complex=torch.complex128):
    all_values = []
    for i, scale in enumerate(area_scales):
        distance_from_edge = min(i, len(area_scales) - i - 1) + 1
        scaled_A = A_param * scale
        U = composite_unitary_torch(phi_param, scaled_A, device=device, dtype=dtype_complex)
        inner = torch.trace(H.conj().T @ U)
        F = torch.abs(inner) / 2.0
        # Add fidelity to the list
        all_values.append((1 - F) * distance_from_edge * 1000)
        # Compute derivatives
        current = F
        for _ in range(N_derivs):
            grad = torch.autograd.grad(current, A_param, create_graph=True, retain_graph=True)[0]
            all_values.append(grad)
            current = grad
    return all_values

# ----------------------------------------------------------------------------
# 5) CMA-ES objective (fidelity only)
# ----------------------------------------------------------------------------
def cmaes_objective(params):
    phi_list = params[:len(params) // 2]
    A = params[len(params) // 2:]
    # H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    U = unitary_function_np(A, phi_list, eps=0.0)
    inner = np.trace(np.conj(X.T) @ U[:2, :2])
    F = np.abs(inner) / 2.0
    return 1.0 - F**2

# ----------------------------------------------------------------------------
# 6) Modified gradient descent with multi-point derivative evaluation
# ----------------------------------------------------------------------------
def minimize_derivatives(run_id, phi_init, A_init, N_derivs=2, max_iters=2000, device='cpu'):
    dtype_real = torch.float64
    dtype_complex = torch.complex128
    # Area scaling points
    area_scales = [0.8, .85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    phi_param = torch.nn.Parameter(torch.tensor(phi_init, dtype=dtype_real, device=device))
    A_param = torch.nn.Parameter(torch.tensor(A_init, dtype=dtype_real, device=device))
    optimizer = torch.optim.Adam([phi_param, A_param], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    H = torch.tensor([[1, 1], [1, -1]], dtype=dtype_complex, device=device) / np.sqrt(2)
    patience, min_imp = 200, 1e-6
    no_imp, best_loss = 0, float('inf')
    best_params, best_values, best_F, best_norm = None, None, 0.0, float('inf')

    for it in tqdm(range(max_iters), desc=f"GD run {run_id}", position=run_id, leave=False):
        optimizer.zero_grad()
        all_values = compute_multi_point_derivatives(
            phi_param, A_param, N_derivs, area_scales, H, device, dtype_complex
        )
        nominal_idx = area_scales.index(1.0)
        F = all_values[nominal_idx * (N_derivs + 1)] / (min(nominal_idx, len(area_scales) - nominal_idx - 1) + 1)  # Fidelity at scale=1.0
        value_scalars = [val.item() if hasattr(val, 'item') else val for val in all_values]
        norm = np.linalg.norm(value_scalars)
        loss = sum(val**2 for val in all_values) / len(all_values) + F * 10000
        loss_val = loss.item()
        if best_loss - loss_val > min_imp:
            best_loss, no_imp = loss_val, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([phi_param, A_param], 1.0)
        optimizer.step()
        scheduler.step(loss)
        with torch.no_grad(): 
            phi_param.data %= 2*math.pi
            A_param.data %= 2*math.pi
        if norm < best_norm:
            best_norm = norm
            best_params = (phi_param.detach().cpu().numpy(), A_param.detach().cpu().numpy())
            best_F = F.item()
            best_values = value_scalars

    # Save best parameters for this GD run
    if best_params:
        phi_f, A_f = best_params
        out_dir = './data'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{RUN_NAME}_{run_id}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump({'phases': phi_f.tolist(), 'area': float(A_f)}, f)
        return True, phi_f, A_f, best_F, best_values, best_norm
    return False, None, None, 0.0, None, None

# ----------------------------------------------------------------------------
# 7) GD task wrapper
# ----------------------------------------------------------------------------
def gd_task(args):
    run_id, phi, A, Nd, device = args
    res = minimize_derivatives(run_id, phi, A, Nd, device=device)
    return run_id, res

# ----------------------------------------------------------------------------
# 8) Main hybrid parallel loop
# ----------------------------------------------------------------------------
def hybrid_parallel(N_pulses=13, N_derivs=2, n_cand=None, fidelity_target=0.999995, device='cpu'):
    n_cand = n_cand or cpu_count()
    tqdm.write(f"Running CMA-ES to generate {n_cand} candidates...")
    cma_sols, attempts = [], 0
    
    # From Table I - complete population transfer
    rabi_frequencies = np.array(
        [31.651, 44.988, 69.97, 60.608, 66.029, 68.771, 69.562, 66.971]
    )

    rabi_frequencies = (
        rabi_frequencies / 7000
    )  # Convert to GHz from MHz and normalise by ω01

    phases = np.array(
        [0.1779, 0.0499, 0.1239, 0.2538, 0.2886, 0.1688,0.1645, 0.1234]
    ) * np.pi
    
    with Pool(cpu_count()) as pool:
        while len(cma_sols) < n_cand and attempts < 3*n_cand:
            attempts +=1
            p0 = np.concatenate([rabi_frequencies, phases])
            es = CMAEvolutionStrategy(p0,0.05,{'popsize':50,'maxiter':1000,'tolfun':1e-10,'verb_disp':0})
            losses = [cmaes_objective(p0)]  # Evaluate known good point
            sols = es.ask() 
            losses += pool.map(cmaes_objective, sols)
            sols = [p0] + sols  # Add p0 to beginning of solutions
            es.tell(sols, losses)
            for _ in range(500):
                print(f"  CMA iteration {es.countiter} (attempt {attempts})")
                sols = es.ask(); 
                losses = pool.map(cmaes_objective, sols)
                print(f"  Evaluated {len(sols)} candidates, min loss: {min(losses):.6f}")
                es.tell(sols, losses)
                if min(losses) < (1-fidelity_target**2): break
            opt = es.result.xbest; 
            F_c, _ = compute_fidelity_derivatives(opt[:N_pulses],opt[:N_pulses:],N_derivs,device)
            if F_c>=fidelity_target:
                tqdm.write(f"  Candidate {len(cma_sols)+1}: F={F_c:.6f}, A={opt[-1]:.6f}")
                cma_sols.append((len(cma_sols)+1,opt[:-1],opt[-1]))
        if not cma_sols:
            tqdm.write("No CMA candidates found."); return None
        tqdm.write("Spawning parallel GD runs...")
        args = [(cid,phi,A,N_derivs,device) for cid,phi,A in cma_sols]
        results=[]
    with Pool(1) as pool:
        for run_id,res in tqdm(pool.imap_unordered(gd_task,args), total=len(args), desc='Overall GD', position=0):
            results.append((run_id,*res))
    tqdm.write("Results from all GD runs:")
    best=None
    for run_id,success,phi,A,F_v,values,nrm in sorted(results):
        status = f"F={F_v:.6f}, norm={nrm:.3e}, A={A:.6f}" if success else "Failed"
        tqdm.write(f"Run {run_id}: {status}")
        if success and (not best or nrm<best[-1]): best=(run_id,phi,A,F_v,values,nrm)
    if best:
        cid,phi_b,A_b,F_b,values_b,nrb = best
        phases = phi_b.tolist()+[0];
        phases = [round(p, 3) for p in phases]
        tqdm.write(f"Best run #{cid}: F={F_b:.6f}, norm={nrb:.3e}\nPhases={phases}\nArea={A_b:.6f}")
        tqdm.write(f"Multi-point values: {values_b}")
    return best

# ----------------------------------------------------------------------------
# 9) Execution
# ----------------------------------------------------------------------------
if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")
    hybrid_parallel(device=device, N_pulses=13, N_derivs=2, n_cand=36, fidelity_target=0.999995)
        

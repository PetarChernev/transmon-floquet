#!/usr/bin/env python3
"""
**OUTDATED: currently not updated to use floquet_propagator_square_rabi_one_period**

Hybrid optimization of a composite pulse sequence to approximate an arbitrary unitary
using a combination of CMA-ES and gradient descent.

Uses a Floquet-based approach to compute the unitary evolution of a transmon qubit
under a composite pulse sequence, and optimizes the fidelity of the resulting unitary
with respect to a target unitary.
This script is designed to run on a CUDA-capable GPU, but can also run on CPU
if CUDA is not available.
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

RUN_NAME = "H_5x3_L2_5000+10000 fidelity weight"
# Ensure CUDA-friendly multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

warnings.filterwarnings('ignore')


n_levels = 6
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)

energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)
energies = torch.tensor(energies, dtype=torch.float64)
couplings = torch.tensor(couplings, dtype=torch.complex128)



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
    
    with Pool(cpu_count()) as pool:
        while len(cma_sols) < n_cand and attempts < 3*n_cand:
            attempts +=1
            p0 = np.concatenate([np.random.random(N_pulses), np.random.random(N_pulses)])
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
        

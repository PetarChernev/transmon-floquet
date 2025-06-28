## Current Implementation

I have an imlementation of multi-chain SGLD. 


`sgld.py`
```
# Hybrid SGLD-GA optimization for pulse sequences
# SGLD for continuous parameters (rabi, phases)
# Genetic Algorithm for discrete parameters (pulse durations)

from cmath import sqrt
import torch
from torch import einsum
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from dataclasses import dataclass
from tqdm.auto import tqdm, trange

from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import (
    floquet_propagator_square_sequence_stroboscopic,
)
from transmon.transmon_floquet_propagator_parallel import floquet_propagator_square_sequence_stroboscopic_vectorized

##############################################################################
# fixed chip parameters (define once)
##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_levels = 6
# find EJ/EC ratio that gives target anharmonicity
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

energies = torch.tensor(energies, dtype=torch.float64, device=device)
couplings = torch.tensor(couplings, dtype=torch.complex128, device=device)

omega_d = 1.0  # nominal drive frequency, rad s^-1
floquet_cutoff: int = 25

U_TARGET = torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble, device=device) / sqrt(2)

TWO_PI = 2.0 * torch.pi

##############################################################################
# loss function
##############################################################################

def U_pulse_sequence(rabi, phases, pulse_durations_periods, omega=omega_d):
    return floquet_propagator_square_sequence_stroboscopic_vectorized(
        rabi,
        phases,
        pulse_durations_periods,
        energies,
        couplings,
        omega,
        floquet_cutoff,
    )

def propagator_loss(rabi, phases, pulse_durations_periods):
    """Loss function based on gate fidelity."""
    U = U_pulse_sequence(rabi, phases, pulse_durations_periods)
    
    # Project onto the first two levels
    M = U_TARGET.conj().T @ U[:2, :2]
    tr_MMdag = torch.trace(M @ M.conj().T)
    tr_M = torch.trace(M)
    fidelity = (tr_MMdag + torch.abs(tr_M) ** 2) / 6.0
    
    # Add penalty for leakage to higher states
    higher_state_loss = torch.sum(torch.abs(U[2:, :2])) * 0.1
    
    # Return loss (minimize 1 - fidelity)
    return (1.0 - fidelity.real) + higher_state_loss

##############################################################################
# SGLD implementation for continuous parameters only
##############################################################################

class SGLDChain:
    """Single SGLD chain with temperature control."""
    
    def __init__(self, pulse_durations, temperature, lr_base=1e-2):
        self.pulse_durations = pulse_durations
        self.n_pulses = len(pulse_durations)
        self.temperature = temperature
        self.lr_base = lr_base
        
        # Initialize continuous parameters only
        self.rabi = torch.nn.Parameter(
            torch.rand(self.n_pulses, dtype=torch.float64, device=device) * 2.0,
            requires_grad=True
        )
        self.phases = torch.nn.Parameter(
            torch.rand(self.n_pulses, dtype=torch.float64, device=device) * TWO_PI,
            requires_grad=True
        )
        
        # Track history
        self.loss_history = []
        self.param_history = []
        
    def step(self, clip_grad=5.0):
        """Perform one SGLD step."""

        # Zero gradients
        if self.rabi.grad is not None:
            self.rabi.grad.zero_()
        if self.phases.grad is not None:
            self.phases.grad.zero_()

        
        # Compute loss and gradients
        loss = propagator_loss(self.rabi, self.phases, self.pulse_durations)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([self.rabi, self.phases], clip_grad)
        
        # SGLD update with temperature-scaled noise
        lr = self.lr_base
        noise_scale = np.sqrt(2 * lr * self.temperature)
        
 
        with torch.no_grad():
            # Update rabi with noise
            self.rabi -= lr * self.rabi.grad
            self.rabi += noise_scale * torch.randn_like(self.rabi)
            
            # Update phases with noise
            self.phases -= lr * self.phases.grad
            self.phases += noise_scale * torch.randn_like(self.phases)
            
            # Wrap phases to [0, 2π]
            with torch.no_grad():
                self.phases[:] = self.phases % TWO_PI
        
        # Record history
        self.loss_history.append(loss.item())
        self.param_history.append({
            'rabi': self.rabi.detach().clone().cpu().numpy(),
            'phases': self.phases.detach().clone().cpu().numpy(),
            'loss': loss.item()
        })
        
        return loss.item()

class MultiModalSGLD:
    """Multi-chain SGLD for finding multiple local minima."""
    
    def __init__(self, pulse_durations, n_chains=20, temp_range=(0.001, 0.1), lr_base=1e-2):
        self.pulse_durations = torch.tensor(pulse_durations, dtype=torch.int64, device=device)
        self.temperatures = np.logspace(
            np.log10(temp_range[0]), 
            np.log10(temp_range[1]), 
            n_chains
        )
        self.chains = [SGLDChain(self.pulse_durations, temp, lr_base) for temp in self.temperatures]
        self.n_chains = n_chains
        self.unique_solutions = []
        
    def run(self, n_steps=1000, swap_interval=50, verbose=False):
        """Run SGLD exploration with a tqdm progress bar."""

        # --- change this single line ---------------------------------
        for step in trange(n_steps, desc="Overall SGLD"):          # ← remove “disable=…”
        # -------------------------------------------------------------

            # Update all chains
            losses = []
            for chain in self.chains:
                loss = chain.step()
                losses.append(loss)

            # Periodic swapping between adjacent chains
            if step % swap_interval == 0 and step > 0:
                self._try_swaps()

            # Optional textual log (unchanged)
            if verbose and step % 100 == 0:
                avg_loss = np.mean(losses)
                min_loss = np.min(losses)
                print(f"Step {step:5d}: avg_loss={avg_loss:.6f}, min_loss={min_loss:.6f}")

        # Collect and cluster final solutions
        self._cluster_solutions()


        
    def _try_swaps(self):
        """Try to swap parameters between adjacent temperature chains."""
        for i in range(self.n_chains - 1):
            chain1, chain2 = self.chains[i], self.chains[i + 1]
            
            # Compute losses
            loss1 = propagator_loss(chain1.rabi, chain1.phases, self.pulse_durations).item()
            loss2 = propagator_loss(chain2.rabi, chain2.phases, self.pulse_durations).item()
            
            # Metropolis criterion for swap
            delta = (loss2 - loss1) * (1/chain1.temperature - 1/chain2.temperature)
            
            if delta < 0 or np.random.rand() < np.exp(-delta):
                # Swap parameters
                chain1.rabi.data, chain2.rabi.data = chain2.rabi.data.clone(), chain1.rabi.data.clone()
                chain1.phases.data, chain2.phases.data = chain2.phases.data.clone(), chain1.phases.data.clone()
    
    def _cluster_solutions(self, eps=0.5, min_samples=3):
        """Cluster solutions to identify unique local minima."""
        all_solutions = []
        all_losses = []
        
        for chain in self.chains:
            # Take last 20% of history (after burn-in)
            n_samples = len(chain.param_history)
            start_idx = int(0.8 * n_samples)
            
            for params in chain.param_history[start_idx:]:
                # Concatenate rabi and phases for clustering
                solution = np.concatenate([params['rabi'], params['phases']])
                all_solutions.append(solution)
                all_losses.append(params['loss'])
        
        if not all_solutions:
            return
            
        all_solutions = np.array(all_solutions)
        all_losses = np.array(all_losses)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_solutions)
        
        # Extract unique solutions (cluster centers)
        unique_clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # -1 means noise
                unique_clusters[label].append((all_solutions[i], all_losses[i]))
        
        # Get best solution from each cluster
        self.unique_solutions = []
        for label, solutions in unique_clusters.items():
            # Find solution with lowest loss in cluster
            solutions.sort(key=lambda x: x[1])
            best_sol, best_loss = solutions[0]
            
            # Split back into rabi and phases
            n_pulses = len(self.pulse_durations)
            rabi = best_sol[:n_pulses]
            phases = best_sol[n_pulses:]
            
            self.unique_solutions.append({
                'rabi': rabi,
                'phases': phases,
                'loss': best_loss,
                'cluster_size': len(solutions)
            })
        
        # Sort by loss
        self.unique_solutions.sort(key=lambda x: x['loss'])
        
    def get_fitness_metrics(self):
        """Compute fitness metrics for the duration configuration."""
        if not self.unique_solutions:
            return {
                'best_loss': float('inf'),
                'n_unique_minima': 0,
                'convergence_rate': 0.0,
                'avg_final_loss': float('inf'),
                'loss_diversity': 0.0
            }
        
        # Best loss achieved
        best_loss = self.unique_solutions[0]['loss']
        
        # Number of unique minima found
        n_unique_minima = len(self.unique_solutions)
        
        # Convergence rate (average improvement over time)
        all_improvements = []
        for chain in self.chains:
            if len(chain.loss_history) > 10:
                initial_loss = np.mean(chain.loss_history[:10])
                final_loss = np.mean(chain.loss_history[-10:])
                improvement = (initial_loss - final_loss) / initial_loss
                all_improvements.append(improvement)
        convergence_rate = np.mean(all_improvements) if all_improvements else 0.0
        
        # Average final loss across all chains
        final_losses = [chain.loss_history[-1] for chain in self.chains if chain.loss_history]
        avg_final_loss = np.mean(final_losses) if final_losses else float('inf')
        
        # Diversity of solutions (std of unique minima losses)
        unique_losses = [sol['loss'] for sol in self.unique_solutions]
        loss_diversity = np.std(unique_losses) if len(unique_losses) > 1 else 0.0
        
        return {
            'best_loss': best_loss,
            'n_unique_minima': n_unique_minima,
            'convergence_rate': convergence_rate,
            'avg_final_loss': avg_final_loss,
            'loss_diversity': loss_diversity
        }

```

`transmon_floquet_propagator_parallel.py`
```
import numpy as np
import torch

from typing import Optional, Sequence, Union

from transmon.transmon_dynamics_qutip import pulse_sequence_qutip

def floquet_propagators_square_rabi(
    rabi_frequencies: torch.Tensor,
    phases: torch.Tensor,
    energies: torch.Tensor, 
    couplings: torch.Tensor,
    omega_d: torch.Tensor,
    floquet_cutoff: int
) -> torch.Tensor:
    """
    Builds the propagators for a sequence period of the cosine drive with frequency omega_d
    Parameters:
        rabi_frequencies: Tensor of Rabi frequencies for each pulse (shape (n,)).
        phases: Tensor of phases for each pulse (shape (n,)).
        energies: Tensor of energies of the system (shape (d,)).
        couplings: Tensor of couplings between states (shape (d, d)).
        omega_d: Drive frequency (scalar or tensor).
        floquet_cutoff: Fourier cutoff used in Floquet formalism M.
    Returns:
        A PyTorch tensor representing the total propagator after applying all pulses in sequence.
        (shape (n, d, d), where n is the number of pulses and d is the dimension of the physical Hilbert space).
    """
    H_Fs = build_floquet_hamiltonians(
        rabi_frequencies, 
        phases,
        energies, 
        couplings,
        omega_d,
        floquet_cutoff
    )
    return get_physical_propagators(H_Fs, floquet_cutoff, omega_d)


# --- small helper -----------------------------------------------------------------
def _batched_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker product between a 2-D matrix A (S×S) and a
    *batch* of square matrices B (n×d×d).

    Returns
    -------
    Tensor with shape (n, S*d, S*d):
        kron(A, B[i]) for each i.
    """
    S = A.size(0)
    d = B.size(-1)
    # (n, S, S, d, d)
    K = (A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
         * B.unsqueeze(1).unsqueeze(2))
    # reorder so the (row-block,row-intra, col-block,col-intra) axes are adjacent
    return K.permute(0, 1, 3, 2, 4).reshape(B.size(0), S * d, S * d)


# --- main routine -----------------------------------------------------------------
def build_floquet_hamiltonians(
    rabi_frequencies: torch.Tensor,    # (n,)
    phases:           torch.Tensor,    # (n,)
    energies:         torch.Tensor,    # (d,)
    couplings:        torch.Tensor,    # (d,d)
    omega_d:          float | torch.Tensor,
    M:                int,
) -> torch.Tensor:
    """
    Fully-vectorised construction of Floquet Hamiltonians for a *large*
    Fourier cut-off `M`, using only GPU tensor ops (no Python loops).

    Output: (n, (2*M+1)*d, (2*M+1)*d)
    """
    # --------------------------------------------------------------------------
    device  = energies.device
    d       = energies.numel()
    n_pulse = rabi_frequencies.numel()
    S       = 2 * M + 1                      # number of Fourier blocks
    N       = S * d                          # full Floquet dimension

    cdtype = couplings.dtype
    energies  = energies.to(cdtype)
    couplings = couplings.to(cdtype)

    # --------------------------------------------------------------------------
    # pulse-dependent block building bricks
    C0 = torch.diag(energies).to(cdtype)                 # (d,d) – same for every pulse

    rabi   = 0.5 * rabi_frequencies.to(energies.real.dtype).view(n_pulse, 1, 1)
    e_iphi = torch.exp(1j * phases.to(energies.real.dtype)).view(n_pulse, 1, 1)

    C1  = (rabi * couplings * e_iphi).to(cdtype)         # (n,d,d)
    Cm1 = (rabi * couplings * e_iphi.conj()).to(cdtype)  # (n,d,d)

    # --------------------------------------------------------------------------
    # matrices that act in the Fourier index space  (size S×S)
    Id_S     = torch.eye(S,  dtype=cdtype, device=device)                    # Iₛ
    diag_m   = torch.diag(torch.arange(-M, M + 1, device=device)).to(cdtype) # diag(m)
    sub_diag = torch.diag(torch.ones(S - 1, device=device, dtype=cdtype), -1)  # idx = +1
    sup_diag = torch.diag(torch.ones(S - 1, device=device, dtype=cdtype),  1)  # idx = −1

    # --------------------------------------------------------------------------
    # terms that are identical for all pulses
    H_static = (
        torch.kron(Id_S, C0) +
        torch.kron(diag_m * omega_d, torch.eye(d, dtype=cdtype, device=device))
    )                                   # (N,N)  – no pulse index yet

    # --------------------------------------------------------------------------
    # pulse-dependent super- and sub-diagonal terms (batched Kronecker products)
    H_sub   = _batched_kron(sub_diag, C1)   # (n,N,N)  idx = +1  (row > col)
    H_super = _batched_kron(sup_diag, Cm1)  # (n,N,N)  idx = −1

    # --------------------------------------------------------------------------
    # final assembly, broadcasting the static part across pulses
    H_F = H_sub + H_super                                   # (n,N,N)
    H_F += H_static                                         # broadcast add

    return H_F



def get_physical_propagators(
    H_F:           torch.Tensor,        # (..., N, N)
    floquet_cutoff: int,                #  M
    omega_d:       Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the physical propagator U(T,0) from one **or a batch** of
    Floquet Hamiltonians – entirely on the GPU and without looping over m.

    Parameters
    ----------
    H_F : (..., N, N) complex tensor
        Floquet Hamiltonian(s); the leading dimensions (if any) are treated
        as batch dimensions.
    floquet_cutoff : int
        Fourier cut-off M used to build H_F  ⇒  N = (2 M+1) d.
    omega_d : float or (broadcastable) tensor
        Drive frequency ω_d (T = 2π / ω_d).

    Returns
    -------
    U_phys : (..., d, d) complex tensor
        Physical propagator(s) at one drive period T.
    """
    # ------------ sizes ----------------------------------------------------
    S = 2 * floquet_cutoff + 1               # number of Fourier blocks
    N = H_F.shape[-1]
    d = N // S                               # physical Hilbert-space dimension

    # ------------ Floquet evolution operator --------------------------------
    T_period = 2 * torch.pi / omega_d
    U_F = torch.matrix_exp(-1j * H_F * T_period)      # same batch shape as H_F

    # ------------ reshape into 5-D block structure --------------------------
    # (..., S_row, d_row, S_col, d_col)
    blocks = U_F.reshape(*U_F.shape[:-2], S, d, S, d)

    # ------------ pick column block m' = 0  (index M) -----------------------
    col0_blocks = blocks[..., :, :, floquet_cutoff, :]   # (..., S, d, d)

    # ------------ sum over the row-block index m ----------------------------
    U_phys = col0_blocks.sum(dim=-3)                     # (..., d, d)

    return U_phys


# --------------------------------------------------------------------------- #
# Helper: batched exponentiation-by-squaring for a *different* power per item #
# --------------------------------------------------------------------------- #
def _batch_matrix_power(mats: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    n, d, _ = mats.shape
    result  = torch.eye(d, dtype=mats.dtype, device=mats.device)\
                  .expand(n, d, d).clone()
    base    = mats.clone()
    exp     = exps.clone()

    while torch.any(exp):
        odd_mask = (exp & 1).bool()
        if odd_mask.any():
            result[odd_mask] = torch.bmm(base[odd_mask], result[odd_mask])
        exp >>= 1
        if torch.any(exp):
            base = torch.bmm(base, base)
    return result
# ---------------------------------------------------------------------------------

def floquet_propagator_square_sequence_stroboscopic_vectorized(
    rabi_frequencies:        torch.Tensor,   # (n,)
    phases:                  torch.Tensor,   # (n,)
    pulse_durations_periods: torch.Tensor,   # (n,) ints
    energies:                torch.Tensor,   # (d,)
    couplings:               torch.Tensor,   # (d,d)
    omega_d:                 Union[float, torch.Tensor],
    floquet_cutoff:          int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Vectorised total propagator for a stroboscopic square-pulse sequence.
    Now handles the single-pulse case without the batched helpers.
    """
    if device is None:
        device = energies.device

    # ---- build the single-period propagator(s) ---------------------------------
    U_single = floquet_propagators_square_rabi(
        rabi_frequencies.to(device),
        phases.to(device),
        energies.to(device),
        couplings.to(device),
        omega_d,
        floquet_cutoff,
    )                                         # shape (n, d, d)

    n, d, _ = U_single.shape

    # ---- special-case: only one pulse ------------------------------------------
    if n == 1:
        return torch.matrix_power(U_single[0], int(pulse_durations_periods.item()))

    # ---- general batched case --------------------------------------------------
    U_powered = _batch_matrix_power(U_single, pulse_durations_periods)   # (n, d, d)

    # chain product in physical order: last pulse acts first
    total_U = torch.linalg.multi_dot(U_powered.flip(0).unbind(0))

    return total_U

```


## The Problem

Currently there is a large amount of overhead due to GPU-CPU syncronization - the chains are evaluated sequentially, losses and param values are synced to CPU at every iteration.

## Proposed solution

I want to rework this code so that we compute all the chains in parallel on the GPU by using batches, so instead of individual calls with 1d tensors for the rabi frequencies and phases, we have 1 call with 2d tensors which return the loss for all chains simultaniously.

Here's an overview of the needed steps that have been identified:


## Road-map of required changes

### 1 (done)  Vectorise the propagator code one level higher

```
def floquet_propagator_square_sequence_batch(
    rabi:                    torch.Tensor,   # (B, P)
    phases:                  torch.Tensor,   # (B, P)
    pulse_durations_periods: torch.Tensor,   # (P,)  – identical for all chains
    energies:                torch.Tensor,   # (d,)
    couplings:               torch.Tensor,   # (d,d)
    omega_d:                 Union[float, torch.Tensor],
    floquet_cutoff:          int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Vectorised total propagator U(T_total, 0) for a *batch* of B chains,
    each consisting of P square pulses that repeat for an integer number
    of drive periods.

    Parameters
    ----------
    rabi, phases : (B, P)  – per–chain continuous parameters
    pulse_durations_periods : (P,)  – **same** integer duration per pulse
    energies, couplings, omega_d, floquet_cutoff : as before
    device : optional override; defaults to `energies.device`

    Returns
    -------
    U_total : (B, d, d)  – physical propagator of every chain
    """
    # ----------------------- housekeeping ----------------------------------
    if device is None:
        device = energies.device

    rabi   = rabi.to(device)
    phases = phases.to(device)
    durations = pulse_durations_periods.to(device, dtype=torch.long)

    B, P = rabi.shape
    d    = energies.numel()

    # ----------------------- single-period propagators ---------------------
    # Flatten (B,P) → (B*P,) and build all Floquet propagators in one call
    U_single_flat = floquet_propagators_square_rabi(
        rabi.reshape(-1),
        phases.reshape(-1),
        energies,
        couplings,
        omega_d,
        floquet_cutoff,
    )                                           # (B*P, d, d)

    # ----------------------- raise to integer powers -----------------------
    # Repeat durations for every chain, flatten again for the helper
    exps_flat = durations.expand(B, P).reshape(-1)           # (B*P,)

    U_powered_flat = _batch_matrix_power(                    # (B*P, d, d)
        U_single_flat, exps_flat
    )
    
    # ----------------------- chain multiplication --------------------------
    if P == 1:
        return U_powered_flat.view(B, d, d)     # (B, d, d)

    # reverse pulse order: last pulse acts first
    U_rev = U_powered_flat.reshape(B, P, d, d).flip(1)   # (B, P, d, d)

    total_U = torch.eye(d, dtype=U_rev.dtype, device=device) \
                  .expand(B, d, d).clone()

    for p in range(P):            # multiply **on the right**
        total_U = torch.bmm(total_U, U_rev[:, p])

    return total_U
```

### 2   Make one big `nn.Parameter` per variable class

```python
class BatchedChains(torch.nn.Module):
    def __init__(self, n_chains, n_pulses, temperatures, lr):
        super().__init__()
        self.rabi   = torch.nn.Parameter(torch.rand(n_chains, n_pulses, dtype=fp64))
        self.phases = torch.nn.Parameter(torch.rand(n_chains, n_pulses, dtype=fp64) * 2π)
        self.temperatures = temperatures.view(-1, 1)              # (B,1)
        self.lr = lr
```

All further maths is plain tensor algebra; the per-chain operations are
just broadcasts along dim 0.

### 3   Compute the loss for *all* chains at once

```python
U  = floquet_propagator_square_sequence_batch(...)
M  = torch.einsum('bij,ij->bij', U[:, :2, :2], U_TARGET.conj().T)
losses = 1 - fidelity(U) + leakage_penalty(U)          # (B,)
total_loss = losses.sum()
total_loss.backward()
```

This single backward pass produces `rabi.grad` and `phases.grad`
of shape `(B, P)` – exactly what you need.

### 4   Vectorised SGLD update

```python
eta = torch.randn_like(self.rabi)                      # (B,P)
noise_scale = (2 * self.lr * self.temperatures).sqrt() # (B,1)
with torch.no_grad():
    self.rabi   -= self.lr * self.rabi.grad + noise_scale * eta
    self.phases -= self.lr * self.phases.grad + noise_scale * torch.randn_like(self.phases)
    self.phases %= 2*π
```

### 5   Chain swapping without sync

* keep the last `losses` tensor;
* perform swaps by indexing:

  ```python
  swap_mask = (torch.rand(B-1, device=losses.device) < swap_probs)
  idx = torch.arange(B, device=losses.device)
  idx[:-1][swap_mask], idx[1:][swap_mask] = idx[1:][swap_mask], idx[:-1][swap_mask]
  self.rabi   = self.rabi[idx]
  self.phases = self.phases[idx]
  losses      = losses[idx]
  ```

No `.item()`, no `.cpu()`.

### 6   Logging without blocking

* Keep rolling buffers on GPU:

  ```python
  self.loss_history[:, step] = losses      # pre-allocated (B, n_steps)
  ```

* Move to CPU **after** training, or every 500 steps if you must.


## Task

Your task is to implement step 1. of this procedure. Implement the `floquet_propagator_square_sequence_batch` function. Can it be done without modifying any of the existing code? If so, please do it this way. If not, you are free to modify the existing code as well to facilitate the batch computation.


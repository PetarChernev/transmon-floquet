from cmath import sqrt
import time
import numpy as np
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import torch
from sklearn.cluster import DBSCAN
from tqdm.auto import trange

from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator_parallel import (
    floquet_propagator_square_sequence_stroboscopic_vectorized,
    floquet_propagator_square_sequence_batch,  # <— NEW
)

# -----------------------------------------------------------------------------
# Fixed chip parameters (define once)
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_levels: int = 6
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, couplings = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

dtype_real = torch.float32
dtype_complex = torch.complex64

energies = torch.tensor(energies, dtype=dtype_real, device=device)
couplings = torch.tensor(couplings, dtype=dtype_complex, device=device)

omega_d: float = 1.0  # nominal drive frequency, rad s⁻¹
floquet_cutoff: int = 25

U_TARGET = (
    torch.tensor([[1, 1], [1, -1]], dtype=dtype_complex, device=device) / sqrt(2)
)

TWO_PI = 2.0 * torch.pi



def propagator_loss_batch(
    rabi: torch.Tensor,  # (B,P)
    phases: torch.Tensor,  # (B,P)
    pulse_durations_periods: torch.Tensor,  # (P,)
) -> torch.Tensor:  # (B,)
    """Vectorised loss for a batch of *B* chains."""

    U = floquet_propagator_square_sequence_batch(
        rabi,
        phases,
        pulse_durations_periods,
        energies,
        couplings,
        omega_d,
        floquet_cutoff,
    )  # (B,d,d)

    U_2 = U[:, :2, :2]  # first two levels (B,2,2)

    # M_b = U†_target · U_2  – broadcasting over batch dim
    M = torch.matmul(U_TARGET.conj().T, U_2)  # (B,2,2)

    tr_MMdag = torch.einsum("bij,bji->b", M, M.conj())  # (B,)
    tr_M = torch.einsum("bii->b", M)  # (B,)
    fidelity = (tr_MMdag + torch.abs(tr_M) ** 2) / 6.0  # (B,)

    # Leakage: everything outside the computational sub‑space
    higher_state_loss = torch.sum(torch.abs(U[:, 2:, :2]), dim=(-2, -1)) * 0.1

    return (1.0 - fidelity.real) + higher_state_loss  # (B,)


# =============================================================================
#  B A T C H E D   S G L D   C H A I N S
# =============================================================================

class BatchedChains(torch.nn.Module):
    """Maintain (B,P) parameter matrices and perform one SGLD step."""

    def __init__(
        self,
        pulse_durations: torch.Tensor,  # (P,) int
        temperatures: Sequence[float] | torch.Tensor,  # len = B
        lr_base: float = 1e-2,
    ) -> None:
        super().__init__()

        if not isinstance(temperatures, torch.Tensor):
            temperatures = torch.tensor(temperatures, dtype=dtype_real, device=device)
        self.register_buffer("temperatures", temperatures.view(-1, 1))
        self.register_buffer("pulse_durations", pulse_durations)


        B = self.temperatures.shape[0]
        P = self.pulse_durations.numel()

        # Continuous parameters (shared learnable tensors)
        self.rabi = torch.nn.Parameter(torch.rand(B, P, dtype=dtype_real, device=device) * 2.0)
        self.phases = torch.nn.Parameter(
            torch.rand(B, P, dtype=dtype_real, device=device) * TWO_PI
        )

        self.lr_base = lr_base

        # Lightweight logging (lists of numpy arrays)
        self.loss_history: list[np.ndarray] = []  # one (B,) vector per step

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _wrap_phases(self):
        self.phases %= TWO_PI

    # ---------------------------------------------------------------------
    def step(self, clip_grad: float = 5.0) -> torch.Tensor:
        """One simultaneous SGLD update for *all* chains.

        Returns
        -------
        losses : (B,) tensor – snapshot before the update (detached).
        """
        # -------- forward & backward ------------------------------------
        losses = propagator_loss_batch(
            self.rabi, self.phases, self.pulse_durations
        )  # (B,)

        total_loss = losses.sum()

        # clear stale gradients (if any)
        if self.rabi.grad is not None:
            self.rabi.grad.zero_()
        if self.phases.grad is not None:
            self.phases.grad.zero_()
        total_loss.backward()     

        torch.nn.utils.clip_grad_norm_([self.rabi, self.phases], clip_grad)

        # -------- SGLD parameter update ---------------------------------

        with torch.no_grad():
            # SGLD drift
            self.rabi   -= self.lr_base * self.rabi.grad
            self.phases -= self.lr_base * self.phases.grad

            # SGLD diffusion (temperature-dependent)
            noise_scale = torch.sqrt(2.0 * self.lr_base * self.temperatures)   # (B,1)
            self.rabi   += noise_scale * torch.randn_like(self.rabi)
            self.phases += noise_scale * torch.randn_like(self.phases)

            # keep parameters in bounds
            self.rabi.clamp_(0.0, 4.0)           # avoid runaway amplitudes

            # -------- minimal logging ---------------------------------------
        self.loss_history.append(losses.detach().cpu().numpy())
        return losses.detach()  # (B,)


# =============================================================================
#  B A T C H E D   M U L T I ‑ M O D A L   S G L D
# =============================================================================

class BatchedMultiModalSGLD:
    """Parallel‑tempering SGLD implemented with a single batched module."""

    def __init__(
        self,
        pulse_durations: Sequence[int],
        n_chains: int = 20,
        temp_range: Tuple[float, float] = (0.001, 0.1),
        lr_base: float = 1e-2,
    ) -> None:
        self.pulse_durations = torch.tensor(pulse_durations, device=device)

        self.temperatures_np = np.logspace(np.log10(temp_range[0]), np.log10(temp_range[1]), n_chains)
        self.model = BatchedChains(self.pulse_durations, self.temperatures_np, lr_base)
        self.n_chains = n_chains

        # bookkeeping for swaps
        self._temperature_tensor = self.model.temperatures.squeeze(1)  # (B,)

        # cluster analysis containers (filled after `run`)
        self.unique_solutions: list[dict] = []

    # ---------------------------------------------------------------------
    def run(self, n_steps: int = 1000, swap_interval: int = 50, verbose: bool = False):
        for step in trange(n_steps, desc="Batched SGLD"):
            losses = self.model.step()  # (B,)

            if step % swap_interval == 0 and step > 0:
                self._try_swaps(losses)

            if verbose and step % 100 == 0:
                print(
                    f"Step {step:5d}: avg_loss={losses.mean():.6f}, min_loss={losses.min():.6f}"
                )

        self._cluster_solutions()

    # ---------------------------------------------------------------------
    def _try_swaps(self, losses: torch.Tensor):
        """Metropolis swaps between adjacent temperature chains (vectorised)."""
        temp_inv = 1.0 / self._temperature_tensor  # (B,)

        # Δ = (L_{i+1} − L_i) (1/T_i − 1/T_{i+1})
        delta = (losses[1:] - losses[:-1]) * (temp_inv[:-1] - temp_inv[1:])
        swap_mask = (delta < 0) | (torch.rand_like(delta) < torch.exp(-delta))

        if swap_mask.any():
            idx = torch.arange(self.n_chains, device=device)
            swap_idx = torch.nonzero(swap_mask, as_tuple=False).flatten()
            for i in swap_idx:
                idx[i], idx[i + 1] = idx[i + 1], idx[i]

            # re‑index parameters and temperatures *in‑place*
            self.model.rabi.data = self.model.rabi.data[idx]
            self.model.phases.data = self.model.phases.data[idx]
            self.model.temperatures = self.model.temperatures[idx]
            self._temperature_tensor = self._temperature_tensor[idx]

    # ---------------------------------------------------------------------
    def _cluster_solutions(self, eps: float = 0.5, min_samples: int = 3):
        """Very lightweight clustering: we take the **final** parameters only."""
        final_rabi = self.model.rabi.detach().cpu().numpy()   # (B,P)
        final_phas = self.model.phases.detach().cpu().numpy()  # (B,P)
        final_loss = np.stack(self.model.loss_history)[-1]     # (B,)

        all_solutions = np.hstack([final_rabi, final_phas])    # (B,2P)
        normalized_solutions = np.hstack([
            (final_rabi   - final_rabi.mean(0)) / final_rabi.std(0),
            (final_phas   - final_phas.mean(0)) / final_phas.std(0)
        ])
        clustering = DBSCAN(eps=1.5, min_samples=2).fit(normalized_solutions)

        clusters: dict[int, list[Tuple[np.ndarray, float]]] = defaultdict(list)
        for sol, loss, label in zip(all_solutions, final_loss, clustering.labels_):
            if label >= 0:
                clusters[label].append((sol, float(loss)))

        self.unique_solutions = []
        P = len(self.pulse_durations)
        for label, sol_list in clusters.items():
            sol_list.sort(key=lambda t: t[1])   # by loss
            best_sol, best_loss = sol_list[0]
            self.unique_solutions.append({
                "rabi": best_sol[:P],
                "phases": best_sol[P:],
                "loss": best_loss,
                "cluster_size": len(sol_list),
            })
        self.unique_solutions.sort(key=lambda d: d["loss"])

    # ---------------------------------------------------------------------
    def get_best_solution(self):
        if not self.unique_solutions:
            return None
        return self.unique_solutions[0]


if __name__ == "__main__":
    # Example usage
    pulse_durations = [5, 10, 15, 20, 5, 10, 15, 20]  # example durations
    sgld = BatchedMultiModalSGLD(
        pulse_durations=pulse_durations,
        n_chains=15,
        temp_range=(0.001, 0.1),
        lr_base=1e-2,
    )
    sgld.run(n_steps=500, swap_interval=50, verbose=True)

    best_solution = sgld.get_best_solution()
    print("Best solution found:", best_solution)
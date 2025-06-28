"""robust_pulse_optimizer.py

Refactored module for robust single‑qubit gate optimisation on a transmon
with **adaptive precision** and **restart strategies**.

The optimisation logic is split into two layers:

* **`_PulseGDRun`** – self‑contained gradient‑descent engine that starts from a
  given parameter set and runs until one of the following conditions holds:

  * **single‑precision phase:** *switch‑criterion* **OR** plateau reached
  * **double‑precision phase:** all detunings meet `fidelity_target` **OR** plateau reached

  On exit it returns the best loss, associated fidelity/leak vectors, final
  parameters, whether convergence was achieved, and whether precision was
  upgraded.

* **`RandomRobustPulseOptimizer`** – high‑level scheduler that repeatedly draws
  **random** pulse durations / Rabi amplitudes / phases, launches a
  `_PulseGDRun`, and stops as soon as a run converges.  Its public API is
  identical to the former `RobustPulseOptimizer`.

Example
-------
```python
from robust_pulse_optimizer import RandomRobustPulseOptimizer

opt = RandomRobustPulseOptimizer(
        n_pulses=8,
        detunings=[0.95, 0.98, 1.00, 1.02, 1.05],
        fidelity_target=0.9999,
        plateau_patience=150,
    )

best = opt.run(max_steps_per_restart=10_000)
print(best["fidelities"], best["double_precision"])
```
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Dict

import torch
from torch import Tensor, einsum
from tqdm.auto import tqdm

# Local modules ----------------------------------------------------------------
from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator_parallel import (
    floquet_propagator_square_sequence_omega_batch,
)

__all__ = ["RandomRobustPulseOptimizer"]

################################################################################
# Gradient‑descent engine – single run, no restarts
################################################################################
class _PulseGDRun:
    """Encapsulates a *single* gradient‑descent trajectory.

    The run obeys the state machine described in the module doc‑string:

        single‑precision → (switch OR plateau) → double‑precision → (success OR plateau)
    """

    def __init__(
        self,
        parent: "_BasePulseOptimizer",
        rabi: Tensor,
        phases: Tensor,
        pulse_durations: Tensor,
        max_steps: int,
        desc: str = "",
    ) -> None:
        self.p = parent  # shortcut to configuration/state in the parent optimiser
        self.rabi = rabi
        self.phases = phases
        self.pulse_durations = pulse_durations
        self.max_steps = max_steps
        self.desc = desc or "GD run"

        # optimiser / scheduler are created here so that they survive precision switch
        self.opt, self.sched = self._new_optimisers()

        # bookkeeping
        self.best_loss = float("inf")
        self.best_stats: List[Tuple[float, float]] | None = None
        self.best_rabi: Tensor | None = None
        self.best_phases: Tensor | None = None
        self.best_precision_double = False

    # ------------------------------------------------------------------ public
    def run(self) -> Dict:
        plateau_counter, prev_loss = 0, float("inf")
        bar = tqdm(range(self.max_steps), desc=self.desc, leave=False, position=0)   
        info_bar = tqdm(total=0, position=1, bar_format='{desc}', leave=False)          # line below the main bar
        for step in bar:
            # forward & backward -------------------------------------------------
            self.opt.zero_grad()
            loss, stats, propagators = self._loss_and_stats()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.rabi, self.phases], 5.0)
            self.opt.step()
            self.sched.step(loss)

            # update best -------------------------------------------------------
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_stats = stats
                self.best_rabi = self.rabi.detach().clone()
                self.best_phases = self.phases.detach().clone()
                self.best_precision_double = self.p._precision_switched

            # ---------- adaptive plateau bookkeeping -------------------------
            central_idx = self.p._central_idx
            fid_central = 1.0 - stats[central_idx][1]
            # aggressive far from 1, gentle near 1
            patience_eff = max(20, int(self.p.plateau_patience / (1 - fid_central + 1e-3)))
            if not self.p._precision_switched:
                # double‑precision phase: plateau patience is disabled
                patience_eff = min(300, patience_eff)
            tol_eff      = self.p.plateau_tol * (1 + loss.item()) * (1 + fid_central)     

            improvement = prev_loss - loss.item()
            plateau_counter = plateau_counter + 1 if improvement < tol_eff else 0
            prev_loss = loss.item()


            # fidelity of central point for bar ---------------------------------
            central_idx = self.p._central_idx
            fid_central = 1.0 - stats[central_idx][1]
            bar.set_postfix(loss=f"{self.best_loss:.3e}", F_central=f"{fid_central:.4f}")      # keep bar compact
            info_bar.display(f"tol={tol_eff:.2e}  patience_eff={patience_eff}  plateau={plateau_counter}\r")  # prints on its own line

            # ----- control flow ------------------------------------------------
            if not self.p._precision_switched:
                # single‑precision phase
                if self.p._should_switch_precision(stats, propagators):
                    self._switch_to_double()
                    plateau_counter = 0  # reset
                    bar.write("↻ Switched to double precision …")
                    continue  # continue optimisation in higher precision
                if plateau_counter >= patience_eff:
                    break  # plateau – give up this run
            else:
                # double‑precision phase
                if self.p._success(stats):
                    return self._finish(converged=True, step=step)
                if plateau_counter >= patience_eff:
                    break  # plateau – give up this run
            # end for step
        return self._finish(converged=False, step=self.max_steps)

    # ---------------------------------------------------------------- private
    def _finish(self, converged: bool, step: int) -> Dict:
        leak_tensor = torch.tensor([s[0] for s in self.best_stats])
        F_tensor = 1.0 - torch.tensor([s[1] for s in self.best_stats])
        return {
            "converged": converged,
            "loss": self.best_loss,
            "leak": leak_tensor.tolist(),
            "fidelities": F_tensor.tolist(),
            "rabi": self.best_rabi.cpu().tolist(),
            "phases": self.best_phases.cpu().tolist(),
            "pulse_durations": self.pulse_durations.cpu().tolist(),
            "double_precision": self.best_precision_double,
            "steps": step,
        }

    def _loss_and_stats(self):
        return self.p._loss_and_stats(self.rabi, self.phases, self.pulse_durations)

    def _switch_to_double(self):
        self.rabi, self.phases = self.p._switch_to_double(self.rabi, self.phases)
        self.opt, self.sched = self._new_optimisers()

    def _new_optimisers(self):
        opt = torch.optim.Adam([self.rabi, self.phases], lr=self.p.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=self.p.plateau_patience // 4,
            cooldown=10,
        )
        return opt, sched

################################################################################
# Base class – shared math utilities / configuration
################################################################################
@dataclass
class _BasePulseOptimizer:
    """Holds all shared configuration and helper methods."""

    # ===== Model & target ====================================================
    n_levels: int = 6
    anharmonicity: float = -0.0429
    U_target: Tensor = field(
        default_factory=lambda: torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / torch.sqrt(torch.tensor(2.0))
    )

    # ===== Pulse‑sequence ====================================================
    pulse_length_options: Sequence[int] = tuple(range(0, 11))
    pulse_length_middle = 5
    n_pulses: int = 13

    # ===== Robustness settings ===============================================
    detunings: Sequence[float] = (0.95, 0.98, 1.0, 1.02, 1.05)
    weights: Sequence[float] = (1.0, 10.0, 100.0, 10.0, 1.0)
    fidelity_target: float = 0.999
    n_derivs: int = 0
    deriv_weights: Sequence[float] = (0.1, 1e-4)

    # ===== Optimisation hyper‑params =========================================
    plateau_patience: int = 100
    plateau_tol: float = 1e-6
    lr: float = 1e-2

    # ===== Runtime / HW ======================================================
    device: torch.device | str = ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_real: torch.dtype = torch.float32
    dtype_complex: torch.dtype = torch.complex64
    floquet_cutoff: int = 20
    seed: int | None = None

    auto_precision: bool = True

    # ===== internal ----------------------------------------------------------
    energies: Tensor = field(init=False, repr=False)
    couplings: Tensor = field(init=False, repr=False)
    _precision_switched: bool = field(init=False, default=False, repr=False)
    _central_idx: int = field(init=False, repr=False)

    # ---------------------------------------------------------------- public
    def _prepare_shared(self):
        # RNG setup
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        # transmon parameters
        ej_ec = TransmonCore.find_EJ_EC_for_anharmonicity(self.anharmonicity)
        energies, couplings = TransmonCore.compute_transmon_parameters(
            self.n_levels, n_charge=30, EJ_EC_ratio=ej_ec
        )
        self.energies = torch.tensor(energies, dtype=self.dtype_real, device=self.device)
        self.couplings = torch.tensor(couplings, dtype=self.dtype_complex, device=self.device)
        # U_target dtype/device
        self.U_target = self.U_target.to(self.dtype_complex).to(self.device)
        # central idx helper
        self._central_idx = list(self.detunings).index(1.0) if 1.0 in self.detunings else len(self.detunings) // 2

    # ---------------------------------------------------------------- math utils
    def _loss_and_stats(self, rabi: Tensor, phases: Tensor, pulse_durations: Tensor):
        detunings_t = torch.tensor(self.detunings, dtype=self.dtype_real, device=self.device)
        propagators = floquet_propagator_square_sequence_omega_batch(
            rabi,
            phases,
            pulse_durations,
            self.energies,
            self.couplings,
            detunings_t,
            self.floquet_cutoff,
        )
        F = self._fidelity(propagators[:, :2, :2])
        leak = propagators[:, 2:, :2].abs().sum(dim=(-2, -1))
        infid = 1.0 - F
        weights_t = torch.tensor(self.weights, dtype=self.dtype_real, device=self.device)
        total_loss = torch.dot(weights_t, leak) + torch.dot(weights_t, infid)
        if self.n_derivs >= 1:
            (dF_dw,) = torch.autograd.grad(F, self._omegas_detunings_tensor(), grad_outputs=torch.ones_like(F), create_graph=True)
            total_loss += self.deriv_weights[0] * torch.dot(weights_t, dF_dw)
        if self.n_derivs >= 2:
            (d2F_dw2,) = torch.autograd.grad(dF_dw, self._omegas_detunings_tensor(), grad_outputs=torch.ones_like(dF_dw), create_graph=True)
            total_loss += self.deriv_weights[1] * torch.dot(weights_t, d2F_dw2)
        stats = [(leak[i].item(), infid[i].item()) for i in range(leak.numel())]
        return total_loss, stats, propagators.detach()

    def _fidelity(self, U: Tensor) -> Tensor:
        M = torch.matmul(self.U_target.conj().T.to(self.dtype_complex), U)
        tr_MMdag = einsum("bij,bji->b", M, M.conj())
        tr_M = einsum("bii->b", M)
        return (tr_MMdag.abs() + torch.abs(tr_M) ** 2) / 6.0

    def _omegas_detunings_tensor(self) -> Tensor:
        return torch.tensor(self.detunings, dtype=self.dtype_real, device=self.device, requires_grad=self.n_derivs > 0)

    def _should_switch_precision(self, stats: List[Tuple[float, float]], propagators: Tensor) -> bool:
        if self._precision_switched:
            return False
        fidelities = 1.0 - torch.tensor([infid for (_, infid) in stats], dtype=self.dtype_real, device=self.device)
        return torch.all(fidelities > .9).item()
    
    def _switch_to_double(self, rabi: Tensor, phases: Tensor) -> Tuple[Tensor, Tensor]:
        self.dtype_real, self.dtype_complex, self._precision_switched = torch.float64, torch.complex128, True
        self.energies = self.energies.double()
        self.couplings = self.couplings.to(torch.complex128)
        self.U_target = self.U_target.to(torch.complex128)
        return torch.nn.Parameter(rabi.detach().double()), torch.nn.Parameter(phases.detach().double())

    def _success(self, stats: List[Tuple[float, float]]) -> bool:
        infid_tensor = torch.tensor([s[1] for s in stats])
        return torch.all(1.0 - infid_tensor >= self.fidelity_target).item()

################################################################################
# Top‑level optimiser with random restarts
################################################################################
@dataclass
class RandomRobustPulseOptimizer(_BasePulseOptimizer):
    """Random‑restart optimiser using the `_PulseGDRun` engine."""

    max_restarts: int = 100

    # ---------------------------------------------------------------- public
    def run(self, max_steps_per_restart: int = 10_000) -> Dict:
        self._prepare_shared()
        global_best: Dict | None = None

        for restart in range(self.max_restarts):
            rabi, phases, pulse_durations = self._random_initial_params()
            gd = _PulseGDRun(
                parent=self,
                rabi=rabi,
                phases=phases,
                pulse_durations=pulse_durations,
                max_steps=max_steps_per_restart,
                desc=f"Restart {restart}",
            )
            result = gd.run()

            # update global best --------------------------------------------------
            if global_best is None or result["loss"] < global_best["loss"]:
                global_best = result | {"restart": restart}  # merge dicts (py3.9+)

            # exit on success ----------------------------------------------------
            if result["converged"]:
                return global_best
        # after all restarts ----------------------------------------------------
        raise RuntimeError(
            f"Failed to meet fidelity_target after {self.max_restarts} restarts. "
            f"Best loss = {global_best['loss']:.3e} (restart {global_best['restart']})"
        )

    # ---------------------------------------------------------------- private
    def _random_initial_params(self) -> Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor]:
        """
        Draw a fresh random pulse-duration sequence together with random Rabi
        amplitudes and phases.

        Returns
        -------
        rabi            : torch.nn.Parameter, shape (n_pulses,)
        phases          : torch.nn.Parameter, shape (n_pulses,)
        pulse_durations : torch.Tensor        (int64), shape (n_pulses,)
        """
        # --- pulse durations --------------------------------------------------
        pulse_durations = torch.tensor(
            random.choices(self.pulse_length_options, k=self.n_pulses),
            dtype=torch.int64,
            device=self.device,
        )

        # --- random initial Rabi & phases ------------------------------------
        #   • Rabi: Uniform in [0.75, 1.25]  (close to 1 keeps time-evolution well-behaved)
        #   • Phase: Uniform in [0, 2π)
        rabi_init = 0.75 + 0.5 * torch.rand(self.n_pulses, dtype=self.dtype_real, device=self.device)
        phase_init = (2.0 * torch.pi) * torch.rand(self.n_pulses, dtype=self.dtype_real, device=self.device)

        # Wrap in Parameter so Adam can optimise them
        rabi_param   = torch.nn.Parameter(rabi_init)
        phase_param  = torch.nn.Parameter(phase_init)

        return rabi_param, phase_param, pulse_durations
    
    
if __name__ == "__main__":
    # Example usage
    optimizer = RandomRobustPulseOptimizer(
        n_pulses=8,
        detunings=[0.95, 0.98, 1.00, 1.02, 1.05],
        fidelity_target=0.9999,
        plateau_patience=3,
        plateau_tol=1,
    )
    
    best_solution = optimizer.run(max_steps_per_restart=10_000)
    print("Best solution found:")
    print(best_solution)
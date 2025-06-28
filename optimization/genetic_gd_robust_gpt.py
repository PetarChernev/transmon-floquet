################################################################################
# Genetic-algorithm optimiser with restart-aware fitness
################################################################################
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Dict

import torch
from torch import Tensor, einsum
from tqdm.auto import tqdm

# Local modules ----------------------------------------------------------------
from optimization.gradient_descent_robustness import _BasePulseOptimizer, _PulseGDRun
from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator_parallel import (
    floquet_propagator_square_sequence_omega_batch,
)

@dataclass
class GeneticRobustPulseOptimizer(_BasePulseOptimizer):
    """
    Robust-pulse optimiser that first explores the search space with a simple
    genetic algorithm (GA) and then refines every candidate with the existing
    `_PulseGDRun` gradient engine.

    The GA works on *whole pulse sequences* (Rabi amplitudes, phases, pulse
    durations).  A generation therefore consists of

        ┌ evaluate (short GD runs → fitness = best loss)
        ├ elitist selection                         (keep best p%)
        ├ k-point crossover                         (uniform over gene-type)
        └ mutation                                  (Gaussian for floats,
                                                    random pick for durations)

    The process is repeated until either

    * one individual reaches the requested `fidelity_target`, **or**
    * the budget of generations is exhausted – in this case the best
      individual so far is returned and an exception is raised.
    """

    # ===== GA hyper-parameters ==============================================
    population_size: int = 32
    elite_frac: float = 0.25          # share of population copied unchanged
    mutation_prob: float = 0.10       # probability that *each gene* mutates
    mutation_std_rabi: float = 0.05   # σ of Gaussian noise for rabi
    mutation_std_phase: float = 0.10  # σ (rad) for phase noise
    max_generations: int = 40
    # GD budget each generation ----------
    max_steps_per_individual: int = 2_000

    # ---------------------------------------------------------------- public
    def run(self, max_steps_per_restart: int | None = None) -> Dict:  # noqa: D401
        """
        Parameters
        ----------
        max_steps_per_restart
            Ignored (kept for API compatibility with `RandomRobustPulseOptimizer`).
            The GA uses its own `max_steps_per_individual`.
        """
        self._prepare_shared()
        rng = torch.random.manual_seed(self.seed) if self.seed is not None else None

        # -------- helper lambdas --------------------------------------------
        def _random_individual():
            """Return freshly drawn (rabi, phase, duration) tensors."""
            return self._random_initial_params()

        def _evaluate(individual):
            """Run a (possibly short) GD refinement and return its result dict."""
            rabi, phases, pulse_durations = individual
            gd = _PulseGDRun(
                parent=self,
                rabi=rabi,
                phases=phases,
                pulse_durations=pulse_durations,
                max_steps=self.max_steps_per_individual,
                desc="GA-eval",
            )
            return gd.run()

        def _fitness(result):
            """Smaller is better → use total loss as the fitness value."""
            return result["loss"]

        # -------- population init -------------------------------------------
        population = [_random_individual() for _ in range(self.population_size)]
        results: list[Dict] = [_evaluate(ind) for ind in population]
        global_best = min(results, key=_fitness)

        # -------- evolutionary loop -----------------------------------------
        for gen in range(self.max_generations):
            # --- convergence test ------------------------------------------
            if global_best["converged"]:
                global_best["generation"] = gen
                return global_best

            # --- selection --------------------------------------------------
            num_elite = max(1, int(self.elite_frac * self.population_size))
            elite_idx = sorted(range(self.population_size), key=lambda i: _fitness(results[i]))[:num_elite]
            elite = [population[i] for i in elite_idx]

            # --- generate offspring ----------------------------------------
            offspring: list[Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor]] = []
            while len(offspring) < self.population_size - num_elite:
                pa, pb = random.sample(elite, 2)            # parents
                child = self._crossover(pa, pb)
                child = self._mutate(child)
                offspring.append(child)

            population = elite + offspring

            # --- evaluate new generation -----------------------------------
            results = [_evaluate(ind) for ind in population]
            cand_best = min(results, key=_fitness)
            if _fitness(cand_best) < _fitness(global_best):
                global_best = cand_best | {"generation": gen}

            # --- progress read-out (optional) ------------------------------
            tqdm.write(
                f"[Gen {gen:02d}] "
                f"best loss = {global_best['loss']:.3e} "
                f"(F_central = {global_best['fidelities'][self._central_idx]:.4f})"
            )

        # -------- budget exhausted -----------------------------------------
        raise RuntimeError(
            f"Genetic optimiser failed after {self.max_generations} generations "
            f"(best loss = {global_best['loss']:.3e})."
        )

    # ---------------------------------------------------------------- private
    # -- genetic operators ----------------------------------------------------
    def _crossover(
        self,
        parent_a: Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor],
        parent_b: Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor],
    ):
        """Uniform crossover for each gene (pulse index)."""
        rabi_a, phase_a, dur_a = parent_a
        rabi_b, phase_b, dur_b = parent_b

        mask = torch.rand(self.n_pulses, device=self.device) < 0.5
        # Rabi & phase are continuous – blend → keeps magnitudes near parents
        rabi_child = torch.where(mask, rabi_a.detach(), rabi_b.detach()).clone()
        phase_child = torch.where(mask, phase_a.detach(), phase_b.detach()).clone()
        dur_child = torch.where(mask, dur_a.detach(), dur_b.detach()).clone()

        return (
            torch.nn.Parameter(rabi_child),
            torch.nn.Parameter(phase_child),
            dur_child,
        )

    def _mutate(
        self,
        individual: Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor],
    ):
        """Add Gaussian noise to Rabi / phase or re-draw a duration."""
        rabi, phase, dur = individual

        # ---- continuous genes ---------------------------------------------
        if self.mutation_prob > 0:
            m_rabi = torch.rand_like(rabi) < self.mutation_prob
            m_phase = torch.rand_like(phase) < self.mutation_prob
            rabi.data[m_rabi] += self.mutation_std_rabi * torch.randn_like(rabi[m_rabi])
            phase.data[m_phase] += self.mutation_std_phase * torch.randn_like(phase[m_phase])

        # ---- discrete genes (durations) -----------------------------------
        if self.mutation_prob > 0:
            m_dur = torch.rand_like(dur.float()) < self.mutation_prob
            for idx in torch.where(m_dur)[0]:
                dur[idx] = random.choice(self.pulse_length_options)

        return individual

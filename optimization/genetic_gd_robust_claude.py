"""genetic_robust_pulse_optimizer.py
Genetic algorithm-based optimizer for robust single-qubit gate optimization.

This module extends the base optimizer with a genetic algorithm that:
1. Maintains a population of pulse sequences
2. Optimizes each individual using gradient descent (_PulseGDRun)
3. Evolves the population through selection, crossover, and mutation
"""

from __future__ import annotations
import json
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm

# Import from the base module
from gradient_descent_robustness import _BasePulseOptimizer, _PulseGDRun


@dataclass
class Individual:
    """Represents a single individual in the genetic algorithm population."""
    rabi: Tensor
    phases: Tensor
    pulse_durations: Tensor
    fitness: float = float('inf')
    leak: List[float] = field(default_factory=list)
    fidelities: List[float] = field(default_factory=list)
    converged: bool = False
    double_precision: bool = False
    
    def clone(self) -> Individual:
        """Create a deep copy of this individual."""
        return Individual(
            rabi=self.rabi.clone().detach(),
            phases=self.phases.clone().detach(),
            pulse_durations=self.pulse_durations.clone().detach(),
            fitness=self.fitness,
            leak=self.leak.copy() if self.leak else [],
            fidelities=self.fidelities.copy() if self.fidelities else [],
            converged=self.converged,
            double_precision=self.double_precision
        )
        
    def to_json(self) -> Dict:
        """Convert the individual to a JSON-serializable dictionary."""
        return {
            "rabi": self.rabi.cpu().tolist(),
            "phases": self.phases.cpu().tolist(),
            "pulse_durations": self.pulse_durations.cpu().tolist(),
            "fitness": self.fitness,
            "leak": self.leak,
            "fidelities": self.fidelities,
            "converged": self.converged,
            "double_precision": self.double_precision
        }


@dataclass
class GeneticRobustPulseOptimizer(_BasePulseOptimizer):
    """Genetic algorithm-based optimizer using gradient descent for local optimization."""
    
    # Genetic algorithm parameters
    population_size: int = 20
    n_generations: int = 50
    elite_size: int = 4
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    mutation_strength: float = 0.1
    pulse_mutation_prob: float = 0.1
    candidates: List[Individual] = field(default_factory=list)
    
    # Gradient descent parameters per individual
    max_steps_per_individual: int = 2000
    
    def run(self) -> Dict:
        """Run the genetic algorithm optimization."""
        self._prepare_shared()
        
        # Initialize population
        population = self._initialize_population()
        
        # Track best individual across all generations
        global_best: Optional[Individual] = None
        
        generations = self.n_generations if self.n_generations is not None else 1000000000
        with tqdm(range(generations), desc="Generations") as pbar:
            for generation in pbar:
                # Optimize each individual with gradient descent
                population = self._optimize_population(population, generation)
                
                # Update global best
                best_in_gen = min(population, key=lambda x: x.fitness)
                if global_best is None or best_in_gen.fitness < global_best.fitness:
                    global_best = best_in_gen.clone()
                
                pbar.set_postfix(
                    best_loss=f"{global_best.fitness:.3e}",
                    best_fid=f"{min(global_best.fidelities):.6f}" if global_best.fidelities else "N/A"
                )
                
                # Check for convergence
                if global_best.converged:
                    print(f"Converged at generation {generation}!")
                    break
                
                # Evolve population (except for last generation)
                if generation < generations - 1:
                    population = self._evolve_population(population)
        
        # Return best solution
        if global_best is None:
            raise RuntimeError("No valid solution found")
            
        return {
            "converged": global_best.converged,
            "loss": global_best.fitness,
            "leak": global_best.leak,
            "fidelities": global_best.fidelities,
            "rabi": global_best.rabi.cpu().tolist(),
            "phases": global_best.phases.cpu().tolist(),
            "pulse_durations": global_best.pulse_durations.cpu().tolist(),
            "double_precision": global_best.double_precision,
            "generation": generation,
        }
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            rabi, phases, pulse_durations = self._random_initial_params()
            individual = Individual(
                rabi=rabi.detach(),
                phases=phases.detach(),
                pulse_durations=pulse_durations
            )
            population.append(individual)
        return population
    
    def _optimize_population(self, population: List[Individual], generation: int) -> List[Individual]:
        """Run gradient descent optimization for each individual in the population."""
        optimized_population = []
        
        
        for i, individual in enumerate(population):
            # Reset precision for each run
            self._precision_switched = False
            self.dtype_real = torch.float32
            self.dtype_complex = torch.complex64
            self.energies = self.energies.float()
            self.couplings = self.couplings.to(torch.complex64)
            self.U_target = self.U_target.to(torch.complex64)
            
            # Create parameters for gradient descent
            rabi_param = torch.nn.Parameter(individual.rabi.clone())
            phases_param = torch.nn.Parameter(individual.phases.clone())
            
            # Run gradient descent
            gd = _PulseGDRun(
                parent=self,
                rabi=rabi_param,
                phases=phases_param,
                pulse_durations=individual.pulse_durations,
                max_steps=self.max_steps_per_individual,
                desc=f"Gen {generation}, Individual {i}"
            )
            
            result = gd.run()
            
            # Create optimized individual
            opt_individual = Individual(
                rabi=torch.tensor(result["rabi"], dtype=torch.float32, device=self.device),
                phases=torch.tensor(result["phases"], dtype=torch.float32, device=self.device),
                pulse_durations=individual.pulse_durations,
                fitness=result["loss"],
                leak=result["leak"],
                fidelities=result["fidelities"],
                converged=result["converged"],
                double_precision=result["double_precision"]
            )
            
            optimized_population.append(opt_individual)
            if all(f > .9 for f in opt_individual.fidelities):
                self.candidates.append(opt_individual.to_json())
        with open(f"candidates/generation_{generation}.json", "w+") as f:
            json.dump([ind.to_json() for ind in optimized_population], f, indent=2)
        if self.candidates:
            with open(f"candidates/candidates_{i}.json", "w+") as f:
                json.dump(self.candidates, f)
            self.candidates = []  # Clear candidates after saving
        return optimized_population
    
    def _evolve_population(self, population: List[Individual]) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness (lower is better)
        
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        
        # Elite selection
        new_population = [ind.clone() for ind in sorted_pop[:self.elite_size]]
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            if random.random() < self.mutation_prob:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_prob:
                child2 = self._mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover that also handles parents of different lengths."""
        len1, len2 = len(parent1.rabi), len(parent2.rabi)
        max_len, min_len = max(len1, len2), min(len1, len2)

        # ------------------------------------------------------------------ helpers
        def pad_short(short: Individual, long: Individual) -> Tuple[Tensor, Tensor, Tensor]:
            """Embed the shorter parent in the longer one's length."""
            offset = 0 if torch.rand(1).item() < 0.5 else max_len - min_len  # flush left or right
            rabi   = long.rabi.clone()
            phases = long.phases.clone()
            pulses = long.pulse_durations.clone()
            rabi[offset:offset+min_len]   = short.rabi
            phases[offset:offset+min_len] = short.phases
            pulses[offset:offset+min_len] = short.pulse_durations
            return rabi, phases, pulses

        # ----------------------------------------------------------------- pad / align
        if len1 == len2:
            p1_rabi, p1_phases, p1_pulses = parent1.rabi, parent1.phases, parent1.pulse_durations
            p2_rabi, p2_phases, p2_pulses = parent2.rabi, parent2.phases, parent2.pulse_durations
        elif len1 < len2:
            p1_rabi, p1_phases, p1_pulses = pad_short(parent1, parent2)
            p2_rabi, p2_phases, p2_pulses = parent2.rabi, parent2.phases, parent2.pulse_durations
        else:  # len2 < len1
            p2_rabi, p2_phases, p2_pulses = pad_short(parent2, parent1)
            p1_rabi, p1_phases, p1_pulses = parent1.rabi, parent1.phases, parent1.pulse_durations

        # ------------------------------------------------------------------ crossover
        mask = torch.rand(max_len, device=self.device) < 0.5
        child1_rabi   = torch.where(mask, p1_rabi, p2_rabi)
        child2_rabi   = torch.where(mask, p2_rabi, p1_rabi)
        child1_phases = torch.where(mask, p1_phases, p2_phases)
        child2_phases = torch.where(mask, p2_phases, p1_phases)
        child1_pulses = torch.where(mask, p1_pulses, p2_pulses)
        child2_pulses = torch.where(mask, p2_pulses, p1_pulses)

        # ------------------------------------------------------------ choose lengths
        def truncate(arr_r, arr_p, arr_d, target_len):
            if target_len == max_len:
                return arr_r, arr_p, arr_d
            if torch.rand(1).item() < 0.5:  # flush left
                sl = slice(0, target_len)
            else:                           # flush right
                sl = slice(max_len - target_len, max_len)
            return arr_r[sl], arr_p[sl], arr_d[sl]

        len_child1 = int(random.choice([len1, len2]))
        len_child2 = int(random.choice([len1, len2]))
        child1_rabi, child1_phases, child1_pulses = truncate(child1_rabi, child1_phases, child1_pulses, len_child1)
        child2_rabi, child2_phases, child2_pulses = truncate(child2_rabi, child2_phases, child2_pulses, len_child2)

        # ------------------------------------------------------------- build objects
        child1 = Individual(
            rabi=child1_rabi.clone().detach(),
            phases=child1_phases.clone().detach(),
            pulse_durations=child1_pulses.clone().detach(),
        )
        child2 = Individual(
            rabi=child2_rabi.clone().detach(),
            phases=child2_phases.clone().detach(),
            pulse_durations=child2_pulses.clone().detach(),
        )
        return child1, child2

    
    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        mutated = individual.clone()
        
        # Mutate Rabi frequencies
        rabi_noise = torch.randn_like(mutated.rabi) * self.mutation_strength
        mutated.rabi = torch.clamp(mutated.rabi + rabi_noise, min=0.1, max=2.0)
        
        # Mutate phases (with wrapping)
        phase_noise = torch.randn_like(mutated.phases) * self.mutation_strength * 2 * torch.pi
        mutated.phases = (mutated.phases + phase_noise) % (2 * torch.pi)
        
        # Mutate pulse durations (discrete)
        for i in range(len(mutated.pulse_durations)):
            pulse_mutation_chance = random.random()
            if pulse_mutation_chance < self.pulse_mutation_prob / len(mutated.pulse_durations):
                new_duration = random.choice(self.pulse_length_options)
                
                # Add an new pulse
                if random.random() < 0.5:  # 50% chance to add a new pulse to the beginning
                    mutated.pulse_durations = torch.cat(
                        [torch.tensor([new_duration], device=self.device), mutated.pulse_durations]
                    )
                    mutated.rabi = torch.cat(
                        [torch.tensor([0.75 + 0.5 * random.random()], device=self.device), mutated.rabi]
                    )
                    mutated.phases = torch.cat(
                        [torch.tensor([2.0 * torch.pi * random.random()], device=self.device), mutated.phases]
                    )
                else:
                    mutated.pulse_durations = torch.cat(
                        [mutated.pulse_durations[:i], torch.tensor([new_duration], device=self.device), mutated.pulse_durations[i:]]
                    )
                    mutated.rabi = torch.cat(
                        [mutated.rabi[:i], torch.tensor([0.75 + 0.5 * random.random()], device=self.device), mutated.rabi[i:]]
                    )
                    mutated.phases = torch.cat(
                        [mutated.phases[:i], torch.tensor([2.0 * torch.pi * random.random()], device=self.device), mutated.phases[i:]]
                    )
            elif pulse_mutation_chance < self.pulse_mutation_prob:
                mutated.pulse_durations[i] += max(0, random.choice(self.pulse_length_options) - self.pulse_length_middle)
        
        return mutated
    
    def _random_initial_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate random initial parameters."""
        pulse_durations = torch.tensor(
            random.choices(self.pulse_length_options, k=self.n_pulses),
            dtype=torch.int64,
            device=self.device,
        )
        
        rabi_init = 0.75 + 0.5 * torch.rand(
            self.n_pulses, dtype=torch.float32, device=self.device
        )
        phase_init = 2.0 * torch.pi * torch.rand(
            self.n_pulses, dtype=torch.float32, device=self.device
        )
        
        return rabi_init, phase_init, pulse_durations


# Example usage
if __name__ == "__main__":
    optimizer = GeneticRobustPulseOptimizer(
        n_pulses=13,
        detunings=[0.95, 0.98, 1.00, 1.02, 1.05],
        fidelity_target=0.95,
        plateau_patience=20,
        plateau_tol=.01,
        # Genetic algorithm parameters
        population_size=20,
        n_generations=None,
        elite_size=4,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.3,
        mutation_strength=0.1,
        pulse_mutation_prob=0.15,
        max_steps_per_individual=2000,
    )
    
    with open("candidates.pkl", "wb+") as f:
        pickle.dump(optimizer.candidates, f)
    
    best_solution = optimizer.run()
    print("\nBest solution found:")
    print(f"Converged: {best_solution['converged']}")
    print(f"Loss: {best_solution['loss']:.6e}")
    print(f"Fidelities: {best_solution['fidelities']}")
    print(f"Leak: {best_solution['leak']}")
    print(f"Generation: {best_solution['generation']}")
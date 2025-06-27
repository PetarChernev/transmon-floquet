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

##############################################################################
# fixed chip parameters (define once)
##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_levels = 6
# find EJ/EC ratio that gives target anharmonicity
EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
energies, lambdas_full = TransmonCore.compute_transmon_parameters(
    n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
)

energies = torch.tensor(energies, dtype=torch.float64, device=device)
lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128, device=device)

omega_d = 1.0  # nominal drive frequency, rad s^-1
floquet_cutoff: int = 25

U_TARGET = torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble, device=device) / sqrt(2)

TWO_PI = 2.0 * torch.pi

##############################################################################
# loss function
##############################################################################

def U_pulse_sequence(rabi, phases, pulse_durations_periods, omega=omega_d):
    return floquet_propagator_square_sequence_stroboscopic(
        rabi,
        phases,
        pulse_durations_periods,
        energies,
        lambdas_full,
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

##############################################################################
# Genetic Algorithm for pulse durations
##############################################################################

@dataclass
class Individual:
    """Individual in the genetic algorithm population."""
    durations: np.ndarray
    fitness_metrics: Optional[Dict] = None
    fitness_score: float = float('inf')
    sgld_results: Optional[List] = None

class GeneticAlgorithm:
    """Genetic algorithm for optimizing pulse durations."""
    
    def __init__(self, n_pulses=8, population_size=20, duration_range=(1, 15),
                 mutation_rate=0.2, crossover_rate=0.7, elite_fraction=0.2):
        self.n_pulses = n_pulses
        self.population_size = population_size
        self.duration_range = duration_range
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(elite_fraction * population_size)
        
        # Initialize population
        self.population = self._initialize_population()
        self.generation = 0
        self.best_individual = None
        self.history = []
        
    def _initialize_population(self):
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            durations = np.random.randint(
                self.duration_range[0], 
                self.duration_range[1] + 1, 
                self.n_pulses
            )
            population.append(Individual(durations=durations))
        return population
    
    def evaluate_individual(self, individual, sgld_steps=500, n_chains=15, verbose=False):
        """Evaluate an individual by running SGLD with its duration configuration."""
        if verbose:
            print(f"  Evaluating durations: {individual.durations}")
        
        # Run SGLD with these durations
        start_time = time.time()
        sgld = MultiModalSGLD(
            pulse_durations=individual.durations,
            n_chains=n_chains,
            temp_range=(0.0001, 0.05),
            lr_base=5e-3
        )
        sgld.run(n_steps=sgld_steps, verbose=False)
        
        # Get fitness metrics
        metrics = sgld.get_fitness_metrics()
        individual.fitness_metrics = metrics
        individual.sgld_results = sgld.unique_solutions[:5]  # Store top 5 solutions
        
        # Compute composite fitness score (lower is better)
        # Prioritize: low loss, many unique minima, fast convergence
        fitness = (
            metrics['best_loss'] * 1000 +  # Heavily weight best loss
            1.0 / (metrics['n_unique_minima'] + 1) +  # Reward finding many minima
            (1.0 - metrics['convergence_rate']) * 10 +  # Reward fast convergence
            metrics['avg_final_loss'] * 100  # Consider average performance
        )
        individual.fitness_score = fitness
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"    Best loss: {metrics['best_loss']:.6f}, "
                  f"Unique minima: {metrics['n_unique_minima']}, "
                  f"Time: {elapsed:.1f}s")
        
        return fitness
    
    def evaluate_population(self, verbose=True):
        """Evaluate all individuals in the population."""
        if verbose:
            print(f"\nGeneration {self.generation}: Evaluating {len(self.population)} individuals...")
        
        for i, individual in enumerate(self.population):
            if individual.fitness_score == float('inf'):  # Only evaluate if not already done
                self.evaluate_individual(individual, verbose=verbose and i < 3)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness_score)
        self.best_individual = self.population[0]
        
        if verbose:
            print(f"\nBest individual: durations={self.best_individual.durations}")
            print(f"  Fitness score: {self.best_individual.fitness_score:.6f}")
            print(f"  Best loss: {self.best_individual.fitness_metrics['best_loss']:.6f}")
            print(f"  Unique minima: {self.best_individual.fitness_metrics['n_unique_minima']}")
    
    def crossover(self, parent1, parent2):
        """Create offspring through crossover."""
        if np.random.rand() > self.crossover_rate:
            return parent1.durations.copy(), parent2.durations.copy()
        
        # Two-point crossover
        points = sorted(np.random.choice(self.n_pulses, 2, replace=False))
        child1 = parent1.durations.copy()
        child2 = parent2.durations.copy()
        
        child1[points[0]:points[1]] = parent2.durations[points[0]:points[1]]
        child2[points[0]:points[1]] = parent1.durations[points[0]:points[1]]
        
        return child1, child2
    
    def mutate(self, durations):
        """Apply mutation to durations."""
        mutated = durations.copy()
        
        for i in range(self.n_pulses):
            if np.random.rand() < self.mutation_rate:
                # Random mutation strategies
                strategy = np.random.choice(['change', 'swap', 'shift'])
                
                if strategy == 'change':
                    # Change to random value
                    mutated[i] = np.random.randint(
                        self.duration_range[0], 
                        self.duration_range[1] + 1
                    )
                elif strategy == 'swap' and i < self.n_pulses - 1:
                    # Swap with neighbor
                    mutated[i], mutated[i+1] = mutated[i+1], mutated[i]
                else:  # shift
                    # Small change up or down
                    change = np.random.choice([-2, -1, 1, 2])
                    mutated[i] = np.clip(
                        mutated[i] + change, 
                        self.duration_range[0], 
                        self.duration_range[1]
                    )
        
        return mutated
    
    def evolve_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            new_population.append(
                Individual(durations=self.population[i].durations.copy())
            )
        
        # Create rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            tournament_size = 3
            parents = []
            for _ in range(2):
                tournament = np.random.choice(self.population, tournament_size, replace=False)
                winner = min(tournament, key=lambda x: x.fitness_score)
                parents.append(winner)
            
            # Crossover
            child1_dur, child2_dur = self.crossover(parents[0], parents[1])
            
            # Mutation
            child1_dur = self.mutate(child1_dur)
            child2_dur = self.mutate(child2_dur)
            
            # Add to new population
            new_population.append(Individual(durations=child1_dur))
            if len(new_population) < self.population_size:
                new_population.append(Individual(durations=child2_dur))
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def run(self, n_generations=10, verbose=True):
        """Run the genetic algorithm with a tqdm bar over generations."""
        print("Starting Genetic Algorithm for pulse duration optimization...")
        print(f"Population size: {self.population_size}, Generations: {n_generations}")
        
        for gen in trange(n_generations, desc="GA generations"):
            # Evaluate current generation
            self.evaluate_population(verbose=verbose)
            
            # Record history
            self.history.append({
                'generation': self.generation,
                'best_fitness': self.best_individual.fitness_score,
                'best_loss': self.best_individual.fitness_metrics['best_loss'],
                'best_durations': self.best_individual.durations.copy(),
                'best_n_minima': self.best_individual.fitness_metrics['n_unique_minima']
            })
            
            # Evolve to next generation (except for last generation)
            if gen < n_generations - 1:
                self.evolve_generation()
        
        return self.best_individual

##############################################################################
# Combined optimization with refinement
##############################################################################

def refine_best_solutions(ga_result, n_refine_steps=1000, n_chains=30):
    """Refine the best solutions found by GA with extended SGLD."""
    print("\n" + "="*50)
    print("Refining best solution with extended SGLD...")
    print(f"Best durations: {ga_result.durations}")
    
    # Run extended SGLD with best durations
    sgld = MultiModalSGLD(
        pulse_durations=ga_result.durations,
        n_chains=n_chains,
        temp_range=(0.00001, 0.01),  # Lower temperature for refinement
        lr_base=1e-3
    )
    
    sgld.run(n_steps=n_refine_steps, verbose=True)
    
    # Get refined solutions
    best_solutions = sgld.unique_solutions[:5]
    
    print(f"\nFound {len(sgld.unique_solutions)} unique refined solutions")
    for i, sol in enumerate(best_solutions):
        print(f"\nRefined solution {i+1}:")
        print(f"  Loss: {sol['loss']:.6f}")
        print(f"  Rabi: {np.round(sol['rabi'], 4)}")
        print(f"  Phases: {np.round(sol['phases'], 4)}")
    
    return sgld, best_solutions

##############################################################################
# Main optimization script
##############################################################################

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run genetic algorithm to find optimal pulse durations
    ga = GeneticAlgorithm(
        n_pulses=8,
        population_size=20,
        duration_range=(1, 15),
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_fraction=0.2
    )
    
    # Run GA optimization
    best_individual = ga.run(n_generations=10, verbose=True)
    
    # Plot GA evolution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    generations = [h['generation'] for h in ga.history]
    best_losses = [h['best_loss'] for h in ga.history]
    plt.plot(generations, best_losses, 'b-o')
    plt.xlabel('Generation')
    plt.ylabel('Best Loss')
    plt.title('GA Evolution: Best Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    n_minima = [h['best_n_minima'] for h in ga.history]
    plt.plot(generations, n_minima, 'r-o')
    plt.xlabel('Generation')
    plt.ylabel('Number of Unique Minima')
    plt.title('GA Evolution: Solution Diversity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ga_evolution.png')
    plt.show()
    
    # Refine best solutions
    sgld_refined, refined_solutions = refine_best_solutions(
        best_individual, 
        n_refine_steps=2000,
        n_chains=30
    )
    
    # Save results
    import pickle
    results = {
        'ga_history': ga.history,
        'best_durations': best_individual.durations,
        'best_fitness_metrics': best_individual.fitness_metrics,
        'refined_solutions': refined_solutions,
        'all_unique_solutions': sgld_refined.unique_solutions
    }
    
    with open('hybrid_sgld_ga_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*50)
    print("Optimization complete!")
    print(f"Best pulse durations: {best_individual.durations}")
    print(f"Total duration: {np.sum(best_individual.durations)} periods")
    print(f"Best loss achieved: {refined_solutions[0]['loss']:.6f}")
    print(f"Results saved to 'hybrid_sgld_ga_results.pkl'")
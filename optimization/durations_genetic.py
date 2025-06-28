
##############################################################################
# Genetic Algorithm for pulse durations
##############################################################################

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import trange

from optimization.sgld import BatchedMultiModalSGLD



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
        sgld = BatchedMultiModalSGLD(
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

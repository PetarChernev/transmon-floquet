
##############################################################################
# Combined optimization with refinement
##############################################################################

from matplotlib import pyplot as plt
import numpy as np
import torch

from optimization.durations_genetic import GeneticAlgorithm
from optimization.sgld import BatchedMultiModalSGLD


def refine_best_solutions(ga_result, n_refine_steps=1000, n_chains=30):
    """Refine the best solutions found by GA with extended SGLD."""
    print("\n" + "="*50)
    print("Refining best solution with extended SGLD...")
    print(f"Best durations: {ga_result.durations}")
    
    # Run extended SGLD with best durations
    sgld = BatchedMultiModalSGLD(
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
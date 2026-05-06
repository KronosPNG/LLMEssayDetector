import random
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
import copy

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class Individual:
    """Represents a single hyperparameter combination."""
    hyperparameters: Dict[str, Any]
    fitness: float = None
    
    def __lt__(self, other):
        """For sorting - higher fitness is better."""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness > other.fitness


class GeneticAlgorithmOptimizer:
    """Genetic algorithm for hyperparameter optimization."""
    
    def __init__(
        self,
        hyperparameter_space: Dict[str, List[Any]],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        random_seed: int = None,
        verbose: bool = True,
    ):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            hyperparameter_space: Dictionary mapping hyperparameter names to lists of possible values
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene (0-1)
            crossover_rate: Probability of crossover when creating offspring (0-1)
            elite_ratio: Fraction of top performers to preserve unchanged (0-1)
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.hyperparameter_space = hyperparameter_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.verbose = verbose
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.history = []
        self.best_individual = None
    
    def _create_random_individual(self) -> Individual:
        """Create an individual with random hyperparameters."""
        hyperparameters = {}
        for param_name, param_values in self.hyperparameter_space.items():
            hyperparameters[param_name] = random.choice(param_values)
        return Individual(hyperparameters=hyperparameters)
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        return [self._create_random_individual() for _ in range(self.population_size)]
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Single-point crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        param_names = list(self.hyperparameter_space.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        child1_params = {}
        child2_params = {}
        
        for i, param_name in enumerate(param_names):
            if i < crossover_point:
                child1_params[param_name] = parent1.hyperparameters[param_name]
                child2_params[param_name] = parent2.hyperparameters[param_name]
            else:
                child1_params[param_name] = parent2.hyperparameters[param_name]
                child2_params[param_name] = parent1.hyperparameters[param_name]
        
        return Individual(hyperparameters=child1_params), Individual(hyperparameters=child2_params)
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual by randomly changing some hyperparameters.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = copy.deepcopy(individual)
        
        for param_name in self.hyperparameter_space.keys():
            if random.random() < self.mutation_rate:
                mutated.hyperparameters[param_name] = random.choice(
                    self.hyperparameter_space[param_name]
                )
        
        return mutated
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """
        Select an individual using tournament selection.
        
        Args:
            population: Current population
            tournament_size: Number of individuals in each tournament
            
        Returns:
            Selected individual
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """
        Create next generation using elitism, crossover, and mutation.
        
        Args:
            population: Current population
            
        Returns:
            New population
        """
        # Sort by fitness (best first)
        sorted_pop = sorted(population, reverse=True)
        
        # Elitism: keep the best individuals
        new_population = [copy.deepcopy(ind) for ind in sorted_pop[:self.elite_size]]
        
        # Create offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population[:self.population_size]
    
    def optimize(self, fitness_function: Callable[[Dict[str, Any]], float]) -> Individual:
        """
        Run the genetic algorithm optimization.
        
        Args:
            fitness_function: Function that takes a hyperparameter dict and returns a fitness score.
                             Higher values are considered better.
        
        Returns:
            Best individual found
        """
        population = self._initialize_population()
        if self.verbose:
            print(
                f"Initialized population with {len(population)} individuals | "
                f"generations={self.generations} | population_size={self.population_size}"
            )
        
        for generation in range(self.generations):
            if self.verbose:
                print(f"\n=== Generation {generation + 1}/{self.generations} ===")
                print("Evaluating population...")

            # Evaluate fitness for population
            population_iterator = enumerate(population, start=1)
            if self.verbose and tqdm is not None:
                population_iterator = tqdm(
                    population_iterator,
                    total=len(population),
                    desc=f"Generation {generation + 1}/{self.generations}",
                    leave=False,
                )

            for index, individual in population_iterator:
                if individual.fitness is None:
                    individual.fitness = fitness_function(individual.hyperparameters)
                if self.verbose:
                    print(
                        f"  Individual {index:02d}/{len(population):02d} | "
                        f"fitness={individual.fitness:.6f} | "
                        f"hyperparameters={individual.hyperparameters}"
                    )
            
            # Track best individual
            best_in_gen = max(population, key=lambda x: x.fitness)
            if self.best_individual is None or best_in_gen.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(best_in_gen)
                if self.verbose:
                    print(f"New global best found: {self.best_individual.fitness:.6f}")
            
            avg_fitness = np.mean([ind.fitness for ind in population])
            self.history.append({
                'generation': generation,
                'best_fitness': best_in_gen.fitness,
                'avg_fitness': avg_fitness,
                'best_hyperparameters': copy.deepcopy(best_in_gen.hyperparameters)
            })
            
            print(f"Generation {generation + 1}/{self.generations} | "
                  f"Best: {best_in_gen.fitness:.6f} | "
                  f"Avg: {avg_fitness:.6f}")
            
            # Create next generation
            if generation < self.generations - 1:
                if self.verbose:
                    print("Creating next generation with elitism, crossover, and mutation...")
                population = self._create_next_generation(population)
        
        return self.best_individual
    
    def get_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.history

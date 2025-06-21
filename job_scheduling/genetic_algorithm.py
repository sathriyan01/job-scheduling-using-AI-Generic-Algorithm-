import numpy as np
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import copy

class GeneticAlgorithm:
    def __init__(self, jobs: Dict, machines: Dict, job_ids: List, 
                 population_size: int = 100, generations: int = 500,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 elitism_ratio: float = 0.1):
        # Initialization code remains the same as your working version
        self.jobs = jobs
        self.machines = machines
        self.job_ids = job_ids
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio

    def initialize_population(self) -> List[Dict]:
        """Create initial population of random schedules."""
        population = []
        for _ in range(self.population_size):
            schedule = {machine: [] for machine in self.machines}
            
            # Shuffle job IDs and assign to machines
            shuffled_jobs = random.sample(self.job_ids, len(self.job_ids))
            for job_id in shuffled_jobs:
                machine = self.jobs[job_id]['assigned_machine']
                schedule[machine].append(job_id)
            
            # Ensure dependencies are met
            for machine, job_sequence in schedule.items():
                for i in range(len(job_sequence)):
                    job_id = job_sequence[i]
                    dependency = self.jobs[job_id]['dependency']
                    if dependency and dependency in job_sequence and job_sequence.index(dependency) > i:
                        dep_index = job_sequence.index(dependency)
                        job_sequence[i], job_sequence[dep_index] = job_sequence[dep_index], job_sequence[i]
            
            population.append(schedule)
        return population

    def fitness(self, schedule: Dict) -> float:
        """Calculate fitness of a schedule."""
        total_penalty = 0
        priority_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        
        for machine, job_sequence in schedule.items():
            current_time = 0
            for job_id in job_sequence:
                job = self.jobs[job_id]
                completion_time = current_time + job['processing_time']
                
                # Deadline penalty
                if completion_time > job['deadline']:
                    tardiness = completion_time - job['deadline']
                    total_penalty += tardiness * priority_weights[job['priority']] * 2
                
                # Reward early completion
                if completion_time <= job['deadline']:
                    earliness = job['deadline'] - completion_time
                    total_penalty -= earliness * priority_weights[job['priority']]
                
                current_time = completion_time
        
        # Validate schedule
        if not self.validate_schedule(schedule):
            total_penalty += 1000
        
        return -total_penalty  # Higher is better

    def validate_schedule(self, schedule: Dict) -> bool:
        """Validate schedule dependencies."""
        job_positions = {}
        for machine, sequence in schedule.items():
            for pos, job_id in enumerate(sequence):
                job_positions[job_id] = (machine, pos)
        
        for job_id, job_info in self.jobs.items():
            if job_info['dependency'] and job_info['dependency'] not in job_positions:
                return False
        return True

    def selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        child1 = {m: [] for m in self.machines}
        child2 = {m: [] for m in self.machines}
        
        for machine in self.machines:
            pt = random.randint(1, min(len(parent1[machine]), len(parent2[machine]))-1)
            child1[machine] = parent1[machine][:pt] + [j for j in parent2[machine] if j not in parent1[machine][:pt]]
            child2[machine] = parent2[machine][:pt] + [j for j in parent1[machine] if j not in parent2[machine][:pt]]
        
        return child1, child2

    def mutate(self, schedule: Dict) -> Dict:
        """Mutation operator."""
        if random.random() > self.mutation_rate:
            return schedule
            
        mutated = copy.deepcopy(schedule)
        m1, m2 = random.sample(list(self.machines.keys()), 2)
        
        if len(mutated[m1]) > 1:
            i, j = random.sample(range(len(mutated[m1])), 2)
            mutated[m1][i], mutated[m1][j] = mutated[m1][j], mutated[m1][i]
        
        return mutated

    def evolve(self) -> Tuple[Dict, float, List[float]]:
        """Run genetic algorithm evolution."""
        population = self.initialize_population()
        best_fitness = -float('inf')
        best_schedule = None
        fitness_history = []
        
        for _ in tqdm(range(self.generations), desc="Evolving"):
            # Evaluate
            fitness_scores = [self.fitness(ind) for ind in population]
            current_best = max(fitness_scores)
            
            # Update best
            if current_best > best_fitness:
                best_fitness = current_best
                best_schedule = copy.deepcopy(population[np.argmax(fitness_scores)])
            
            fitness_history.append(best_fitness)
            
            # Selection
            parents = self.selection(population, fitness_scores)
            
            # Elitism
            elite_size = max(1, int(self.elitism_ratio * self.population_size))
            elite = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_size]
            next_gen = [copy.deepcopy(ind) for ind, _ in elite]
            
            # Crossover/Mutation
            while len(next_gen) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self.crossover(p1, p2)
                next_gen.extend([self.mutate(c1), self.mutate(c2)])
            
            population = next_gen[:self.population_size]
        
        return best_schedule, best_fitness, fitness_history
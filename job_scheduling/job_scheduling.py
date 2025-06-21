import pandas as pd
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from utils import load_jobs_data, validate_schedule
import json

def visualize_schedule(schedule: dict, jobs: dict):
    """Visualize the schedule using a Gantt chart."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    machines = sorted(schedule.keys())
    colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    
    for i, machine in enumerate(machines):
        current_time = 0
        for job_id in schedule[machine]:
            job = jobs[job_id]
            duration = job['processing_time']
            priority = job['priority']
            
            # Plot the job as a horizontal bar
            ax.barh(i, duration, left=current_time, color=colors[priority], edgecolor='black')
            
            # Add job ID text
            ax.text(current_time + duration/2, i, job_id, ha='center', va='center', color='white')
            
            # Mark deadline with a vertical line
            deadline = job['deadline']
            if deadline < current_time + duration:
                ax.axvline(x=deadline, color='red', linestyle='--', alpha=0.5)
                ax.text(deadline, i + 0.2, f'Deadline {job_id}', color='red', fontsize=8)
            
            current_time += duration
    
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel('Time')
    ax.set_title('Job Schedule Gantt Chart')
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    
    # Create legend for priorities
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='High Priority'),
        Patch(facecolor='orange', label='Medium Priority'),
        Patch(facecolor='green', label='Low Priority')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def print_schedule_stats(schedule: dict, jobs: dict):
    """Print statistics about the schedule."""
    total_tardiness = 0
    missed_deadlines = 0
    total_jobs = 0
    
    for machine, job_sequence in schedule.items():
        current_time = 0
        print(f"\nMachine {machine} Schedule:")
        for job_id in job_sequence:
            job = jobs[job_id]
            duration = job['processing_time']
            completion_time = current_time + duration
            deadline = job['deadline']
            tardiness = max(0, completion_time - deadline)
            
            if tardiness > 0:
                missed_deadlines += 1
                total_tardiness += tardiness
            
            print(f"  {job_id}: Start={current_time}, End={completion_time}, "
                  f"Deadline={deadline}, Tardiness={tardiness}, "
                  f"Priority={job['priority']}, Dependency={job['dependency']}")
            
            current_time = completion_time
            total_jobs += 1
    
    print(f"\nSummary:")
    print(f"Total Jobs Scheduled: {total_jobs}")
    print(f"Jobs Missing Deadlines: {missed_deadlines}")
    print(f"Total Tardiness: {total_tardiness}")
    print(f"Schedule Validity: {'Valid' if validate_schedule(schedule, jobs) else 'Invalid'}")

def save_schedule(schedule: dict, file_path: str):
    """Save schedule to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(schedule, f, indent=4)

def main():
    # Load data
    jobs, machines, job_ids = load_jobs_data('data/job_scheduling_dataset.csv')
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        jobs=jobs,
        machines=machines,
        job_ids=job_ids,
        population_size=100,
        generations=200,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_ratio=0.1
    )
    
    best_schedule, best_fitness, fitness_history = ga.evolve()
    
    # Print results
    print("\nBest Schedule Found:")
    print_schedule_stats(best_schedule, jobs)
    print(f"\nBest Fitness Score: {best_fitness}")
    
    # Visualize results
    visualize_schedule(best_schedule, jobs)
    
    # Plot fitness history
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title('Fitness Score Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True)
    plt.show()
    
    # Save the best schedule
    save_schedule(best_schedule, 'best_schedule.json')

if __name__ == "__main__":
    main()
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

def load_jobs_data(file_path: str) -> Tuple[Dict, Dict, List]:
    """Load job data from CSV file and return jobs, machines, and job_ids."""
    df = pd.read_csv("data\job_scheduling_dataset.csv")
    
    jobs = {}
    machines = {}
    
    for _, row in df.iterrows():
        job_id = row['Job_ID']
        jobs[job_id] = {
            'processing_time': int(row['Processing_Time']),
            'priority': row['Priority'],
            'deadline': int(row['Deadline']),
            'assigned_machine': row['Assigned_Machine'],
            'dependency': None if row['Dependency'] == 'None' else row['Dependency']
        }
        
        machine = row['Assigned_Machine']
        if machine not in machines:
            machines[machine] = []
    
    job_ids = list(jobs.keys())
    
    return jobs, machines, job_ids

def calculate_priority_weight(priority: str) -> int:
    """Convert priority to numerical weight."""
    weights = {'High': 3, 'Medium': 2, 'Low': 1}
    return weights.get(priority, 1)

def validate_schedule(schedule: Dict, jobs: Dict) -> bool:
    """Check if schedule meets all dependencies."""
    for machine, job_sequence in schedule.items():
        for i, job_id in enumerate(job_sequence):
            dependency = jobs[job_id]['dependency']
            if dependency:
                # Check if dependency is scheduled before this job
                found = False
                for prev_job in job_sequence[:i]:
                    if prev_job == dependency:
                        found = True
                        break
                if not found:
                    # Check other machines
                    for other_machine, other_sequence in schedule.items():
                        if other_machine == machine:
                            continue
                        if dependency in other_sequence:
                            dep_index = other_sequence.index(dependency)
                            current_machine_jobs = schedule[machine]
                            if i > 0 and current_machine_jobs[i-1] == dependency:
                                found = True
                                break
                if not found:
                    return False
    return True
import numpy as np
import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from algo_aco import AntColonyOptimization
from algo_hba import HoneyBeeAlgorithm
from algo_pso import ParticleSwarmOptimization
from algo_tlb import ThrottledLoadBalancer
from cal_metrics import calculate_metrics

# Seed for reproducibility
np.random.seed(57)

# Number of tasks
num_tasks = 100

# Task characteristics: [CPU usage, Memory usage, GPU usage, Execution time]
tasks = np.random.randint(1, 10, (num_tasks, 4))

# Cloud providers' characteristics: [CPU capacity, Memory capacity, GPU capacity, Cost per unit resource]
cloud_providers = np.array([
    [1000, 2000, 500, 0.02],  # Provider 1
    [1500, 1500, 600, 0.025], # Provider 2
    [1200, 1800, 550, 0.018]  # Provider 3
])

# Each task will be assigned to a cloud provider with initial random allocation
task_allocation = np.random.choice(3, num_tasks)



# Initialize and run ACO
aco = AntColonyOptimization(tasks, cloud_providers)
best_allocation_aco, best_cost_aco = aco.run()

# Initialize and run PSO
pso = ParticleSwarmOptimization(tasks, cloud_providers)
best_allocation_pso, best_cost_pso = pso.run()

# # Initialize and run TLB
tlb = ThrottledLoadBalancer(tasks, cloud_providers)
best_allocation_tlb, best_cost_tlb = tlb.run()

# # Initialize and run HBA
hba = HoneyBeeAlgorithm(tasks, cloud_providers)
best_allocation_hba, best_cost_hba = hba.run()

metrics_aco = calculate_metrics(best_allocation_aco, tasks, cloud_providers)
metrics_aco["cost"] = best_cost_aco
metrics_pso = calculate_metrics(best_allocation_pso, tasks, cloud_providers)
metrics_pso["cost"] = best_cost_pso
metrics_tlb = calculate_metrics(best_allocation_tlb, tasks, cloud_providers)
metrics_tlb["cost"] = best_cost_tlb
metrics_hba = calculate_metrics(best_allocation_hba, tasks, cloud_providers)
metrics_hba["cost"] = best_cost_hba

print(f"Metrics (ACO): {metrics_aco}")
print(f"Metrics (PSO): {metrics_pso}")
print(f"Metrics (TLB): {metrics_tlb}")
print(f"Metrics (HBA): {metrics_hba}")


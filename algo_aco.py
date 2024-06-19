import numpy as np

class AntColonyOptimization:
    def __init__(self, tasks, cloud_providers, iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
        self.tasks = tasks
        self.cloud_providers = cloud_providers
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.num_tasks = len(tasks)
        self.num_providers = len(cloud_providers)
        self.pheromone = np.ones((self.num_tasks, self.num_providers))

    def fitness(self, allocation):
        cost = 0
        for i in range(self.num_providers):
            allocated_tasks = self.tasks[allocation == i]
            if len(allocated_tasks) == 0:
                continue
            resource_usage = np.sum(allocated_tasks[:, :3], axis=0)  # Only sum the first three columns
            if np.any(resource_usage > self.cloud_providers[i, :-1]):
                return float('inf')  # Overloaded provider
            cost += np.sum(resource_usage * self.cloud_providers[i, -1])
        return cost

    def run(self):
        best_allocation = None
        best_cost = float('inf')
        
        for iteration in range(self.iterations):
            allocations = []
            costs = []

            for ant in range(self.num_tasks):
                allocation = np.zeros(self.num_tasks, dtype=int)
                for task in range(self.num_tasks):
                    probabilities = (self.pheromone[task] ** self.alpha) * (1 / (self.fitness(allocation) + 1e-10) ** self.beta)
                    probabilities /= np.sum(probabilities)
                    allocation[task] = np.random.choice(self.num_providers, p=probabilities)
                
                allocations.append(allocation)
                costs.append(self.fitness(allocation))
            
            best_iteration_allocation = allocations[np.argmin(costs)]
            best_iteration_cost = np.min(costs)
            
            if best_iteration_cost < best_cost:
                best_cost = best_iteration_cost
                best_allocation = best_iteration_allocation
            
            self.pheromone *= (1 - self.evaporation_rate)
            for i in range(self.num_tasks):
                self.pheromone[i, best_iteration_allocation[i]] += 1 / (best_iteration_cost + 1e-10)
        
        return best_allocation, best_cost
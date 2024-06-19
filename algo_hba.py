import numpy as np

class HoneyBeeAlgorithm:
    def __init__(self, tasks, cloud_providers, iterations=100, num_bees=30, num_scouts=5, elite_sites=2, non_elite_sites=3):
        self.tasks = tasks
        self.cloud_providers = cloud_providers
        self.iterations = iterations
        self.num_bees = num_bees
        self.num_scouts = num_scouts
        self.elite_sites = elite_sites
        self.non_elite_sites = non_elite_sites
        self.num_tasks = len(tasks)
        self.num_providers = len(cloud_providers)

    def fitness(self, allocation):
        cost = 0
        for i in range(self.num_providers):
            allocated_tasks = self.tasks[allocation == i]
            resource_usage = np.sum(allocated_tasks[:, :3], axis=0)
            if np.any(resource_usage > self.cloud_providers[i, :-1]):
                return float('inf')  # Overloaded provider
            cost += np.sum(resource_usage * self.cloud_providers[i, -1])
        return cost

    def run(self):
        best_allocation = None
        best_cost = float('inf')
        
        for iteration in range(self.iterations):
            scouts = [np.random.randint(self.num_providers, size=self.num_tasks) for _ in range(self.num_scouts)]
            costs = [self.fitness(scout) for scout in scouts]
            
            elite_scouts = sorted(range(self.num_scouts), key=lambda x: costs[x])[:self.elite_sites]
            non_elite_scouts = sorted(range(self.num_scouts), key=lambda x: costs[x])[self.elite_sites:self.elite_sites+self.non_elite_sites]

            for elite in elite_scouts:
                for _ in range(self.num_bees // 2):
                    neighbor = scouts[elite].copy()
                    neighbor[np.random.randint(self.num_tasks)] = np.random.randint(self.num_providers)
                    neighbor_cost = self.fitness(neighbor)
                    if neighbor_cost < costs[elite]:
                        scouts[elite] = neighbor
                        costs[elite] = neighbor_cost
            
            for non_elite in non_elite_scouts:
                for _ in range(self.num_bees // 4):
                    neighbor = scouts[non_elite].copy()
                    neighbor[np.random.randint(self.num_tasks)] = np.random.randint(self.num_providers)
                    neighbor_cost = self.fitness(neighbor)
                    if neighbor_cost < costs[non_elite]:
                        scouts[non_elite] = neighbor
                        costs[non_elite] = neighbor_cost
            
            best_iteration_cost = min(costs)
            best_iteration_allocation = scouts[costs.index(best_iteration_cost)]
            
            if best_iteration_cost < best_cost:
                best_cost = best_iteration_cost
                best_allocation = best_iteration_allocation
        
        return best_allocation, best_cost

# Initialize and run HBA
# hba = HoneyBeeAlgorithm(tasks, cloud_providers)
# best_allocation_hba, best_cost_hba = hba.run()
# print(f"Best Cost (HBA): {best_cost_hba}")
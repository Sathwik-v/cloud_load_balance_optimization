import numpy as np

class ThrottledLoadBalancer:
    def __init__(self, tasks, cloud_providers, max_tasks_per_provider=50):
        self.tasks = tasks
        self.cloud_providers = cloud_providers
        self.max_tasks_per_provider = max_tasks_per_provider
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
        allocation = np.zeros(self.num_tasks, dtype=int)
        task_count = np.zeros(self.num_providers, dtype=int)

        for i in range(self.num_tasks):
            min_cost = float('inf')
            best_provider = 0

            for j in range(self.num_providers):
                if task_count[j] < self.max_tasks_per_provider:
                    allocation[i] = j
                    cost = self.fitness(allocation)
                    if cost < min_cost:
                        min_cost = cost
                        best_provider = j

            allocation[i] = best_provider
            task_count[best_provider] += 1

        return allocation, self.fitness(allocation)

# Initialize and run TLB
# tlb = ThrottledLoadBalancer(tasks, cloud_providers)
# best_allocation_tlb, best_cost_tlb = tlb.run()
# print(f"Best Cost (TLB): {best_cost_tlb}")
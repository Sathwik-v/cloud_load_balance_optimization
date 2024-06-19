import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, tasks, cloud_providers, iterations=100, num_particles=30, w=0.5, c1=1.5, c2=1.5):
        self.tasks = tasks
        self.cloud_providers = cloud_providers
        self.iterations = iterations
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_tasks = len(tasks)
        self.num_providers = len(cloud_providers)

        self.positions = np.random.randint(self.num_providers, size=(self.num_particles, self.num_tasks))
        self.velocities = np.random.rand(self.num_particles, self.num_tasks)
        self.p_best_positions = self.positions.copy()
        self.p_best_costs = np.full(self.num_particles, float('inf'))
        self.g_best_position = None
        self.g_best_cost = float('inf')

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
        for iteration in range(self.iterations):
            for i in range(self.num_particles):
                cost = self.fitness(self.positions[i])
                if cost < self.p_best_costs[i]:
                    self.p_best_costs[i] = cost
                    self.p_best_positions[i] = self.positions[i].copy()
                if cost < self.g_best_cost:
                    self.g_best_cost = cost
                    self.g_best_position = self.positions[i].copy()
            
            for i in range(self.num_particles):
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * np.random.rand() * (self.p_best_positions[i] - self.positions[i])
                    + self.c2 * np.random.rand() * (self.g_best_position - self.positions[i])
                )
                self.positions[i] = np.round(self.positions[i] + self.velocities[i]).astype(int) % self.num_providers

        return self.g_best_position, self.g_best_cost

# Initialize and run PSO
# pso = ParticleSwarmOptimization(tasks, cloud_providers)
# best_allocation_pso, best_cost_pso = pso.run()
# print(f"Best Cost (PSO): {best_cost_pso}")
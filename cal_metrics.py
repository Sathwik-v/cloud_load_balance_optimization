import numpy as np

def calculate_metrics(allocation, tasks, cloud_providers):
    response_times = []
    throughput = 0
    resource_utilization = np.zeros((len(cloud_providers), 3))  # [CPU, Memory, GPU]
    performance = 0

    for i in range(len(cloud_providers)):
        allocated_tasks = tasks[allocation == i]
        if len(allocated_tasks) == 0:
            continue
        resource_usage = np.sum(allocated_tasks[:, :3], axis=0)
        utilization = resource_usage / cloud_providers[i, :-1]
        response_time = np.mean(allocated_tasks[:, -1])
        throughput += len(allocated_tasks) / np.sum(allocated_tasks[:, -1])
        resource_utilization[i] = utilization
        performance += len(allocated_tasks) / (response_time * cloud_providers[i, -1])

        response_times.append(response_time)

    avg_response_time = np.mean(response_times)
    overall_throughput = throughput / len(cloud_providers)
    avg_resource_utilization = np.mean(resource_utilization, axis=0)
    overall_performance = performance / len(cloud_providers)

    return {
        "average_response_time": avg_response_time, 
        "throughput":overall_throughput, 
        "resource_utilization": avg_resource_utilization,
        "overall_performance": overall_performance
    }
import matplotlib.pyplot as plt

def plot_pareto_front(archive):
    """
    Plot the Pareto front using matplotlib.

    Args:
    - archive (Archive): Archive object containing the paths and their evaluation results.
    """
    # Extract distance and time values from the archive
    distances = []
    times = []
    for path, result in archive.paths_results_archive:
        distances.append(result.get('Distance', 0))
        times.append(result.get('Time', 0))

    # Plot Pareto front
    plt.scatter(distances, times)
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Pareto Front')
    plt.show()

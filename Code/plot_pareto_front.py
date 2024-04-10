import matplotlib.pyplot as plt

def plot_pareto_front(archive):
    """
    Plot the Pareto front using matplotlib.

    Args:
    - archive (Archive): Archive object containing the paths and their evaluation results.
    """

    # Extract distance and time values from the archive
    distances = [result.get('Distance', 0) for _, result in archive.paths_results_archive]
    times = [result.get('Time', 0) for _, result in archive.paths_results_archive]

    # Plot Pareto front
    plt.scatter(distances, times)
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Pareto Front')
    plt.show()

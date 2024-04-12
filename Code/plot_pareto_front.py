import matplotlib.pyplot as plt

def plot_pareto_front(archive):
    """
    Plot the Pareto front using matplotlib.

    Args:
    - archive (ParetoArchive): Archive object containing the paths and their evaluation results.
    """

    # Print contents of the Pareto archive for inspection
    print("Contents of Pareto Archive:")
    for path, result in archive.pareto_archive:
        print(f"Path: {path}, Result: {result}")

    # Extract distance and time values from the archive
    distances = [result.get('Distance', 0) for _, result in archive.pareto_archive]
    times = [result.get('Time', 0) for _, result in archive.pareto_archive]

    # Plot Pareto front
    plt.scatter(distances, times)
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Pareto Front')
    plt.show()
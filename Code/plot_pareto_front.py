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

    """# Plot Pareto front
    plt.plot(distances, times, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Pareto Front')
    plt.grid(True)"""
    # Plot Pareto front
    plt.scatter(distances, times)
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Pareto Front')

    # Check if the plot represents a Pareto front
    if is_pareto_front(distances, times):
        print("The plot represents a Pareto front.")
    else:
        print("The plot does not represent a Pareto front.")

    plt.show()


def is_pareto_front(distances, times):
    """
    Check if the given set of distances and times represents a Pareto front.

    Args:
    - distances (list): List of distance values.
    - times (list): List of time values.

    Returns:
    - bool: True if the given set represents a Pareto front, False otherwise.
    """
    pareto_front = True
    for i in range(len(distances)):
        for j in range(len(distances)):
            if i != j:
                # Check if point i dominates point j
                if distances[i] <= distances[j] and times[i] <= times[j] and (
                        distances[i] < distances[j] or times[i] < times[j]):
                    pareto_front = False
                    break
        if not pareto_front:
            break
    return pareto_front
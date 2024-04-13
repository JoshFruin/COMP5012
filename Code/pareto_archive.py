class ParetoArchive:

    def __init__(self):
        # Initialise the archive
        self.pareto_archive = []

    def add_result(self, path, result):
        # Create a tuple and then add it to the archive list
        path_result = (path, result)
        self.pareto_archive.append(path_result)

    def archive_print_results(self):
        for path, res in self.pareto_archive:
            print(res)

    def contains(self, solution):
        """
        Check if a solution already exists in the Pareto archive.

        Args:
        - solution: The solution to check for existence.

        Returns:
        - bool: True if the solution exists in the archive, False otherwise.
        """
        for path, _ in self.pareto_archive:
            if path == solution:
                return True
        return False



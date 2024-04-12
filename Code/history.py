
class History:
    def __init__(self):
        self.paths_results_history = []

    def add_solution(self, path, result):
        path_Result = (path, result)
        self.paths_results_history.append(path_Result)

    # rename to history
    def print_path(self):
        print(self.paths_results_history)

    def clear_history(self):
        """Clears all path and result data from the archive."""
        self.paths_results_history = []  # Reset to an empty list
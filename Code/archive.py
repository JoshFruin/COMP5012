
class Archive:
    def __init__(self):
        self.paths_results_archive = []

    def add_solution(self, path, result):
        self.paths_results_archive.append((path, result))

    def print_path(self):
        print(self.paths_results_archive)

    def clear_archive(self):
        """Clears all path and result data from the archive."""
        self.paths_results_archive = []  # Reset to an empty list
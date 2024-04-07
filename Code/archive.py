
class Archive:
    def __init__(self):
        self.paths_results_archive = []

    def add_solution(self, path, result):
        self.paths_results_archive.append((path, result))

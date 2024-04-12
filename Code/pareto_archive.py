
class ParetoArchive:

    def __init__(self):
        # initialise the archive
        self.pareto_archive = []

    def add_result(self, path, result):
        # create tuple then add to archive list
        path_result = (path, result)
        self.pareto_archive.append(path_result)

    def archive_print_results(self):
        for path, res in self.pareto_archive:
            print(res)
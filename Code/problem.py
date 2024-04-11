import matplotlib.pyplot as plt
import networkx as nx


class ShortestPathProblem:

    def __init__(self, problemMap):
        # networkX graph representing the city, nodes are places, edges are roads
        self.problemMap = problemMap

    def displayMap(self):
        print("Drawing Graph")
        positions = nx.spring_layout(self.problemMap, iterations=10)
        print("middle")
        nx.draw(self.problemMap, positions, with_labels=True, node_size=300)
        print("Layout Calculated")
        plt.show()

    def speedSwitcher(self, choice):
        switcher = {
            0: 0,
            1: 20,
            2: 30,
            3: 50,
            4: 90,
            5: 100,
            6: 120,
        }
        return switcher.get(choice, "Invalid edge")

    # evaluates the path/solution
    def evaluate(self, path):
        # set distance and time objectives
        totalTime = 0
        totalDist = 0

        # iterate through nodes
        for i in range(len(path)-1):
            startNode = path[i]  # get the start & end nodes
            endNode = path[i + 1]
            edgeData = self.problemMap.get_edge_data(startNode, endNode)  # work out edge data between them

            # get the distance and sL between nodes and speed limit of edge
            distance = edgeData.get("length", 0)
            speed_rank = edgeData.get("car", 0)

            # speed is in the form of 0-6, use speedSwitcher to get actual edge speed
            speed_limit = self.speedSwitcher(speed_rank)
            time = distance / speed_limit if speed_limit > 0 else 0

            totalDist += distance
            totalTime += time

        return {"Distance": totalDist, "Time": totalTime}

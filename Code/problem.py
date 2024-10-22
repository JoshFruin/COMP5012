import matplotlib.pyplot as plt
import networkx as nx


class ShortestPathProblem:

    def __init__(self, problemMap):
        # networkX graph representing the city, nodes are places, edges are roads
        self.problemMap = problemMap

    def displayMap(self):
        print("Drawing Graph")
        positions = nx.spring_layout(self.problemMap, iterations=100)
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
        totalCo2 = 0

        # iterate through nodes
        for i in range(len(path) - 1):
            startNode = path[i]  # get the start & end nodes
            endNode = path[i + 1]
            edgeData = self.problemMap.get_edge_data(startNode, endNode)  # work out edge data between them

            # get the distance and sL between nodes and speed limit of edge
            distance_m = edgeData.get("length", 0)
            speed_rank = edgeData.get("car", 0)
            co2 = edgeData.get('co2_emissions', 0)

            # speed is in the form of 0-6, use speedSwitcher to get actual edge speed
            speed_limit = self.speedSwitcher(speed_rank)
            distance_km = distance_m / 1000

            time_h = distance_km / speed_limit if speed_limit > 0 else 0
            time_s = time_h * 3600

            totalDist += distance_m
            totalTime += time_s
            totalCo2 += co2

        return {"Distance": totalDist, "Time": totalTime, "Co2_Emission": totalCo2}

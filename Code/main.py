# load necessary libraries
import networkx
import pandas as pd
import networkx as nx
import problem
from Code.ant_colony_optimiser import AntColony
from Code.archive import Archive

# load san fransisco map data
nodes_df = pd.read_csv("nodes_l.csv")
edges_df = pd.read_csv("edges_l.csv")

# print the data to test
print(nodes_df)
print(edges_df)

##### Test Graph ####

testGraph = nx.complete_graph(10)

networkMap = nx.Graph()

for index, row in nodes_df.iterrows():
    networkMap.add_node(
        row['node_id'],
        longitude=row['longitude'],
        latitude=row['latitude'],
        altitude=row['altitude']
    )

for index, row in edges_df.iterrows():
    networkMap.add_edge(
        # row['edge_id'],
        row['source'],  # source node id
        row['target'],  # target node id
        length=row['length'],
        car=row['car'],
        car_reverse=row['car_reverse'],
        bike=row['bike'],
        bike_reverse=row['bike_reverse'],
        foot=row['foot']
    )

# create and set objects
prob = problem.ShortestPathProblem(networkMap)
prob.displayMap()
archive = Archive()
optimiser = AntColony(graph=networkMap, num_ants=100)

# stores iterations results
progress_results = []

iterations = 10
sourceNode = 65282506
targetNode = 338898805

for i in range(iterations):
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob)
    print("Iteration complete")
    best_path, best_result = optimiser.get_best_path()  # Assuming you have this function
    progress_results.append(best_result)  # Or another metric you prefer
    optimiser.archive.clear()  # Clear archive for next iteration

print("YAY")

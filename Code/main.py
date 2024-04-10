# load necessary libraries
import pandas as pd
import networkx as nx
import problem
from ant_colony_optimiser import AntColony
from archive import Archive
from Mutation import select_mutation
from ant_colony_optimiser import AntColony
from archive import Archive

# load san fransisco map data
nodes_df = pd.read_csv("nodes_l.csv")
edges_df = pd.read_csv("edges_l.csv")

# print the data to test
print(nodes_df.dtypes)
print(edges_df.dtypes)

# ----------------------------    Test Graph

testGraph = nx.complete_graph(10)

# ------------------------
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
        row['source'],  # source node id
        row['target'],  # target node id
        edge_id=row['edge_id'],
        length=row['length'],
        car=row['car'],
        car_reverse=row['car_reverse'],
        bike=row['bike'],
        bike_reverse=row['bike_reverse'],
        foot=row['foot']
    )

# create and set objects
prob = problem.ShortestPathProblem(networkMap)
# prob.displayMap()
archive = Archive()
optimiser = AntColony(graph=networkMap, num_ants=100)

# stores iterations results
progress_results = []

mutation_type = int(input("Select mutation type (1: random selection, 2: swap, 3: insertion, 4: inversion): "))

mutation_func = select_mutation(mutation_type)
mutation_rate = 0.1  # Adjust as needed

# changeables
iterations = 2
sourceNode = 440853802
targetNode = 65316450

for i in range(iterations):

    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob, mutation_rate=mutation_rate,  mutation_func=mutation_func)
    print("\n Iteration complete \n")
    best_paths = optimiser.get_best_path()  # Assuming you have this function
    for path, result in best_paths:
        progress_results.append(result)  # Or another metric you prefer
    optimiser.archive.clear_archive()  # Clear archive for next iteration

print("YAY")
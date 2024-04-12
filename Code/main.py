# load necessary libraries
import pandas as pd
import networkx as nx
import problem
from Code.ant_colony_optimiser import AntColony
from Code.history import History
from Code.pareto_archive import ParetoArchive
import random


# load san fransisco map data
nodes_df = pd.read_csv("nodes_l.csv")
edges_df = pd.read_csv("edges_l.csv")

# print the data to test
print(nodes_df.dtypes)
print(edges_df.dtypes)

# ----------------------------    Test Graph ---------------------

# Jobs for tomorrow
# Make the test graph have the same attributes for nods and edges as the real graph, sos not to change any of the functions.
# test with test graph
# add more randomness to the ants ADD AN ACTUAL EXPLORATION RATE

# Create an empty graph
testG = nx.Graph()

# Add 20 nodes
testG.add_nodes_from(range(1, 21))
#
# Add edges between all pairs of nodes with random lengths and times
#for u in range(1, 21):
#    for v in range(u + 1, 21):
#        length = random.randint(1, 20)  # Random length
#        time = random.randint(1, 6)     # Random time between 1 and 6
#        testG.add_edge(u, v, length=length, time=time)  # Assign random length and time to each edge#
#
# Print edge data
#for u, v, data in testG.edges(data=True):
#    print(f"Edge between nodes {u} and {v} with length {data['length']} and time {data['time']}")

# Draw the graph (optional)
#nx.draw(testG, with_labels=True)

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

history = History()
optimiser = AntColony(graph=networkMap, num_ants=100)
pareto_front_archive = ParetoArchive()

# stores iterations results
iterations_best_results = []

# changeable variables
iterations = 50
sourceNode = 65328679
targetNode = 65303655

for i in range(iterations):

    # run the optimiser for 1 iteration
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob)
    print("\n Iteration complete \n")
    # get the iterations best path results for sexy pareto graph, WARNING DOES NOT WORK
    iterations_best_results.append(optimiser.get_best_path())
    # Clear Ant path history's for next iteration
    optimiser.history.clear_history()
    print("Iterations archive contains: ", iterations_best_results)

print("YAY")
pareto_front_archive.archive_print_results()

# load necessary libraries
import pandas as pd
import networkx as nx
import problem
from Code.ant_colony_optimiser import AntColony
from Code.history import History
from Code.pareto_archive import ParetoArchive


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
history = History()
optimiser = AntColony(graph=networkMap, num_ants=250)
pareto_front_archive = ParetoArchive()

# stores iterations results
progress_results = []

# changeable variables
iterations = 5
sourceNode = 65328679
targetNode = 258967500

for i in range(iterations):

    # run the optimiser for 1 iteration
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob)
    print("\n Iteration complete \n")
    # get the iterations best path results for sexy pareto graph, WARNING DOES NOT WORK
    iterations_best_results = optimiser.get_best_path()
    # Clear Ant path history's for next iteration
    optimiser.history.clear_history()
    # print("Iterations archive contains: ", iterations_best_results)

print("YAY")
pareto_front_archive.archive_print_results()

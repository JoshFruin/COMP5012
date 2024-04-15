# load necessary libraries
import pandas as pd
import problem
from Code.ant_colony_optimiser import AntColony
from Code.history import History
from Code.pareto_archive import ParetoArchive
from Code.Map import MapMaker

# load san fransisco map data
nodes_df = pd.read_csv("nodes_tavi.csv")
edges_df = pd.read_csv("edges_tavi.csv")

# print the data to test
print(nodes_df.dtypes)
print(edges_df.dtypes)

# create the networkX map with nodes and edge data
my_map = MapMaker(nodes_df, edges_df)
my_map.add_nodes()
my_map.add_edges()

# create and set objects
prob = problem.ShortestPathProblem(my_map.network_map)
history = History()
optimiser = AntColony(graph=my_map.network_map, num_ants=100)
pareto_front_archive = ParetoArchive()

# stores iterations results
iterations_best_results = []

# changeable variables
iterations = 50
sourceNode = 290344782
targetNode = 6848266087

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

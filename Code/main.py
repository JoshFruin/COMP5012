# load necessary libraries
import pandas as pd
import problem
import matplotlib.pyplot as plt
from Code.ant_colony_optimiser import AntColony
from Code.history import History
from Code.pareto_archive import ParetoArchive
from Code.Map import MapMaker

# load san fransisco map data
nodes_df = pd.read_csv("nodes_test.csv")
edges_df = pd.read_csv("edges_test.csv")

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
pareto_front_archive = ParetoArchive()
optimiser = AntColony(graph=my_map.network_map, pareto_Archive=pareto_front_archive, num_ants=100)

# stores iterations results
iterations_best_results = []
iteration_distances = []

# changeable variables
iterations = 100
# 290344782
sourceNode = 5
# 6848266087
targetNode = 90

optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob, iterations=iterations)

#for i in range(iterations):

    # run the optimiser for 1 iteration
 #   iteration_distances.append(optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob))
  #  print("\n Iteration complete \n")
    # get the iterations best path results for sexy pareto graph, WARNING DOES NOT WORK
   # iterations_best_results.append(optimiser.get_best_path())
    # Clear Ant path history's for next iteration
    #optimiser.history.clear_history()
    # print("Iterations archive contains: ", iterations_best_results)

print("YAY")
pareto_front_archive.archive_print_results()

# create lists for plotting
archive_time_values = []
archive_co2_values = []

# iterate through archive, fill the plotting lists
for _, results in pareto_front_archive.pareto_archive:
    time_val = results['Time']
    co2_val = results['Co2_Emission']

    archive_time_values.append(time_val)
    archive_co2_values.append(co2_val)

# plot the pareto front
plt.scatter(archive_time_values, archive_co2_values)
plt.xlabel("Time in seconds")
plt.ylabel("Co2 emissions")
plt.title("Pareto Front of Time vs Co2 Emissions")
plt.show()

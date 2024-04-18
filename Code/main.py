# load necessary libraries
import pandas as pd
import problem
import matplotlib.pyplot as plt
from Code.ant_colony_optimiser import AntColony
from Code.history import History
from Code.pareto_archive import ParetoArchive
from Code.Map import MapMaker


# -------------------TO LAUREN/ MARKER ------------------------
# Firstly hello, I hope you're having a lovely day :)
# If you want to change any of the weights for heuristics, exploration, evaporation
# this is all in the ant_colony_optimisaiton.py file in the constructor.
# changing the number of iterations can be done in this main file under the 'changeable variables' comment
# source and target nodes can also be changed here.
# to select new source and target nodes please refer to the node ids in the nodes_test.csv, they range from 1-100
# changing the number of ants can be done in the arguments when the ACO object is created in main, default 100
# running 1 iteration takes roughly 2 seconds so expect 1000 iterations to take a while to complete
# Graphs will output after use of the averages for distance, time and co2 along with the pareto front.
# Thank you for using out ACO!
#------------------------------------------------------------------

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
sourceNode = 76
# 6848266087
targetNode = 90

optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob, iterations=iterations)

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

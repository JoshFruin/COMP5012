# load necessary libraries
import pandas as pd
import problem
import matplotlib.pyplot as plt
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
pareto_front_archive = ParetoArchive()
optimiser = AntColony(graph=my_map.network_map, pareto_Archive=pareto_front_archive, num_ants=100)

# stores iterations results
iterations_best_results = []

# changeable variables
iterations = 200
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
    # print("Iterations archive contains: ", iterations_best_results)

print("YAY")
<<<<<<< HEAD
#%%
# test graph
import networkx as nx
import random

# Create an empty graph
testG = nx.Graph()

# Add 20 nodes
testG.add_nodes_from(range(1, 21))

# Add edges between all pairs of nodes with random lengths and times
for u in range(1, 21):
    for v in range(u + 1, 21):
        length = random.randint(1, 20)  # Random length
        time = random.randint(1, 6)     # Random time between 1 and 6
        testG.add_edge(u, v, length=length, time=time)  # Assign random length and time to each edge

# Print edge data
for u, v, data in testG.edges(data=True):
    print(f"Edge between nodes {u} and {v} with length {data['length']} and time {data['time']}")

# Draw the graph (optional)
nx.draw(testG, with_labels=True)

#%%
=======
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
>>>>>>> James-Dev

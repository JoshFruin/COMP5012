# load necessary libraries
import pandas as pd
import problem
import matplotlib.pyplot as plt
from ant_colony_optimiser import AntColony
from history import History
from pareto_archive import ParetoArchive
from Map import MapMaker
#%%
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
#%%
# create and set objects
prob = problem.ShortestPathProblem(my_map.network_map) #my_map.network_map
history = History()
pareto_front_archive = ParetoArchive()
optimiser = AntColony(graph=my_map.network_map, pareto_Archive=pareto_front_archive, num_ants=100) #my_map.network_map

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
    # print("Iterations archive contains: ", iterations_best_results)

print("YAY")

#%%
# test graph
import networkx as nx
import random

# Create an empty graph
G = nx.Graph()

# Add 100 nodes
# Add nodes with node IDs
for i in range(1, 101):
    G.add_node(i, node_id=i)


# Add edges between all pairs of nodes with random distances and speed limits
for u in range(1, 101):
    for v in range(u + 1, 101):
        distance = random.randint(1, 100)  # Random distance in kilometers
        speed_limit_key = random.randint(0,6)# Random speed limit between 0 and 6 km/h
        G.add_edge(u, v, distance=distance, speed_limit=speed_limit)
        
# Print edge data
for u, v, data in G.edges(data=True):
    print(f"Edge between nodes {u} and {v} with distance {data['distance']} km and speed limit {data['speed_limit']} km/h")


# Draw the graph (optional)
nx.draw(G, with_labels=True)

#%%
# plot the pareto archive (showing the pareto front)
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

#%%

import matplotlib.pyplot as plt
import numpy as np

# Assuming pheromones is a dictionary containing pheromone levels for each edge
# Calculate maximum and minimum pheromone levels
max_pheromone = max(pheromones.values())
min_pheromone = min(pheromones.values())

# Normalize pheromone levels to range between 0 and 1
normalized_pheromones = {edge: (pheromone - min_pheromone) / (max_pheromone - min_pheromone) for edge, pheromone in pheromones.items()}

# Create a grid representing the edges of your graph
# For simplicity, assume the grid is a 2D array where each cell corresponds to an edge
# Replace this with the actual grid representing your graph
grid_size = (10, 10)  # Adjust grid size as needed
grid = np.zeros(grid_size)

# Assign normalized pheromone levels to the grid
for edge, pheromone in normalized_pheromones.items():
    x, y = edge  # Extract edge coordinates
    grid[x, y] = pheromone

# Plot the heatmap
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()  # Add colorbar to show pheromone levels
plt.title('Pheromone Trails Heatmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


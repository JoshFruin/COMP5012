import pandas as pd
import networkx as nx
import random
from ant_colony_optimiser import AntColony
from pareto_archive import ParetoArchive
#from Mutation import select_mutation, mutate_solution
import plot_pareto_front  # Import the module for plotting Pareto front
from problem import ShortestPathProblem

# Load San Francisco map data
nodes_df = pd.read_csv(r"C:\Users\Ciaran Keating\Documents\GitHub\COMP5012\Code\nodes_l.csv")
edges_df = pd.read_csv(r"C:\Users\Ciaran Keating\Documents\GitHub\COMP5012\Code\edges_l.csv")

# Print the data to test
print(nodes_df.dtypes)
print(edges_df.dtypes)

# Create the network map
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
        row['source'],  # Source node id
        row['target'],  # Target node id
        edge_id=row['edge_id'],
        length=row['length'],
        car=row['car'],
        car_reverse=row['car_reverse'],
        bike=row['bike'],
        bike_reverse=row['bike_reverse'],
        foot=row['foot']
    )

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


# Create problem instance and set objects
prob = ShortestPathProblem(networkMap)
pareto_front_archive = ParetoArchive()
optimiser = AntColony(graph=networkMap, num_ants=100)

# Mutation parameters
"""mutation_type = int(input("Select mutation type (1: random selection, 2: swap, 3: insertion, 4: inversion): ")) # Gives user choice what mutation they want for testing
mutation_func = select_mutation(mutation_type)"""
mutation_rate = 0.1  # Adjust as needed

# Other Hyperparameters
iterations = 2
sourceNode = 440853802
targetNode = 65316450

# Collect results after each iteration
for i in range(iterations):
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob) #, mutation_rate=mutation_rate, mutation_func=mutation_func)
    print(f"\n Iteration complete \n")
    # get the iterations best path results for sexy pareto graph, WARNING DOES NOT WORK
    iterations_best_results = optimiser.get_best_path()
    # Clear Ant path history's for next iteration
    optimiser.history.clear_history()
    # print("Iterations archive contains: ", iterations_best_results)

print("YAY")

# Plot Pareto front at the end of the loop
pareto_front_archive.archive_print_results()

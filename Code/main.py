# load necessary libraries
import pandas as pd
import networkx as nx
import problem
from ant_colony_optimiser import AntColony
from archive import Archive

# load san francisco map data
nodes_df = pd.read_csv("nodes_l.csv")
edges_df = pd.read_csv("edges_l.csv")

# ----------------------------    Test Graph
# Display basic information about the dataframes
print("Nodes DataFrame:")
print(nodes_df.head())
print("\nEdges DataFrame:")
print(edges_df.head())

# Construct the graph
networkMap = nx.Graph()

# Add nodes to the graph
for index, row in nodes_df.iterrows():
    networkMap.add_node(
        row['node_id'],
        longitude=row['longitude'],
        latitude=row['latitude'],
        altitude=row['altitude']
    )

# Add edges to the graph
for index, row in edges_df.iterrows():
    print(f"Processing edge {index+1}/{len(edges_df)}")  # Print edge being processed
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

    # Check if edge data is being found correctly
    edge_data = networkMap.get_edge_data(row['source'], row['target'])
    if edge_data is None:
        print("Edge data not found. Check your graph representation.")
    else:
        print("Edge data found successfully.")

# Display basic information about the graph
print("\nGraph Information:")
print("Number of nodes:", networkMap.number_of_nodes())
print("Number of edges:", networkMap.number_of_edges())

# Print some edge data to verify
print("\nSample Edge Data:")
for edge in networkMap.edges(data=True):
    print("Edge:", edge)

# print the data to test
print(nodes_df.dtypes)
print(edges_df.dtypes)
print("Data test successful!")

# create and set objects
prob = problem.ShortestPathProblem(networkMap)
# prob.displayMap()
archive = Archive()
optimiser = AntColony(graph=networkMap, num_ants=100)

# stores iterations results
progress_results = []

iterations = 10
sourceNode = 440853802
targetNode = 338898805

for i in range(iterations):
    print(f"Iteration {i + 1}")
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob)
    print("Iteration complete")
    best_path, best_result = optimiser.get_best_path()  # Assuming you have this function
    progress_results.append(best_result)  # Or another metric you prefer
    optimiser.archive.clear()  # Clear archive for next iteration

print("YAY")

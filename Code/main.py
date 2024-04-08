# load necessary libraries
import pandas as pd
import networkx as nx
import problem
from ant_colony_optimiser import AntColony
from archive import Archive
from problem import ShortestPathProblem

# load san francisco map data
nodes_df = pd.read_csv("nodes_l.csv")
edges_df = pd.read_csv("edges_l.csv")

# Display basic information about the dataframes
print("Nodes DataFrame:")
print(nodes_df.head())
print("\nEdges DataFrame:")
print(edges_df.head())

# Construct the graph
networkMap = nx.Graph()

# Add nodes to the graph
print("\nAdding nodes to the graph:")
for index, row in nodes_df.iterrows():
    print("Adding node:", row['node_id'])
    networkMap.add_node(
        row['node_id'],
        longitude=row['longitude'],
        latitude=row['latitude'],
        altitude=row['altitude']
    )

# Add edges to the graph
print("\nAdding edges to the graph:")
total_edges = len(edges_df)
for idx, row in edges_df.iterrows():
    # Fill missing values with 0
    row.fillna(0, inplace=True)

    print(f"Processing edge {idx + 1}/{total_edges}")
    print("Adding edge from", row['source'], "to", row['target'])
    if idx >= total_edges:
        print("Reached end of edges DataFrame. Exiting loop.")
        break
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

# Create the ShortestPathProblem object
prob = ShortestPathProblem(networkMap)

# Visualize the graph
prob.visualize_graph()

# Verify edge data
print("\nVerifying Edge Data:")
for edge in networkMap.edges(data=True):
    if 'length' not in edge[2]:
        print("Edge data not found for edge:", edge)
        print("Check your graph representation.")
        break
else:
    print("Edge data found for all edges.")

# Display basic information about the graph
print("\nGraph Information:")
print("Number of nodes:", networkMap.number_of_nodes())
print("Number of edges:", networkMap.number_of_edges())

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
    print(f"\nIteration {i + 1}")
    optimiser.run(source_node=sourceNode, target_node=targetNode, problem=prob)
    print("Iteration complete")
    best_path, best_result = optimiser.get_best_path()  # Assuming you have this function
    progress_results.append(best_result)  # Or another metric you prefer
    optimiser.archive.clear()  # Clear archive for next iteration

print("YAY")

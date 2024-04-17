# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:43:10 2024

@author: russe
"""
# generate csv's that relate to a networkx graph that has 100 nodes with all nodes connected by an edge that 
# have the same data attributes as taken from an example osm file.

import pandas as pd
import random

# Number of nodes
num_nodes = 100

# Generate nodes data
nodes_data = {
    'id': list(range(1, num_nodes + 1)),  # Node IDs
    'longitude': [random.uniform(-180, 180) for _ in range(num_nodes)],  # Random longitude values
    'latitude': [random.uniform(-90, 90) for _ in range(num_nodes)]  # Random latitude values
}

# Generate edges data
edges_data = {
    'edge_id': [],  # Edge IDs
    'source_node_id': [],  # Source node IDs
    'target_node_id': [],  # Target node IDs
    'length': [],  # Length values
    'car': [],  # Car values
    'car_reverse': [],  # Car reverse values
    'bike': [],  # Bike values
    'bike_reverse': [],  # Bike reverse values
    'foot': []  # Foot values
}

# Generate edges between every pair of nodes
edge_id = 1
for u in range(1, num_nodes + 1):
    for v in range(u + 1, num_nodes + 1):
        edges_data['edge_id'].append(edge_id)
        edges_data['source_node_id'].append(u)
        edges_data['target_node_id'].append(v)
        edges_data['length'].append(random.randint(1, 100))
        edges_data['car'].append(random.randint(1, 6))
        edges_data['car_reverse'].append(random.randint(1, 6))
        edges_data['bike'].append(random.randint(1, 6))
        edges_data['bike_reverse'].append(random.randint(1, 6))
        edges_data['foot'].append(random.randint(1, 6))
        edge_id += 1

# Create DataFrames
nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)

# Save DataFrames to CSV files
nodes_df.to_csv('nodes.csv', index=False)
edges_df.to_csv('edges.csv', index=False)



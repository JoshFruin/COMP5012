# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:49:30 2024

@author: russe
"""

# data checking 
#%%
import pandas as pd

# Get unique nodes from the edges DataFrame
unique_nodes_in_edges = set(edges_df['source']).union(set(edges_df['target']))

# Check if all nodes from the nodes DataFrame are present in the unique_nodes_in_edges set
missing_nodes = set(nodes_df['node_id']) - unique_nodes_in_edges

if len(missing_nodes) == 0:
    print("All nodes have at least one edge attached to them.")
else:
    print("There are nodes without edges attached to them:")
    print(missing_nodes)
#%%
# has data been parsed properly from csv's to netX
# Check if all nodes from DataFrame are present in the NetworkX graph
missing_nodes = set(nodes_df['node_id']) - set(networkMap.nodes())

if len(missing_nodes) == 0:
    print("All nodes from the DataFrame are present in the NetworkX graph.")
else:
    print("Nodes from the DataFrame not found in the NetworkX graph:")
    print(missing_nodes)

# Check if all edges from DataFrame are present in the NetworkX graph
missing_edges = set(zip(edges_df['source'], edges_df['target'])) - set(networkMap.edges())

if len(missing_edges) == 0:
    print("All edges from the DataFrame are present in the NetworkX graph.")
else:
    print("Edges from the DataFrame not found in the NetworkX graph:")
    print(missing_edges)
#%%
# print edge data of edges that were not parsed properly for comparison

# Filter the edges DataFrame to select rows with edges not found in the NetworkX graph
missing_edges_df = edges_df[~edges_df.apply(lambda row: (row['source'], row['target']) in missing_edges, axis=1)]

# Print the filtered DataFrame
print("Edges not parsed to the NetworkX graph:")
print(missing_edges_df)
# most data not parsed into networkx graph
#%%
# Check for duplicate rows where all values in each column are the same
duplicate_rows = edges_df[edges_df.duplicated(keep=False)] # no true duplicates
#%%
# check data types

print(edges_df.dtypes)
#%%
# print the row of the first missing edge for inspection
# Locate the row where the source node is 65317750 and the target node is 65290263
specific_row = edges_df[(edges_df['source'] == 65317750) & (edges_df['target'] == 65290263)]

# Print the specific row
print("The line of the edges DataFrame where the source node is 65317750 and the target node is 65290263:")
print(specific_row)
#%%
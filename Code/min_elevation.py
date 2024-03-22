# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:37:10 2024

@author: russe
"""

import networkx as nx
import numpy as np
from problem import ShortestPathProblem, speedSwitcher
from elevation import ElevationData

# Initialize elevation data for San Francisco map (assuming it's instantiated in your code)
elevation_data = ElevationData("North_America")

# Function to compute total elevation change along a path
def compute_total_elevation_change(path):
    total_elevation_change = 0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        # Retrieve elevation at start and end coordinates
        start_lat, start_lon = G.nodes[start_node]['latitude'], G.nodes[start_node]['longitude']
        end_lat, end_lon = G.nodes[end_node]['latitude'], G.nodes[end_node]['longitude']
        start_elevation = elevation_data.altitude(start_lat, start_lon)
        end_elevation = elevation_data.altitude(end_lat, end_lon)
        # Compute elevation change and accumulate
        elevation_change = abs(end_elevation - start_elevation)
        total_elevation_change += elevation_change
    return total_elevation_change

# Your ant colony optimization algorithm here, generating paths

# Example path
path = [1, 2, 3, 4, 5]  # Example path, replace with generated paths from your algorithm

# Compute total elevation change along the path
total_elevation_change = compute_total_elevation_change(path)

print("Total elevation change along path:", total_elevation_change)

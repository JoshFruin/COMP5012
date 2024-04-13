# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:58:47 2024

@author: russe
"""

# this script parses .osm files into two csv's: edges and nodes. Also extracts length of edges info and accesibility info
# at the moment these accessiblity criterea are defaulted to 0.
        
#%%
import xml.etree.ElementTree as ET
import csv

# Define mapping for accessibility values
accessibility_mapping = {
    'residential': 1,
    'tertiary': 2,
    'secondary': 3,
    'primary': 4,
    'trunk': 5,
    'motorway': 6,
    'cycleway': 3,  # Example mapping for bike accessibility
    # Add more mappings as needed
}

# Parse .osm file
tree = ET.parse('tavi.osm')
root = tree.getroot()

# Open CSV files for writing
with open('nodes.csv', 'w', newline='') as nodes_file, open('edges.csv', 'w', newline='') as edges_file:
    nodes_writer = csv.writer(nodes_file)
    edges_writer = csv.writer(edges_file)

    # Write header row for nodes.csv
    nodes_writer.writerow(['id', 'longitude', 'latitude', 'altitude'])

    # Write header row for edges.csv
    edges_writer.writerow(['id', 'source_node_id', 'target_node_id', 'length', 'car', 'car_reverse', 'bike', 'bike_reverse', 'foot'])

    # Extract nodes information
    for node in root.findall('.//node'):
        node_id = node.attrib['id']
        longitude = node.attrib['lon']
        latitude = node.attrib['lat']
        altitude = node.attrib.get('altitude', '')  # You may need to handle altitude differently if it's not provided in the .osm file
        nodes_writer.writerow([node_id, longitude, latitude, altitude])

    # Extract edges information
    for way in root.findall('.//way'):
        way_id = way.attrib['id']
        nodes = [nd.attrib['ref'] for nd in way.findall('.//nd')]
        length = len(nodes)  # = length of edge 
        # Initialize accessibility values
        car = 0
        car_reverse = 0
        bike = 0
        bike_reverse = 0
        foot = 0
        # Extract accessibility information for each edge
        for tag in way.findall('.//tag'):
            if tag.attrib.get('k') == 'highway':
                # Map the value of the 'highway' tag to the corresponding accessibility code
                highway_type = tag.attrib.get('v')
                if highway_type in accessibility_mapping:
                    car = accessibility_mapping[highway_type]
                    car_reverse = accessibility_mapping[highway_type]
            elif tag.attrib.get('k') == 'cycleway':
                bike = 1  # Example value for bike accessibility
                bike_reverse = 1  # Example value for reverse direction
            elif tag.attrib.get('k') == 'foot':
                foot = 1  # Example value for foot accessibility
        # Write edge information to edges.csv
        edges_writer.writerow([way_id, nodes[0], nodes[-1], length, car, car_reverse, bike, bike_reverse, foot])
#%%

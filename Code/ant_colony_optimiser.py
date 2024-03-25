# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:20 2024

@author: russe
"""

##### format for ant-colony optimiser

# Initialize ants
ants = initialize_ants()

# Main loop of the ant-colony optimization algorithm
for iteration in range(max_iterations):
    # Construct solutions
    for ant in ants:
        ant.construct_solution()

    # Evaluate solutions
    for ant in ants:
        ant.evaluate_solution()

    # Update pheromone
    update_pheromone(ants)

    # Evaporation
    evaporate_pheromone()

# Obtain Pareto front from the solutions found by ants
pareto_front = extract_pareto_front(ants)

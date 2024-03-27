# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:29:08 2024

@author: russe
"""
#%%
# Create the objective function
objective_function = LengthTimeObjectiveFunction(ShortestPathProblem(networkMap))

# Create the Ant Colony Optimization instance
aco = AntColony(networkMap, objective_function)

# Run the optimization
aco.run(num_iterations=100)

# Plot the Pareto front
archive_objectives = np.array(aco.archive.objective_values)
plt.scatter(archive_objectives[:, 0], archive_objectives[:, 1])
plt.xlabel('Length')
plt.ylabel('Time')
plt.title('Pareto Front for Multi-Objective Shortest Path Problem')
plt.show()
#%%
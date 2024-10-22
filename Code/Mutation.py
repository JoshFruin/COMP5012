"""
This File contains all the mutations we could find that could work for our optimisation problem so that we can find, which mutation will give us
the optimal result.
"""
import numpy as np
import random

def random_selection_mutation(solution):
  """
  Performs random selection mutation on a solution path.

  Args:
      solution: A list representing the current solution path.

  Returns:
      A new list with the mutated solution path.
  """
  if random.random() < mutation_rate:  # Apply mutation with a probability
    city1 = random.randint(0, len(solution) - 1)
    city2 = random.randint(0, len(solution) - 1)
    while city1 == city2:
      city2 = random.randint(0, len(solution) - 1)  # Ensure different cities
    solution[city1], solution[city2] = solution[city2], solution[city1]
  return solution

def swap_mutation(solution):
  """
  Performs swap mutation on a solution path.

  Args:
      solution: A list representing the current solution path.

  Returns:
      A new list with the mutated solution path.
  """
  if random.random() < mutation_rate:
    city1 = random.randint(0, len(solution) - 1)
    city2 = random.randint(0, len(solution) - 1)
    while city1 == city2:
      city2 = random.randint(0, len(solution) - 1)  # Ensure different cities
    solution[city1], solution[city2] = solution[city2], solution[city1]
  return solution

def insertion_mutation(solution):
  """
  Performs insertion mutation on a solution path.

  Args:
      solution: A list representing the current solution path.

  Returns:
      A new list with the mutated solution path.
  """
  if random.random() < mutation_rate:
    city_to_move = random.randint(0, len(solution) - 1)
    solution.pop(city_to_move)
    insert_position = random.randint(0, len(solution))
    solution.insert(insert_position, city_to_move)
  return solution

def inversion_mutation(solution):
  """
  Performs inversion mutation on a solution path.

  Args:
      solution: A list representing the current solution path.

  Returns:
      A new list with the mutated solution path.
  """
  if random.random() < mutation_rate:
    start = random.randint(0, len(solution) - 2)
    end = random.randint(start + 1, len(solution) - 1)
    subsequence = solution[start:end]
    subsequence.reverse()
    solution[start:end] = subsequence
  return solution

# Example usage (assuming you have a solution path as a list)
mutation_rate = 0.05  # Adjust mutation rate as needed
mutated_solution = random_selection_mutation(solution.copy())  # Operate on a copy
# Or use any other mutation function (swap_mutation, insertion_mutation, inversion_mutation)

def evaporate_pheromones(pheromones, evaporation_rate):
  """
  Evaporates pheromone levels on all edges by a given rate.

  Args:
  - pheromones (list of lists): Matrix representing pheromone levels on edges.
  - evaporation_rate (float): Rate at which pheromones evaporate.

  Returns:
  - updated_pheromones (list of lists): Matrix with updated pheromone levels.
  """
  for i in range(len(pheromones)):
    for j in range(len(pheromones[i])):
      pheromones[i][j] *= (1 - evaporation_rate)
  return pheromones


# Example usage
pheromones = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
evaporation_rate = 0.1
new_pheromones = evaporate_pheromones(pheromones, evaporation_rate)
print(new_pheromones)


def global_update_pheromones(pheromones, best_solution, evaporation_rate, delta_pheromone):
  """
  Performs global pheromone update, reinforcing pheromones on edges of the best solution found.

  Args:
  - pheromones (list of lists): Matrix representing pheromone levels on edges.
  - best_solution (list of tuples): Edges of the best solution found.
  - evaporation_rate (float): Rate at which pheromones evaporate.
  - delta_pheromone (float): Amount of pheromone to deposit on edges of the best solution.

  Returns:
  - updated_pheromones (list of lists): Matrix with updated pheromone levels.
  """
  for i in range(len(pheromones)):
    for j in range(len(pheromones[i])):
      if (i, j) in best_solution or (j, i) in best_solution:
        pheromones[i][j] = (1 - evaporation_rate) * pheromones[i][j] + delta_pheromone
  return pheromones


# Example usage
pheromones = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
best_solution = [(0, 1), (1, 2), (2, 0)]
evaporation_rate = 0.1
delta_pheromone = 1.0
updated_pheromones = global_update_pheromones(pheromones, best_solution, evaporation_rate, delta_pheromone)
print(updated_pheromones)


def adapt_parameters(pheromones, evaporation_rate, convergence_rate):
  """
  Adapts parameters (e.g., evaporation rate) based on the convergence rate.

  Args:
  - pheromones (list of lists): Matrix representing pheromone levels on edges.
  - evaporation_rate (float): Rate at which pheromones evaporate.
  - convergence_rate (float): Measure of convergence.

  Returns:
  - adapted_pheromones (list of lists): Matrix with adapted pheromone levels.
  - adapted_evaporation_rate (float): Adapted evaporation rate.
  """
  # Example adaptive mechanism
  if convergence_rate < 0.2:
    evaporation_rate *= 0.9
  elif convergence_rate > 0.8:
    evaporation_rate *= 1.1
  return pheromones, evaporation_rate


# Example usage
pheromones = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
evaporation_rate = 0.1
convergence_rate = 0.3
adapted_pheromones, adapted_evaporation_rate = adapt_parameters(pheromones, evaporation_rate, convergence_rate)
print(adapted_pheromones)
print("Adapted evaporation rate:", adapted_evaporation_rate)

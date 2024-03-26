# Mutations
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

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:20 2024

@author: R and J
"""
import random
import archive


# %%
######################

# this is the functional class, others are drafts or templates
def dominates(u, v):
    """
    Checks if solution 'u' dominates solution 'v' in a multi-objective context.
    """
    return (u["Distance"] <= v["Distance"] and u["Time"] <= v["Time"] and
            (u["Distance"] < v["Distance"] or u["Time"] < v["Time"]))


class AntColony:

    def __init__(self, graph, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5):
        """
        Initialize the Ant Colony Optimization algorithm.

        Args:
        - graph: Graph representing the problem domain.
        - objective_function: Function to evaluate the quality of a solution.
        - num_ants (int): Number of ants in the colony (default is 10).
        - alpha (float): Weight of pheromone in ant decision making (default is 1).
        - beta (float): Weight of heuristic information in ant decision making (default is 2).
        - evaporation_rate (float): Rate at which pheromones evaporate (default is 0.5).
        """
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromones = {node: {neighbor: 1 for neighbor in graph.neighbors(node)} for node in graph.nodes}
        self.archive = archive.Archive()  # init the ant paths archive
        self.distance_weight = 0.5  # the importance of distance vs time, scale: 0-1
        self.time_weight = 0.5

    def run(self, source_node, target_node, problem):

        # runs all the ants in the iteration
        for anti in range(self.num_ants):
            self.run_ant(source_node, target_node, problem)

        # update all pheromones
        self.update_pheromones()

    def initialize_pheromones(self):
        """
        Initialize pheromone levels on edges of the graph.

        Returns:
        - pheromones (dict): Dictionary storing pheromone levels on edges.
        """
        pheromones = {}  # Dict that stores pheromone levels as a tuple
        for edge in self.graph.edges:  # Iterates through all edges
            u, v = edge  # Set edge source and target IDs
            pheromones[(u, v)] = 1  # Start edge pheromone with a uniform base value of 1
        return pheromones  # Keep for now, perhaps not needed

    def _select_next_node(self, ant):
        """
        Select the next node for the ant to move to based on pheromone levels and heuristic information.

        Args:
        - ant (dict): Ant's information including current node and visited nodes.

        Returns:
        - next_node: Next node selected for the ant to move to.
        """
        neighbors = self.graph.neighbors(ant['current_node'])  # Get current node and look at neighboring nodes
        unvisited = [node for node in neighbors if
                     node not in ant['visited']]  # Create list of unvisited neighboring nodes

        probabilities = {}  # Initialize probability dictionary
        total_prob = 0  # Initialize variable (should end up as 1)

        for node in unvisited:  # Loop through unvisited nodes
            edge_data = self.graph.get_edge_data(ant['current_node'],
                                                 node)  # Get edge data between current node and target unvisited node
            distance = edge_data.get('length', 1)  # Default to 1 if no data is available
            speed_limit = edge_data.get('car', 0)  # We need to change these to equal our CSV column names

            # Heuristics
            time = distance / speed_limit if speed_limit > 0 else 0  # Calculate time
            heuristic = (1 / distance) * self.distance_weight + (
                    1 / time) * self.time_weight  # Create heuristic to guide ants

            # Pheromone Influence
            pheromone = self.pheromones.get((ant['current_node'], node), 1)

            probabilities[node] = ((pheromone ** self.alpha) * (
                    heuristic ** self.beta))  # Probability of traveling to target node using pheromone importance and heuristic
            total_prob += probabilities[node]  # Should be 1

        # Probabilistic Selection
        if total_prob > 0:  # Check there are valid nodes to move to
            nodes = list(probabilities.keys())  # Extract key values and convert to list
            node_weights = [probabilities[node] / total_prob for node in nodes]  # Normalized probabilities
            next_node = random.choices(nodes, weights=node_weights)[0]  # Random aspect to guided node choice
        else:
            # If all probabilities were 0 (e.g., trapped ant), choose randomly
            next_node = random.choice(unvisited)

        return next_node

    def run_ant(self, start_node, target_node, problem):
        """
        Simulate the movement of an ant from a start node to the target node.

        Args:
        - start_node: Node ID from which the ant starts its journey.
        - problem: Problem instance to evaluate the solution path.

        Returns:
        - result: Result of the ant's journey based on the problem evaluation.
        """
        ant = {'current_node': start_node, 'visited': [start_node], 'distance': 0,
               'time': 0}  # Initialize list of visited nodes with the start node as it's been visited

        while ant['current_node'] != target_node:  # While ant has not reached the target node, it selects the next node
            next_node = self._select_next_node(ant)  # Need the move_ant
            self._move_ant(ant, next_node)

        # Final Evaluation Here:
        path = ant['visited']  # The complete path taken by the ant, nodes visited
        result = problem.evaluate(path)  # Use your ShortestPathProblem class
        self.archive.add_solution(path, result)

    def _move_ant(self, ant, next_node):
        """
        Move the ant to the next node and update its state.

        Args:
        - ant (dict): Ant's information including current node and visited nodes.
        - next_node: Next node to which the ant will move.
        """
        ant['visited'].append(next_node)  # Adding next node to the visited
        ant['current_node'] = next_node  # Updating the current node and attached variables

        # Update distance and time traveled
        edge_data = self.graph.get_edge_data(ant['current_node'], next_node)
        distance = edge_data.get('length', 0)  # Check variables with the CSV
        print("Hello")
        speed_limit = edge_data.get('car', 0)  # Check variables with the CSV + use speed switcher
        time = distance / speed_limit if speed_limit > 0 else 0

        ant['distance'] += distance  # Updates distance and time for that specific ant
        ant['time'] += time

    def update_pheromones(self):
        """
        Update pheromone levels on edges based on the ant's traversal path and objective values.
        """
        for edge, data in self.graph.edges(data=True):
            # ... (evaporation code)
            u, v = edge
            pheromone_update = (1 - self.evaporation_rate) * self.pheromones.get((u, v), 0)  # Evaporation

            for path, result in self.archive.paths_results_archive:  # Iterate through archive
                if edge in path:
                    # Use result[0] = distance and result[1] = time
                    # works out path quality to modify the pheromone update
                    quality = self.distance_weight / result[0] + self.time_weight / result[1]
                    pheromone_update += quality

                self.pheromones[(u, v)] = pheromone_update

    def get_best_path(self):
        """
        Retrieves the best path(s) from the archive based on Pareto dominance.
        """
        pareto_optimal_solutions = []
        for path, result in self.archive.paths_results_archive:
            dominated = False
            for other_path, other_result in self.archive.paths_results_archive:
                if dominates(result, other_result):  # Check if our result is dominated
                    dominated = True
                    break

            if not dominated:
                pareto_optimal_solutions.append((path, result))

        return pareto_optimal_solutions

    # %%

# %%

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:20 2024

@author: R and J with additions by Josh
"""
import random
import history
import pareto_archive

# this is the functional class, others are drafts or templates
def dominates(u, v):
    """
    Checks if solution 'u' dominates solution 'v' in a multi-objective context.
    """
    return (u["Distance"] <= v["Distance"] and u["Time"] <= v["Time"] and
            (u["Distance"] < v["Distance"] or u["Time"] < v["Time"]))


class AntColony:

    def __init__(self, graph, num_ants=250, alpha=1, beta=2, evaporation_rate=0.5):
        """
        Initialize the Ant Colony Optimization algorithm.

        Args:
        - graph: Graph representing the problem domain.
        - objective_function: Function to evaluate the quality of a solution.
        - num_ants (int): Number of ants in the colony (default is 10).
        - alpha (float): Weight of pheromone in ant decision-making (default is 1).
        - beta (float): Weight of heuristic information in ant decision-making (default is 2).
        - evaporation_rate (float): Rate at which pheromones evaporate (default is 0.5).
        """
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        # self.pheromones = {node: {neighbor: 1 for neighbor in graph.neighbors(node)} for node in graph.nodes}
        self.pheromones = self.initialize_pheromones()
        self.history = history.History()  # init the ant paths history
        self.distance_weight = 0.5  # the importance of distance vs time, scale: 0-1
        self.time_weight = 0.5
        self.pareto_archive = pareto_archive.ParetoArchive()  # initialise the archive

    def run(self, source_node, target_node, problem):

        # runs all the ants in the iteration
        for anti in range(self.num_ants):
            self.run_ant(source_node, target_node, problem)
            # print("Complete Ant Cycle \n")

        # update all pheromones
        self.update_Ph()

    def initialize_pheromones(self):
        """
        Initialize pheromone levels on edges of the graph.

        Returns:
        - pheromones (dict): Dictionary storing pheromone levels on edges.
        """
        pheromones = {}  # Dict that stores pheromone levels as a tuple
        for edge in self.graph.edges:  # Iterates through all edges
            node1, node2 = edge  # Set edge source and target IDs
            pheromones[(node1, node2)] = 0  # Start edge pheromone with a uniform base value of 1

        print("Starting Pheromones initialised")
        return pheromones  # Keep for now, perhaps not needed

    def _select_next_node(self, ant, problem):
        """
        Select the next node for the ant to move to based on pheromone levels and heuristic information.

        Args:
        - ant (dict): Ant's information including current node and visited nodes.

        Returns:
        - next_node: Next node selected for the ant to move to.
        """
        neighbors = self.graph.neighbors(ant['current_node'])  # Get current node and look at neighboring nodes
        all_neighbors = [node for node in neighbors]
        unvisited = [node for node in neighbors if
                     node not in ant['visited']]  # Create list of unvisited potential next neighboring nodes

        u_probabilities = {}  # Initialize probability dictionary
        total_prob = 0  # Initialize variable (should end up as 1)

        if len(unvisited) > 0:
            for node in unvisited:  # Loop through unvisited nodes

                edge_data = self.graph.get_edge_data(ant['current_node'],
                                                     node)  # Get edge data between current node and
                # print(edge_data)

                # target unvisited node
                distance = edge_data.get('length', 0)  # Default to 1 if no data is available
                speed_limit_key = edge_data.get('car', 0)  # We need to change these to equal our CSV column names

                if speed_limit_key > 0 and distance > 0:

                    # Heuristics
                    speed_limit = problem.speedSwitcher(speed_limit_key)

                    time = distance / speed_limit if speed_limit > 0 else 0  # Calculate time
                    heuristic = (1 / distance) * self.distance_weight + (
                            1 / time) * self.time_weight  # Create heuristic to guide ants

                    # Pheromone Influence
                    pheromone = self.pheromones.get((ant['current_node'], node), 0)

                    u_probabilities[node] = ((pheromone ** self.alpha) * (
                            heuristic ** self.beta))  # Probability of traveling to target node using pheromone
                    # importance and heuristic

                    total_prob += u_probabilities[node]  # Should be 1

                else:
                    continue  # NEED TO COME UP WITH A BETTER IDEA HERE, THE ANTS ARE JUST TERMINATING

        # else:
        #   break

        # Probabilistic Selection
        if total_prob > 0:  # Check there are valid nodes to move to
            nodes = list(u_probabilities.keys())  # Extract key values and convert to list
            node_weights = [u_probabilities[node] / total_prob for node in nodes]  # Normalized probabilities
            # LOOK UP HOW THIS RANDOM FUNCTION WORKS, MAKE AN EXPLORATION RATE.
            next_node = random.choices(nodes, weights=node_weights)[0]  # Random aspect to guided node choice
        else:
            # If all probabilities were 0 or all nodes inaccessible (e.g., trapped ant), choose randomly
            # can we do this? It could choose an inaccessible node.

            next_node = random.choice(all_neighbors)

        return next_node

    def mutate_solution(self, solution):
        """
        Apply Random Selection mutation to the solution.

        Args:
        - solution (list): The solution (path) to be mutated.

        Returns:
        - mutated_solution (list): The mutated solution.
        """
        # Make a copy of the original solution
        mutated_solution = solution[:]

        # Perform mutation by randomly selecting a node to replace
        if mutated_solution:
            # Choose a random index within the solution length
            mutate_index = random.randint(0, len(mutated_solution) - 1)

            # Generate a random node to replace the node at mutate_index
            new_node = random.choice(list(self.graph.nodes()))

            # Replace the node at mutate_index with the new node
            mutated_solution[mutate_index] = new_node

        return mutated_solution

    def run_ant(self, start_node, target_node, problem):
        """
        Simulate the movement of an ant from a start node to the target node.

        Args:
        - start_node: Node ID from which the ant starts its journey.
        - target_node: Node ID representing the target destination.
        - problem: Problem instance to evaluate the solution path.
        """
        ant = {'current_node': start_node, 'visited': [start_node], 'distance': 0, 'time': 0}

        while ant['current_node'] != target_node:
            next_node = self._select_next_node(ant, problem)
            self._move_ant(ant, next_node, problem)

        path = ant['visited']
        result = problem.evaluate(path)

        # Apply mutation to the solution before adding it to the history
        mutated_solution = self.mutate_solution(path)
        self.history.add_solution(mutated_solution, problem.evaluate(mutated_solution))

    def _move_ant(self, ant, next_node, problem):
        """
        Move the ant to the next node and update its state.

        Args:
        - ant (dict): Ant's information including current node and visited nodes.
        - next_node: Next node to which the ant will move.
        """
        # Update distance and time traveled
        edge_data = self.graph.get_edge_data(ant['current_node'], next_node)
        distance = edge_data.get('length', 0)  # Check variables with the CSV
        speed_limit_key = edge_data.get('car', 0)  # Check variables with the CSV + use speed switcher

        speed_limit = problem.speedSwitcher(speed_limit_key)
        time = distance / speed_limit if speed_limit > 0 else 0

        ant['visited'].append(next_node)  # Add next_node to the ant's visited list
        ant['current_node'] = next_node  # Update the current node
        # print("Ant moved to: ", ant['current_node'])
        ant['distance'] += distance  # Updates distance and time for that specific ant
        ant['time'] += time

    def update_pheromones(self):
        """
        Update pheromone levels on edges based on the ant's traversal path and objective values.
        """
        # for every edge in the graph
        for edge in self.graph.edges():

            # print(type(edge))
            # extrapolate the edges nodes
            node1 = edge[0]
            node2 = edge[1]
            # node1, node2, _ = edge
            # pheromone update equation
            pheromone_update = (1 - self.evaporation_rate) * self.pheromones.get((node1, node2), 0)  # Evaporation

            # for every path and result in the archive
            for path, result in self.history.paths_results_history:  # Iterate through archive
                # for each node pair in the path
                for i in range(len(path) - 1):  # Iterate using indices
                    node1 = path[i]
                    node2 = path[i + 1]
                    # if the node pair is the same node pair as an edge in the graph
                    if (node1, node2) == edge:
                        # works out path quality to modify the pheromone update
                        print(result)
                        quality = self.distance_weight / result['Distance'] + self.time_weight / result['Time']
                        pheromone_update += quality

                    self.pheromones[(node1, node2)] = pheromone_update

    def update_Ph(self):

        max_distance = max(result['Distance'] for _, result in self.history.paths_results_history)
        max_time = max(result['Time'] for _, result in self.history.paths_results_history)

        for path, result in self.history.paths_results_history:  # Iterate through ants path history
            # for each node pair in the path
            for node1, node2 in zip(path, path[1:]):
                # convert to tuple
                edge = (node1, node2)
                # get existing pheromone level for the edge
                pheromone_level = self.pheromones.get(edge, 0)
                # Evaporation, this method ensures that if edges have been used multiple times eg stuck ants,
                # they get evaporated multiple times
                pheromone_level *= (1 - self.evaporation_rate)

                # Scale down the distance and time values
                scaled_distance = result['Distance'] / max_distance
                scaled_time = result['Time'] / max_time

                # Adjust the scaling factor based on your problem domain
                scaling_factor = 10

                # Update the pheromone level
                pheromone_level += scaling_factor * (
                            self.distance_weight * scaled_distance + self.time_weight * scaled_time)

                self.pheromones[edge] = pheromone_level

                # quality = self.distance_weight / result['Distance'] + self.time_weight / result['Time']
                # pheromone_level += quality

                # self.pheromones[edge] = pheromone_level

    def get_best_path(self):
        """
        Retrieves the best path(s) from the archive based on Pareto dominance.
        """
        iteration_pareto_archive = []

        # gets a path and the evaluated path result from the history
        for path_new, result_new in self.history.paths_results_history:
            dominated = False  # assume new solution is suitable for the archive
            if len(self.pareto_archive.pareto_archive) > 0:
                # checks against every result in the archive
                for archive_path, archive_result in self.pareto_archive.pareto_archive:
                    if dominates(archive_result, result_new):  # Check if archive_result dominates new result
                        dominated = True  # new result cannot enter archive
                        break

            if not dominated:
                # create new list to store results to be removed
                remove_from_archive = []

                # checks if the result dominates anything in the archive
                for archive_path, archive_result in self.pareto_archive.pareto_archive:
                    if dominates(result_new, archive_result):
                        # remove archive_result from pareto_archive, add it so the removal list
                        removal_entry = (archive_path, archive_result)
                        remove_from_archive.append(removal_entry)
                        for r in iteration_pareto_archive:
                            if dominates(result_new, r):
                                iteration_pareto_archive.remove(r)
                    else:
                        continue

                # Looks through the remove from archive list and removes the path and result from the archive
                for path, result in remove_from_archive:
                    try:
                        self.pareto_archive.pareto_archive.remove((path, result))  # Try to remove an exact match
                    except ValueError:  # Handle cases where the tuple might not be present
                        pass

                # add it to the iterations archive
                iteration_pareto_archive.append(result_new)

                # append the archive with the new non dominated result
                self.pareto_archive.pareto_archive.append((path_new, result_new))

        print("archive contains: ")
        self.pareto_archive.archive_print_results()

        # return this iterations archive
        return iteration_pareto_archive
        # returns all the non dominating solutions of that iteration

    # %%

# %%
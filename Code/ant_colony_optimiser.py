# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:20 2024

@author: R and J
"""
import random
import history
import matplotlib.pyplot as plt


# %%
######################

# this is the functional class, others are drafts or templates
def dominates(u, v):
    """
    Checks if solution 'u' dominates solution 'v' in a multi-objective context.
    """
    return (u["Distance"] <= v["Distance"] and
            u["Time"] <= v["Time"] and
            u["Co2_Emission"] <= v["Co2_Emission"] and
            (u["Distance"] < v["Distance"] or
             u["Time"] < v["Time"] or
             u["Co2_Emission"] < v["Co2_Emission"]))


class AntColony:

    def __init__(self, graph, pareto_Archive, num_ants=250, alpha=1, beta=2, evaporation_rate=0.5):
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
        self.distance_weight = 0.1  # the importance of distance vs time, scale: 0-1
        self.time_weight = 0.1
        self.co2_emission_weight = 0.9
        self.pareto_archive = pareto_Archive  # assign the archive
        self.exploration_rate = 0.1
        self.approx_max_distance_m = 100
        self.approx_max_co2 = 25
        self.approx_max_time = 5
        self.approx_min_distance = 1
        self.approx_min_time = 0.1
        self.approx_min_co2 = 9
        self.iteration_distances = []
        self.iteration_times = []
        self.iteration_co2_emissions = []

    def run(self, source_node, target_node, problem, iterations):

        for i in range(iterations):
            # runs all the ants in the iteration
            for anti in range(self.num_ants):
                self.run_ant(source_node, target_node, problem)
                # print("Complete Ant Cycle \n")

            # update all pheromones
            self.update_Ph()
            # print pheromones dictionary
            # print(self.pheromones)
            # update archive
            self.get_best_path()
            # add to the distances list for a graph
            self.averages(iterations)
            # clear the ant path history
            self.history.clear_history()
            print("\n Iteration complete \n")

    def initialize_pheromones(self):
        """
        Initialize pheromone levels on edges of the graph.

        Returns:
        - pheromones (dict): Dictionary storing pheromone levels on edges.
        """
        pheromones = {}  # Dict that stores pheromone levels as a tuple
        for edge in self.graph.edges:  # Iterates through all edges
            node1, node2 = edge  # Set edge source and target IDs
            pheromones[(node1, node2)] = 0  # Start edge pheromone with a uniform base value of 0

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
        all_valid_neighbors = [node for node in all_neighbors if self.graph.get_edge_data(ant['current_node'], node)['car'] != 0]
        unvisited = [node for node in all_neighbors if
                     node not in ant['visited']]  # Create list of unvisited potential next neighboring nodes

        u_probabilities = {}  # Initialize probability dictionary
        total_prob = 0  # Initialize variable (should end up as 1)

        if len(unvisited) > 0:
            for node in unvisited:  # Loop through unvisited nodes

                edge_data = self.graph.get_edge_data(ant['current_node'], node)  # Get edge data between current and next node
                # print(edge_data)

                # target unvisited node
                distance = edge_data.get('length', 0)  # Default to 1 if no data is available
                speed_limit_key = edge_data.get('car', 0)  # We need to change these to equal our CSV column names
                co2_emissions = edge_data.get('co2_emissions', 0)  # get the co2 emission data

                if speed_limit_key > 0 and distance > 0:

                    speed_limit = problem.speedSwitcher(speed_limit_key)
                    distance_km = distance / 1000

                    time_h = distance_km / speed_limit if speed_limit > 0 else 0  # Calculate time
                    time_s = time_h * 3600

                    normalised_distance = (distance - self.approx_min_distance) / (self.approx_max_distance_m - self.approx_min_distance)
                    normalised_time = (time_s - self.approx_min_time) / (self.approx_max_time - self.approx_min_time)
                    normalised_co2 = (co2_emissions - self.approx_min_co2) / (self.approx_max_co2 - self.approx_min_co2)

                    # calculate heuristic based on inverse normalised attributes and their weights
                    heuristic = (self.distance_weight * (1 - normalised_distance)) + (self.time_weight * (1 - normalised_time)) + (
                            self.co2_emission_weight * (1 - normalised_co2))  # Create heuristic to guide ants

                    # Pheromone Influence
                    pheromone = self.pheromones.get((ant['current_node'], node), 0)

                    u_probabilities[node] = ((pheromone ** self.alpha) * (
                            heuristic ** self.beta))  # Probability of traveling to target node using pheromone
                    # importance and heuristic

                    total_prob += u_probabilities[node]  # Should be 1

                else:
                    continue

        # Probabilistic Selection
        if total_prob > 0:  # Check there are valid nodes to move to
            nodes = list(u_probabilities.keys())  # Extract key values and convert to list
            node_weights = [u_probabilities[node] / total_prob for node in nodes]  # Normalized probabilities
            # LOOK UP HOW THIS RANDOM FUNCTION WORKS, MAKE AN EXPLORATION RATE.

            # Exploration vs. Exploitation
            if random.random() < self.exploration_rate:  # Exploration
                next_node = random.choice(all_neighbors)  # Choose randomly from all neighbors
            else:  # Exploitation
                next_node = random.choices(nodes, weights=node_weights)[0]

        else:
            # need to make this so that it only
            next_node = random.choice(all_valid_neighbors)
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
               'time': 0, 'co2_emissions': 0}  # Initialize list of visited nodes with the start node as it's been
        # visited

        while ant['current_node'] != target_node:  # While ant has not reached the target node, it selects the next node
            next_node = self._select_next_node(ant, problem)  # Need the move_ant
            self._move_ant(ant, next_node, problem)

        # Final Evaluation Here:
        path = ant['visited']  # The complete path taken by the ant, nodes visited

        result = problem.evaluate(path)  # Use your ShortestPathProblem class
        self.history.add_solution(path, result)  # add the ant path to the history

    def _move_ant(self, ant, next_node, problem):
        """
        Move the ant to the next node and update its state.

        Args:
        - ant (dict): Ant's information including current node and visited nodes.
        - next_node: Next node to which the ant will move.
        """

        # Update distance and time traveled
        edge_data = self.graph.get_edge_data(ant['current_node'], next_node)
        distance_meters = edge_data.get('length', 0)  # Check variables with the CSV
        speed_limit_key = edge_data.get('car', 0)  # Check variables with the CSV + use speed switcher
        co2_emissions = edge_data.get('co2_emissions', 0)

        speed_limit_kmh = problem.speedSwitcher(speed_limit_key)

        distance_km = distance_meters / 1000
        time_hours = distance_km / speed_limit_kmh if speed_limit_kmh > 0 else 0
        time_seconds = time_hours * 3600

        ant['visited'].append(next_node)  # Add next_node to the ant's visited list
        ant['current_node'] = next_node  # Update the current node
        # print("Ant moved to: ", ant['current_node'])
        ant['distance'] += distance_meters  # Updates distance and time for that specific ant
        ant['time'] += time_seconds
        ant['co2_emissions'] += co2_emissions

    def update_Ph(self):

        # normalising
        max_distance = max(result['Distance'] for _, result in self.history.paths_results_history)
        min_distance = min(result['Distance'] for _, result in self.history.paths_results_history)
        max_co2 = max(result['Co2_Emission'] for _, result in self.history.paths_results_history)
        min_co2 = min(result['Co2_Emission'] for _, result in self.history.paths_results_history)
        max_time = max(result['Time'] for _, result in self.history.paths_results_history)
        min_time = min(result['Time'] for _, result in self.history.paths_results_history)

        for path, result in self.history.paths_results_history:  # Iterate through ants path history
            # for each node pair in the path
            for node1, node2 in zip(path, path[1:]):
                # convert to tuple
                edge = (node1, node2)
                # get existing pheromone level for the edge
                pheromone_level = self.pheromones.get(edge, 0)
                # Evaporation, this method ensures that if edges have been used multiple times e.g. stuck ants,
                # they get evaporated multiple times
                pheromone_level *= (1 - self.evaporation_rate)

                # Scale down the distance and time values
                normalised_distance = (result['Distance'] - min_distance) / (max_distance - min_distance)
                normalised_time = (result['Time'] - min_time) / (max_time - min_time)
                normalised_co2 = (result['Co2_Emission'] - min_co2) / (max_co2 - min_co2)

                # Adjust the scaling factor based on your problem domain, with sf 1 they sit in the 0-2 range
                scaling_factor = 0.1

                # Update the pheromone level
                pheromone_level += scaling_factor * (
                        self.distance_weight * normalised_distance + self.time_weight * normalised_time + self.co2_emission_weight * normalised_co2)

                self.pheromones[edge] = pheromone_level

    def get_best_path(self):
        """
        Retrieves the best path(s) from the archive based on Pareto dominance.
        """

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

                    else:
                        continue

                # Looks through the remove from archive list and removes the path and result from the archive
                for path, result in remove_from_archive:
                    try:
                        self.pareto_archive.pareto_archive.remove((path, result))  # Try to remove an exact match
                    except ValueError:  # Handle cases where the tuple might not be present
                        pass

                # append the archive with the new non dominated result
                self.pareto_archive.pareto_archive.append((path_new, result_new))

        print("archive contains: ")
        self.pareto_archive.archive_print_results()

        # return this iterations archive
        return [archived_result[1] for archived_result in self.pareto_archive.pareto_archive]
        # returns all the non dominating solutions of that iteration

    def averages(self, iterations):

        # Create a lists to store distance, time and co2 from current iteration
        ant_distances = []
        ant_times = []
        ant_co2 = []

        for _, result in self.history.paths_results_history:
            distance = result['Distance']  # Extract distance from the result
            time = result['Time']
            co2 = result['Co2_Emission']
            ant_distances.append(distance)
            ant_times.append(time)
            ant_co2.append(co2)

        average_distance = sum(ant_distances) / len(ant_distances)
        average_time = sum(ant_times) / len(ant_times)
        average_co2 = sum(ant_co2) / len(ant_co2)
        self.iteration_distances.append(average_distance)
        self.iteration_times.append(average_time)
        self.iteration_co2_emissions.append(average_co2)

        if len(self.iteration_distances) == iterations:
            plt.scatter(range(len(self.iteration_distances)), self.iteration_distances)
            plt.xlabel("Iteration")
            plt.ylabel("Average Path Distance (m)")
            plt.title("ACO Optimization Progress")
            plt.show()

            plt.scatter(range(len(self.iteration_times)), self.iteration_times)
            plt.xlabel("Iteration")
            plt.ylabel("Average Path Time (s)")
            plt.title("ACO Optimization Progress")
            plt.show()

            plt.scatter(range(len(self.iteration_co2_emissions)), self.iteration_co2_emissions)
            plt.xlabel("Iteration")
            plt.ylabel("Average Path Co2 Emissions (g/km)")
            plt.title("ACO Optimization Progress")
            plt.show()

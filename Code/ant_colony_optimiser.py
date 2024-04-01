# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:20 2024

@author: russe
"""
#%%
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
#%%

#%%
# if length & duration objective function classes are okay then this would be the updated ACO
# update is in AntColony class in def update_pheromones

class Archive:
    def __init__(self):
        self.objective_values = []

    def add_solution(self, objective_values):
        self.objective_values.append(objective_values)

class Ant:
    def __init__(self, colony, start_node):
        self.colony = colony
        self.current_node = start_node
        self.visited_nodes = [start_node]
        self.objective_values = None

    def move_to_next_node(self):
        next_node = self.select_next_node()
        self.visited_nodes.append(next_node)
        self.current_node = next_node

    def select_next_node(self):
        probabilities = self.calculate_probabilities()
        next_node = np.random.choice(list(self.colony.graph.neighbors(self.current_node)), p=probabilities)
        return next_node

    def calculate_probabilities(self): #calc prob of moving to neighbouring node (run ants function)
        pheromone_values = self.colony.pheromones[self.current_node]
        unvisited_nodes = set(self.colony.graph.nodes) - set(self.visited_nodes)
        probabilities = [pheromone_values[node] ** self.colony.alpha *
                         (1.0 / self.colony.graph.get_edge_data(self.current_node, node)['length']) ** self.colony.beta
                         for node in self.colony.graph.nodes]
        probabilities = [p if i in unvisited_nodes else 0 for i, p in enumerate(probabilities)]
        probabilities /= np.sum(probabilities)
        return probabilities

    def evaluate_objectives(self):
        self.objective_values = self.colony.objective_function.evaluate(self.visited_nodes)

#%%
######################

# this is the functional class, others are drafts or templates
class AntColony:
    
    def __init__(self, graph, objective_function, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5):
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
    self.objective_function = objective_function
    self.num_ants = num_ants
    self.alpha = alpha
    self.beta = beta
    self.evaporation_rate = evaporation_rate
    self.pheromones = {node: {neighbor: 1 for neighbor in graph.neighbors(node)} for node in graph.nodes}
    self.archive = Archive()
    self.target_node = None  # Initialize as None

    
    def set_target(self, target_node):
    """
    Set the target node for the ant colony.

    Args:
    - target_node: Node ID representing the target node.
    """
    self.target_node = target_node

        
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
    unvisited = [node for node in neighbors if node not in ant['visited']]  # Create list of unvisited neighboring nodes

    probabilities = {}  # Initialize probability dictionary
    total_prob = 0  # Initialize variable (should end up as 1)

    for node in unvisited:  # Loop through unvisited nodes
        edge_data = self.graph.get_edge_data(ant['current_node'], node)  # Get edge data between current node and target unvisited node
        distance = edge_data.get('length', 1)  # Default to 1 if no data is available
        speed_limit = edge_data.get('car', 0)  # We need to change these to equal our CSV column names

        # Heuristics
        time = distance / speed_limit if speed_limit > 0 else 0  # Calculate time
        heuristic = (1 / distance) * self.distance_weight + (1 / time) * self.time_weight  # Create heuristic to guide ants

        # Pheromone Influence
        pheromone = self.pheromones.get((ant['current_node'], node), 1)

        probabilities[node] = ((pheromone ** self.alpha) * (heuristic ** self.beta))  # Probability of traveling to target node using pheromone importance and heuristic
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

    def run_ant(self, start_node, problem):
    """
    Simulate the movement of an ant from a start node to the target node.

    Args:
    - start_node: Node ID from which the ant starts its journey.
    - problem: Problem instance to evaluate the solution path.

    Returns:
    - result: Result of the ant's journey based on the problem evaluation.
    """
    ant = {'current_node': start_node, 'visited': [start_node], 'distance': 0, 'time': 0}  # Initialize list of visited nodes with the start node as it's been visited

    while ant['current_node'] != self.target_node:  # While ant has not reached the target node, it selects the next node
        next_node = self._select_next_node(ant)  # Need the move_ant
        self._move_ant(ant, next_node)

    # Final Evaluation Here:
    path = ant['visited']  # The complete path taken by the ant, nodes visited
    result = problem.evaluate(path)  # Use your ShortestPathProblem class
    return result
    
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
    speed_limit = edge_data.get('car', 0)  # Check variables with the CSV + use speed switcher
    time = distance / speed_limit if speed_limit > 0 else 0

    ant['distance'] += distance  # Updates distance and time for that specific ant
    ant['time'] += time 

        
    def update_pheromones(self, ant):
    """
    Update pheromone levels on edges based on the ant's traversal path and objective values.

    Args:
    - ant (dict): Ant's information including visited nodes and objective values.
    """
    for i in range(len(ant.visited) - 1):
        current_node, next_node = ant.visited[i], ant.visited[i + 1]
        edge_data = self.graph.get_edge_data(current_node, next_node)
        pheromone = edge_data.get('pheromone', 0)
        
        # Evaporation
        pheromone *= (1 - self.evaporation_rate)
        
        # Update pheromones based on the length objective
        pheromone_increment_length = 1 / ant.objective_values['distance']
        pheromone += pheromone_increment_length
        
        # Update pheromones based on the duration objective
        pheromone_increment_duration = 1 / ant.objective_values['time']
        pheromone += pheromone_increment_duration
        
        # Update pheromone attribute for the edge
        self.graph[current_node][next_node]['pheromone'] = pheromone
        
        # Symmetric update - update the opposite direction as well
        self.graph[next_node][current_node]['pheromone'] = pheromone


#%%

#%%


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
######################
class AntColony:
    
    def __init__(self, graph, objective_function, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5):
        self.graph = graph
        self.objective_function = objective_function
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromones = {node: {neighbor: 1 for neighbor in graph.neighbors(node)} for node in graph.nodes}
        self.archive = Archive()
        self.target_node = None # initialise as none
    
    def set_target(self, target_node):
        self.target_node = target_node
        
    def initialize_pheromones(self):
        pheromones = {} # dict that stores pheromone levels as a tuple
        for edge in self.graph.edges: # iterates trhough all edges
            u, v = edge # set edge source and target IDs
            pheromones[(u, v)] = 1  # Start edge pheromone with a uniform base value of 1
        return pheromones # keep for now, perhaps not needed
    
    def _select_next_node(self, ant): 
        neighbors = self.graph.neighbors(ant['current_node']) # get current node and looks at neighbouring nodes
        unvisited = [node for node in neighbors if node not in ant['visited']] # creates list of neighbouring nodes that are unvisited
    
        probabilities = {} #init prob dict
        total_prob = 0 # init variable # should end up as 1 
    
        for node in unvisited: # loop through unvisited nodes
            edge_data = self.graph.get_edge_data(ant['current_node'], node) # gets edge data between current node and target unvisited node
            distance = edge_data.get('length', 1)  # Default to 1 if no data is available
            speed_limit = edge_data.get('car', 0) # we need to change these to equal our csv col names
            
            # Heuristics
            time = distance / speed_limit if speed_limit > 0 else 0 # calcs time 
            heuristic = (1 / distance) * self.distance_weight + (1 / time) * self.time_weight # creates heuristic to guide ants
    
            # Pheromone Influence
            pheromone = self.pheromones.get((ant['current_node'], node), 1)
    
            probabilities[node] = ((pheromone ** self.alpha) * (heuristic ** self.beta)) # prob of travelling to target node using pheromone importance and heuristic
            total_prob += probabilities[node] # should be 1
    
            # Probabilistic Selection # 
        if total_prob > 0: # checks there are valid nodes to move to
            nodes = list(probabilities.keys()) # extract key values and converts to list 
            node_weights = [probabilities[node]/total_prob for node in nodes] # Normalized probabilities
            next_node = random.choices(nodes, weights=node_weights)[0] # random aspect to guided node choice
        else: 
            # If all probabilities were 0 (e.g., trapped ant), choose randomly
            next_node = random.choice(unvisited) 
    
        return next_node 
    
    
    def run_ant(self, start_node, problem): # start node = node id # another argument as problem?
        ant = {'current_node': start_node, 'visited': [start_node], 'distance': 0, 'time': 0} # initilises list of visited node with the start node as its been visited
    
        while ant['current_node'] != self.target_node: # while ant has not reached t node, it selects next node # Need to set a target node
            next_node = self._select_next_node(ant) # need the move_ant
            self._move_ant(ant, next_node)
    
        # Final Evaluation Here: 
        path = ant['visited'] # The complete path taken by the ant, nodes visited
        result = problem.evaluate(path)  # Use your ShortestPathProblem class
        return result 
    
    def _move_ant(self, ant, next_node):
        ant['visited'].append(next_node) # adding next node to the visited
        ant['current_node'] = next_node # updating the current node and attached variables
    
        # Update distance and time traveled
        edge_data = self.graph.get_edge_data(ant['current_node'], next_node)
        distance = edge_data.get('length', 0) # check variables with the csv
        speed_limit = edge_data.get('car', 0) # check variables with the csv + use speed switcher
        time = distance / speed_limit if speed_limit > 0 else 0
    
        ant['distance'] += distance # updates distance and time for that specific ant
        ant['time'] += time 

    def update_pheromones(self, ant):
        for i in range(len(ant.visited_nodes) - 1):
            current_node, next_node = ant.visited_nodes[i], ant.visited_nodes[i + 1]
            self.pheromones[current_node][next_node] *= (1 - self.evaporation_rate)
            # Update pheromones based on the length objective
            pheromone_increment_length = 1 / ant.objective_values['Distance']
            self.pheromones[current_node][next_node] += pheromone_increment_length
            # Update pheromones based on the duration objective
            pheromone_increment_duration = 1 / ant.objective_values['Time']
            self.pheromones[current_node][next_node] += pheromone_increment_duration
            # Symmetric update - update the opposite direction as well
            self.pheromones[next_node][current_node] = self.pheromones[current_node][next_node]

#%%


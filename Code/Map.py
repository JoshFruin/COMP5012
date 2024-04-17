import networkx as nx


class MapMaker:

    def __init__(self, nodes, edges):
        self.map = map
        self.nodes_df = nodes
        self.edges_df = edges
        self.network_map = nx.Graph()
        self.base_emissions_per_km = 164  # Average emissions in grams of CO2e per km
        self.optimal_speed = 60  # km/h

    def add_nodes(self):
        for index, row in self.nodes_df.iterrows():
            self.network_map.add_node(
                row['id'],
                longitude=row['longitude'],
                latitude=row['latitude'],
                altitude=row['altitude']
            )

    def add_edges(self):
        for index, row in self.edges_df.iterrows():

            distance_m = row['length']  # Get the distance
            speed_rank = row['car']  # Get speed limit (assuming 'car' is your speed limit column)

            speed_limit_kmh = self.speedSwitcher(speed_rank)
            distance_km = distance_m / 1000

            if speed_limit_kmh > 0:
                co2_emissions_gperkm = self.calculate_co2_emissions(distance_km, speed_limit_kmh)
            else:
                co2_emissions_gperkm = 0  # Handle cases where speed limit data might not be present

            self.network_map.add_edge(
                row['source_node_id'],  # source node id
                row['target_node_id'],  # target node id
                edge_id=row['edge_id'],
                length=row['length'],
                car=row['car'],
                car_reverse=row['car_reverse'],
                bike=row['bike'],
                bike_reverse=row['bike_reverse'],
                foot=row['foot'],
                co2_emissions=co2_emissions_gperkm
            )

    def print_emissions(self):
        for node1, node2, data in self.network_map.edges(data=True):
            co2_emissions = data.get('co2_emissions', 0)
            print(f"Edge ({node1}, {node2}): CO2 Emissions = {co2_emissions} g/km")

    def calculate_co2_emissions(self, distance_km, speed_limit_kmh):

        # Model higher emissions at extremes (simplified quadratic)
        speed_factor = 1 + 0.1 * (abs(speed_limit_kmh - self.optimal_speed) ** 2)

        total_emissions = distance_km * self.base_emissions_per_km * speed_factor

        return total_emissions

    def speedSwitcher(self, choice):
        switcher = {
            0: 0,
            1: 20,
            2: 30,
            3: 50,
            4: 90,
            5: 100,
            6: 120,
        }
        return switcher.get(choice, "Invalid edge")

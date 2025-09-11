import heapq
import math
import random


class CityGraph:
    """Represents the city's road network as a graph."""

    def __init__(self, intersections, roads):
        # The graph is an adjacency list where each key is an intersection
        # and its value is a dictionary of connected intersections and their travel times.
        self.graph = {i: {} for i in intersections}
        self.intersections = intersections
        for road in roads:
            start, end, time = road
            self.graph[start][end] = time
            self.graph[end][start] = time  # Assuming two-way roads

    def get_neighbors(self, intersection):
        return self.graph[intersection]


class AStarDispatch:
    """Implements the A* algorithm for finding the optimal ambulance route."""

    def __init__(self, city_graph, ambulance_locations):
        self.city_graph = city_graph
        self.ambulance_locations = ambulance_locations
        # Use a dictionary to store intersection coordinates for heuristic calculation
        self.coordinates = {
            'A': (0, 0), 'B': (10, 0), 'C': (20, 5), 'D': (5, 10),
            'E': (15, 10), 'F': (25, 15), 'G': (5, 20), 'H': (15, 25),
            'I': (25, 20), 'J': (30, 0), 'K': (35, 10), 'L': (40, 20)
        }

    def heuristic(self, start, goal):
        """Calculates the Euclidean distance between two points."""
        x1, y1 = self.coordinates[start]
        x2, y2 = self.coordinates[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def find_path(self, start_node, goal_node):
        """
        Finds the shortest path from start_node to goal_node using A*.
        Returns the path and its total travel time.
        """
        # A priority queue to store (f_score, node, path)
        open_set = [(0 + self.heuristic(start_node, goal_node), 0, start_node, [start_node])]

        # Dictionaries to store the lowest g_score and a record of the path
        g_scores = {node: float('inf') for node in self.city_graph.intersections}
        g_scores[start_node] = 0

        while open_set:
            f_score, g_score, current_node, path = heapq.heappop(open_set)

            if current_node == goal_node:
                return path, g_score

            for neighbor, travel_time in self.city_graph.get_neighbors(current_node).items():
                tentative_g_score = g_score + travel_time

                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal_node)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, new_path))

        return None, float('inf')

    def find_best_ambulance(self, patient_location):
        """Finds the ambulance with the shortest estimated travel time to the patient."""
        best_ambulance = None
        min_time = float('inf')
        best_path = None

        for ambulance_id, location in self.ambulance_locations.items():
            path, time = self.find_path(location, patient_location)
            if time < min_time:
                min_time = time
                best_ambulance = ambulance_id
                best_path = path

        return best_ambulance, best_path, min_time


#MAIN PROGRAM
if __name__ == "__main__":
    # Define the city's road network
    # Format: (start_intersection, end_intersection, travel_time)
    city_roads = [
        ('A', 'B', 5), ('A', 'D', 10), ('B', 'C', 15), ('B', 'E', 8),
        ('C', 'E', 7), ('C', 'F', 12), ('D', 'E', 6), ('D', 'G', 10),
        ('E', 'H', 9), ('F', 'I', 11), ('G', 'H', 5), ('H', 'I', 6),
        ('H', 'K', 15), ('I', 'L', 14), ('J', 'B', 10), ('J', 'K', 20),
        ('K', 'L', 8)
    ]
    intersections = list(set([i for road in city_roads for i in road[:2]]))

    # Create the graph and the A* dispatch system
    city_graph = CityGraph(intersections, city_roads)

    # Define the starting locations of the ambulances
    ambulance_locations = {
        'Ambulance 1': 'A',
        'Ambulance 2': 'F',
        'Ambulance 3': 'G'
    }

    dispatch_system = AStarDispatch(city_graph, ambulance_locations)

    # Simulate a patient call
    patient_location = 'L'  # Patient is at intersection L

    print(f"Ambulance locations: {ambulance_locations}")
    print(f"\nPatient call received at intersection: {patient_location}")

    # Find the best ambulance to dispatch
    best_ambulance, path, travel_time = dispatch_system.find_best_ambulance(patient_location)

    if path:
        print("\n--- Dispatching Best Ambulance ---")
        print(f"Ambulance to dispatch: {best_ambulance}")
        print(f"Optimal route: {' -> '.join(path)}")
        print(f"Estimated travel time: {travel_time:.2f} minutes")
    else:
        print("\nCould not find a path to the patient's location.")
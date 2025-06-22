import numpy as np
import pandas as pd
from docplex.mp.model import Model
import random
from datetime import datetime, timedelta


class VRPSolver:
    def __init__(self, data_path="data"):
        """Initialize solver with data validation"""
        try:
            # Load nodes and create ID mapping
            self.nodes = pd.read_csv(f"{data_path}/nodes.csv")
            self.node_ids = sorted(self.nodes['id'].unique())
            self.n = len(self.node_ids)
            self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.node_ids)}
            self.idx_to_id = {idx: id_ for id_, idx in self.id_to_idx.items()}

            # Load and validate matrices
            self._load_matrices(data_path)

            # Load additional data
            self.time_windows = pd.read_csv(f"{data_path}/time_windows.csv")
            self.products = pd.read_csv(f"{data_path}/products.csv")
            self.orders = pd.read_csv(f"{data_path}/orders.csv")
            self.vehicles = pd.read_csv(f"{data_path}/vehicles.csv")

            print(f"Initialized with {self.n} nodes (depot: 1, clients: {self.n - 1})")

        except Exception as e:
            print(f"Data loading error: {str(e)}")
            raise

    def _load_matrices(self, data_path):
        """Load and prepare distance/time matrices"""
        try:
            # Load full matrices from CSV
            dist_full = pd.read_csv(f"{data_path}/distance_matrix.csv", header=None).values
            time_full = pd.read_csv(f"{data_path}/time_matrix.csv", header=None).values

            # Initialize properly sized matrices
            self.dist_matrix = np.zeros((self.n, self.n))
            self.time_matrix = np.zeros((self.n, self.n))

            # Map values based on node IDs
            for i, id_i in enumerate(self.node_ids):
                for j, id_j in enumerate(self.node_ids):
                    if id_i < dist_full.shape[0] and id_j < dist_full.shape[1]:
                        self.dist_matrix[i][j] = dist_full[id_i][id_j]
                        self.time_matrix[i][j] = time_full[id_i][id_j]
                    else:
                        # Set large but finite distance for missing entries
                        self.dist_matrix[i][j] = 1e6 if i != j else 0
                        self.time_matrix[i][j] = 1e6 if i != j else 0

            print(f"Created {self.n}x{self.n} matrices from original {dist_full.shape}")

        except Exception as e:
            print(f"Matrix loading error: {str(e)}")
            raise

    def solve_with_cplex(self, k=5):
        """Solve using CPLEX with proper constraint handling"""
        try:
            model = Model('VRP_Optimization')

            # Find depot index
            depot_idx = [i for i, id_ in enumerate(self.node_ids)
                         if self.nodes.loc[self.nodes['id'] == id_, 'type'].iloc[0] == 'depot'][0]

            # Create variables for valid arcs
            x = {}
            for i in range(self.n):
                # Get k nearest valid neighbors
                distances = self.dist_matrix[i]
                valid = [j for j in range(self.n)
                         if i != j and distances[j] < 1e6]
                neighbors = sorted(valid, key=lambda j: distances[j])[:k]

                for j in neighbors:
                    x[(i, j)] = model.binary_var(name=f'x_{i}_{j}')

            # Constraints:
            # 1. Each client visited exactly once
            client_indices = [i for i in range(self.n) if i != depot_idx]
            for j in client_indices:
                incoming = [x[(i, j)] for i in range(self.n) if (i, j) in x]
                if incoming:
                    model.add_constraint(model.sum(incoming) == 1)

            # 2. Flow conservation
            for h in range(self.n):
                outgoing = [x[(h, j)] for j in range(self.n) if (h, j) in x]
                incoming = [x[(i, h)] for i in range(self.n) if (i, h) in x]
                if outgoing and incoming:
                    model.add_constraint(model.sum(outgoing) == model.sum(incoming))

            # Objective: minimize total distance
            model.minimize(model.sum(
                self.dist_matrix[i][j] * x[(i, j)]
                for (i, j) in x
            ))

            # Solve model
            solution = model.solve()
            if solution:
                return self._extract_routes(solution, x, depot_idx)
            return []

        except Exception as e:
            print(f"CPLEX error: {str(e)}")
            return []

    def solve_with_ga(self, population_size=50, generations=100, mutation_rate=0.1):
        """Solve using Genetic Algorithm"""
        try:
            # Find depot index
            depot_idx = [i for i, id_ in enumerate(self.node_ids)
                         if self.nodes.loc[self.nodes['id'] == id_, 'type'].iloc[0] == 'depot'][0]
            client_indices = [i for i in range(self.n) if i != depot_idx]

            # Initialize population
            population = []
            for _ in range(population_size):
                individual = client_indices.copy()
                random.shuffle(individual)
                individual = [depot_idx] + individual + [depot_idx]
                population.append(individual)

            # Evolution loop
            for _ in range(generations):
                # Evaluation
                fitness = [self._route_cost(ind) for ind in population]

                # Tournament selection
                new_pop = []
                for _ in range(population_size):
                    candidates = random.sample(list(zip(population, fitness)), 3)
                    winner = min(candidates, key=lambda x: x[1])[0]
                    new_pop.append(winner.copy())

                # Order-1 Crossover
                for i in range(0, len(new_pop) - 1, 2):
                    p1, p2 = new_pop[i][1:-1], new_pop[i + 1][1:-1]
                    c1, c2 = self._ox_crossover(p1, p2)
                    new_pop[i] = [depot_idx] + c1 + [depot_idx]
                    new_pop[i + 1] = [depot_idx] + c2 + [depot_idx]

                # Mutation
                for ind in new_pop:
                    if random.random() < mutation_rate:
                        idx1, idx2 = random.sample(range(1, len(ind) - 1), 2)
                        ind[idx1], ind[idx2] = ind[idx2], ind[idx1]

                population = new_pop

            # Return best solution
            best = min(population, key=lambda x: self._route_cost(x))
            return self._split_routes(best, depot_idx)

        except Exception as e:
            print(f"GA error: {str(e)}")
            return []

    def _route_cost(self, route):
        """Calculate total distance of a route"""
        total = 0
        for i in range(1, len(route)):
            from_node = route[i - 1]
            to_node = route[i]
            if from_node < len(self.dist_matrix) and to_node < len(self.dist_matrix):
                total += self.dist_matrix[from_node][to_node]
        return total

    def _ox_crossover(self, parent1, parent2):
        """Order-1 crossover operation"""
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))

        # Initialize children
        child1 = [None] * size
        child2 = [None] * size

        # Copy segments between a and b
        child1[a:b] = parent1[a:b]
        child2[a:b] = parent2[a:b]

        # Fill remaining positions
        for child, parent in [(child1, parent2), (child2, parent1)]:
            ptr = b
            for gene in parent[b:] + parent[:b]:
                if gene not in child[a:b]:
                    child[ptr % size] = gene
                    ptr += 1

        return child1, child2

    def _split_routes(self, giant_tour, depot_idx):
        """Split giant tour into vehicle routes"""
        routes = []
        current_route = [depot_idx]

        for node in giant_tour[1:-1]:  # Skip depot at start/end
            current_route.append(node)
            # Simple splitting logic - can be enhanced
            if len(current_route) >= 5:  # Max 4 clients per route
                current_route.append(depot_idx)
                routes.append(current_route)
                current_route = [depot_idx]

        # Add last route if not empty
        if len(current_route) > 1:
            current_route.append(depot_idx)
            routes.append(current_route)

        return routes

    def _extract_routes(self, solution, x, depot_idx):
        """Extract routes from CPLEX solution"""
        routes = []
        visited = set()
        n = self.n

        for _ in range(len(self.vehicles)):
            route = [depot_idx]
            current = depot_idx
            while True:
                next_node = None
                for j in range(n):
                    if (current, j) in x and solution.get_value(x[(current, j)]) > 0.5 and j not in visited:
                        next_node = j
                        break
                if next_node is None or next_node == depot_idx:
                    break
                route.append(next_node)
                visited.add(next_node)
                current = next_node

            if len(route) > 1:
                route.append(depot_idx)
                routes.append(route)

        return routes
from solver import VRPSolver
from visualization import Visualizer
import pandas as pd
import time


def validate_data(solver):
    """Validate input data before solving"""
    print("\n=== Data Validation ===")

    # 1. Check nodes
    if solver.n < 2:
        print("Error: Not enough nodes (need at least 1 depot and 1 client)")
        return False

    # 2. Check matrices
    if solver.dist_matrix.shape != (solver.n, solver.n):
        print(f"Error: Distance matrix shape {solver.dist_matrix.shape} != {solver.n}x{solver.n}")
        return False

    # 3. Check orders
    if len(solver.orders) == 0:
        print("Warning: No delivery orders found")

    print("Data validation passed")
    return True


def main():
    try:
        # Initialize solver with data
        print("Initializing solver...")
        solver = VRPSolver()

        # Validate data
        if not validate_data(solver):
            return

        # Solve with CPLEX
        print("\n=== Solving with CPLEX ===")
        start_time = time.time()
        cplex_routes = solver.solve_with_cplex()
        cplex_time = time.time() - start_time
        print(f"CPLEX found {len(cplex_routes)} routes in {cplex_time:.2f} seconds")

        # Solve with GA
        print("\n=== Solving with Genetic Algorithm ===")
        start_time = time.time()
        ga_routes = solver.solve_with_ga()
        ga_time = time.time() - start_time
        print(f"GA found {len(ga_routes)} routes in {ga_time:.2f} seconds")

        # Compare results
        print("\n=== Results Comparison ===")
        results = []
        if cplex_routes:
            cplex_dist = sum(solver._route_cost(route) for route in cplex_routes)
            results.append(('CPLEX', cplex_time, cplex_dist, len(cplex_routes)))
        if ga_routes:
            ga_dist = sum(solver._route_cost(route) for route in ga_routes)
            results.append(('Genetic Algorithm', ga_time, ga_dist, len(ga_routes)))
        def analyze_routes(name, routes):
            route_lengths = [solver._route_cost(r) for r in routes]
            client_counts = [len(r) - 1 for r in routes]  # без учёта депо
            return {
                'Method': name,
                'Avg Route Length': sum(route_lengths) / len(route_lengths),
                'Max Route Length': max(route_lengths),
                'Min Route Length': min(route_lengths),
                'Avg Clients per Route': sum(client_counts) / len(client_counts),
                'Route Length Variance': pd.Series(route_lengths).var()
            }

        print("\n=== Detailed Analysis ===")
        details = []
        if cplex_routes:
            details.append(analyze_routes('CPLEX', cplex_routes))
        if ga_routes:
            details.append(analyze_routes('Genetic Algorithm', ga_routes))

        detailed_df = pd.DataFrame(details)
        print(detailed_df.to_string(index=False))

        # Optional: Percentage difference in distance
        if cplex_routes and ga_routes:
            perc_diff = 100 * (ga_dist - cplex_dist) / cplex_dist
            print(f"\nGA solution is {perc_diff:+.2f}% {'worse' if perc_diff > 0 else 'better'} than CPLEX in total distance.")

        if results:
            df = pd.DataFrame(results, columns=['Method', 'Time (s)', 'Distance', 'Routes'])
            print("\n", df.to_string(index=False))

        # Visualize solutions
        visualizer = Visualizer(solver.nodes)
        if cplex_routes:
            visualizer.plot_map(cplex_routes, "cplex_routes.html")
        if ga_routes:
            visualizer.plot_map(ga_routes, "ga_routes.html")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
    # After getting routes in main()
    visualizer = Visualizer(solver.nodes)

    # Interactive HTML map
    print("\nGenerating interactive map...")
    visualizer.plot_interactive_map(cplex_routes, "cplex_routes.html")
    visualizer.plot_interactive_map(ga_routes, "ga_routes.html")

    # Static comparison plot
    print("\nGenerating static comparison plot...")
    all_routes = (cplex_routes if cplex_routes else []) + (ga_routes if ga_routes else [])
    if all_routes:
        visualizer.plot_static_map(all_routes)


if __name__ == "__main__":
    main()
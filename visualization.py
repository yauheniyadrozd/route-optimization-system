import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


class Visualizer:
    def __init__(self, nodes):
        """Initialize with node data"""
        self.nodes = nodes
        self.coords = list(zip(nodes['lat'], nodes['lon']))

    def get_route_color(self, idx):
        """Get distinct color for each route"""
        colors = list(mcolors.TABLEAU_COLORS.values())
        return colors[idx % len(colors)]

    def plot_interactive_map(self, routes, save_path="vrp_solution.html"):
        """Create interactive Folium map with routes and markers"""
        if not routes:
            print("No routes to visualize")
            return

        # Create base map centered on depot
        depot_coord = self.coords[0]
        m = folium.Map(location=depot_coord, zoom_start=13)

        # Add depot marker
        folium.Marker(
            depot_coord,
            popup="Depot",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)

        # Define colors for different routes
        colors = ['blue', 'green', 'purple', 'orange', 'darkred',
                  'lightred', 'beige', 'darkblue', 'darkgreen',
                  'cadetblue', 'pink', 'lightblue']

        # Draw each route
        for i, route in enumerate(routes):
            if len(route) < 2:
                continue

            # Get valid coordinates
            route_coords = [self.coords[node] for node in route
                            if node < len(self.coords)]

            # Add route line
            folium.PolyLine(
                route_coords,
                color=colors[i % len(colors)],
                weight=3,
                opacity=0.8,
                tooltip=f"Route {i + 1}"
            ).add_to(m)

            # Add markers for each stop
            for seq, node in enumerate(route):
                if node >= len(self.coords):
                    continue

                popup_text = f"Depot" if node == 0 else f"Client {node} (Stop {seq})"

                folium.CircleMarker(
                    location=self.coords[node],
                    radius=6 if node == 0 else 5,
                    color=colors[i % len(colors)],
                    fill=True,
                    fill_opacity=0.7,
                    popup=popup_text,
                    tooltip=popup_text
                ).add_to(m)

        # Save to HTML file
        m.save(save_path)
        print(f"Interactive map saved to {save_path}")
        return m

    def plot_static_map(self, routes):
        """Create static matplotlib plot of routes"""
        plt.figure(figsize=(12, 10))

        # Plot all nodes
        x, y = zip(*self.coords)
        plt.scatter(x[1:], y[1:], c='blue', s=50, label='Clients')
        plt.scatter(x[0], y[0], c='red', marker='s', s=100, label='Depot')

        # Plot each route with arrows
        colors = plt.cm.tab10.colors  # Different colors for each route
        for i, route in enumerate(routes):
            if len(route) < 2:
                continue

            # Get valid coordinates
            valid_route = [node for node in route if node < len(self.coords)]
            route_x, route_y = zip(*[self.coords[node] for node in valid_route])

            # Plot route line
            plt.plot(route_x, route_y,
                     color=colors[i % len(colors)],
                     linestyle='-',
                     marker='o',
                     markersize=5,
                     linewidth=2,
                     label=f'Route {i + 1}')

            # Add direction arrows
            for j in range(len(valid_route) - 1):
                dx = route_x[j + 1] - route_x[j]
                dy = route_y[j + 1] - route_y[j]
                plt.arrow(route_x[j], route_y[j],
                          dx * 0.9, dy * 0.9,  # Shorter arrow
                          color=colors[i % len(colors)],
                          head_width=0.0005,
                          head_length=0.001,
                          length_includes_head=True)

        plt.title("Vehicle Routing Problem Solution", fontsize=14)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save and show
        plt.savefig("vrp_solution.png", dpi=300)
        print("Static map saved to vrp_solution.png")
        plt.show()

    def plot_graph(self, routes):
        """Create static matplotlib plot of routes"""
        if not routes or not self.coords:
            print("No valid routes to plot")
            return

        plt.figure(figsize=(10, 8))

        # Plot all nodes
        x, y = zip(*self.coords)
        plt.scatter(x[1:], y[1:], c='blue', label='Clients')
        plt.scatter(x[0], y[0], c='red', marker='s', s=100, label='Depot')

        # Plot each route
        for i, route in enumerate(routes):
            try:
                valid_nodes = [node for node in route if 0 <= node < len(self.coords)]
                if len(valid_nodes) < 2:
                    continue

                path_x, path_y = zip(*[self.coords[node] for node in valid_nodes])
                plt.plot(path_x, path_y, marker='o',
                         label=f'Route {i + 1}',
                         linestyle='-', linewidth=2)

            except Exception as e:
                print(f"Error plotting route {i}: {str(e)}")
                continue

        plt.title("Vehicle Routing Solution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
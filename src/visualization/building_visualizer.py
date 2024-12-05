"""Module for visualizing WiFi signal strength and building materials."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects
from typing import List, Tuple, Optional, Dict
from src.physics.materials import Material, MATERIALS
import os
import seaborn as sns

class BuildingVisualizer:
    """Visualizes WiFi signal strength and building materials."""
    
    def __init__(self, width: float = 50.0, height: float = 50.0, resolution: float = 0.5):
        """Initialize the building visualizer.
        
        Args:
            width (float): Width of the building in meters
            height (float): Height of the building in meters
            resolution (float): Grid resolution in meters
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.materials_grid = None
        self.walls = []  # List of (material, x, y, w, h) tuples
        self.material_colors = {
            'concrete': '#808080',  # Gray
            'glass': '#ADD8E6',    # Light blue
            'wood': '#8B4513',     # Saddle brown
            'drywall': '#F5F5F5',  # White smoke
            'metal': '#C0C0C0',    # Silver
        }
        
    def add_material(self, material: Material, x: float, y: float, w: float, h: float):
        """Add a material to the building at specified location.
        
        Args:
            material (Material): Material to add
            x (float): X coordinate of bottom-left corner
            y (float): Y coordinate of bottom-left corner
            w (float): Width of material
            h (float): Height of material
        """
        if self.materials_grid is None:
            rows = int(self.height / self.resolution)
            cols = int(self.width / self.resolution)
            self.materials_grid = [[None for _ in range(cols)] for _ in range(rows)]
            
        # Store wall for visualization
        self.walls.append((material, x, y, w, h))
        
        # Convert coordinates to grid indices
        x1 = int(x / self.resolution)
        y1 = int(y / self.resolution)
        x2 = int((x + w) / self.resolution)
        y2 = int((y + h) / self.resolution)
        
        # Ensure indices are within bounds
        x1 = max(0, min(x1, len(self.materials_grid[0])))
        x2 = max(0, min(x2, len(self.materials_grid[0])))
        y1 = max(0, min(y1, len(self.materials_grid)))
        y2 = max(0, min(y2, len(self.materials_grid)))
        
        # Add material to grid
        for i in range(y1, y2):
            for j in range(x1, x2):
                self.materials_grid[i][j] = material

    def plot_signal_strength(self, rssi_grid, points, ap_locations, output_path):
        """Plot WiFi signal strength heatmap.
        
        Args:
            rssi_grid: 2D numpy array of RSSI values
            points: List of (x, y) coordinates used for sampling
            ap_locations: Either a single (x, y) tuple for one AP or a dict of AP names to locations
            output_path: Path to save the plot
        """
        # Create figure with extra space for legend
        plt.figure(figsize=(14, 8))
        
        # Create main plot area
        main_ax = plt.gca()
        
        # Convert points to grid coordinates
        points = np.array(points)
        x_unique = np.unique(points[:, 0])
        y_unique = np.unique(points[:, 1])
        
        # Create heatmap with interpolation
        im = plt.imshow(rssi_grid, 
                     extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                     origin='lower',
                     cmap='RdYlBu_r',
                     aspect='equal',
                     interpolation='gaussian')  # Add interpolation
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Signal Strength (dBm)')
        cbar.ax.tick_params(labelsize=9)
        
        # Create legend elements for materials
        material_patches = []
        seen_materials = set()
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', alpha=0.3, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
        
        # Plot materials and walls
        for material, x, y, w, h in self.walls:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', alpha=0.3)
            main_ax.add_patch(rect)
        
        # Plot AP locations and create legend elements
        ap_artists = []
        if isinstance(ap_locations, dict):
            for ap_name, loc in ap_locations.items():
                # Plot the AP marker (bigger size for the actual marker)
                artist = plt.plot(loc[0], loc[1], 'r*', markersize=40, zorder=10)[0]
                # Create a separate artist for the legend with original size
                legend_artist = plt.plot([], [], 'r*', markersize=15)[0]
                ap_artists.append((legend_artist, ap_name))
                
                # Add label with white background
                plt.text(loc[0], loc[1], ap_name[-1],  # Just get the number from APX
                        fontsize=12,
                        color='black',
                        bbox=dict(facecolor='white', 
                                edgecolor='black',
                                alpha=1.0,
                                pad=0.5),
                        ha='center',
                        va='center',
                        zorder=11)  # Higher zorder than the marker
                
        elif ap_locations is not None:
            # For single AP plots, extract the AP number from the output path
            ap_num = '1'  # default
            if 'output_path' in locals():
                import re
                match = re.search(r'coverage_AP(\d+)\.png', output_path)
                if match:
                    ap_num = match.group(1)
            
            # Plot the AP marker (bigger size for the actual marker)
            artist = plt.plot(ap_locations[0], ap_locations[1], 'r*', markersize=40, zorder=10)[0]
            # Create a separate artist for the legend with original size
            legend_artist = plt.plot([], [], 'r*', markersize=15)[0]
            ap_artists.append((legend_artist, f'AP{ap_num}'))
            
            # Add label with white background for single AP
            plt.text(ap_locations[0], ap_locations[1], ap_num,
                    fontsize=12,
                    color='black',
                    bbox=dict(facecolor='white', 
                            edgecolor='black',
                            alpha=1.0,
                            pad=0.5),
                    ha='center',
                    va='center',
                    zorder=11)
        
        # Add legends
        # Materials legend
        material_legend = plt.legend(handles=material_patches, title='Materials',
                                   bbox_to_anchor=(1.15, 1), loc='upper left',
                                   fontsize=9)
        plt.gca().add_artist(material_legend)
        
        # AP legend
        ap_legend = plt.legend([artist for artist, _ in ap_artists],
                             [name for _, name in ap_artists],
                             title='Access Points',
                             bbox_to_anchor=(1.15, 0.6), loc='upper left',
                             fontsize=9)
        
        plt.title('WiFi Signal Strength Map')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save plot with higher resolution
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def plot_access_points(self, ax, access_points):
        """Plot access points on the map."""
        if access_points:
            ap_x = [ap.x for ap in access_points]
            ap_y = [ap.y for ap in access_points]
            # Increase marker size to 800 (from default ~100)
            scatter = ax.scatter(ap_x, ap_y, c='yellow', marker='*', s=800, zorder=5, edgecolor='black', linewidth=1.5)
            # Add white labels with black edge for visibility
            for idx, (x, y) in enumerate(zip(ap_x, ap_y)):
                ax.text(x, y, f'{idx+1}', color='white', fontweight='bold',
                        ha='center', va='center', zorder=6, fontsize=12,
                        path_effects=[patheffects.withStroke(linewidth=3, 
                                    foreground='black')])
            return scatter
        return None

    def plot_coverage(self, signal_grid, access_points, output_path, title=None):
        """Plot WiFi coverage with materials and access points."""
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        # Plot the interpolated signal strength
        x = np.linspace(0, self.width, signal_grid.shape[1])
        y = np.linspace(0, self.height, signal_grid.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Plot signal strength with slightly lower zorder
        plt.pcolormesh(X, Y, signal_grid, shading='auto', cmap='jet', zorder=1)
        plt.colorbar(label='Signal Strength (dBm)')

        # Plot materials with separate legend (zorder=2)
        material_patches = []
        material_labels = []
        for material_name, material_value in MATERIALS.items():
            mask = np.zeros_like(self.materials_grid, dtype=bool)
            for i in range(len(self.materials_grid)):
                for j in range(len(self.materials_grid[0])):
                    if isinstance(self.materials_grid[i][j], Material):
                        if self.materials_grid[i][j].name == material_name:
                            mask[i][j] = True
            
            if np.any(mask):
                color = self.material_colors.get(material_name, 'gray')
                plt.pcolormesh(X, Y, np.where(mask, 1, np.nan), 
                             color=color, alpha=0.5, zorder=2)
                material_patches.append(plt.Rectangle((0,0), 1, 1, 
                                     fc=color, alpha=0.5))
                material_labels.append(material_name.title())

        # Plot access points with much larger stars (zorder=10)
        if access_points:
            # Handle both dictionary of locations and list of AP objects
            if isinstance(access_points, dict):
                ap_x = [loc[0] for loc in access_points.values()]
                ap_y = [loc[1] for loc in access_points.values()]
                ap_names = list(access_points.keys())
            else:
                ap_x = [ap.x for ap in access_points]
                ap_y = [ap.y for ap in access_points]
                ap_names = [f"AP{i+1}" for i in range(len(access_points))]
            
            print("AP Coordinates:")
            for i, (x, y) in enumerate(zip(ap_x, ap_y)):
                print(f"{ap_names[i]}: ({x}, {y})")
            
            # Plot the stars
            scatter = plt.scatter(ap_x, ap_y, c='red', marker='*', s=4000, 
                                zorder=10, edgecolor='black', linewidth=2)
            
            # Try a different way to add labels
            for idx, (x, y) in enumerate(zip(ap_x, ap_y)):
                # Create a text box with white background
                textbox = ax.text(x, y, str(idx+1),
                                fontsize=36,
                                color='black',
                                bbox=dict(facecolor='white', 
                                        edgecolor='black',
                                        alpha=1.0,
                                        pad=0.5),
                                ha='center',
                                va='center',
                                zorder=20)
                print(f"Added label {idx+1} at position ({x}, {y})")
            
            # Force drawing update
            plt.draw()
        
        # Create and position the materials legend
        materials_legend = plt.legend(material_patches, material_labels,
                                    title='Materials', bbox_to_anchor=(1.15, 1),
                                    loc='upper left')
        plt.gca().add_artist(materials_legend)

        # Add AP legend if there are multiple APs
        if len(access_points) > 1:
            ap_legend = plt.legend([scatter], ['Access Points'],
                                 bbox_to_anchor=(1.15, 0.9),
                                 loc='upper left')
            plt.gca().add_artist(ap_legend)

        plt.title(title if title else 'WiFi Coverage Map')
        plt.xlabel('Width (m)')
        plt.ylabel('Height (m)')
        
        # Ensure the aspect ratio is equal
        plt.axis('equal')
        plt.tight_layout()
        
        # Save plot with higher resolution
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def plot_signal_statistics(self, rssi_by_ap, output_dir):
        """Generate separate statistical plots for WiFi signal analysis."""
        self.plot_average_signal_strength(rssi_by_ap, output_dir)
        self.plot_coverage_area(rssi_by_ap, output_dir)
        self.plot_signal_distribution(rssi_by_ap, output_dir)

    def plot_average_signal_strength(self, rssi_by_ap, output_dir):
        """Create a bar plot showing average RSSI values for each AP."""
        plt.figure(figsize=(10, 6))
        ap_means = [np.mean(rssi) for rssi in rssi_by_ap.values()]
        ap_names = list(rssi_by_ap.keys())
        
        bars = plt.bar(ap_names, ap_means)
        plt.axhline(y=-70, color='r', linestyle='--', label='Good Signal (-70 dBm)')
        plt.axhline(y=-80, color='y', linestyle='--', label='Fair Signal (-80 dBm)')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.title('Average Signal Strength by Access Point')
        plt.xlabel('Access Point')
        plt.ylabel('Average RSSI (dBm)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_signal_strength.png'))
        plt.close()

    def plot_coverage_area(self, rssi_by_ap, output_dir):
        """Create a grouped bar plot showing coverage area percentages."""
        thresholds = [-70, -80]  # Good and Fair signal thresholds
        coverage_data = []
        
        for ap_name, rssi_values in rssi_by_ap.items():
            coverage = []
            total_points = len(rssi_values)
            for threshold in thresholds:
                coverage_count = np.sum(np.array(rssi_values) >= threshold)
                coverage.append((coverage_count / total_points) * 100)
            coverage_data.append(coverage)
        
        coverage_data = np.array(coverage_data)
        ap_names = list(rssi_by_ap.keys())
        x = np.arange(len(ap_names))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, coverage_data[:, 0], width, label='Good Signal (≥ -70 dBm)')
        plt.bar(x + width/2, coverage_data[:, 1], width, label='Fair Signal (≥ -80 dBm)')
        
        plt.xlabel('Access Point')
        plt.ylabel('Coverage Percentage (%)')
        plt.title('WiFi Coverage Analysis by Access Point')
        plt.xticks(x, ap_names)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i in range(len(ap_names)):
            plt.text(i - width/2, coverage_data[i, 0], f'{coverage_data[i, 0]:.1f}%',
                    ha='center', va='bottom')
            plt.text(i + width/2, coverage_data[i, 1], f'{coverage_data[i, 1]:.1f}%',
                    ha='center', va='bottom')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_area.png'))
        plt.close()

    def plot_signal_distribution(self, rssi_by_ap, output_dir):
        """Create distribution plots (histogram + KDE) for signal strength."""
        plt.figure(figsize=(12, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(rssi_by_ap)))
        
        # Plot KDE for each AP using flattened values
        for (ap_name, rssi_grid), color in zip(rssi_by_ap.items(), colors):
            rssi_values = rssi_grid.flatten()  # Flatten the grid to 1D array
            sns.kdeplot(data=rssi_values, label=ap_name, color=color)
        
        # Add threshold lines
        plt.axvline(x=-70, color='r', linestyle='--', label='Good Signal (-70 dBm)')
        plt.axvline(x=-80, color='y', linestyle='--', label='Fair Signal (-80 dBm)')
        
        plt.title('Signal Strength Distribution by Access Point')
        plt.xlabel('RSSI (dBm)')
        plt.ylabel('Density')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        # Adjust layout with extra space for legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        plt.savefig(os.path.join(output_dir, 'signal_distribution.png'))
        plt.close()

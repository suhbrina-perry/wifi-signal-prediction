"""Main script for WiFi signal strength prediction."""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.building_visualizer import BuildingVisualizer
from src.data_collection.wifi_data_collector import WiFiDataCollector
from src.physics.materials import MATERIALS, SignalPath

def save_run_info(args, run_dir):
    """Save run configuration and metadata.
    
    Args:
        args: Command line arguments
        run_dir: Directory to save run information
    """
    # Create run info dictionary
    run_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'building_width': args.width,
            'building_height': args.height,
            'ap_location': {'x': args.ap_x, 'y': args.ap_y},
            'sampling_resolution': args.resolution,
        },
        'materials_used': list(MATERIALS.keys()),
        'mode': 'building_layout' if args.building_layout else 
               'materials_info' if args.materials else 
               'signal_strength'
    }
    
    # Save run info
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=4)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WiFi Signal Strength Prediction')
    
    # Operation modes
    parser.add_argument('--building-layout', action='store_true',
                      help='Generate and display building layout only')
    parser.add_argument('--signal-strength', action='store_true',
                      help='Generate signal strength heatmap')
    parser.add_argument('--materials', action='store_true',
                      help='Show material properties and effects')
    
    # Building dimensions
    parser.add_argument('--width', type=float, default=20.0,
                      help='Building width in meters (default: 20.0)')
    parser.add_argument('--height', type=float, default=15.0,
                      help='Building height in meters (default: 15.0)')
    
    # Access point location
    parser.add_argument('--ap-x', type=float, default=5.0,
                      help='Access point X coordinate in meters (default: 5.0)')
    parser.add_argument('--ap-y', type=float, default=3.0,
                      help='Access point Y coordinate in meters (default: 3.0)')
    
    # Sampling resolution
    parser.add_argument('--resolution', type=int, default=80,
                      help='Number of sample points along width (default: 80)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path (default: auto-generated in results directory)')
    
    return parser.parse_args()

def create_building(visualizer, args):
    """Create building layout with walls, doors, and windows.
    
    Args:
        visualizer (BuildingVisualizer): Building visualizer instance
        args: Command line arguments
    """
    # Wall thicknesses
    CONCRETE_THICKNESS = 0.3
    DRYWALL_THICKNESS = 0.15
    DOOR_WIDTH = 1.0
    WINDOW_WIDTH = 2.0
    WINDOW_HEIGHT = 1.5
    
    # Add rooms and materials
    # Outer walls (concrete)
    visualizer.add_material(MATERIALS['concrete'], 0, 0, CONCRETE_THICKNESS, args.height)  # Left wall
    visualizer.add_material(MATERIALS['concrete'], 0, args.height-CONCRETE_THICKNESS, args.width, CONCRETE_THICKNESS)  # Top wall
    visualizer.add_material(MATERIALS['concrete'], args.width-CONCRETE_THICKNESS, 0, CONCRETE_THICKNESS, args.height)  # Right wall
    visualizer.add_material(MATERIALS['concrete'], 0, 0, args.width, CONCRETE_THICKNESS)  # Bottom wall
    
    # Inner walls (drywall)
    visualizer.add_material(MATERIALS['drywall'], args.width/2, CONCRETE_THICKNESS, 
                          DRYWALL_THICKNESS, args.height-2*CONCRETE_THICKNESS)  # Vertical divider
    visualizer.add_material(MATERIALS['drywall'], CONCRETE_THICKNESS, args.height/2, 
                          args.width-2*CONCRETE_THICKNESS, DRYWALL_THICKNESS)  # Horizontal divider
    
    # Windows (glass)
    visualizer.add_material(MATERIALS['glass'], args.width/4, 0, WINDOW_WIDTH, CONCRETE_THICKNESS)  # Bottom window
    visualizer.add_material(MATERIALS['glass'], 3*args.width/4, args.height-CONCRETE_THICKNESS, 
                          WINDOW_WIDTH, CONCRETE_THICKNESS)  # Top window
    visualizer.add_material(MATERIALS['glass'], args.width-CONCRETE_THICKNESS, args.height/3, 
                          CONCRETE_THICKNESS, WINDOW_HEIGHT)  # Right window
    
    # Doors (wood)
    door_x = args.width/2 - DOOR_WIDTH/2
    visualizer.add_material(MATERIALS['wood'], door_x, args.height/4, DOOR_WIDTH, 0.1)  # Bottom door
    visualizer.add_material(MATERIALS['wood'], door_x, 3*args.height/4, DOOR_WIDTH, 0.1)  # Top door

def show_building_layout(args, run_dir):
    """Display building layout without signal strength."""
    visualizer = BuildingVisualizer(width=args.width, height=args.height)
    create_building(visualizer, args)
    
    plt.figure(figsize=(12, 8))
    
    # Plot materials
    for material, x, y, w, h in visualizer.walls:
        color = visualizer.material_colors.get(material.name.lower(), '#FFFFFF')
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', alpha=0.7)
        plt.gca().add_patch(rect)
        if w > 1.0 or h > 1.0:
            plt.text(x + w/2, y + h/2, material.name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)
    
    # Plot AP location
    plt.plot(args.ap_x, args.ap_y, 'r*', markersize=15, label='Access Point')
    
    plt.xlim(0, args.width)
    plt.ylim(0, args.height)
    plt.title('Building Layout with Access Point')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal')
    
    # Save plot
    output_path = os.path.join(run_dir, 'building_layout.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Building layout saved to: {output_path}")
    plt.close()

def show_materials_info(run_dir):
    """Display information about available materials and their properties."""
    # Create output string
    output = []
    output.append("\nAvailable Materials and Properties:")
    output.append("-" * 80)
    output.append(f"{'Material':<15} {'Relative Permittivity':<25} {'Conductivity (S/m)':<20} {'Default Thickness (m)'}")
    output.append("-" * 80)
    
    for name, material in MATERIALS.items():
        output.append(f"{material.name:<15} {material.relative_permittivity:<25.2f} {material.conductivity:<20.3f} {material.thickness:.3f}")
    
    output.append("\nMaterial Effects on 2.4 GHz WiFi Signal:")
    output.append("-" * 80)
    for name, material in MATERIALS.items():
        attenuation = material.calculate_attenuation(2.4e9)
        output.append(f"{material.name:<15} Attenuation: {attenuation:.1f} dB through {material.thickness:.3f}m thickness")
    
    # Print to console
    print('\n'.join(output))
    
    # Save to file
    with open(os.path.join(run_dir, 'materials_info.txt'), 'w') as f:
        f.write('\n'.join(output))
    print(f"\nMaterial information saved to: {os.path.join(run_dir, 'materials_info.txt')}")

def main():
    """Run the main WiFi signal strength prediction demo."""
    args = parse_args()
    
    # Create results directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save run information
    save_run_info(args, results_dir)
    
    if args.building_layout:
        show_building_layout(args, results_dir)
        return
        
    if args.materials:
        show_materials_info(results_dir)
        return
    
    # Default: generate signal strength heatmap
    visualizer = BuildingVisualizer(width=args.width, height=args.height)
    collector = WiFiDataCollector()
    
    # Create building
    create_building(visualizer, args)
    
    # Generate sample points
    x = np.linspace(0, args.width, args.resolution)
    y = np.linspace(0, args.height, int(args.resolution * args.height/args.width))
    X, Y = np.meshgrid(x, y)
    points = list(zip(X.flatten(), Y.flatten()))
    
    # Set AP location
    ap_location = (args.ap_x, args.ap_y)
    
    # Collect RSSI samples
    rssi_values = collector.collect_samples(points, ap_location, visualizer.materials_grid)
    
    # Plot and save results
    output_path = os.path.join(results_dir, 'signal_strength.png')
    visualizer.plot_signal_strength(rssi_values, points, ap_location, output_path)
    print(f"Signal strength heatmap saved to: {output_path}")
    
    # Save sample points and RSSI values
    np.savez(os.path.join(results_dir, 'signal_data.npz'),
             points=points,
             rssi_values=rssi_values)

if __name__ == '__main__':
    main()

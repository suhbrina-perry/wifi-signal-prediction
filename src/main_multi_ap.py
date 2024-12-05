"""Main script for WiFi signal strength prediction with multiple APs."""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.building_visualizer import BuildingVisualizer
from src.data_collection.wifi_data_collector import WiFiDataCollector
from src.physics.materials import MATERIALS, SignalPath

def create_office_layout(visualizer, args):
    """Create a complex office layout with multiple rooms.
    
    Args:
        visualizer (BuildingVisualizer): Building visualizer instance
        args: Command line arguments
    """
    # Material thicknesses
    CONCRETE_THICKNESS = 0.3
    DRYWALL_THICKNESS = 0.15
    DOOR_WIDTH = 1.0
    WINDOW_WIDTH = 2.0
    WINDOW_HEIGHT = 1.5
    
    # Outer walls (concrete)
    visualizer.add_material(MATERIALS['concrete'], 0, 0, CONCRETE_THICKNESS, args.height)  # Left wall
    visualizer.add_material(MATERIALS['concrete'], 0, args.height-CONCRETE_THICKNESS, args.width, CONCRETE_THICKNESS)  # Top wall
    visualizer.add_material(MATERIALS['concrete'], args.width-CONCRETE_THICKNESS, 0, CONCRETE_THICKNESS, args.height)  # Right wall
    visualizer.add_material(MATERIALS['concrete'], 0, 0, args.width, CONCRETE_THICKNESS)  # Bottom wall
    
    # Meeting rooms (top)
    room_width = args.width / 4 - CONCRETE_THICKNESS
    room_height = args.height / 3
    for i in range(4):
        x = i * args.width/4 + CONCRETE_THICKNESS
        # Room dividers
        if i < 3:
            visualizer.add_material(MATERIALS['drywall'], (i+1)*args.width/4, CONCRETE_THICKNESS, 
                                  DRYWALL_THICKNESS, room_height)
        # Add glass windows
        visualizer.add_material(MATERIALS['glass'], x + room_width/2 - WINDOW_WIDTH/2, 
                              room_height - CONCRETE_THICKNESS, WINDOW_WIDTH, CONCRETE_THICKNESS)
        # Add wooden doors
        visualizer.add_material(MATERIALS['wood'], x + room_width/2 - DOOR_WIDTH/2, 
                              room_height + DRYWALL_THICKNESS, DOOR_WIDTH, DRYWALL_THICKNESS)
    
    # Open office area (middle)
    visualizer.add_material(MATERIALS['drywall'], 0, room_height, args.width, DRYWALL_THICKNESS)
    visualizer.add_material(MATERIALS['drywall'], 0, 2*room_height, args.width, DRYWALL_THICKNESS)
    
    # Private offices (bottom)
    office_width = args.width / 5 - CONCRETE_THICKNESS
    for i in range(5):
        x = i * args.width/5 + CONCRETE_THICKNESS
        # Office dividers
        if i < 4:
            visualizer.add_material(MATERIALS['drywall'], (i+1)*args.width/5, 2*room_height, 
                                  DRYWALL_THICKNESS, room_height - CONCRETE_THICKNESS)
        # Add wooden doors
        visualizer.add_material(MATERIALS['wood'], x + office_width/2 - DOOR_WIDTH/2, 
                              2*room_height - DRYWALL_THICKNESS, DOOR_WIDTH, DRYWALL_THICKNESS)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WiFi Signal Strength Prediction')
    
    # Building dimensions
    parser.add_argument('--width', type=float, default=30.0,
                      help='Building width in meters (default: 30.0)')
    parser.add_argument('--height', type=float, default=20.0,
                      help='Building height in meters (default: 20.0)')
    
    # Sampling resolution
    parser.add_argument('--resolution', type=int, default=100,
                      help='Number of sample points along width (default: 100)')
    
    return parser.parse_args()

def main():
    """Run the main WiFi signal strength prediction demo."""
    args = parse_args()
    
    # Create results directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize visualizer and collector
    visualizer = BuildingVisualizer(width=args.width, height=args.height)
    collector = WiFiDataCollector()
    
    # Create complex office layout
    create_office_layout(visualizer, args)
    
    # Define multiple AP locations for optimal coverage
    ap_locations = [
        (args.width/4, args.height/4),      # Bottom left
        (3*args.width/4, args.height/4),    # Bottom right
        (args.width/4, 3*args.height/4),    # Top left
        (3*args.width/4, 3*args.height/4),  # Top right
    ]
    
    # Generate sample points
    x = np.linspace(0, args.width, args.resolution)
    y = np.linspace(0, args.height, int(args.resolution * args.height/args.width))
    X, Y = np.meshgrid(x, y)
    points = list(zip(X.flatten(), Y.flatten()))
    
    # Collect RSSI samples for each AP
    all_rssi_values = []
    for ap_location in ap_locations:
        rssi_values = collector.collect_samples(points, ap_location, visualizer.materials_grid)
        all_rssi_values.append(rssi_values)
    
    # Combine RSSI values (take maximum at each point)
    combined_rssi = np.maximum.reduce(all_rssi_values)
    
    # Plot and save results
    output_path = os.path.join(results_dir, 'signal_strength_multi_ap.png')
    visualizer.plot_signal_strength(combined_rssi, points, ap_locations[0], output_path)
    print(f"Signal strength heatmap saved to: {output_path}")
    
    # Save sample points and RSSI values
    np.savez(os.path.join(results_dir, 'signal_data_multi_ap.npz'),
             points=points,
             rssi_values=combined_rssi,
             ap_locations=ap_locations)

if __name__ == '__main__':
    main()

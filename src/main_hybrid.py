"""Main script for WiFi signal strength prediction combining real and simulated data."""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.building_visualizer import BuildingVisualizer
from src.data_collection.wifi_data_collector import WiFiDataCollector
from src.physics.materials import MATERIALS, SignalPath
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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

def load_real_data(data_path):
    """Load and preprocess real WiFi data.
    
    Args:
        data_path (str): Path to the WiFi data CSV file
        
    Returns:
        tuple: X (features) and y (RSSI values) for each AP
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Group by AP and create separate datasets
    ap_data = {}
    for ap in df['ssid'].unique():
        ap_df = df[df['ssid'] == ap]
        X = ap_df[['x', 'y']].values
        y = ap_df['rssi'].values
        ap_data[ap] = (X, y)
    
    return ap_data

def train_models(ap_data):
    """Train Random Forest models for each AP.
    
    Args:
        ap_data (dict): Dictionary containing X and y data for each AP
        
    Returns:
        tuple: Trained models and scalers for each AP
    """
    models = {}
    scalers = {}
    
    for ap, (X, y) in ap_data.items():
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        models[ap] = model
        scalers[ap] = scaler
    
    return models, scalers

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
    
    # Data path
    parser.add_argument('--data-path', type=str, 
                      default='results/run_20241119_102048/data/wifi_data.csv',
                      help='Path to real WiFi data')
    
    return parser.parse_args()

def main():
    """Run the main WiFi signal strength prediction demo."""
    args = parse_args()
    
    # Create results directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and process real data
    ap_data = load_real_data(args.data_path)
    models, scalers = train_models(ap_data)
    
    # Initialize visualizer and collector
    visualizer = BuildingVisualizer(width=args.width, height=args.height)
    collector = WiFiDataCollector()
    
    # Create complex office layout
    create_office_layout(visualizer, args)
    
    # Generate sample points
    x = np.linspace(0, args.width, args.resolution)
    y = np.linspace(0, args.height, int(args.resolution * args.height/args.width))
    X, Y = np.meshgrid(x, y)
    points = list(zip(X.flatten(), Y.flatten()))
    
    # Predict RSSI values using trained models
    all_rssi_values = []
    for ap, model in models.items():
        # Scale points
        points_array = np.array(points)
        points_scaled = scalers[ap].transform(points_array)
        
        # Predict RSSI values
        rssi_values = model.predict(points_scaled)
        
        # Adjust predictions using physics-based model
        physics_rssi = collector.collect_samples(points, (args.width/2, args.height/2), visualizer.materials_grid)
        
        # Combine predictions (weighted average)
        combined_rssi = 0.7 * rssi_values + 0.3 * physics_rssi
        all_rssi_values.append(combined_rssi)
    
    # Combine RSSI values (take maximum at each point)
    combined_rssi = np.maximum.reduce(all_rssi_values)
    
    # Plot and save results
    output_path = os.path.join(results_dir, 'signal_strength_hybrid.png')
    visualizer.plot_signal_strength(combined_rssi, points, (args.width/2, args.height/2), output_path)
    print(f"Signal strength heatmap saved to: {output_path}")
    
    # Save predictions and model info
    np.savez(os.path.join(results_dir, 'predictions.npz'),
             points=points,
             rssi_values=combined_rssi,
             ap_locations=list(models.keys()))
    
    # Save run information
    run_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'building_width': args.width,
            'building_height': args.height,
            'resolution': args.resolution,
            'data_source': args.data_path
        },
        'materials_used': list(MATERIALS.keys()),
        'access_points': list(models.keys())
    }
    
    with open(os.path.join(results_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)

if __name__ == '__main__':
    main()

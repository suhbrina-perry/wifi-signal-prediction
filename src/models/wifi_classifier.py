"""WiFi signal strength prediction models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class WiFiSignalPredictor:
    """WiFi signal strength prediction model."""
    
    def __init__(self):
        """Initialize the predictor with multiple models."""
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5, weights='distance')
        }
        self.scalers = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def prepare_features(self, data):
        """Prepare features for model training.
        
        Args:
            data: DataFrame with columns ['x', 'y', 'rssi', 'ssid']
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Extract features
        X = data[['x', 'y']].copy()
        
        # Add derived features
        X['distance_from_center'] = np.sqrt((X['x'] - X['x'].mean())**2 + 
                                          (X['y'] - X['y'].mean())**2)
        X['angle'] = np.arctan2(X['y'] - X['y'].mean(), 
                               X['x'] - X['x'].mean())
        
        return X, data['rssi'].values
        
    def train(self, data, ap_name):
        """Train models for a specific access point.
        
        Args:
            data: DataFrame with columns ['x', 'y', 'rssi', 'ssid']
            ap_name: Name of the access point
            
        Returns:
            Dictionary with training results
        """
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[ap_name] = scaler
        
        results = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'actual': y_test,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                }
            }
            
            # Store feature importance for RF
            if name == 'rf':
                self.feature_importance[ap_name] = dict(zip(X.columns, 
                                                          model.feature_importances_))
        
        return results
    
    def save_models(self, output_dir):
        """Save trained models and scalers.
        
        Args:
            output_dir: Directory to save models
        """
        # Create models directory if it doesn't exist
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save models and scalers
        for ap_name in self.scalers.keys():
            ap_dir = os.path.join(models_dir, ap_name)
            os.makedirs(ap_dir, exist_ok=True)
            
            # Save scaler
            scaler_path = os.path.join(ap_dir, 'scaler.joblib')
            joblib.dump(self.scalers[ap_name], scaler_path)
            
            # Save models
            for model_name, model in self.models.items():
                model_path = os.path.join(ap_dir, f'{model_name}.joblib')
                joblib.dump(model, model_path)

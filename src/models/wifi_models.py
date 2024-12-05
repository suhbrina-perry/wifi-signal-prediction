from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class WiFiModelTrainer:
    def __init__(self):
        """Initialize the WiFi model trainer."""
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        self.trained_models = {}
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target values
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def train_and_evaluate(self, X, y):
        """Train and evaluate all models.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            dict: Results for each model
        """
        results = {}
        
        for name, model in self.models.items():
            # Train the model
            model.fit(X, y)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            # Perform cross-validation
            cv_rmse = -cross_val_score(model, X, y, 
                                     scoring='neg_root_mean_squared_error',
                                     cv=5)
            
            results[name] = {
                'metrics': {
                    'rmse': rmse,
                    'r2': r2
                },
                'cv_results': {
                    'mean_rmse': cv_rmse.mean(),
                    'std_rmse': cv_rmse.std()
                },
                'predictions': y_pred
            }
        
        return results
        
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train: Training feature matrix
            y_train: Training target values
            
        Returns:
            object: Trained model
        """
        print(f"\nTraining {model_name.upper()} model...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
        
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            dict: Evaluation metrics
        """
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Perform cross-validation
        cv_rmse = -cross_val_score(model, X_test, y_test, 
                                 scoring='neg_root_mean_squared_error',
                                 cv=5)
        
        return {
            'metrics': metrics,
            'cv_results': {
                'mean_rmse': cv_rmse.mean(),
                'std_rmse': cv_rmse.std()
            },
            'predictions': y_pred
        }
        
    def predict(self, model_name, X):
        """Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model to use
            X: Feature matrix
            
        Returns:
            array: Predicted values
        """
        model = self.trained_models[model_name]
        return model.predict(X)
        
    def cross_validate(self, model_name, X, y, cv=5):
        """Perform cross-validation for a model.
        
        Args:
            model_name (str): Name of the model to validate
            X: Feature matrix
            y: Target values
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        model = self.models[model_name]
        
        # Perform cross-validation for RMSE
        cv_rmse = -cross_val_score(model, X, y, 
                                 scoring='neg_root_mean_squared_error',
                                 cv=cv)
                                 
        # Perform cross-validation for R2
        cv_r2 = cross_val_score(model, X, y,
                              scoring='r2',
                              cv=cv)
        
        return {
            'mean_rmse': cv_rmse.mean(),
            'std_rmse': cv_rmse.std(),
            'mean_r2': cv_r2.mean(),
            'std_r2': cv_r2.std()
        }

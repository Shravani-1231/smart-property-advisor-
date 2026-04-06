"""
Machine Learning Model Training Module for Smart Property Advisor
Trains and evaluates property price prediction models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class PropertyPriceModel:
    """Property Price Prediction Model"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False
        
        # Available models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
    
    def _preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features for model training/prediction"""
        df = df.copy()
        
        # Drop non-feature columns
        if 'property_id' in df.columns:
            df = df.drop('property_id', axis=1)
        if 'price' in df.columns:
            df = df.drop('price', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['property_type', 'location_tier']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[col] = df[col].apply(
                            lambda x: x if x in self.label_encoders[col].classes_ 
                            else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'price') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        X = self._preprocess_features(df, fit=True)
        y = df[target_col].values
        
        self.feature_columns = X.columns.tolist()
        
        return X.values, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.model = self.models[self.model_type]
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        self.is_trained = True
        
        print(f"\nModel Performance:")
        print(f"Train MAE: ${metrics['train_mae']:,.2f}")
        print(f"Test MAE: ${metrics['test_mae']:,.2f}")
        print(f"Train R²: {metrics['train_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        X = self._preprocess_features(features, fit=False)
        
        # Ensure columns match training data
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X.values)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


class ModelComparison:
    """Compare different models"""
    
    @staticmethod
    def compare_models(df: pd.DataFrame, models: list = None) -> pd.DataFrame:
        """Compare multiple models"""
        if models is None:
            models = ['random_forest', 'gradient_boosting', 'linear', 'ridge']
        
        results = []
        
        for model_type in models:
            print(f"\n{'='*50}")
            print(f"Training {model_type}...")
            print('='*50)
            
            model = PropertyPriceModel(model_type=model_type)
            metrics = model.train(df)
            metrics['model_type'] = model_type
            results.append(metrics)
        
        return pd.DataFrame(results)


def train_and_save_model(data_path: str = "data/property_data.csv",
                         model_path: str = "models/property_price_model.pkl",
                         model_type: str = 'random_forest') -> PropertyPriceModel:
    """Train and save a model"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Train model
    model = PropertyPriceModel(model_type=model_type)
    metrics = model.train(df)
    
    # Save model
    model.save_model(model_path)
    
    # Print feature importance
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        print("\nTop 10 Feature Importances:")
        print(importance_df.head(10))
    
    return model


if __name__ == "__main__":
    # Generate data if it doesn't exist
    if not os.path.exists("data/property_data.csv"):
        from data_generator import generate_and_save_dataset
        generate_and_save_dataset()
    
    # Train and save model
    model = train_and_save_model()
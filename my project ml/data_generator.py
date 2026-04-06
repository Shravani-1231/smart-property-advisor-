"""
Data Generation Module for Smart Property Advisor
Generates synthetic real estate data for model training
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


class PropertyDataGenerator:
    """Generates synthetic real estate property data"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Location multipliers (city tier affects price)
        self.location_tiers = {
            'Tier 1': 2.5,      # Major metros
            'Tier 2': 1.5,      # Secondary cities
            'Tier 3': 1.0       # Smaller cities
        }
        
        # Property type base prices
        self.property_types = {
            'Apartment': 1.0,
            'House': 1.3,
            'Villa': 2.0,
            'Penthouse': 2.5,
            'Studio': 0.6,
            'Duplex': 1.4
        }
        
    def generate_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic property data"""
        
        # Generate base features
        data = {
            'property_id': range(1, n_samples + 1),
            'square_feet': np.random.normal(1500, 600, n_samples).clip(300, 8000).astype(int),
            'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, 
                                        p=[0.1, 0.25, 0.35, 0.2, 0.08, 0.02]),
            'bathrooms': np.random.choice([1, 2, 3, 4, 5], n_samples,
                                         p=[0.15, 0.4, 0.3, 0.12, 0.03]),
            'age_years': np.random.exponential(15, n_samples).clip(0, 100).astype(int),
            'distance_to_city_center': np.random.exponential(10, n_samples).clip(0.5, 50),
            'property_type': np.random.choice(
                list(self.property_types.keys()), 
                n_samples,
                p=[0.3, 0.25, 0.15, 0.05, 0.15, 0.1]
            ),
            'location_tier': np.random.choice(
                list(self.location_tiers.keys()),
                n_samples,
                p=[0.2, 0.4, 0.4]
            ),
            'has_garden': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_pool': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'has_garage': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'furnished': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'floor': np.random.choice([0, 1, 2, 3, 4, 5, 10, 15, 20, 25], n_samples,
                                     p=[0.1, 0.15, 0.15, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03]),
            'crime_rate': np.random.beta(2, 5, n_samples) * 100,  # 0-100 scale
            'school_rating': np.random.normal(7, 1.5, n_samples).clip(1, 10),
            'hospital_distance': np.random.exponential(5, n_samples).clip(0.5, 30),
            'shopping_distance': np.random.exponential(3, n_samples).clip(0.2, 15),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate price based on features
        df['price'] = self._calculate_price(df)
        
        return df
    
    def _calculate_price(self, df: pd.DataFrame) -> pd.Series:
        """Calculate property price based on features"""
        
        # Base price per square foot
        base_price_per_sqft = 200
        
        # Calculate price components
        prices = df['square_feet'] * base_price_per_sqft
        
        # Apply property type multiplier
        type_multipliers = df['property_type'].map(self.property_types)
        prices *= type_multipliers
        
        # Apply location tier multiplier
        location_multipliers = df['location_tier'].map(self.location_tiers)
        prices *= location_multipliers
        
        # Bedroom premium
        bedroom_premium = 1 + (df['bedrooms'] - 2) * 0.08
        prices *= bedroom_premium
        
        # Bathroom premium
        bathroom_premium = 1 + (df['bathrooms'] - 1) * 0.05
        prices *= bathroom_premium
        
        # Age depreciation (newer = more expensive)
        age_factor = 1 - (df['age_years'] / 100) * 0.3
        prices *= age_factor
        
        # Distance penalty (further from center = cheaper)
        distance_factor = 1 - (df['distance_to_city_center'] / 50) * 0.25
        prices *= distance_factor
        
        # Amenities premium
        amenities_premium = (
            1 + 
            df['has_garden'] * 0.08 +
            df['has_pool'] * 0.12 +
            df['has_garage'] * 0.06 +
            df['furnished'] * 0.05
        )
        prices *= amenities_premium
        
        # Floor premium (higher floors = more expensive for apartments)
        floor_premium = np.where(
            df['property_type'].isin(['Apartment', 'Penthouse']),
            1 + df['floor'] * 0.015,
            1 + df['floor'] * 0.005
        )
        prices *= floor_premium
        
        # Crime rate penalty
        crime_factor = 1 - (df['crime_rate'] / 100) * 0.2
        prices *= crime_factor
        
        # School rating premium
        school_factor = 1 + (df['school_rating'] - 5) * 0.03
        prices *= school_factor
        
        # Hospital distance penalty
        hospital_factor = 1 - (df['hospital_distance'] / 30) * 0.1
        prices *= hospital_factor
        
        # Shopping distance bonus (closer = better)
        shopping_factor = 1 - (df['shopping_distance'] / 15) * 0.05
        prices *= shopping_factor
        
        # Add random noise
        noise = np.random.normal(1, 0.1, len(df))
        prices *= noise
        
        # Ensure positive prices
        prices = prices.clip(50000, 5000000)
        
        return prices.round(2)
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """Save generated data to CSV"""
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load data from CSV"""
        return pd.read_csv(filepath)


def generate_and_save_dataset(output_dir: str = "data", n_samples: int = 5000) -> Tuple[pd.DataFrame, str]:
    """Generate and save dataset"""
    generator = PropertyDataGenerator()
    df = generator.generate_data(n_samples)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "property_data.csv")
    generator.save_data(df, filepath)
    
    return df, filepath


if __name__ == "__main__":
    df, path = generate_and_save_dataset()
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"\nPrice Statistics:")
    print(df['price'].describe())
    print(f"\nFirst 5 rows:")
    print(df.head())
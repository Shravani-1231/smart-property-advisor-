"""Quick test script for the model"""
import sys
sys.path.append('src')

from model_trainer import PropertyPriceModel
import pandas as pd

# Load model
model = PropertyPriceModel()
model.load_model('models/property_price_model.pkl')

# Test prediction
test_data = pd.DataFrame([{
    'square_feet': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'age_years': 5,
    'distance_to_city_center': 3,
    'property_type': 'House',
    'location_tier': 'Tier 2',
    'has_garden': 1,
    'has_pool': 0,
    'has_garage': 1,
    'furnished': 0,
    'floor': 0,
    'crime_rate': 20,
    'school_rating': 8,
    'hospital_distance': 2,
    'shopping_distance': 1
}])

prediction = model.predict(test_data)[0]
print(f"Test Property: 2000 sqft House, 3 bed, 2 bath in Tier 2 location")
print(f"Predicted Price: ${prediction:,.2f}")
print("\nModel test successful!")
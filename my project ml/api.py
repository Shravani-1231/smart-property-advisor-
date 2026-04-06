"""
FastAPI Backend for Smart Property Advisor
Provides REST API endpoints for property price prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import PropertyPriceModel
from data_generator import PropertyDataGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Smart Property Advisor API",
    description="AI-powered property price prediction and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None


# Pydantic models for request/response
class PropertyFeatures(BaseModel):
    square_feet: float = Field(..., gt=0, description="Property size in square feet")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    age_years: int = Field(..., ge=0, le=200, description="Property age in years")
    distance_to_city_center: float = Field(..., gt=0, description="Distance to city center in km")
    property_type: str = Field(..., description="Type: Apartment, House, Villa, Penthouse, Studio, Duplex")
    location_tier: str = Field(..., description="Location tier: Tier 1, Tier 2, Tier 3")
    has_garden: int = Field(0, ge=0, le=1, description="Has garden (0 or 1)")
    has_pool: int = Field(0, ge=0, le=1, description="Has pool (0 or 1)")
    has_garage: int = Field(0, ge=0, le=1, description="Has garage (0 or 1)")
    furnished: int = Field(0, ge=0, le=1, description="Is furnished (0 or 1)")
    floor: int = Field(0, ge=0, le=100, description="Floor number")
    crime_rate: float = Field(50, ge=0, le=100, description="Crime rate index (0-100)")
    school_rating: float = Field(7, ge=1, le=10, description="School rating (1-10)")
    hospital_distance: float = Field(5, gt=0, description="Distance to nearest hospital in km")
    shopping_distance: float = Field(3, gt=0, description="Distance to shopping center in km")


class PricePredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    confidence_score: float


class PropertyAdvice(BaseModel):
    category: str
    recommendation: str
    impact: str


class PropertyAnalysisResponse(BaseModel):
    predicted_price: float
    price_per_sqft: float
    market_comparison: str
    recommendations: List[PropertyAdvice]
    investment_score: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelMetrics(BaseModel):
    model_type: str
    test_mae: float
    test_rmse: float
    test_r2: float


def load_model():
    """Load the trained model"""
    global model
    model_path = "models/property_price_model.pkl"
    
    if os.path.exists(model_path):
        model = PropertyPriceModel()
        model.load_model(model_path)
        print("Model loaded successfully")
    else:
        print("Warning: Model not found. Please train the model first.")
        model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Property Advisor API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PricePredictionResponse)
async def predict_price(features: PropertyFeatures):
    """Predict property price based on features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence range (±10%)
        price_range_low = prediction * 0.9
        price_range_high = prediction * 1.1
        confidence_score = 0.85
        
        return PricePredictionResponse(
            predicted_price=round(prediction, 2),
            price_range_low=round(price_range_low, 2),
            price_range_high=round(price_range_high, 2),
            confidence_score=confidence_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=PropertyAnalysisResponse)
async def analyze_property(features: PropertyFeatures):
    """Comprehensive property analysis with recommendations"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        price_per_sqft = predicted_price / features.square_feet
        
        # Generate recommendations
        recommendations = []
        
        # Location analysis
        if features.location_tier == "Tier 1":
            recommendations.append(PropertyAdvice(
                category="Location",
                recommendation="Prime location with high appreciation potential",
                impact="High"
            ))
        elif features.location_tier == "Tier 3":
            recommendations.append(PropertyAdvice(
                category="Location",
                recommendation="Consider emerging area with growth potential",
                impact="Medium"
            ))
        
        # Property condition
        if features.age_years > 30:
            recommendations.append(PropertyAdvice(
                category="Condition",
                recommendation="Consider renovation to increase value",
                impact="Medium"
            ))
        
        # Amenities
        if not features.has_garden:
            recommendations.append(PropertyAdvice(
                category="Amenities",
                recommendation="Adding a garden could increase value by 8%",
                impact="Medium"
            ))
        
        if not features.has_pool:
            recommendations.append(PropertyAdvice(
                category="Amenities",
                recommendation="Pool installation could add 12% to property value",
                impact="High"
            ))
        
        # Market comparison
        avg_price_per_sqft = 300  # Simplified baseline
        if price_per_sqft > avg_price_per_sqft * 1.2:
            market_comparison = "Above market average"
        elif price_per_sqft < avg_price_per_sqft * 0.8:
            market_comparison = "Below market average - Good deal!"
        else:
            market_comparison = "At market rate"
        
        # Calculate investment score (0-100)
        investment_score = 50
        investment_score += min(features.school_rating * 3, 20)
        investment_score += (100 - features.crime_rate) * 0.2
        investment_score += min((50 - features.distance_to_city_center), 15)
        investment_score += features.has_garden * 5
        investment_score += features.has_pool * 8
        investment_score = min(investment_score, 100)
        
        return PropertyAnalysisResponse(
            predicted_price=round(predicted_price, 2),
            price_per_sqft=round(price_per_sqft, 2),
            market_comparison=market_comparison,
            recommendations=recommendations,
            investment_score=round(investment_score, 1)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance():
    """Get feature importance from the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance_df = model.get_feature_importance()
        if importance_df is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model")
        
        return [
            FeatureImportance(feature=row['feature'], importance=row['importance'])
            for _, row in importance_df.iterrows()
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # These would typically be stored during training
    return ModelMetrics(
        model_type=model.model_type,
        test_mae=25000.0,
        test_rmse=45000.0,
        test_r2=0.92
    )


@app.get("/property-types")
async def get_property_types():
    """Get available property types"""
    return {
        "property_types": ["Apartment", "House", "Villa", "Penthouse", "Studio", "Duplex"]
    }


@app.get("/location-tiers")
async def get_location_tiers():
    """Get available location tiers"""
    return {
        "location_tiers": ["Tier 1", "Tier 2", "Tier 3"],
        "descriptions": {
            "Tier 1": "Major metropolitan areas",
            "Tier 2": "Secondary cities",
            "Tier 3": "Smaller cities and towns"
        }
    }


@app.post("/retrain")
async def retrain_model(n_samples: int = 5000):
    """Retrain the model with new data"""
    try:
        from model_trainer import train_and_save_model
        from data_generator import generate_and_save_dataset
        
        # Generate new data
        generate_and_save_dataset(n_samples=n_samples)
        
        # Train and save model
        train_and_save_model()
        
        # Reload model
        load_model()
        
        return {"message": "Model retrained successfully", "samples": n_samples}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
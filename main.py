from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

COLUMNS = ["latitude", "longitude", "price", "minimum_nights",
           "number_of_reviews", "reviews_per_month",
           "calculated_host_listings_count", "availability_365",
           "neighbourhood_group", "neighbourhood"]
model = joblib.load("Model_Pipeline.pkl")

class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night, must be positive")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
    neighbourhood_group: str = Field(..., min_length=1, description="Borough or neighbourhood group")
    neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")

# 1. Serve index.html directly on the root URL
@app.get('/')
def serve_home():
    return FileResponse("index.html")

# 2. Mount individual static asset files so index.html can load them
@app.get('/style.css')
def serve_css():
    return FileResponse("style.css")

@app.get('/script.js')
def serve_js():
    return FileResponse("script.js")

@app.post('/predict')
def predict(features: Features):
    # Fixed: Use model_dump() for Pydantic v2
    row = pd.DataFrame([features.model_dump()], columns=COLUMNS)
    prediction = model.predict(row)
    probability = model.predict_proba(row)

    return {
        'Predicted_room_type': str(prediction[0]),
        'probability': probability.tolist()[0]  # Fixed: .tolist() instead of .to_list()
    }

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict  # Ensure this import is correct
import sys
import os

app = FastAPI()

# Define the input data model for request validation
class InputData(BaseModel):
    features: list[float]

# Define the predict endpoint
@app.post("/predict")
async def predict_route(input_data: InputData):
    try:
        # Path to the trained model
        model_path = "/app/models/trained_model.pkl"
        # Reshape the input features into the correct format for the model (2D array)
        features_reshaped = [input_data.features]  # List of lists to represent one sample

        # Get predictions
        predictions = predict(features_reshaped, model_path)

        # Return the predictions as a JSON response
        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}

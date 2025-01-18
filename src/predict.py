import mlflow
import mlflow.sklearn
import pickle
import numpy as np
import argparse
import os

def predict(input_data, model_path):
    # Load the trained model
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = mlflow.sklearn.load_model(model_path)  # If using MLflow model logging
    
    # Ensure the input data is in the correct shape (1 row, N features)
    features = np.array(input_data).reshape(1, -1)
    
    # Make prediction using the trained model
    prediction = model.predict(features)
    
    # Return the prediction
    return prediction.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input data for prediction (as a list or numpy array)")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    args = parser.parse_args()
    
    # Convert the input data string to a list or numpy array
    try:
        input_data = eval(args.data)  # Convert string input to list/array
        if not isinstance(input_data, (list, np.ndarray)):
            raise ValueError("Input data must be a list or numpy array.")
    except Exception as e:
        raise ValueError(f"Invalid input data format. {str(e)}")
    
    # Predict using the trained model
    prediction = predict(input_data, args.model)
    
    print(f"Prediction: {prediction}")


import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import argparse
import os

def train_model(data_path, model_output_path):
    # Validate data file path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file at {data_path} not found.")
    
    data = pd.read_csv(data_path)
    
    if data.shape[1] < 2:
        raise ValueError("Data must have at least one feature and one target column.")
    
    # Split the data into features (X) and target (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model accuracy: {acc}")

        # Save the model locally using pickle
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        
        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed data")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()
    
    # Train the model with the given arguments
    train_model(args.data, args.output)

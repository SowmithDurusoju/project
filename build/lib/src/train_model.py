import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import argparse

def train_model(data_path, model_output_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model accuracy: {acc}")

        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed data")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()
    train_model(args.data, args.output)

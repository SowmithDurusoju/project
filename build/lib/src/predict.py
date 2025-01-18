import pickle
import pandas as pd

def predict(input_data, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    predictions = model.predict(pd.DataFrame(input_data))
    return predictions.tolist()

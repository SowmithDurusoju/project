import pandas as pd
import argparse

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    # Preprocessing steps
    data.to_csv(output_path, index=False)
    print(f"Data processed and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw data")
    parser.add_argument("--output", required=True, help="Path to save processed data")
    args = parser.parse_args()
    preprocess_data(args.input, args.output)

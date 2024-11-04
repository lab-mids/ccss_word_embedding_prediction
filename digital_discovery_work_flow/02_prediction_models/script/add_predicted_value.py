# add_predicted_value.py
import pandas as pd
import argparse


def add_predicted_value(input_file, source_column):
    df = pd.read_csv(input_file)
    df["Predicted_Value"] = df[source_column]
    df.to_csv(input_file, index=False)  # Overwriting the original file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a Predicted_Value column to CSV files."
    )
    parser.add_argument("--input_file", required=True, help="Input CSV file")
    parser.add_argument(
        "--source_column", required=True, help="Source column to copy data from"
    )
    args = parser.parse_args()

    add_predicted_value(args.input_file, args.source_column)

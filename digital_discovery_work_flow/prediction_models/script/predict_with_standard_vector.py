import argparse
import pandas as pd
import numpy as np
from matnexus import VecGenerator, VecVisualizer


class MaterialVectorAnalysis:
    def __init__(self, model_path, data_paths, results_path, property_list):
        self.model = VecGenerator.Word2VecModel.load(model_path)
        self.data_paths = data_paths
        self.results_path = results_path
        self.property_list = property_list
        self.calculator = VecGenerator.MaterialSimilarityCalculator(
            self.model, property_list=self.property_list
        )
        self.visualizer = VecVisualizer.Word2VecVisualizer(self.model)
        self.datasets = []
        self.results_df = None

    @staticmethod
    def convert_vector_column_to_ndarray(df, column_name):
        def convert_vector(vec_str):
            vec_str_cleaned = vec_str.replace("\n", "").replace("  ", " ")
            return np.fromstring(vec_str_cleaned[1:-1], sep=" ")

        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].apply(convert_vector)
        return df_copy

    @staticmethod
    def filter_rows_by_training_data(df, training_data_values):
        training_data_str_values = [
            str(value) if isinstance(value, list) else value
            for value in training_data_values
        ]
        filtered_df = df[df["Training Data"].isin(training_data_str_values)]
        return filtered_df

    def load_and_prepare_data(self):
        self.datasets = [pd.read_csv(path) for path in self.data_paths]

        self.datasets[2] = self.datasets[2][
            (abs(self.datasets[2]["Current_at_850mV"]) >= 0.05)
            & (abs(self.datasets[2]["Current_at_850mV"] <= 1))
        ].copy()

        self.results_df = pd.read_csv(self.results_path)
        self.results_df = self.convert_vector_column_to_ndarray(
            self.results_df, "Standard Vector"
        )

    def calculate_similarity(
        self,
        target_data_index,
        elements,
        target_vec_column="Standard Vector",
        training_data_values=[["data1", "data2"]],
    ):
        filtered_data = self.filter_rows_by_training_data(
            self.results_df, training_data_values
        )
        target_vec = filtered_data[target_vec_column].values[0]
        return self.calculator.calculate_similarity_from_dataframe(
            self.datasets[target_data_index],
            elements,
            target_vec=target_vec,
            experimental_indicator_column="Current_at_850mV",
            experimental_indicator_func=lambda x: x,
        )

    def plot_similarity(self, df, elements, cmap_groups):
        return self.visualizer.plot_similarity_scatter(
            df, elements=elements, cmap_groups=cmap_groups
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prediction using standard vector method."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the Word2Vec model."
    )
    parser.add_argument(
        "--data_paths", nargs="+", required=True, help="Paths to the dataset CSV files."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the optimization results CSV file.",
    )
    parser.add_argument(
        "--property_list",
        nargs="+",
        required=True,
        help="List of properties for similarity calculation.",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        required=True,
        help="List of elements for similarity calculation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV file to save similarity results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the MaterialVectorAnalysis class
    analysis = MaterialVectorAnalysis(
        model_path=args.model_path,
        data_paths=args.data_paths,
        results_path=args.results_path,
        property_list=args.property_list,
    )

    # Load and prepare data
    analysis.load_and_prepare_data()

    # Calculate similarity for the third dataset (index 2) using the specified elements
    df_similarity = analysis.calculate_similarity(
        target_data_index=2, elements=args.elements
    )

    # Save the results to the output file
    df_similarity.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()

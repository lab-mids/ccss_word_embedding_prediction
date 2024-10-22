import argparse
import pandas as pd
import numpy as np


class PredictionAnalyzer:
    def __init__(self, df_configs):
        self.df_configs = df_configs

    def analyze(self):
        all_metrics = {}
        for i, config in enumerate(self.df_configs):
            df_name = f"DataFrame {i + 1}"
            df = config['df']
            actual_col = config['actual_col']
            predicted_col = config['predicted_col']
            filter_col = config.get('filter_col')
            filter_value = config.get('filter_value')
            metrics_to_skip = config.get('metrics_to_skip', [])

            metrics = {}
            actual = df[actual_col]
            predicted = df[predicted_col]

            possible_metrics = {
                'Mean (Actual)': lambda actual, predicted: actual.mean(),
                'Mean (Predicted)': lambda actual, predicted: predicted.mean(),
                'Standard Deviation (Actual)': lambda actual, predicted: actual.std(),
                'Standard Deviation (Predicted)': lambda actual,
                                                         predicted: predicted.std(),
                'Minimum (Actual)': lambda actual, predicted: actual.min(),
                'Minimum (Predicted)': lambda actual, predicted: predicted.min(),
                'Mean Absolute Error (MAE)': lambda actual, predicted: np.mean(
                    np.abs(actual - predicted)),
                'Root Mean Square Error (RMSE)': lambda actual, predicted: np.sqrt(
                    np.mean((actual - predicted) ** 2)),
                'Overall coefficient of determination (R^2)': lambda actual,
                                                                     predicted: 1 - (
                            np.sum((actual - predicted) ** 2) / np.sum(
                        (actual - np.mean(actual)) ** 2)),
                'Overall Correlation (r)': lambda actual, predicted:
                np.corrcoef(actual, predicted)[0, 1],
            }

            if filter_col and filter_value is not None:
                filtered_actual = actual[df[filter_col] < filter_value]
                filtered_predicted = predicted[df[filter_col] < filter_value]
                if not filtered_actual.empty:
                    possible_metrics[
                        f'Correlation (r) for {filter_col} < {filter_value}'] = lambda \
                        actual, predicted: \
                    np.corrcoef(filtered_actual, filtered_predicted)[0, 1]

            for metric_name, metric_func in possible_metrics.items():
                if metric_name not in metrics_to_skip:
                    try:
                        metrics[metric_name] = metric_func(actual, predicted)
                    except:
                        metrics[metric_name] = None

                all_metrics[metric_name] = all_metrics.get(metric_name, {})
                all_metrics[metric_name][df_name] = metrics.get(metric_name)

        results_df = pd.DataFrame(all_metrics).T
        results_df.columns = [config.get('name', f"DataFrame {i + 1}") for i, config in
                              enumerate(self.df_configs)]
        return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze predictions and compute metrics.")
    parser.add_argument('--data_paths', nargs='+', required=True, help="Paths to the CSV files.")
    parser.add_argument('--names', nargs='+', required=True, help="Names for the datasets.")
    parser.add_argument('--actual_col', type=str, required=True, help="Name of the column for actual values.")
    parser.add_argument('--predicted_col', type=str, required=True, help="Name of the column for predicted values.")
    parser.add_argument('--filter_col', type=str, help="Column to apply the filter (optional).")
    parser.add_argument('--filter_value', type=float, help="Value for filtering data (optional).")
    parser.add_argument('--metrics_to_skip', nargs='+', help="List of metrics to skip (optional).")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save the results.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Handle optional arguments: filter and metrics_to_skip
    filter_col = args.filter_col if args.filter_col else None
    filter_value = args.filter_value if args.filter_value is not None else None
    metrics_to_skip = args.metrics_to_skip if args.metrics_to_skip else []

    # Load datasets
    dataframes = [pd.read_csv(path) for path in args.data_paths]

    # Prepare configuration for each dataframe
    df_configs = [
        {
            'df': dataframes[i],
            'name': args.names[i],
            'actual_col': args.actual_col,
            'predicted_col': args.predicted_col,
            'filter_col': filter_col,
            'filter_value': filter_value,
            'metrics_to_skip': metrics_to_skip
        }
        for i in range(len(dataframes))
    ]

    # Initialize and run the analyzer
    analyzer = PredictionAnalyzer(df_configs)
    results = analyzer.analyze()

    # Save the results to the output file
    results.to_csv(args.output_file, index=True)

if __name__ == "__main__":
    main()
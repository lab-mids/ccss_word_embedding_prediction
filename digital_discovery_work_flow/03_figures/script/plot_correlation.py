import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from typing import List, Optional
import seaborn as sns


class CorrelationPlotter:
    def __init__(
        self, dataframes: List[pd.DataFrame], names: Optional[List[str]] = None
    ):
        self.dataframes = dataframes
        self.names = (
            names
            if names is not None
            else [f"Dataframe {i+1}" for i in range(len(dataframes))]
        )

    def filter_data(self, filter_func):
        self.dataframes = [filter_func(df) for df in self.dataframes]

    def plot_correlation(
        self, x_col: str, y_col: str, combined: bool = True, show_fit_line: bool = False
    ):
        sns.set(style="whitegrid")
        if combined:
            fig, ax = plt.subplots(
                figsize=(10, 10)
            )  # Create a figure and a single axis
            colors = sns.color_palette("mako", len(self.dataframes))
            for df, name, color in zip(self.dataframes, self.names, colors):
                self._plot_single_df(
                    df, x_col, y_col, name, show_fit_line, ax=ax, color=color
                )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend(loc="best")
        else:
            fig, axs = plt.subplots(
                len(self.dataframes), 1, figsize=(10, len(self.dataframes) * 5)
            )
            if len(self.dataframes) == 1:
                axs = [axs]  # Make sure axs is always a list, even with one subplot
            for df, ax, name in zip(self.dataframes, axs, self.names):
                self._plot_single_df(df, x_col, y_col, name, show_fit_line, ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend()
            plt.tight_layout()
        return fig  # Return the figure object instead of showing it

    def _plot_single_df(self, df, x_col, y_col, name, show_fit_line, ax, color=None):
        ax.scatter(df[x_col], df[y_col], label=name, alpha=0.7, color=color)
        if show_fit_line:
            slope, intercept, r_value, _, _ = linregress(df[x_col], df[y_col])
            line = slope * df[x_col] + intercept
            ax.plot(
                df[x_col],
                line,
                color=color if color is not None else "r",
                label=f"Fit: r={r_value:.2f}",
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot correlation between two variables with optional filtering and fit lines."  # noqa
    )
    parser.add_argument(
        "--data_paths", nargs="+", required=True, help="Paths to the dataset CSV files."
    )
    parser.add_argument(
        "--names", nargs="+", required=True, help="Names for the datasets."
    )
    parser.add_argument(
        "--x_col", type=str, required=True, help="Name of the column for the X-axis."
    )
    parser.add_argument(
        "--y_col", type=str, required=True, help="Name of the column for the Y-axis."
    )
    parser.add_argument(
        "--combined",
        type=bool,
        default=True,
        help="Plot all datasets on a combined plot.",
    )
    parser.add_argument(
        "--show_fit_line",
        type=bool,
        default=False,
        help="Show the fit line on the plots.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Optional filter for the data (e.g., 'Current_at_850mV < -0.2').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for saving the plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    dataframes = [pd.read_csv(path) for path in args.data_paths]

    # Initialize the plotter
    plotter = CorrelationPlotter(dataframes, names=args.names)

    # Apply filter if provided
    if args.filter:
        plotter.filter_data(lambda df: df.query(args.filter))

    # Generate the plot
    fig = plotter.plot_correlation(
        x_col=args.x_col,
        y_col=args.y_col,
        combined=args.combined,
        show_fit_line=args.show_fit_line,
    )

    # Save the plot to the output file
    fig.savefig(args.output_file, bbox_inches="tight")


if __name__ == "__main__":
    main()

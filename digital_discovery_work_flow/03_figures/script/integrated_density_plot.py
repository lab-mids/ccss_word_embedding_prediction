import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


class IntegratedDensityCombinedXPlot:
    def __init__(
        self,
        filepaths,
        columns,
        plot_types,
        legend_labels=None,
        fontsize=14,
        subplot_width=7,
        subplot_height=5,
    ):
        self.filepaths = filepaths
        self.columns = columns
        self.plot_types = plot_types
        self.legend_labels = (
            legend_labels
            if legend_labels
            else [col for i, col in enumerate(columns) if plot_types[i] == "x_only"]
        )
        self.fontsize = fontsize
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height
        self.dataframes = [pd.read_csv(fp) for fp in self.filepaths]
        self.direction = None  # Will store the direction vector from the first subplot

    def plot_all(self):
        figsize = (2 * self.subplot_width, self.subplot_height)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        axes = np.atleast_1d(axes)
        subplot_labels = [f"({chr(97 + i)})" for i in range(2)]
        plot_index = 0
        combined_ax = axes[1]
        secondary_y = None
        color_map = plt.get_cmap("tab10", len(self.legend_labels))
        x_only_index = 0
        secondary_legends = []

        for i, (df, column, plot_type) in enumerate(
            zip(self.dataframes, self.columns, self.plot_types)
        ):
            x = df["x"].values
            y = df["y"].values
            current = df[column].values

            if i == 0 and plot_type == "xy":
                min_index = np.argmin(current)
                max_index = np.argmax(current)
                min_point = (x[min_index], y[min_index])
                max_point = (x[max_index], y[max_index])
                dx = max_point[0] - min_point[0]
                dy = max_point[1] - min_point[1]
                direction = np.array([dx, dy])
                self.direction = direction / np.linalg.norm(direction)

            norm_direction = self.direction
            x_range = (np.min(x), np.max(x))
            y_range = (np.min(y), np.max(y))
            start_point = min_point - norm_direction * (
                np.abs(min_point[0] - x_range[0]) + np.abs(min_point[1] - y_range[0])
            )
            end_point = max_point + norm_direction * (
                np.abs(max_point[0] - x_range[1]) + np.abs(max_point[1] - y_range[1])
            )
            num_points = 200
            line_x = np.linspace(start_point[0], end_point[0], num_points)
            line_y = np.linspace(start_point[1], end_point[1], num_points)
            line_current = griddata((x, y), current, (line_x, line_y), method="linear")

            if plot_type == "xy":
                ax = axes[0]
                sc = ax.scatter(line_x, line_y, c=line_current, cmap="plasma", s=50)
                ax.scatter(x, y, c="gray", alpha=0.3)
                ax.set_xlabel("x", fontsize=self.fontsize)
                ax.set_ylabel("y", fontsize=self.fontsize)
                ax.tick_params(axis="both", which="major", labelsize=self.fontsize - 2)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(
                    f"Interpolated {column} (mA/cm²)", fontsize=self.fontsize
                )
                cbar.ax.tick_params(labelsize=self.fontsize - 2)
                ax.text(
                    -0.15,
                    1.05,
                    subplot_labels[plot_index],
                    transform=ax.transAxes,
                    fontsize=self.fontsize + 2,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )
                plot_index += 1
            elif plot_type == "x_only":
                label = self.legend_labels[x_only_index]
                if column == "Similarity":
                    if secondary_y is None:
                        secondary_y = combined_ax.twinx()
                        secondary_y.set_ylabel("Similarity", fontsize=self.fontsize)
                    (similarity_line,) = secondary_y.plot(
                        line_x,
                        line_current,
                        label=label,
                        color=color_map(x_only_index),
                        lw=2,
                    )
                    secondary_y.tick_params(
                        axis="both", which="major", labelsize=self.fontsize - 2
                    )
                    secondary_legends.append(similarity_line)
                else:
                    combined_ax.plot(
                        line_x,
                        line_current,
                        label=label,
                        color=color_map(x_only_index),
                        lw=2,
                    )
                x_only_index += 1

        handles, labels = combined_ax.get_legend_handles_labels()
        combined_ax.set_xlabel("x", fontsize=self.fontsize)
        combined_ax.set_ylabel("Current_at_850mV (mA/cm²)", fontsize=self.fontsize)
        combined_ax.legend(
            handles + secondary_legends,
            labels + [legend.get_label() for legend in secondary_legends],
            loc="lower right",
            fontsize=self.fontsize - 2,
        )
        combined_ax.tick_params(axis="both", which="major", labelsize=self.fontsize - 2)
        combined_ax.text(
            -0.1,
            1.05,
            subplot_labels[plot_index],
            transform=combined_ax.transAxes,
            fontsize=self.fontsize + 2,
            fontweight="bold",
            va="center",
            ha="right",
        )
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated Density Combined X Plot with multiple files and plot types."  # noqa
    )
    parser.add_argument(
        "--filepaths", nargs="+", required=True, help="List of file paths to CSV files."
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="List of column names for each file.",
    )
    parser.add_argument(
        "--plot_types",
        nargs="+",
        required=True,
        help="List of plot types ('xy' or 'x_only') for each file.",
    )
    parser.add_argument(
        "--legend_labels",
        nargs="*",
        help="List of custom legend labels for x_only plots.",
    )
    parser.add_argument(
        "--fontsize", type=int, default=14, help="Font size for text elements."
    )
    parser.add_argument(
        "--subplot_width",
        type=float,
        default=7,
        help="Width of each subplot in inches.",
    )
    parser.add_argument(
        "--subplot_height",
        type=float,
        default=5,
        help="Height of each subplot in inches.",
    )
    parser.add_argument(
        "--output_file", required=True, help="Output file name for the plot."
    )
    args = parser.parse_args()

    plotter = IntegratedDensityCombinedXPlot(
        filepaths=args.filepaths,
        columns=args.columns,
        plot_types=args.plot_types,
        legend_labels=args.legend_labels,
        fontsize=args.fontsize,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
    )
    fig = plotter.plot_all()
    fig.savefig(args.output_file, bbox_inches="tight")

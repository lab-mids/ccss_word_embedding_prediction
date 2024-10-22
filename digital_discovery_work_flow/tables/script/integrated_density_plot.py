import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import argparse


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
        """
        Initializes the class with multiple files and respective plot types.

        :param filepaths: List of file paths to CSV files.
        :param columns: List of column names for each file to be used for color-coding.
        :param plot_types: List of plot types ('xy' or 'x_only') corresponding to each file. # noqa
        :param legend_labels: List of custom legend labels for x_only plots (optional).
        :param fontsize: Font size for all text elements (labels, ticks, legend).
        :param subplot_width: Width of each subplot in inches.
        :param subplot_height: Height of each subplot in inches.
        """
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
        # Create figure with space for one xy plot and one subplot for combined x plots
        figsize = (2 * self.subplot_width, self.subplot_height)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Ensure axes is a 1D list even if there's only one plot
        axes = np.atleast_1d(axes)

        subplot_labels = [
            f"({chr(97 + i)})" for i in range(2)
        ]  # Only two plots (xy and combined x)
        plot_index = 0

        combined_ax = axes[1]  # Combined x plot at the second position
        secondary_y = None
        color_map = plt.get_cmap(
            "tab10", len(self.legend_labels)
        )  # Use different colors for combined x plots

        x_only_index = 0  # To keep track of legend labels for x_only plots
        secondary_legends = (
            []
        )  # To collect legend entries for secondary y-axis (Similarity)

        for i, (df, column, plot_type) in enumerate(
            zip(self.dataframes, self.columns, self.plot_types)
        ):
            x = df["x"].values
            y = df["y"].values
            current = df[column].values

            # If this is the first xy plot,
            # calculate the direction based on max and min points
            if i == 0 and plot_type == "xy":
                min_index = np.argmin(current)
                max_index = np.argmax(current)
                min_point = (x[min_index], y[min_index])
                max_point = (x[max_index], y[max_index])

                # Compute the direction vector from min to max point
                dx = max_point[0] - min_point[0]
                dy = max_point[1] - min_point[1]
                direction = np.array([dx, dy])
                self.direction = direction / np.linalg.norm(
                    direction
                )  # Store the normalized direction vector

            # Use the stored direction for all subplots
            norm_direction = self.direction

            # Determine the full range of x and y for the line
            x_range = (np.min(x), np.max(x))
            y_range = (np.min(y), np.max(y))

            # Calculate start and end points based on the direction
            start_point = min_point - norm_direction * (
                np.abs(min_point[0] - x_range[0]) + np.abs(min_point[1] - y_range[0])
            )
            end_point = max_point + norm_direction * (
                np.abs(max_point[0] - x_range[1]) + np.abs(max_point[1] - y_range[1])
            )

            # Generate the extended line coordinates along the max-min direction
            num_points = 200
            line_x = np.linspace(start_point[0], end_point[0], num_points)
            line_y = np.linspace(start_point[1], end_point[1], num_points)
            line_current = griddata((x, y), current, (line_x, line_y), method="linear")

            if plot_type == "xy":
                # Plot x and y with the color-coded column
                ax = axes[0]
                sc = ax.scatter(line_x, line_y, c=line_current, cmap="plasma", s=50)
                ax.scatter(
                    x, y, c="gray", alpha=0.3
                )  # Plot original data points in gray
                ax.set_xlabel("x", fontsize=self.fontsize)
                ax.set_ylabel("y", fontsize=self.fontsize)

                # Set tick parameters
                ax.tick_params(axis="both", which="major", labelsize=self.fontsize - 2)

                # Colorbar with a customized label
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(
                    f"Interpolated {column} (mA/cm²)", fontsize=self.fontsize
                )
                cbar.ax.tick_params(labelsize=self.fontsize - 2)

                # Add subplot labels outside the plot area
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
                # Combine all x_only plots in the same subplot
                # with different line colors
                label = self.legend_labels[x_only_index]
                if column == "Similarity":
                    # Add a secondary y-axis for 'Similarity' values
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

                x_only_index += 1  # Move to the next legend label for x_only plots

        # Collect primary axis legends
        handles, labels = combined_ax.get_legend_handles_labels()

        # Set up the combined x plot at the bottom
        combined_ax.set_xlabel("x", fontsize=self.fontsize)
        combined_ax.set_ylabel("Current_at_850mV (mA/cm²)", fontsize=self.fontsize)

        # Show legend for both primary and secondary y-axes
        combined_ax.legend(
            handles + secondary_legends,
            labels + [legend.get_label() for legend in secondary_legends],
            loc="lower right",
            fontsize=self.fontsize - 2,
        )

        combined_ax.tick_params(axis="both", which="major", labelsize=self.fontsize - 2)

        # Add label to the combined x plot
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Integrated Density Combined X Plots"
    )
    parser.add_argument(
        "--filepaths", nargs="+", required=True, help="List of file paths to CSV files"
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="List of columns to be used for color-coding",
    )
    parser.add_argument(
        "--plot_types",
        nargs="+",
        required=True,
        help="List of plot types for each file (xy or x_only)",
    )
    parser.add_argument(
        "--legend_labels",
        nargs="+",
        help="Custom legend labels for x_only plots (optional)",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=14,
        help="Font size for all text elements (default: 14)",
    )
    parser.add_argument(
        "--subplot_width",
        type=int,
        default=7,
        help="Width of each subplot in inches (default: 7)",
    )
    parser.add_argument(
        "--subplot_height",
        type=int,
        default=5,
        help="Height of each subplot in inches (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the plot (PDF format)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create plotter object
    plotter = IntegratedDensityCombinedXPlot(
        filepaths=args.filepaths,
        columns=args.columns,
        plot_types=args.plot_types,
        legend_labels=args.legend_labels,
        fontsize=args.fontsize,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
    )

    # Generate plot and save to PDF
    fig = plotter.plot_all()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()

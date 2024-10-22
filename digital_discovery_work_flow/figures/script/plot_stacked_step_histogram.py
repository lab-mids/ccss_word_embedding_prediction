import argparse
import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_step_histogram(dataframes, system_names, bins=10,
                                column_name='Current_at_850mV'):
    fig, ax = plt.subplots(figsize=(10, 8))
    assert len(system_names) == len(
        dataframes), "Each dataframe needs a corresponding system name."

    markers = ['o', 's', '^', 'p', '*', 'H', 'D', 'X', '<', '>']  # Define marker styles

    for data_index, (df, system_name) in enumerate(zip(dataframes, system_names)):
        if column_name in df.columns:
            line_color = plt.cm.tab10((data_index + 5) % 10)
            n, bins, patches = plt.hist(df[column_name], bins=bins, histtype='step',
                                        linewidth=2,
                                        label=system_name, color=line_color)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            plt.plot(bin_centers, n, linestyle='',
                     marker=markers[data_index % len(markers)],
                     color=line_color)  # Add markers

    plt.xlabel(column_name, fontsize=15)  # Set x-axis label size
    plt.ylabel('Count', fontsize=15)  # Set y-axis label size
    plt.xticks(fontsize=12)  # Set x-tick label size
    plt.yticks(fontsize=12)  # Set y-tick label size
    plt.legend(fontsize=12)  # Set legend text size
    plt.grid(False)

    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stacked step histogram for current density")
    parser.add_argument("--data_paths", nargs='+', type=str, required=True,
                        help="Paths to the CSV files containing data")
    parser.add_argument("--system_names", nargs='+', type=str, required=True,
                        help="Names of the systems corresponding to the data files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the plot PDF")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins for the histogram")
    parser.add_argument("--column_name", type=str, default='Current_at_850mV',
                        help="Column name for the data to be plotted")
    return parser.parse_args()


def main():
    args = parse_args()

    dataframes = [pd.read_csv(path) for path in args.data_paths]
    fig = plot_stacked_step_histogram(dataframes=dataframes,
                                      system_names=args.system_names,
                                      bins=args.bins,
                                      column_name=args.column_name)
    fig.savefig(args.output_path, bbox_inches='tight')


if __name__ == "__main__":
    main()
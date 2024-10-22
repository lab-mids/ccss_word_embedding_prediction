import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


class CurrentDensityPlotter:
    def __init__(self, dataframes, system_names):
        self.dataframes = dataframes
        self.system_names = system_names
        self.marker_styles = ['o', 's', '^', 'p', '*', 'H', 'D', 'X', '<', '>']

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 4))
        system_offset = 4

        current_position = 0
        legend_handles = []

        for data_index, data in enumerate(self.dataframes):
            if 'Current_at_850mV' not in data.columns:
                continue

            q25 = data['Current_at_850mV'].quantile(0.25)
            q75 = data['Current_at_850mV'].quantile(0.75)
            min_val = data['Current_at_850mV'].min()
            max_val = data['Current_at_850mV'].max()

            line_color = plt.cm.tab10((data_index + 5) % 10)
            marker_style = self.marker_styles[data_index % len(self.marker_styles)]

            plt.hlines(current_position, min_val, max_val, color=line_color, lw=2)
            plt.plot([min_val, q25, q75, max_val], [current_position] * 4,
                     marker_style, color=line_color, markersize=6)

            plt.text(min_val, current_position - 1.2, f'Min: {min_val:.2f}',
                     verticalalignment='bottom', horizontalalignment='center',
                     color=line_color, fontsize=12)
            plt.text(q25, current_position + 0.1, f'25%: {q25:.2f}',
                     verticalalignment='bottom', horizontalalignment='center',
                     color=line_color, fontsize=12)
            plt.text(q75, current_position + 0.1, f'75%: {q75:.2f}',
                     verticalalignment='bottom', horizontalalignment='center',
                     color=line_color, fontsize=12)
            plt.text(max_val, current_position - 1.2, f'Max: {max_val:.2f}',
                     verticalalignment='bottom', horizontalalignment='center',
                     color=line_color, fontsize=12)

            line_handle = mlines.Line2D([], [], color=line_color,
                                        marker=marker_style, markersize=6,
                                        label=self.system_names[data_index], lw=2)
            legend_handles.append(line_handle)

            current_position += system_offset

        plt.ylim([-4, current_position + 4])
        plt.xlabel('Current_at_850mV (mA)', fontsize=15)
        plt.tick_params(axis='x', labelsize=12)
        plt.yticks([])
        plt.gca().xaxis.grid(False)
        plt.gca().yaxis.grid(False)
        plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1),
                   fontsize='large')
        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot current density at 850mV for different systems")
    parser.add_argument("--data_paths", nargs='+', type=str, required=True,
                        help="Paths to the CSV files containing data")
    parser.add_argument("--system_names", nargs='+', type=str, required=True,
                        help="Names of the systems corresponding to the data files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the plot PDF")
    return parser.parse_args()


def main():
    args = parse_args()

    dataframes = [pd.read_csv(path) for path in args.data_paths]
    plotter = CurrentDensityPlotter(dataframes=dataframes,
                                    system_names=args.system_names)
    fig = plotter.plot()
    fig.savefig(args.output_path, bbox_inches='tight')


if __name__ == "__main__":
    main()
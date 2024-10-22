from matnexus import VecVisualizer
import argparse
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot similarity scatter for dataset elements"
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the CSV files containing data",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        required=True,
        help="Elements to plot in the format 'data_label:element1,element2,...'",
    )
    parser.add_argument("--x_labels", nargs="+", required=True, help="X-axis labels")
    parser.add_argument("--y_labels", nargs="+", required=True, help="Y-axis labels")
    parser.add_argument(
        "--legend_labels", type=str, help="Legend labels as a dictionary"
    )
    parser.add_argument("--cmap_groups", type=str, help="Color maps as a dictionary")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for the PDF file"
    )
    parser.add_argument(
        "--ncols", type=int, default=3, help="Number of columns in the subplot grid"
    )
    parser.add_argument(
        "--nrows", type=int, default=1, help="Number of rows in the subplot grid"
    )
    parser.add_argument(
        "--subplot_labels",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to include subplot labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the datasets
    data_dict = {
        f"data{i+1}": pd.read_csv(path) for i, path in enumerate(args.data_paths)
    }

    # Parse the elements argument
    elements = [
        (elem.split(":")[0], elem.split(":")[1].split(",")) for elem in args.elements
    ]

    # Prepare labels
    x_labels = {"x": args.x_labels}
    y_labels = {"y": args.y_labels}

    # Parse optional legend labels and color maps
    legend_labels = eval(args.legend_labels) if args.legend_labels else None
    cmap_groups = eval(args.cmap_groups) if args.cmap_groups else None

    # Initialize visualizer
    visualizer = VecVisualizer.Word2VecVisualizer(None)

    # Generate the plot
    fig = visualizer.plot_similarity_scatter(
        data_dict=data_dict,
        elements=elements,
        x_labels=x_labels,
        y_labels=y_labels,
        legend_labels=legend_labels,
        cmap_groups=cmap_groups,
        ncols=args.ncols,
        nrows=args.nrows,
        subplot_labels=args.subplot_labels,
    )

    # Save the plot
    fig.savefig(args.output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()

import argparse
from matnexus import VecGenerator, VecVisualizer
import plotly.io as py

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize material vectors based on Word2Vec model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Word2Vec model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the generated plot PDF")
    parser.add_argument("--material_list", nargs='+', required=True,
                        help="List of materials to visualize")
    parser.add_argument("--property_list", nargs='+', required=True,
                        help="List of properties to include in the visualization")
    parser.add_argument("--marker_size", type=int, default=25,
                        help="Size of the markers in the plot")
    parser.add_argument("--axisfont_size", type=int, default=25,
                        help="Font size for the axis labels")
    parser.add_argument("--tickfont_size", type=int, default=20,
                        help="Font size for the tick labels")
    parser.add_argument("--width", type=int, default=2800, help="Width of the plot")
    parser.add_argument("--height", type=int, default=1400, help="Height of the plot")
    parser.add_argument("--show_legend", type=str2bool, nargs='?', const=True,
                        default=False, help="Whether to show legend in the plot")
    parser.add_argument("--textfont_size", type=int, default=25,
                        help="Font size for the text in the plot")

    # parser.add_argument("--random_state", type=int, default=42,
    #                     help="Random state variable for t-SNE")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load the Word2Vec model
    model = VecGenerator.Word2VecModel.load(args.model_path)

    # Create Word2VecVisualizer instance
    visualizer = VecVisualizer.Word2VecVisualizer(model)

    # Visualize material vectors
    fig_materials = visualizer.plot_vectors(
        material_list=args.material_list,
        property_list=args.property_list,
        marker_size=args.marker_size,
        axisfont_size=args.axisfont_size,
        tickfont_size=args.tickfont_size,
        width=args.width,
        height=args.height,
        show_legend=args.show_legend,
        textfont_size=args.textfont_size,
        # random_state=args.random_state
    )

    # Save the visualization to a PDF file
    fig_materials.write_image(args.output_path, engine="kaleido")


if __name__ == "__main__":
    main()

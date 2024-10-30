import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class ElementConcentrationPlotter:
    periodic_table = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Fl": 114,
        "Mc": 115,
        "Lv": 116,
        "Ts": 117,
        "Og": 118,
    }

    def __init__(self, dataframes):
        self.dataframes = dataframes
        self.element_color_map = {}
        self.element_marker_map = {}
        self.marker_styles = ["s", "p", "X", "<", ">"]

    @staticmethod
    def adjust_positions(start, count, offset):
        return [start + i * offset for i in range(count)]

    def get_sorted_elements(self, elements):
        # Sort elements alphabetically instead of by periodic table order
        return sorted(elements)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        element_offset = 0.4
        system_offset = 0.6

        current_position = 0
        marker_counter = 0
        # Reverse the order of dataframes to plot data1 at the top
        for data_index, data in enumerate(reversed(self.dataframes)):
            elements_in_data = [
                col for col in data.columns if col in self.periodic_table
            ]
            sorted_elements = self.get_sorted_elements(elements_in_data)
            positions = self.adjust_positions(
                current_position, len(sorted_elements), element_offset
            )

            system_label = "-".join(sorted_elements)

            for i, element in enumerate(sorted_elements):
                position = positions[i]
                if element not in self.element_color_map:
                    self.element_color_map[element] = plt.cm.tab10(
                        len(self.element_color_map) % 10
                    )
                    self.element_marker_map[element] = self.marker_styles[
                        marker_counter % len(self.marker_styles)
                    ]
                    marker_counter += 1

                line_color = self.element_color_map[element]
                marker_shape = self.element_marker_map[element]
                min_val = int(data[element].min() * 100)
                max_val = int(data[element].max() * 100)

                plt.hlines(position, min_val, max_val, color=line_color, lw=2)
                plt.plot(
                    min_val,
                    position,
                    marker_shape,
                    markersize=10,
                    color=line_color,
                    clip_on=False,
                )
                plt.plot(
                    max_val,
                    position,
                    marker_shape,
                    markersize=10,
                    color=line_color,
                    clip_on=False,
                )

                # Adjust text to prevent crossing axis
                offset = 1.5  # Adjust this value as needed to avoid crossing the axes
                plt.text(
                    min_val - offset,
                    position,
                    f"{min_val}",
                    verticalalignment="center",
                    horizontalalignment="right",
                    color=line_color,
                    fontsize=15,
                )
                plt.text(
                    max_val + offset,
                    position,
                    f"{max_val}",
                    verticalalignment="center",
                    horizontalalignment="left",
                    color=line_color,
                    fontsize=15,
                )

            plt.text(
                -10,
                sum(positions) / len(positions),
                system_label,
                verticalalignment="center",
                horizontalalignment="right",
                fontsize=15,
                color="black",
                weight="bold",
            )

            if data_index < len(self.dataframes) - 1:
                current_position += (
                    len(sorted_elements) * element_offset + system_offset
                )

        plt.xlabel("Concentration (%)", fontsize=15)
        plt.xlim(0, 100)
        plt.tick_params(axis="x", labelsize=12)
        plt.yticks([])
        plt.gca().xaxis.grid(False)
        plt.gca().yaxis.grid(False)

        # Creating custom legend handles with adjusted text size
        custom_lines = [
            mlines.Line2D(
                [0],
                [0],
                color=color,
                marker=self.element_marker_map[elem],
                markersize=8,
                linestyle="-",
                lw=3,
            )
            for elem, color in self.element_color_map.items()
        ]
        plt.legend(
            custom_lines,
            self.element_color_map.keys(),
            loc="lower right",
            fontsize="large",
        )  # Adjusted legend text size

        return fig

    import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create horizontal range plot for element concentration"
    )
    parser.add_argument(
        "--data1", type=str, required=True, help="Path to the first dataset CSV file"
    )
    parser.add_argument(
        "--data2", type=str, required=True, help="Path to the second dataset CSV file"
    )
    parser.add_argument(
        "--data3", type=str, required=True, help="Path to the third dataset CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for the plot PDF file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data1 = pd.read_csv(args.data1)
    data2 = pd.read_csv(args.data2)
    data3 = pd.read_csv(args.data3)

    plotter = ElementConcentrationPlotter([data1, data2, data3])
    fig = plotter.plot()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()

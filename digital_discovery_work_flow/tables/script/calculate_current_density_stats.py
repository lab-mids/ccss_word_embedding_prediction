import argparse
import pandas as pd


class CurrentDensityStatsCalculator:
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

    def __init__(self, target_column):
        self.target_column = target_column

    def load_and_process_data(self, filepaths):
        all_results = []
        for filepath in filepaths:
            df = pd.read_csv(filepath)
            if self.target_column not in df.columns:
                raise ValueError(
                    f"The target column '{self.target_column}' does not exist in the dataframe."  # noqa
                )
            system_name = "-".join(
                sorted([col for col in df.columns if col in self.periodic_table])
            )
            result = self.calculate_stats_and_correlation(df[self.target_column], df)
            result_df = pd.DataFrame([result])
            result_df.insert(0, "System", system_name)
            all_results.append(result_df)

        return pd.concat(all_results, ignore_index=True)

    def calculate_stats_and_correlation(self, target_data, df):
        stats = {
            "Mean Current (mA)": target_data.mean(),
            "Standard Deviation (mA)": target_data.std(),
            "Minimum Current (mA)": target_data.min(),
            "25% Quantile (mA)": target_data.quantile(0.25),
            "Median (mA)": target_data.median(),
            "75% Quantile (mA)": target_data.quantile(0.75),
            "Maximum Current (mA)": target_data.max(),
        }

        correlations = {
            f"Correlation with {element}": target_data.corr(df[element])
            for element in df.columns
            if element in self.periodic_table
        }

        return {**stats, **correlations}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate current density statistics and element correlations."
    )
    parser.add_argument(
        "--filepaths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the CSV files containing the data.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="The column representing the current density to analyze.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the resulting statistics CSV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the calculator with the target column
    cd_stats_calculator = CurrentDensityStatsCalculator(
        target_column=args.target_column
    )

    # Process the data
    results_df = cd_stats_calculator.load_and_process_data(args.filepaths)

    # Save the results to a CSV file
    results_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()

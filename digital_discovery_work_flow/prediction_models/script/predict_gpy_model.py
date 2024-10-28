import pandas as pd
import numpy as np
import argparse
import GPy


class MaterialAnalysis:
    def __init__(self, target_column):
        self.elements_list = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]
        self.target_column = target_column

    def detect_material_columns(self, df):
        return [col for col in df.columns if col in self.elements_list]

    def prepare_data(self, data):
        material_columns = self.detect_material_columns(data)
        X = data[material_columns].values
        y = data[self.target_column].values[:, None]
        return X, y

    def standardize_columns(self, data, element_columns):
        for col in element_columns:
            if col not in data.columns:
                data[col] = 0
        return data[element_columns + [col for col in data.columns if
                                       col not in element_columns]]

    def load_and_prepare_datasets(self, filenames):
        datasets = [pd.read_csv(filename) for filename in filenames]
        combined_data = pd.concat(datasets)
        all_element_columns = self.detect_material_columns(combined_data)
        standardized_datasets = []

        for data in datasets:
            standardized_data = self.standardize_columns(data, all_element_columns).copy()
            standardized_datasets.append(standardized_data)

        return standardized_datasets

    def train_and_test_model(self, train_datasets, test_dataset):
        X_train_list = []
        Y_train_list = []

        for data in train_datasets:
            X, Y = self.prepare_data(data)
            X_train_list.append(X)
            Y_train_list.append(Y)

        X_train = np.vstack(X_train_list)
        Y_train = np.vstack(Y_train_list)

        X_test, Y_test = self.prepare_data(test_dataset)

        kernel = GPy.kern.RBF(input_dim=X_train.shape[1])
        gp_model = GPy.models.GPRegression(X_train, Y_train, kernel)
        gp_model.optimize(messages=False)

        Y_pred, _ = gp_model.predict(X_test)
        test_dataset['Predicted_Current_at_850mV'] = Y_pred.flatten()

        return test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Material Analysis Tool for Training and Testing GP Models.")
    parser.add_argument("--filenames", nargs='+', required=True, help="List of input CSV files for training.")
    parser.add_argument("--target_column", default="Current_at_850mV", help="Target column for predictions.")
    parser.add_argument("--output_file", required=True, help="Output CSV file with predictions.")
    args = parser.parse_args()

    # Initialize MaterialAnalysis with specified target column
    analysis = MaterialAnalysis(target_column=args.target_column)

    # Load and prepare datasets
    datasets = analysis.load_and_prepare_datasets(args.filenames)

    # Train on the first N-1 datasets and test on the last dataset
    trained_data_with_predictions = analysis.train_and_test_model(datasets[:-1], datasets[-1])

    # Save the output file with predictions
    trained_data_with_predictions.to_csv(args.output_file, index=False)
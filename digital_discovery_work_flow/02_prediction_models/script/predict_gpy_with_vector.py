import argparse
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import GPy


class MaterialVectorAnalysis:
    def __init__(self, model_path, target_column):
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
        self.model = Word2Vec.load(model_path)
        self.target_column = target_column

    def detect_material_columns(self, df):
        return [col for col in df.columns if col in self.elements_list]

    def compute_material_vector(self, row):
        material_columns = self.detect_material_columns(pd.DataFrame([row]))
        multiplied_vectors = [
            self.model.wv[element.lower()] * row[element]
            for element in material_columns
            if element in row
        ]
        material_vec = np.mean(multiplied_vectors, axis=0)
        return material_vec

    def prepare_data(self, data):
        data["Material_Vector"] = data.apply(self.compute_material_vector, axis=1)
        return (
            np.array(data["Material_Vector"].tolist()),
            data[self.target_column].values[:, None],
        )

    def load_datasets(self, filenames):
        return [pd.read_csv(filename) for filename in filenames]

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

        test_dataset["Predicted_" + self.target_column] = Y_pred.flatten()

        return test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Material Vector Analysis for Training and Testing GP Models with Word2Vec embeddings."  # noqa
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the Word2Vec model file."
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        required=True,
        help="List of input CSV files for training and testing.",
    )
    parser.add_argument(
        "--target_column", required=True, help="Target column for predictions."
    )
    parser.add_argument(
        "--output_file", required=True, help="Output CSV file with predictions."
    )
    args = parser.parse_args()

    # Initialize the analysis with the Word2Vec model path and target column
    analysis = MaterialVectorAnalysis(
        model_path=args.model_path, target_column=args.target_column
    )

    # Load and prepare datasets
    datasets = analysis.load_datasets(args.filenames)

    # Train on the first N-1 datasets and test on the last dataset
    trained_data_with_predictions = analysis.train_and_test_model(
        datasets[:-1], datasets[-1]
    )

    # Save the output file with predictions
    trained_data_with_predictions.to_csv(args.output_file, index=False)

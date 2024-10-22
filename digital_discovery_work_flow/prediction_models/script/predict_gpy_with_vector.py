import argparse
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import GPy


class MaterialVectorAnalysis:
    def __init__(self, model_path, target_column):
        self.elements_list = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
            "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
            "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
            "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
            "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
            "Mc", "Lv", "Ts", "Og"
        ]
        self.model = Word2Vec.load(model_path)
        self.target_column = target_column

    def detect_material_columns(self, df):
        return [col for col in df.columns if col in self.elements_list]

    def compute_material_vector(self, row):
        material_columns = self.detect_material_columns(pd.DataFrame([row]))
        multiplied_vectors = [self.model.wv[element.lower()] * row[element] for element
                              in material_columns if element in row]
        material_vec = np.mean(multiplied_vectors, axis=0)
        return material_vec

    def prepare_data(self, data):
        data['Material_Vector'] = data.apply(self.compute_material_vector, axis=1)
        return np.array(data['Material_Vector'].tolist()), data[self.target_column].values[:, None]

    def load_datasets(self, filenames):
        datasets = [pd.read_csv(filename) for filename in filenames]
        return datasets

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

        test_dataset['Predicted_' + self.target_column] = Y_pred.flatten()

        return test_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPy model with material vectors for current density predictions.")
    parser.add_argument("--train_files", nargs='+', required=True, help="Paths to the training dataset CSV files.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Word2Vec model.")
    parser.add_argument("--target_column", type=str, required=True, help="Target column to predict (e.g., 'Current_at_850mV').")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file to save predictions.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize the MaterialVectorAnalysis class with the Word2Vec model and target column
    analysis = MaterialVectorAnalysis(model_path=args.model_path, target_column=args.target_column)

    # Load the datasets
    train_datasets = analysis.load_datasets(args.train_files)
    test_dataset = pd.read_csv(args.test_file)

    # Train the model and make predictions
    result_dataset = analysis.train_and_test_model(train_datasets, test_dataset)

    # Save the predictions to the output file
    result_dataset.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
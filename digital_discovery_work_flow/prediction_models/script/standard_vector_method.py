import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

class MaterialAnalysis:
    def __init__(self, model_path, data_paths, property_list):
        self.model = Word2Vec.load(model_path)
        self.datasets = [pd.read_csv(path) for path in data_paths]
        self.elements_list = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
            "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
            "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
            "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
            "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
            "Fl", "Mc", "Lv", "Ts", "Og"
        ]
        self.property_list = property_list
        self.results = []

    def detect_material_columns(self, df):
        return [col for col in df.columns if col in self.elements_list]

    def compute_material_vector(self, row):
        material_columns = self.detect_material_columns(row.to_frame().T)
        elements_counts = [(element, row[element]) for element in material_columns]
        multiplied_vectors = [self.model.wv[element.lower()] * count for element, count in elements_counts]
        return np.mean(multiplied_vectors, axis=0)

    def preprocess_data(self):
        all_columns = set.union(*[set(dataset.columns) for dataset in self.datasets])

        for dataset in self.datasets:
            missing_columns = all_columns - set(dataset.columns)
            for column in missing_columns:
                dataset[column] = 0.0

        for i, dataset in enumerate(self.datasets):
            dataset['Material_Vector'] = dataset.apply(lambda row: self.compute_material_vector(row), axis=1)
            self.datasets[i] = dataset

    def similarity_difference(self, weights, similarities, property_vectors, standard_vec):
        weighted_standard_vec = np.dot(weights, property_vectors)
        weighted_standard_vec = weighted_standard_vec.reshape(1, -1)
        similarities_to_weighted_standard = cosine_similarity(standard_vec, weighted_standard_vec).flatten()
        return np.sum((similarities - similarities_to_weighted_standard) ** 2)

    def optimize_weights(self):
        if len(self.datasets) < 3:
            print("Not enough datasets provided.")
            return

        training_datasets = [self.datasets[0], self.datasets[1]]
        data_combined = pd.concat(training_datasets, ignore_index=True)

        test_dataset = self.datasets[2]

        property_vectors = [np.mean([self.model.wv[word] for word in prop.split()], axis=0) for prop in self.property_list]

        material_vectors_combined = np.array(data_combined['Material_Vector'].tolist())
        experimental_indicator = data_combined['Current_at_850mV'].values

        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        initial_weights = np.full(len(self.property_list), 1 / len(self.property_list))

        optimization_result = minimize(
            self.similarity_difference,
            initial_weights,
            args=(experimental_indicator, property_vectors, material_vectors_combined),
            constraints=constraints
        )

        optimal_weights = optimization_result.x
        optimal_standard_vec = np.dot(optimal_weights, property_vectors)

        self.results = {
            "Training Data": ["data1", "data2"],
            "Test Data": "data3",
            "Optimization Fun": optimization_result.fun,
            "Standard Vector": optimal_standard_vec
        }

    def run_analysis(self, output_file):
        self.preprocess_data()
        self.optimize_weights()
        if isinstance(self.results, dict):
            results_list = [self.results]
        else:
            results_list = self.results
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_file, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize weights of properties and obtain a standard vector.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Word2Vec model.")
    parser.add_argument("--data_paths", nargs='+', required=True, help="Paths to the dataset CSV files.")
    parser.add_argument("--property_list", nargs='+', required=True, help="List of properties for optimization.")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file to save optimization results.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize the MaterialAnalysis class with model path, data paths, and property list
    analysis = MaterialAnalysis(model_path=args.model_path, data_paths=args.data_paths, property_list=args.property_list)

    # Run the analysis and save results to the output file
    analysis.run_analysis(output_file=args.output_file)

if __name__ == "__main__":
    main()
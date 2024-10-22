
# Composition-property Extrapolation for Compositionally Complex Solid Solutions Based on Word Embeddings

This repository contains the code and workflow for the paper **"Composition-property extrapolation for compositionally complex solid solutions based on word embeddings: efficient materials discovery"**, submitted to *Digital Discovery*. This workflow guides users through the process of collecting papers, building predictive models, and generating figures and tables used in the paper.

## Repository Structure

```plaintext
digital_discovery_workflow/
├── figures/
│   ├── script/
│   ├── __init__.py
│   ├── config.yaml
│   └── Snakefile
├── prediction_models/
│   ├── script/
│   ├── __init__.py
│   ├── config.yaml
│   └── Snakefile
├── tables/
│   ├── script/
│   ├── __init__.py
│   ├── config.yaml
│   └── Snakefile
├── word2vec_model/
│   ├── script/
│   ├── __init__.py
│   ├── config.yaml
│   └── Snakefile
```

### 1. **Word2Vec Model**

The first step in this workflow is to collect research papers related to electrocatalysts and build a Word2Vec model from these papers. The Word2Vec model will be trained using the scripts located in `word2vec_model`. This model forms the basis for predicting material properties in the later steps.

#### To run:
```bash
cd word2vec_model
snakemake
```

### 2. **Data Source**

The dataset used for model training and validation is available in [Zanendo repository](https://doi.org/TODO). You can download the data from this source to reproduce the results.

### 3. **Prediction Models**

Once the Word2Vec model and the dataset are ready, the next step is to run the prediction models located in the `prediction_models` directory. These scripts will generate the models used in the paper to extrapolate the composition-property relationships for the materials.

#### To run:
```bash
cd prediction_models
snakemake
```

### 4. **Figures**

The figures used in the paper can be reproduced using the scripts in the `figures` directory. This will generate the exact same figures as those shown in the paper.

#### To run:
```bash
cd figures
snakemake
```

### 5. **Tables**

The tables presented in the paper are generated using the scripts in the `tables` directory.

#### To run:
```bash
cd tables
snakemake
```

## Requirements

To reproduce the results, ensure you have the following installed:

- Python 3.8+
- [Snakemake](https://snakemake.readthedocs.io/)
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lab-mids/ccss_word_embedding_prediction.git
   cd digital_discovery_workflow
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Authors

- **Lei Zhang** (Corresponding Author)  
  Interdisciplinary Centre for Advanced Materials Simulation, Ruhr University Bochum  
  E-mail: [lei.zhang-w2i@rub.de](mailto:lei.zhang-w2i@rub.de)  
- **Lars Banko**  
  Chair for Materials Discovery and Interfaces, Institute for Materials, Ruhr University Bochum  
  E-mail: [lars.banko@rub.de](mailto:lars.banko@rub.de)  
- **Wolfgang Schuhmann**  
  Analytical Chemistry -- Center for Electrochemical Sciences (CES), Faculty of Chemistry and Biochemistry, Ruhr University Bochum  
  E-mail: [wolfgang.schuhmann@rub.de](mailto:wolfgang.schuhmann@rub.de)  
- **Alfred Ludwig**  
  Chair for Materials Discovery and Interfaces, Institute for Materials, Ruhr University Bochum  
  E-mail: [alfred.ludwig@rub.de](mailto:alfred.ludwig@rub.de)  
- **Markus Stricker**  
  Interdisciplinary Centre for Advanced Materials Simulation, Ruhr University Bochum  
  E-mail: [markus.stricker@rub.de](mailto:markus.stricker@rub.de)  


## How to Cite

Please cite this repository and the related paper when using this code for your research:

```
@article{zhang2024composition_property,
  author = {Zhang, Lei},
  title = {Composition-property extrapolation for compositionally complex solid solutions based on word embeddings: efficient materials discovery},
  journal = {Digital Discovery},
  year = {2024},
  doi = {TODO}
}
```

## License

This project is licensed under the LGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
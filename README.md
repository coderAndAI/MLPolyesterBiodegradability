# Explainable Random Forest Predictions of Polyester Biodegradability Using High-Throughput Biodegradation Data

This repository contains data, trained models, and code to replicate results for the paper **'Explainable Random Forest Predictions of Polyester Biodegradability Using High-Throughput Biodegradation Data'** by P. L. Jacob, M. I. Parker, D. J. Keddie, V. Taresco, S. M. Howdle and J. D. Hirst.

## Project Structure

```
MLPolyesterBiodegradability/
├── data/                          # Datasets
│   ├── inhouse_data.csv           # In-house polyester trimer data
│   ├── inhouse_trimer_ml_dataset.csv
│   ├── high_throughput_ml_dataset.csv
│   ├── full_pH_data.csv
│   └── pnas.2220021120.sd0*.csv   # Fransen et al. raw data
├── featurisation/                 # Molecular feature engineering
│   └── feature_transformers.py    # Custom sklearn transformers (fingerprints, descriptors)
├── training/                      # Model training scripts
│   ├── inhouse.py                 # Train RF with RDKit fingerprints
│   ├── inhouse_descriptors.py     # Train RF with molecular descriptors
│   └── benchmarking.py            # Compare RF vs NN with statistical tests
├── transfer_learning/             # Transfer learning pipeline
│   ├── transfer_learning.py       # Deep transfer learning (Keras)
│   ├── chain_step1.py             # Step 1: RF on source domain
│   ├── chain_step2.py             # Step 2: RF with transferred features
│   └── domain_similarity.py       # OTDD domain distance calculation
├── analysis/                      # Model interpretation and analysis
│   ├── shap_analysis.py           # SHAP explainability (fingerprints)
│   ├── shap_descriptors.py        # SHAP explainability (descriptors)
│   ├── feature_mapping.py         # Map fingerprint bits to substructures
│   ├── reliability_domain.py      # Model reliability assessment
│   ├── pca.py                     # PCA visualization
│   ├── tsne.py                    # t-SNE visualization
│   ├── ph_time_threshold.py       # pH threshold analysis
│   └── prepare_*.py               # Data preparation scripts
├── models/                        # Trained model files (.joblib)
├── requirements.txt               # Python dependencies
└── LICENSE                        # GNU AGPL v3
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the in-house fingerprint model:
```bash
cd training
python inhouse.py
```

Run benchmarking comparison:
```bash
cd training
python benchmarking.py
```

Run transfer learning pipeline:
```bash
cd transfer_learning
python chain_step1.py
python chain_step2.py
```

Generate SHAP analysis:
```bash
cd analysis
python shap_analysis.py
```

## Transfer Learning Data Source

The transfer learning models were pretrained on data from:

Fransen et al. (2023). *High-throughput experimentation for discovery of biodegradable polyesters*. PNAS. https://doi.org/10.1073/pnas.2220021120

The data is licensed under CC BY-NC-ND 4.0. This repository does not include the original data.

## License

GNU Affero General Public License v3 (AGPL-3.0)

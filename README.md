## Table of Contents
1. [Data](#data)
    - [Augmented](#augmented)
    - [Original](#original)
2. [Notebooks - Dissertation](#notebooks-dissertation)
    - [Augmentation](#augmentation)
    - [Cmatrix](#cmatrix)
    - [Models](#models)
3. [Notebooks - Publication](#notebooks-publication)
    - [Augmentation](#augmentation-1)
    - [Cmatrix](#cmatrix-1)
    - [Models](#models-1)
4. [Source Code](#source-code)
5. [Dependencies](#dependencies)

## Data
### Augmented
Contains augmented datasets.
- `gen-cc-10x-v4.csv`: Augmented data..
- `test-gen-cc-10x-v4.csv`:  Test data for the augmented dataset

### Original
Contains original datasets.
- `cancer and immune cells data - HC05-HC07.xlsx`: Original data of Raman Spectroscopy.
- `HC05_HC07.csv`: Original data in CSV format.

## Notebooks - Dissertation
The `notebooks-dissertation directory` contains Jupyter notebooks used for the dissertation
### Augmentation
Contains notebooks related to data augmentation.
- `data-augmentation-v2.ipynb`: Notebook to generate augmented dataset.

### Cmatrix
Contains notebooks related to confusion matrices.
- `cnn-vit-cmatrix.ipynb`: Notebook for CNN and Vision Transformer confusion matrix.

### Models
Contains notebooks related to different models.
#### BLS
- `ori-bls.ipynb`: Original BLS model notebook.

#### Classical
- `ori-ml.ipynb`: Original machine learning model notebook.

#### CNN
- `ori-cnn-1500-epochs-v2 (2).ipynb`: Original CNN model with 1500 epochs.
- `ori-cnn-500-epochs-v2.ipynb`: Original CNN model with 500 epochs.
- `ori-cnn-800-epochs-v2.ipynb`: Original CNN model with 800 epochs.

#### VIT
- `ori-vit-250-epochs.ipynb`: Original Vision Transformer model with 250 epochs.
- `ori-vit-500-epochs.ipynb`: Original Vision Transformer model with 500 epochs.

## Notebooks - Publication
The `notebooks-publication` directory contains Jupyter notebooks used for publication purposes
### Augmentation
Contains notebooks related to data augmentation for publication.
- `Data-aug.ipynb`: Data augmentation notebook.

### Cmatrix
Contains notebooks related to confusion matrices.
- `1d-ori-aug-bls-cmatrix.ipynb`: Confusion matrix for 1D original and augmented BLS models.
- `1d-ori-aug-cnn-cmatrix.ipynb`: Confusion matrix for 1D original and augmented CNN models.
- `1d-ori-aug-vit-cmatrix.ipynb`: Confusion matrix for 1D original and augmented Vision Transformer models.
- `logo-1d-ori-aug-cnn-cmatrix.xpynb`: Confusion matrix for Leave One Out Group 1D original and augmented CNN models.

### Models
Contains notebooks related to different models for publication.
#### BLS
- `1d-aug-bls-filtered.ipynb`: 1D augmented BLS model with filtering.
- `1d-aug-bls-non-filtered.ipynb`: 1D augmented BLS model without filtering.
- `1d-ori-bls-filtered.ipynb`: 1D original BLS model with filtering.
- `1d-ori-bls-non-filtered.ipynb`: 1D original BLS model without filtering.

#### Classical
- `1D-Aug-ML.ipynb`: 1D augmented machine learning model.
- `1D-Ori-ML.ipynb`: 1D original machine learning model.

#### CNN
- `1d-aug-cnn-filtered.ipynb`: 1D augmented CNN model with filtering.
- `1d-aug-cnn-not-filtered.ipynb`: 1D augmented CNN model without filtering.
- `1D-cnn-ori-nonfiltered.ipynb`: 1D original CNN model without filtering.
- `1D-ori-cnn-filtered.ipynb`: 1D original CNN model with filtering.

#### VIT
- `1D-Aug-VIT-filtered-v3.ipynb`: 1D augmented Vision Transformer model with filtering, version 3.
- `1D-Aug-VIT-nonfiltered-v3.ipynb`: 1D augmented Vision Transformer model without filtering, version 3.
- `1D-Ori-VIT-filtered.ipynb`: 1D original Vision Transformer model with filtering.
- `1D-Ori-VIT-nonfiltered.ipynb`: 1D original Vision Transformer model without filtering.

## Source Code
Contains source code for various scripts and modules.
- `airPLS.py`: AirPLS algorithm implementation.
- `data.py`: Data handling utilities.
- `deep_learning_models.py`: Deep learning model implementations.
- `initial.ipynb`: Initial setup notebook.
- `initial.py`: Initial setup script.
- `machine_learning_models.py`: Machine learning model implementations.
- `main.py`: Main script for running the project.
- `package-list.txt`: List of required packages.
- `preprocess.py`: Data preprocessing utilities.
- `utils_dl_model.py`: Utilities for deep learning models.
- `utils_ml_model.py`: Utilities for machine learning models.
- `utils_visualize_data.py`: Data visualization utilities.
- `vision_transformer copy.py`: Copy of Vision Transformer implementation.
- `vision_transformer.py`: Vision Transformer implementation.

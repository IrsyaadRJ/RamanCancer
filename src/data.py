import pandas as pd
import numpy as np
import random
from airPLS import airPLS
import scipy as scipy
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from preprocess import preprocess


# Set the seed value
seed_value = 7
np.random.seed(seed_value)
random.seed(seed_value)

# Load the dataset
def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

# Preprocess the data
def preprocess_data(data):
    # Preprocess the data
    # Normalized the features
    X_normed = preprocess(data.T)
    X_normed = X_normed.T

    # Extract the target variable
    y = np.array(data['Cell type'])

    # Encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Return the preprocessed data  (Normalized features, encoded labels)
    return X_normed, y_encoded

# Extract the X and y data from the dataset.
def extract_data(data):
    # Extract the features from the data
    X = np.array(data.iloc[:, 4:], dtype = float)

    # Extract the target variable
    y = np.array(data['Cell type'])

    # Encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Return X and y
    return X, y_encoded

# Split the data into train and test sets
def split_data(X, y, test_size=0.33, random_state=7):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Standardize the training and test data
def standardize_data(X_train, X_test):
    # Calculate the mean and standard deviation of the training data
    X_train_standardized = scipy.stats.zscore(X_train, axis=0)
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    # Standardize the test data using the mean and standard deviation of the training data
    X_test_standardized = (X_test - mean_train) / std_train
    return X_train_standardized, X_test_standardized

# Reshape the data 
def reshape_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test

# Apply PCA to extract features from the data
def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    # Use the same PCA model fitted on the training data
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Apply median filter and background subtraction to the data
def apply_filters_and_background_subtraction(X):
    # Reshape the input data to a 2D array if necessary
    if len(X.shape) == 1:
        X = X.reshape((1, -1))
    
    # Apply median filter
    datamedfilt = scipy.ndimage.median_filter(X, size=(1, 5))
    
    # Apply airPLS for background subtraction
    baseline = np.zeros_like(datamedfilt)
    cols = baseline.shape[1]
    for col in range(cols):
        baseline[:, col] = airPLS(datamedfilt[:, col], lambda_=300)
    
    data_bksb = datamedfilt - baseline
    return data_bksb

# Preprocess the raw data
def preprocess_raw_data(data_file, test_set):
    # Load the data
    print("Load the data")
    data = load_data(data_file)
    print(f"Data shape : {data.shape}")

    # Extract the feature and target data
    print("Extract the feature and target data")
    X, y = extract_data(data)
    print(f"X shaped: {X.shape}")
    print(f"y shaped: {y.shape}")

    # Apply filters and background substraction to the features dataset
    print("Apply filters and background substraction to the features dataset")
    X_filtered = apply_filters_and_background_subtraction(X)

    # Split the data into train and test sets
    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, random_state = seed_value, test_size = test_set, stratify = y)

    # Standardize the training and test sets
    print("Standardizing the data...")
    X_train_standardized, X_test_standardized = standardize_data(X_train, X_test)
    
    return X_train_standardized, X_test_standardized, y_train, y_test


def preprocess_cv_raw_data(data_file):
    # Load the data
    print("Load the data")
    data = load_data(data_file)
    print(f"Data shape : {data.shape}")

    # Extract the feature and target data
    print("Extract the feature and target data")
    X, y = extract_data(data)
    print(f"X shaped: {X.shape}")
    print(f"y shaped: {y.shape}")

    # Apply filters and background substraction to the features dataset
    print("Apply filters and background substraction to the features dataset")
    X_filtered = apply_filters_and_background_subtraction(X)

    return X_filtered, y

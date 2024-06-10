import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle


def get_hyperparameters(model):
    class_name_to_model_name = {
        'SVC': 'SVM',
        'AdaBoostClassifier': 'AdaBoost',
        'XGBClassifier': 'XGBoost',
        'GradientBoostingClassifier': 'Gradient Boost',
        'RandomForestClassifier': 'Random Forest',
        'DecisionTreeClassifier': 'Decision Tree',
        'LogisticRegression': 'Logistic Regression',
        'KNeighborsClassifier': 'k-Nearest Neighbors'
    }
    
    hyperparameters = {
        'SVM': ['C', 'kernel', 'degree', 'gamma'],
        'AdaBoost': ['n_estimators', 'learning_rate'],
        'XGBoost': ['n_estimators', 'learning_rate', 'max_depth'],
        'Gradient Boost': ['n_estimators', 'learning_rate', 'max_depth'],
        'Random Forest': ['n_estimators', 'max_depth', 'min_samples_split'],
        'Decision Tree': ['criterion', 'max_depth', 'min_samples_split'],
        'Logistic Regression': ['C', 'penalty', 'solver'],
        'k-Nearest Neighbors': ['n_neighbors', 'weights', 'algorithm']
    }
    
    model_name = class_name_to_model_name[type(model).__name__]
    return {key: getattr(model, key) for key in hyperparameters[model_name]}


def create_filename(name, model):
    hyperparameters = get_hyperparameters(model)
    hyperparameters_str = '_'.join([f'{k}-{v}' for k, v in hyperparameters.items()])
    return f'{name}_{hyperparameters_str}_results.pickle'

def save_ml_results(results, models, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for name, result in results.items():
        model = models[name]
        file_name = create_filename(name, model)
        file_path = os.path.join(folder_path, file_name)
        print(f'Saving {name} model results to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

def load_ml_results(folder_path):
    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pickle'):
            model_name = file_name.split('_')[0]
            file_path = os.path.join(folder_path, file_name)
            print(f'Loading {model_name} model results from {file_path}')
            with open(file_path, 'rb') as f:
                results[model_name] = pickle.load(f)
                
    return results
            

# Print the results for each model
def print_ml_results(ml_results):
    print("Machine Learning Model Results:")
    print("-" * 40)
    for name, result in ml_results.items():
        print(f"{name}: ")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")
    print("-" * 40)

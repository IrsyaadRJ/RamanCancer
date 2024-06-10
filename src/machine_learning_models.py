import numpy as np
import os 
import pickle
import json
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from utils_ml_model import create_filename


# Define the machine learning models
def build_ml_models():
    ml_models = {
        'SVM': SVC(),
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': XGBClassifier(),
        'Gradient Boost': GradientBoostingClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'k-Nearest Neighbors': KNeighborsClassifier()
    }
    return ml_models

# Train and evaluate the machine learning models
def train_and_evaluate_ml_models(models, X_train, y_train, X_test, y_test, cv=5, models_folder='trained_models', results_folder='saved_results'):
    results = {}

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for name, model in models.items():
        model_file_name = create_filename(name, model)
        model_file_path = os.path.join(models_folder, model_file_name)
        results_file_name = model_file_name.replace('.pickle', '_results.json')
        results_file_path = os.path.join(results_folder, results_file_name)

        # Load the model and results if they exist
        if os.path.exists(model_file_path) and os.path.exists(results_file_path):
            with open(model_file_path, 'rb') as f:
                print(f"Loading {name} model...")
                model = pickle.load(f)
            with open(results_file_path, 'rb') as f:
                print(f"Loading {name} results...")
                results[name] = json.load(f)
        else:
            # Train the model, evaluate it, and save the model and results
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            print(f"Evaluating {name} model...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
            cv_accuracy = np.mean(cv_scores)
            cv_accuracy_std = np.std(cv_scores)
            results[name] = {
                'CV': cv,
                'Accuracy': accuracy,
                'Cross-Validation Accuracy': cv_accuracy,
                'Cross-Validation Accuracy STD': cv_accuracy_std}

            with open(model_file_path, 'wb') as f:
                print(f"Saving {name} model...")
                pickle.dump(model, f)
            with open(results_file_path, 'w+') as f:
                print(f"Saving {name} results...")
                json.dump(results[name], f)

    return results
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
import random
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping , LearningRateScheduler
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from data import standardize_data, reshape_data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from vision_transformer import *

# Set the random seed for reproducibility
seed_value = 7
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# CNN Model 1 is a basic CNN architecture with two convolutional layers, followed by max-pooling layers.
# The model uses a dropout layer to prevent overfitting.
# The model servers as a baseline model for the other models.
def cnn_model_1(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3,
              activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 2
def cnn_model_2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 2 version 2 (Decreases the number of filter)
def cnn_model_2_2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=12, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=24, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 3 
def cnn_model_3(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 3 that will be optimized 
def cnn_model_3_opt(input_shape, num_classes, learning_rate, dropout_rate, l2_regularizer):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_regularizer), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_regularizer), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_regularizer), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_regularizer)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_regularizer)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 3 version 2 (increases the regularizer rate)
def cnn_model_3_2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# CNN model 4
def cnn_model_4(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# A function to perform cross-validation with the given hyperparameters
def perform_cross_validation(X, y, learning_rate, dropout_rate, l2_regularizer, 
                             n_splits=5, epochs=100, batch_size=32, cnn_version = 1, val_set = 0.2,
                             do_opt = False,models_folder = "models", results_folder = "results"):
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Initialize lists to store precision, recall, and F1 scores
    precisions = []
    recalls = []
    f1_scores = []
    
    # Initialize lists to store losses and accuracies for each epoch
    mean_training_losses = []
    mean_training_accuracies = []
    mean_val_losses = []   
    mean_val_accuracies = []
    
    fold = 1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)
    
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and validation sets
        print("=" * 40)
        print(f"Fold: {fold}")
        print("Splitting the data")
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]
        
        # Standardize the data
        print("Standardizing the data")
        X_train_full, X_test = standardize_data(X_train_full, X_test)
        
        # Reshape the data
        print("Reshaping the data for deep learning models")
        X_train_full, X_test = reshape_data(X_train_full, X_test)
        
        # Input shape and number of classes for the model
        input_shape = (X_train_full.shape[1], 1)
        num_classes = len(np.unique(y))
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = seed_value, stratify = y_train_full)
        
        try:
            if do_opt:
                model = cnn_model_3_opt(input_shape, num_classes, learning_rate, dropout_rate, l2_regularizer)
            else:
                model = cnn_model_3(input_shape, num_classes)
            if model is not None:
                print("Building the model 3 with the following hyperparameters:")
                print(f"Learning rate: {learning_rate}")
                print(f"Dropout rate: {dropout_rate}") 
                print(f"L2 Regularizer: {l2_regularizer}")
                print(f"Number of epochs: {epochs}")
                print(f"Batch size: {batch_size}")
                # model.summary()
            else:
                print("Failed to build the CNN model.")
        except Exception as e:
            print(f"Error while building the CNN model: {e}")
        
        
        print("Training the model")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
            # Calculate precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Append the scores to the lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # Append the losses and accuracies for each epoch to the respective lists
        for epoch in range(epochs):
            if fold == 1:
                # For the first fold, initialize the lists with the first fold's values
                mean_training_losses.append(history.history['loss'][epoch])
                mean_training_accuracies.append(history.history['accuracy'][epoch])
                mean_val_losses.append(history.history['val_loss'][epoch])
                mean_val_accuracies.append(history.history['val_accuracy'][epoch])
            else:
                # For subsequent folds, add the fold's values to the running sum
                mean_training_losses[epoch] += history.history['loss'][epoch]
                mean_training_accuracies[epoch] += history.history['accuracy'][epoch]
                mean_val_losses[epoch] += history.history['val_loss'][epoch]
                mean_val_accuracies[epoch] += history.history['val_accuracy'][epoch]

        # Save the model
        if do_opt:
            model_filename = f"cnn_v{cnn_version}_opt_{batch_size}_{epochs}_seed{seed_value}.h5"
        else:
            model_filename = f"cnn_v{cnn_version}_{batch_size}_{epochs}_seed{seed_value}.h5"
        model_filepath = os.path.join(models_folder, f"fold_{fold}_split_{n_splits}_{model_filename}")
        model.save(model_filepath)

        # Save the results as a JSON file
        if do_opt:
            results_filename = f"cnn_v{cnn_version}_opt_{batch_size}_{epochs}_seed{seed_value}.json"
        else:
            results_filename = f"cnn_v{cnn_version}_{batch_size}_{epochs}_seed{seed_value}.json"
        results_filepath = os.path.join(results_folder, f"fold_{fold}_split_{n_splits}_{results_filename}")

        results = {
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_scores": f1_scores,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "training_loss": history.history['loss'],
            "training_accuracy": history.history['accuracy'],
            "validation_loss": history.history['val_loss'],
            "validation_accuracy": history.history['val_accuracy'],
        }

        with open(results_filepath, "w") as results_file:
            json.dump(results, results_file, indent=4)
        
        # Append the results to the lists
        accuracies.append(accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Fold {fold}, Accuracy: {accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("=" * 40)
        fold += 1

    # At the end, divide the sums by the number of folds to get the mean
    mean_training_losses = [loss / n_splits for loss in mean_training_losses]
    mean_training_accuracies = [acc / n_splits for acc in mean_training_accuracies]
    mean_val_losses = [loss / n_splits for loss in mean_val_losses]
    mean_val_accuracies = [acc / n_splits for acc in mean_val_accuracies]
    # Convert lists to numpy arrays for easier calculation of mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    mean_test_loss = np.mean(test_losses)
    mean_test_accuracy = np.mean(test_accuracies)

    # Calculate standard deviations
    std_accuracy = np.std(accuracies)
    std_test_loss = np.std(test_losses)
    std_test_accuracy = np.std(test_accuracies)

    std_training_loss = np.std(mean_training_losses)
    std_training_accuracy = np.std(mean_training_accuracies)
    std_val_loss = np.std(mean_val_losses)
    std_val_accuracy = np.std(mean_val_accuracies)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

      
    print(f"test losses: {test_losses}")
    print(f"test accuracy: {test_accuracies}")
    
    print("*" * 40)
    print(f"Mean training loss over stratified {n_splits}-fold cross-validation: {np.mean(mean_training_losses):.4f}")
    print(f"Mean training accuracy over stratified {n_splits}-fold cross-validation: {np.mean(mean_training_accuracies):.4f}")
    print(f"Mean validation loss over stratified {n_splits}-fold cross-validation: {np.mean(mean_val_losses):.4f}")
    print(f"Mean validation accuracy over stratified {n_splits}-fold cross-validation: {np.mean(mean_val_accuracies):.4f}")
    print("*" * 40)
    print(f"Mean accuracy over stratified {n_splits}-fold cross-validation: {mean_accuracy:.4f}")
    print(f"Mean test loss over stratified {n_splits}-fold cross-validation: {mean_test_loss:.4f}")
    print(f"Mean test accuracy over stratified {n_splits}-fold cross-validation: {mean_test_accuracy:.4f}")
    print("*" * 40)

    print(f"Standard deviation of training loss over stratified {n_splits}-fold cross-validation: {np.std(mean_training_losses):.4f}")
    print(f"Standard deviation of training accuracy over stratified {n_splits}-fold cross-validation: {np.std(mean_training_accuracies):.4f}")
    print(f"Standard deviation of validation loss over stratified {n_splits}-fold cross-validation: {np.std(mean_val_losses):.4f}")
    print(f"Standard deviation of validation accuracy over stratified {n_splits}-fold cross-validation: {np.std(mean_val_accuracies):.4f}")
    print("*" * 40)
    print(f"Standard deviation of accuracy over stratified {n_splits}-fold cross-validation: {std_accuracy:.4f}")
    print(f"Standard deviation of test loss over stratified {n_splits}-fold cross-validation: {std_test_loss:.4f}")
    print(f"Standard deviation of test accuracy over stratified {n_splits}-fold cross-validation: {std_test_accuracy:.4f}")
    print("*" * 40)
    
    # Calculate and print the mean and standard deviation of precision, recall, and F1 score
    print(f"Mean precision over stratified {n_splits}-fold cross-validation: {np.mean(precisions):.4f}")
    print(f"Standard deviation of precision over stratified {n_splits}-fold cross-validation: {np.std(precisions):.4f}")
    print(f"Mean recall over stratified {n_splits}-fold cross-validation: {np.mean(recalls):.4f}")
    print(f"Standard deviation of recall over stratified {n_splits}-fold cross-validation: {np.std(recalls):.4f}")
    print(f"Mean F1 score over stratified {n_splits}-fold cross-validation: {np.mean(f1_scores):.4f}")
    print(f"Standard deviation of F1 score over stratified {n_splits}-fold cross-validation: {np.std(f1_scores):.4f}")

    
# Define the directory where the JSON files are stored

    fig, axes = plt.subplots(2, n_splits, figsize=(20, 10))

    for fold in range(1, n_splits+1):
        # Load the JSON file for this fold
        if do_opt:
            cnn_filename = f"cnn_v{cnn_version}_opt_{batch_size}_{epochs}_seed{seed_value}.json"
        else:
            cnn_filename = f"cnn_v{cnn_version}_{batch_size}_{epochs}_seed{seed_value}.json"
        results_filename = f"fold_{fold}_split_{n_splits}_{cnn_filename}"
        
        results_filepath = os.path.join(results_folder, results_filename)
        
        with open(results_filepath, "r") as results_file:
            results = json.load(results_file)
        
        # Plot training and validation loss
        axes[0, fold-1].plot(results["training_loss"], label='Train')
        axes[0, fold-1].plot(results["validation_loss"], label='Validation')
        axes[0, fold-1].set_title(f'Fold {fold} Loss')
        axes[0, fold-1].set_xlabel('Epochs')
        axes[0, fold-1].set_ylabel('Loss')
        axes[0, fold-1].legend()

        # Plot training and validation accuracy
        axes[1, fold-1].plot(results["training_accuracy"], label='Train')
        axes[1, fold-1].plot(results["validation_accuracy"], label='Validation')
        axes[1, fold-1].set_title(f'Fold {fold} Accuracy')
        axes[1, fold-1].set_xlabel('Epochs')
        axes[1, fold-1].set_ylabel('Accuracy')
        axes[1, fold-1].legend()

    plt.tight_layout()
    plt.show()




    
    # Plot training and validation loss per fold
 
    # n_splits = 5  # number of folds
    # fig, axs = plt.subplots(n_splits, 2, figsize=(10, n_splits))  # Adjust size here

    # for i in range(n_splits):
    #     # Plot training and validation loss per fold
    #     axs[i, 0].plot(training_losses[i], label='Training Loss')
    #     axs[i, 0].plot(val_losses[i], label='Validation Loss')
    #     axs[i, 0].set_title(f'Fold {i+1} Loss')
    #     axs[i, 0].set_xlabel('Epochs')
    #     axs[i, 0].set_ylabel('Loss')
    #     axs[i, 0].legend()

    #     # Plot training and validation accuracy per fold
    #     axs[i, 1].plot(training_accuracies[i], label='Training Accuracy')
    #     axs[i, 1].plot(val_accuracies[i], label='Validation Accuracy')
    #     axs[i, 1].set_title(f'Fold {i+1} Accuracy')
    #     axs[i, 1].set_xlabel('Epochs')
    #     axs[i, 1].set_ylabel('Accuracy')
    #     axs[i, 1].legend()

    # plt.tight_layout()
    # plt.show()
    

    # # Create a figure and axes
    # fig, axs = plt.subplots(1, 4, figsize=(16, 6))

    # # Find the global min and max for loss
    # global_loss_min = np.min([np.min(training_losses), np.min(val_losses)])
    # global_loss_max = np.max([np.max(training_losses), np.max(val_losses)])

    # # Find the global min and max for accuracy
    # global_acc_min = np.min([np.min(training_accuracies), np.min(val_accuracies)])
    # global_acc_max = np.max([np.max(training_accuracies), np.max(val_accuracies)])

    # # Create box plots for training and validation losses and accuracies
    # boxplot_training_losses  = axs[0].boxplot(training_losses, meanline=True)
    # axs[0].set_title('Training Losses')

    # boxplot_val_losses = axs[1].boxplot(val_losses, meanline=True )
    # axs[1].set_title('Validation Losses')

    # boxplot_training_accuracies = axs[2].boxplot(training_accuracies, meanline=True)
    # axs[2].set_title('Training Accuracies')

    # boxplot_val_accuracies = axs[3].boxplot(val_accuracies, meanline=True)
    # axs[3].set_title('Validation Accuracies')

    # # Set the y-axis limits for loss subplots with margin
    # axs[0].set_ylim([global_loss_min - 0.1, global_loss_max + 0.1])
    # axs[1].set_ylim([global_loss_min - 0.1, global_loss_max + 0.1])

    # # Set the y-axis limits for accuracy subplots with margin
    # axs[2].set_ylim([global_acc_min - 0.1, global_acc_max + 0.1])
    # axs[3].set_ylim([global_acc_min - 0.1, global_acc_max + 0.1])

    # # Add labels
    # for ax in axs:
    #     ax.set_xlabel('Value')
    #     ax.set_ylabel('Fold')

    # # Layout
    # plt.tight_layout()
    # plt.show()
    
    return mean_val_accuracies  

# Define the objective function for Optuna
def objective(trial, X, y, mdl_folder, rslt_folder):
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)
    l2_regularizer = trial.suggest_float("l2_regularizer", 1e-4, 1e-2, log = True)
    
    avg_val_accuracy = perform_cross_validation(X, y, learning_rate, dropout_rate, l2_regularizer,
                                                do_opt = True, models_folder = mdl_folder, results_folder= rslt_folder )  # Get average validation accuracy
    return avg_val_accuracy  # Return average validation accuracy


# Run Optuna hyperparameter tuning
def optimize_hyperparameters(X_train, y_train, n_trials=50, mdl_folder = "models", rslt_folder = "results"):

    study = optuna.create_study(direction="maximize")  # Maximize the validation accuracy
    # Maximize the validation accuracy
    study.optimize(lambda trial: objective(trial, X_train, y_train, mdl_folder, rslt_folder), n_trials=n_trials)

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (avg_val_accuracy): {best_trial.value}")
    print(f"  Params: {best_trial.params}")


def perform_vit_cross_validation(X, y, patch_size, embed_dim, num_heads, mlp_dim, num_layers, dropout_rate,
                                 n_splits=5, epochs=100, batch_size=32, vit_version = 1, val_set = 0.2,
                                 do_opt = False,models_folder = "models", results_folder = "results"):
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Initialize lists to store precision, recall, and F1 scores
    precisions = []
    recalls = []
    f1_scores = []
    
    # Initialize lists to store losses and accuracies for each epoch
    mean_training_losses = []
    mean_training_accuracies = []
    mean_val_losses = []   
    mean_val_accuracies = []
    
    fold = 1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)
    
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and validation sets
        print("=" * 40)
        print(f"Fold: {fold}")
        print("Splitting the data")
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]
        
        # Standardize the data
        print("Standardizing the data")
        X_train_full, X_test = standardize_data(X_train_full, X_test)
        
        # Reshape the data
        print("Reshaping the data for deep learning models")
        X_train_full, X_test = reshape_data(X_train_full, X_test)
        
        # Input shape and number of classes for the model
        input_shape = (X_train_full.shape[1], 1)
        num_classes = len(np.unique(y))
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = seed_value, stratify = y_train_full)
        num_patches = input_shape[0] // patch_size
        try:
            if do_opt:
                # Create the model
                model = create_vit(input_shape, patch_size, num_patches, num_classes,
                                embed_dim, num_heads, mlp_dim, num_layers, dropout_rate)
                model.summary()
                # Compile the model
                model.compile(
                    optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                # patch_size = 30
                # embed_dim = 8
                # num_heads = 6
                # mlp_dim = 16  # patch_size * patch_size * input_shape[1]
                # num_layers = 3
                # dropout_rate = 0.3

                # Create the model
                print("Building the vit model with the following hyperparameters:")
                print(f"Patch size: {learning_rate}")
                print(f"num_patches: {num_patches}") 
                print(f"embed_dim: {embed_dim}")
                print(f"num_heads: {num_heads}")
                print(f"mlp_dim: {mlp_dim}")
                print(f"num_layers: {num_layers}")
                print(f"dropout_rate: {dropout_rate}")
                model = create_vit(input_shape, patch_size, num_patches, num_classes,
                                embed_dim, num_heads, mlp_dim, num_layers, dropout_rate)
                model.summary()
                # Compile the model
                model.compile(
                    optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            if model is not None:
                "Build VIT"
                # model.summary()
            else:
                print("Failed to build the VIT model.")
        except Exception as e:
            print(f"Error while building the VIT model: {e}")
        
        
        print("Training the model")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
            # Calculate precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Append the scores to the lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # Append the losses and accuracies for each epoch to the respective lists
        for epoch in range(epochs):
            if fold == 1:
                # For the first fold, initialize the lists with the first fold's values
                mean_training_losses.append(history.history['loss'][epoch])
                mean_training_accuracies.append(history.history['accuracy'][epoch])
                mean_val_losses.append(history.history['val_loss'][epoch])
                mean_val_accuracies.append(history.history['val_accuracy'][epoch])
            else:
                # For subsequent folds, add the fold's values to the running sum
                mean_training_losses[epoch] += history.history['loss'][epoch]
                mean_training_accuracies[epoch] += history.history['accuracy'][epoch]
                mean_val_losses[epoch] += history.history['val_loss'][epoch]
                mean_val_accuracies[epoch] += history.history['val_accuracy'][epoch]

        # Save the model
        if do_opt:
            model_filename = f"VIT_v{vit_version}_opt_{batch_size}_{epochs}_seed{seed_value}.h5"
        else:
            model_filename = f"VIT_v{vit_version}_{batch_size}_{epochs}_seed{seed_value}.h5"
        model_filepath = os.path.join(models_folder, f"fold_{fold}_split_{n_splits}_{model_filename}")
        model.save(model_filepath)

        # Save the results as a JSON file
        if do_opt:
            results_filename = f"VIT_v{vit_version}_opt_{batch_size}_{epochs}_seed{seed_value}.json"
        else:
            results_filename = f"VIT_v{vit_version}_{batch_size}_{epochs}_seed{seed_value}.json"
        results_filepath = os.path.join(results_folder, f"fold_{fold}_split_{n_splits}_{results_filename}")

        results = {
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_scores": f1_scores,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "training_loss": history.history['loss'],
            "training_accuracy": history.history['accuracy'],
            "validation_loss": history.history['val_loss'],
            "validation_accuracy": history.history['val_accuracy'],
        }

        with open(results_filepath, "w") as results_file:
            json.dump(results, results_file, indent=4)
        
        # Append the results to the lists
        accuracies.append(accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Fold {fold}, Accuracy: {accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("=" * 40)
        fold += 1

    # At the end, divide the sums by the number of folds to get the mean
    mean_training_losses = [loss / n_splits for loss in mean_training_losses]
    mean_training_accuracies = [acc / n_splits for acc in mean_training_accuracies]
    mean_val_losses = [loss / n_splits for loss in mean_val_losses]
    mean_val_accuracies = [acc / n_splits for acc in mean_val_accuracies]
    # Convert lists to numpy arrays for easier calculation of mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    mean_test_loss = np.mean(test_losses)
    mean_test_accuracy = np.mean(test_accuracies)

    # Calculate standard deviations
    std_accuracy = np.std(accuracies)
    std_test_loss = np.std(test_losses)
    std_test_accuracy = np.std(test_accuracies)

    std_training_loss = np.std(mean_training_losses)
    std_training_accuracy = np.std(mean_training_accuracies)
    std_val_loss = np.std(mean_val_losses)
    std_val_accuracy = np.std(mean_val_accuracies)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

      
    print(f"test losses: {test_losses}")
    print(f"test accuracy: {test_accuracies}")
    
    print("*" * 40)
    print(f"Mean training loss over stratified {n_splits}-fold cross-validation: {np.mean(mean_training_losses):.4f}")
    print(f"Mean training accuracy over stratified {n_splits}-fold cross-validation: {np.mean(mean_training_accuracies):.4f}")
    print(f"Mean validation loss over stratified {n_splits}-fold cross-validation: {np.mean(mean_val_losses):.4f}")
    print(f"Mean validation accuracy over stratified {n_splits}-fold cross-validation: {np.mean(mean_val_accuracies):.4f}")
    print("*" * 40)
    print(f"Mean accuracy over stratified {n_splits}-fold cross-validation: {mean_accuracy:.4f}")
    print(f"Mean test loss over stratified {n_splits}-fold cross-validation: {mean_test_loss:.4f}")
    print(f"Mean test accuracy over stratified {n_splits}-fold cross-validation: {mean_test_accuracy:.4f}")
    print("*" * 40)

    print(f"Standard deviation of training loss over stratified {n_splits}-fold cross-validation: {np.std(mean_training_losses):.4f}")
    print(f"Standard deviation of training accuracy over stratified {n_splits}-fold cross-validation: {np.std(mean_training_accuracies):.4f}")
    print(f"Standard deviation of validation loss over stratified {n_splits}-fold cross-validation: {np.std(mean_val_losses):.4f}")
    print(f"Standard deviation of validation accuracy over stratified {n_splits}-fold cross-validation: {np.std(mean_val_accuracies):.4f}")
    print("*" * 40)
    print(f"Standard deviation of accuracy over stratified {n_splits}-fold cross-validation: {std_accuracy:.4f}")
    print(f"Standard deviation of test loss over stratified {n_splits}-fold cross-validation: {std_test_loss:.4f}")
    print(f"Standard deviation of test accuracy over stratified {n_splits}-fold cross-validation: {std_test_accuracy:.4f}")
    print("*" * 40)
    
    # Calculate and print the mean and standard deviation of precision, recall, and F1 score
    print(f"Mean precision over stratified {n_splits}-fold cross-validation: {np.mean(precisions):.4f}")
    print(f"Standard deviation of precision over stratified {n_splits}-fold cross-validation: {np.std(precisions):.4f}")
    print(f"Mean recall over stratified {n_splits}-fold cross-validation: {np.mean(recalls):.4f}")
    print(f"Standard deviation of recall over stratified {n_splits}-fold cross-validation: {np.std(recalls):.4f}")
    print(f"Mean F1 score over stratified {n_splits}-fold cross-validation: {np.mean(f1_scores):.4f}")
    print(f"Standard deviation of F1 score over stratified {n_splits}-fold cross-validation: {np.std(f1_scores):.4f}")

    
# Define the directory where the JSON files are stored

    fig, axes = plt.subplots(2, n_splits, figsize=(20, 10))

    for fold in range(1, n_splits+1):
        # Load the JSON file for this fold
        if do_opt:
            cnn_filename = f"VIT_v{vit_version}_opt_{batch_size}_{epochs}_seed{seed_value}.json"
        else:
            cnn_filename = f"VIT_v{vit_version}_{batch_size}_{epochs}_seed{seed_value}.json"
        results_filename = f"fold_{fold}_split_{n_splits}_{cnn_filename}"
        
        results_filepath = os.path.join(results_folder, results_filename)
        
        with open(results_filepath, "r") as results_file:
            results = json.load(results_file)
        
        # Plot training and validation loss
        axes[0, fold-1].plot(results["training_loss"], label='Train')
        axes[0, fold-1].plot(results["validation_loss"], label='Validation')
        axes[0, fold-1].set_title(f'Fold {fold} Loss')
        axes[0, fold-1].set_xlabel('Epochs')
        axes[0, fold-1].set_ylabel('Loss')
        axes[0, fold-1].legend()

        # Plot training and validation accuracy
        axes[1, fold-1].plot(results["training_accuracy"], label='Train')
        axes[1, fold-1].plot(results["validation_accuracy"], label='Validation')
        axes[1, fold-1].set_title(f'Fold {fold} Accuracy')
        axes[1, fold-1].set_xlabel('Epochs')
        axes[1, fold-1].set_ylabel('Accuracy')
        axes[1, fold-1].legend()

    plt.tight_layout()
    plt.show()
    
    return mean_val_accuracies  
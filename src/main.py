import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.optimizers import Adam
from data import *
from machine_learning_models import *
from deep_learning_models import *
from vision_transformer import *
from utils_dl_model import *
from utils_ml_model import print_ml_results
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


# Set the seed value.
SEED = 7
np.random.seed(SEED)

# Deep Learning parameters
DL_EPOCH = 500
DL_BATCH_SIZE = 32
DL_CNN_VERSION = 3
DL_TRANSFORMER_VISION_VERSION = 14
DL_CV_FOLD = 5

DO_DL = True
CV_DL = True
OPT_DL = False

DO_CNN = False
DO_TRANSFORMER_VISION = True
DO_ML = False

# Percentage of test set out of the dataset.
TEST_SET = 0.2

# Percentage of validation set out of the training dataset.
VAL_SET = 0.2

# Folder path associated with deep learning models
dl_models_folder = "../Deep_Learning_models/"
dl_metrics_folder = "../Deep_Learning_metrics/"
dl_weights_folder = "../Deep_Learning_weights/"
dl_cv_models_folder = "../Deep_Learning_CV/"
dl_cv_results_folder = "../Deep_Learning_CV_results/"

# Folder path associated with machine learning models
ml_models_folder = "../Machine_Learning_models/"
ml_models_results_folder = "../Machine_Learning_models_results/"

# Model names (Saved in h5 format)
cnn_model_name = f"cnn_v{DL_CNN_VERSION}_{DL_BATCH_SIZE}_{DL_EPOCH}_seed_{SEED}.h5"
transformer_vis_model_name = f"transformer_vision_v{DL_TRANSFORMER_VISION_VERSION}_{DL_BATCH_SIZE}_{DL_EPOCH}_seed_{SEED}.h5"

# Metric filenames
cnn_metrics_filename = f"metrics_{cnn_model_name}.json"
transformer_vis_metrics_filename = f"metrics_{transformer_vis_model_name}.json"

# Weight filenames
cnn_weights_filename = f"weights_{cnn_model_name}.json"
transformer_vis_weights_filename = f"weights_{transformer_vis_model_name}.json"


# Deep Learning models path
if DO_CNN:
    dl_model_path = os.path.join(
        dl_models_folder, cnn_model_name)
elif DO_TRANSFORMER_VISION:
    dl_model_path = os.path.join(
        dl_models_folder, transformer_vis_model_name)


# Deep Learning metrics path
if DO_CNN:
    dl_metrics_path = os.path.join(
        dl_metrics_folder, cnn_metrics_filename)
elif DO_TRANSFORMER_VISION:
    dl_metrics_path = os.path.join(
        dl_metrics_folder, transformer_vis_metrics_filename)


# Deep Learning weights path
if DO_CNN:
    dl_weights_path = os.path.join(
        dl_weights_folder, cnn_weights_filename)

elif DO_TRANSFORMER_VISION:
    dl_weights_path = os.path.join(
        dl_weights_folder, transformer_vis_weights_filename)


def main():

    # Load the data
    data_file = "../Data/HC05_HC07.csv"

    if OPT_DL:
        X_filtered, y = preprocess_cv_raw_data(data_file)
        optimize_hyperparameters(X_filtered, y, 50, dl_cv_models_folder, dl_cv_results_folder)

    else:
        # Train deep learning model with cross validation
        if CV_DL:
            X_filtered, y = preprocess_cv_raw_data(data_file)
        else:
            # Preprocess the raw data
            X_train, X_test, y_train, y_test = preprocess_raw_data(
                data_file, TEST_SET)

    # Do Deep Learning
    if DO_DL:
        dl_weights_path = "../During_train/cnn_v3-250-val_acc0.86.h5"
        if os.path.exists(dl_model_path):
            if DO_TRANSFORMER_VISION:
                model = tf.keras.models.load_model(
                    dl_model_path, custom_objects={'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock})
            elif DO_CNN:
                model = tf.keras.models.load_model(dl_model_path)
                print("Loaded trained model from", dl_model_path)
        elif os.path.exists(dl_weights_path):
            if DO_TRANSFORMER_VISION:
                print("hello")
                # model = create_vit(input_shape, patch_size, num_patches, num_classes,
                #                         embed_dim, num_heads, mlp_dim, num_layers, dropout_rate)
                # model.load_weights(dl_weights_path)
            elif DO_CNN:
                X_train = X_train.reshape(
                    X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                input_shape = (X_train.shape[1], 1)
                num_classes = len(np.unique(y_train))
                learning_rate = 1.4286522423518213e-05
                dropout_rate = 0.2
                l2_regularizer = 0.000417822262290105
                model = cnn_model_3_opt(
                    input_shape, num_classes, learning_rate, dropout_rate, l2_regularizer)
                model.load_weights(dl_weights_path)
                print("Loaded trained model weights from", dl_weights_path)
                # Evaluating the model
                print("Evaluating the CNN model... ")
                # Calculate model loss and accuracy
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                print("Test loss:", test_loss)
                print("Test accuracy:", test_accuracy)
        else:
            if CV_DL:
                # learning_rate = 3.408170980489466e-05
                # dropout_rate = 0.3
                # l2_regularizer = 0.00010124894257897855
                if DO_CNN:
                    learning_rate = 1.4286522423518213e-05
                    dropout_rate = 0.2
                    l2_regularizer = 0.000417822262290105
                    # Start the timer
                    start_time = time.time()
                    perform_cross_validation(X_filtered, y, learning_rate, dropout_rate, l2_regularizer,
                                            n_splits=DL_CV_FOLD, epochs=DL_EPOCH, batch_size=DL_BATCH_SIZE, cnn_version=DL_CNN_VERSION,
                                            do_opt=True, models_folder=dl_cv_models_folder, results_folder=dl_cv_results_folder)
                    # End the timer
                    end_time = time.time()
                    # Calculate the elapsed time
                    elapsed_time = end_time - start_time
                    # Print the elapsed time
                    print("Training time: {:.2f} seconds".format(elapsed_time))
                elif DO_TRANSFORMER_VISION:
                    # Train and evaluate deep learning models
                    print("Building Transformer model...")
                    patch_size = 3
                    embed_dim = 64
                    num_heads = 4
                    mlp_dim = 128  # patch_size * patch_size * input_shape[1]
                    num_layers = 4
                    dropout_rate = 0.3

                    # Create the model
                    perform_vit_cross_validation(X_filtered, y, patch_size, embed_dim, num_heads, mlp_dim, num_layers, dropout_rate,
                                                n_splits=DL_CV_FOLD, epochs=DL_EPOCH, batch_size=DL_BATCH_SIZE, 
                                                vit_version = DL_TRANSFORMER_VISION_VERSION, val_set = 0.2,
                                                do_opt = True, models_folder = "CV_VIT_models", results_folder = "CV_VIT_results")
                    
            else:
                # Reshape the input data for deep learning models
                print("Reshaping the data for CNN model...")
                X_train = X_train.reshape(
                    X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                input_shape = (X_train.shape[1], 1)
                num_classes = len(np.unique(y_train))

                # Splitting the training data to create a validation set
                print("Splitting the training data to create a validation set...")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=VAL_SET, random_state=SEED, stratify=y_train)

                filepath = ""
                
                if DO_CNN:
                    # Train and evaluate deep learning models
                    print("Building CNN model...")
                    # model = cnn_model_3_3(input_shape, num_classes)
                    # learning_rate = 3.408170980489466e-05
                    # dropout_rate = 0.3
                    # l2_regularizer = 0.00010124894257897855
                    learning_rate = 1.4286522423518213e-05
                    dropout_rate = 0.2
                    l2_regularizer = 0.000417822262290105
                    model = cnn_model_3_opt(
                        input_shape, num_classes, learning_rate, dropout_rate, l2_regularizer)
                    
                    filepath = '../During_train/cnn_v3-{epoch:02d}-val_acc{val_accuracy:.2f}.h5'

                elif DO_TRANSFORMER_VISION:
                    # Train and evaluate deep learning models
                    print("Building Transformer model...")
                    # Define model parameters
                    print(f"X_train shape = {X_train.shape}")
                    input_shape = (X_train.shape[1], 1)  # (270,1)
                    # cancer cell lines, monocytes, and T-cells
                    num_classes = len(np.unique(y_train))
                    patch_size = 3
                    num_patches = input_shape[0] // patch_size
                    embed_dim = 64
                    num_heads = 4
                    mlp_dim = 128  # patch_size * patch_size * input_shape[1]
                    num_layers = 4
                    dropout_rate = 0.3

                    # Create the model
                    model = create_vit(input_shape, patch_size, num_patches, num_classes,
                                       embed_dim, num_heads, mlp_dim, num_layers, dropout_rate)
                    model.summary()
                    # Compile the model
                    model.compile(
                        optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    filepath = '../During_train/vit_v11-{epoch:02d}-val_acc{val_accuracy:.2f}.h5'
                
                # Train the model
                print("Training and evaluating the model...")
                # Start the timer
                start_time = time.time()
                model_history = model.fit(X_train, y_train, batch_size=DL_BATCH_SIZE,
                                          epochs=DL_EPOCH, validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)
                # End the timer
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                # Print the elapsed time
                print("Training time: {:.2f} seconds".format(elapsed_time))
                # Compile the model

                # Save the deep learning model
                print("Saving model to ", dl_model_path)
                model.save(dl_model_path)

                # Save the deep learning model's weights
                print("Saving model weights to ", dl_weights_path)
                model.save_weights(dl_weights_path)

                # Evaluating the model
                print("Evaluating the CNN model... ")
                # Calculate model loss and accuracy
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                print("Test loss:", test_loss)
                print("Test accuracy:", test_accuracy)

                # Save the model's metrics
                print("Saving the model metrics to ", dl_metrics_path)
                save_dl_metrics(model_history, test_loss,
                                test_accuracy, dl_metrics_path)

                # Plot the model history
                plot_dl_history(model_history)

        # # Load the model metrics
        # loss, val_loss, accuracy, val_accuracy, test_loss, test_acc = load_dl_metrics(
        #     dl_metrics_path)

        # print(f"Classification Test Loss: {test_loss}")
        # print(
        #     f"Classification Test Accuracy: {test_acc}")

        # # Plot the deep learning metrics
        # plot_dl_metrics(loss, val_loss, accuracy, val_accuracy)

    if DO_ML:
        # Train and evaluate machine learning models
        print("Building machine learning models...")
        ml_models = build_ml_models()
        print("Training and evaluating machine learning models...")
        ml_results = train_and_evaluate_ml_models(
            ml_models, X_train, y_train, X_test, y_test,
            cv=5, models_folder=ml_models_folder,
            results_folder=ml_models_results_folder)
        # Print the results for each model
        print_ml_results(ml_results)


if __name__ == "__main__":
    main()

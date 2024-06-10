import numpy as np
import matplotlib.pyplot as plt
import json

seed_value = 7
np.random.seed(seed_value)

# Save the metrics for classification model
def save_dl_metrics(history, test_loss, test_acc, file_name):
    # Extract the metrics from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Store the metrics in a dictionary
    metrics = {
        'loss': loss,
        'val_loss': val_loss,
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'test_loss': test_loss,
        'test_acc': test_acc
    }

    # Save the dictionary as a JSON file
    with open(file_name, 'w') as outfile:
        json.dump(metrics, outfile)
        
def load_dl_metrics(file_name):
    # Load the classification metrics

    # Load the JSON file and parse it as a Python dictionary
    with open(file_name, 'r') as infile:
        metrics = json.load(infile)

    # Extract the metrics from the dictionary
    loss = metrics['loss']
    val_loss = metrics['val_loss']
    accuracy = metrics['accuracy']
    val_accuracy = metrics['val_accuracy']
    test_loss = metrics['test_loss']
    test_acc = metrics['test_acc']

    return loss, val_loss, accuracy, val_accuracy, test_loss, test_acc
        
def print_model_class_report(X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate Sensitivity, Specificity, Accuracy, and F-1
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))
    
def plot_dl_metrics(loss, val_loss, accuracy, val_accuracy):
    epochs = range(1, len(loss) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss on the first subplot
    ax1.plot(epochs, loss, 'b', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')
    ax1.set_title('Classification Loss per Iteration')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy and val_accuracy  on the second subplot
    ax2.plot(epochs, accuracy, 'b', label='Training Accuracy')
    ax2.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    ax2.set_title('Classification Accuracy per Iteration')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Display the figure
    plt.show()
        
def plot_dl_history(history):
      # Loss function per iteration
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function per iteration')
    plt.legend()
    plt.show()

    # Accuracy per iteration
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per iteration')
    plt.legend()
    plt.show()
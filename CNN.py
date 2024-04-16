# init code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# cifar data 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# split cifar training data into training and validation sets
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, train_size=0.9)

def CNN_model(num_layers):
    """_summary_

    Args:
       num_layers (int): number of convolution layers to experiment with for CNN model

    Returns:
        model (object): CNN model
    """
    # creating CNN base with dense layers, source: Lecs14-15 CNN demo
    model = models.Sequential()
    #first conv layer and max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    #adding additional layers
    for i in range(num_layers-1):
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
    # dropout & dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation= 'softmax'))
    return model

def train_model(model, train_images, validation_images, train_labels, validation_labels):
    """_summary_

    Args:
       model (object): CNN model
       train_images (arraylike): training images
       validation_images (arraylike): validation images
       train_labels (arraylike): training labels
       validation_labels (arraylike): validation labels

    Returns:
        model (object): CNN model
        history (object): training history
    """
    # Compile/train the model
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_data=(validation_images, validation_labels))
    return model, history

def evaluate_model(model, images, labels):
    """_summary_

    Args:
       model (object): CNN model
       images (arraylike): image data
       labels (arraylike): labels

    Returns:
        accuracy (double): accuracy of the model
        F1 (double): F1 of the model
        AUC (double): AUC of the model
    """
    y_pred_probabilities = model.predict(images)
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    accuracy = accuracy_score(labels, y_pred_labels)

    f1 = f1_score(labels, y_pred_labels, average='weighted')

    auc = roc_auc_score(labels, y_pred_probabilities, multi_class='ovr')

    return accuracy, f1, auc

def hyperparameter_tuning(layers_list, train_images, validation_images, train_labels, validation_labels):
    """_summary_

    Args:
       layers_list (list): list of number of layers for experimentation
       train_images (arraylike): training images
       validation_images (arraylike): training images
       train_labels (arraylike): training labels
       validation_labels (arraylike): training labels

    Returns:
        best_num_layers (int): number of convolution layers that resulted in greatest validation accuracy
        best_model (object): CNN model with best performance
    """
    # experimenting with the amount of layers
    best_accuracy = 0
    best_num_layers = None
    best_model = None
    for num_layers in layers_list:
        model = CNN_model(num_layers)
        trained_model, history = train_model(model, train_images, validation_images, train_labels, validation_labels)
            
        accuracy = history.history['val_accuracy'][-1]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_layers = num_layers
            best_model = model
    return best_num_layers, best_model

layers_list = [1, 2, 3, 4]
best_num_layers, best_model = hyperparameter_tuning(layers_list, train_images, validation_images, train_labels, validation_labels)
print(f'Best Number of Layers: {best_num_layers}')


# best_model = CNN_model(best_num_layers)
# train_model(best_model, train_images, validation_images, train_labels, validation_labels)
train_accuracy, train_f1, train_auc = evaluate_model(best_model, validation_images, validation_labels)
test_accuracy, test_f1, test_auc = evaluate_model(best_model, test_images, test_labels)

print(f'Training Accuracy: {train_accuracy}')
print(f'Training F1: {train_f1}')
print(f'Training ROC AUC: {train_auc}')
print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing F1: {test_f1}')
print(f'Testing ROC AUC: {test_auc}')
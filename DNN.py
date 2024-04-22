# init code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
# Used Deep Learning Intro lecture slides as a resource on NN's in general and their features

def DNN_model(input_shape, num_classes, num_layers):
    # Used this source to learn about DNNs in python: https://kavita-ganesan.com/neural-network-intro/
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    for _ in range(num_layers - 1):
        model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    #ChatGPT helped optimize the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None)
    data = data[data[1] != -1]
    labels = data[1].astype(int)
    features = data.drop(columns=[0, 1])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels
# use KFold cross validation to tune the hyperparameter, as required in instructions
def hyperparameter_tuning_dnn(features, labels, num_classes, num_folds, layer_options):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_num_layers = None
    best_accuracy = 0
    features = np.array(features)
    labels = np.array(labels)
    for num_layers in layer_options:
        fold_accuracies = []
        for train_index, test_index in kf.split(features):
            trainX, testX = features[train_index], features[test_index]
            trainY, testY = labels[train_index], labels[test_index]
            model = DNN_model(trainX.shape[1], num_classes, num_layers)
            model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)
            test_pred = model.predict(testX)
            test_pred_labels = np.argmax(test_pred, axis=1)
            accuracy = accuracy_score(testY, test_pred_labels)
            fold_accuracies.append(accuracy)
        mean_accuracy = np.mean(fold_accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_num_layers = num_layers
        print(f'Layers: {num_layers}, Mean K-fold Accuracy: {mean_accuracy:.4f}')
    return best_num_layers, best_accuracy

# Main execution
iyer_features, iyer_labels = load_and_preprocess_data('data/iyer.txt')
cho_features, cho_labels = load_and_preprocess_data('data/cho.txt')
# zero-indexing
iyer_labels = iyer_labels - 1
cho_labels = cho_labels - 1
layers_list = [3, 4, 5, 6]
print('Tuning number of layers for Iyer Data')
best_iyer_layers, iyer_accuracy = hyperparameter_tuning_dnn(iyer_features, iyer_labels, len(np.unique(iyer_labels)), 3, layers_list)
print(f'Best number of layers for Iyer Data: {best_iyer_layers}, with K-fold accuracy: {iyer_accuracy:.4f}')

print('Tuning number of layers for Cho Data')
best_cho_layers, cho_accuracy = hyperparameter_tuning_dnn(cho_features, cho_labels, len(np.unique(cho_labels)), 3, layers_list)
print(f'Best number of layers for Cho Data: {best_cho_layers}, with K-fold accuracy: {cho_accuracy:.4f}')
## Train the model with the best hyperparameter and simulate 3 times to get the average accuracy

def evaluate_model(model, features, labels):
    """_summary_

    Args:
       model (object): DNN model
       images (arraylike): feature/image data
       labels (arraylike): labels

    Returns:
        accuracy (double): accuracy of the model
        F1 (double): F1 of the model
        AUC (double): AUC of the model
    """
    y_pred_probabilities = model.predict(features)
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    accuracy = accuracy_score(labels, y_pred_labels)

    f1 = f1_score(labels, y_pred_labels, average='weighted')

    auc = roc_auc_score(labels, y_pred_probabilities, multi_class='ovr')

    return accuracy, f1, auc

def main(train_features, train_labels, test_features, test_labels, best_num_layers):
    n = 3
    results = {
        "train_accuracy": [],
        "train_f1": [],
        "train_auc": [],
        "test_accuracy": [],
        "test_f1": [],
        "test_auc": []
    }

    for _ in range(n):
        # Split the training data to create a train and a validation set for each iteration
        trainX, valX, trainY, valY = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=None, stratify=train_labels)

        # Build and train the DNN model
        model = DNN_model(trainX.shape[1], len(np.unique(train_labels)), best_num_layers)
        model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)

        # Evaluate the model on the validation set
        train_acc, train_f1, train_auc = evaluate_model(model, valX, valY)
        results["train_accuracy"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["train_auc"].append(train_auc)

        # Evaluate the model on the test set
        test_acc, test_f1, test_auc = evaluate_model(model, test_features, test_labels)
        results["test_accuracy"].append(test_acc)
        results["test_f1"].append(test_f1)
        results["test_auc"].append(test_auc)

    # Print the results
    for metric, values in results.items():
        print(f'Average {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}')
    print('')

# Splitting Iyer dataset
iyer_trainX, iyer_testX, iyer_trainY, iyer_testY = train_test_split(
    iyer_features, iyer_labels, test_size=0.3, random_state=42)

# Splitting Cho dataset
cho_trainX, cho_testX, cho_trainY, cho_testY = train_test_split(
    cho_features, cho_labels, test_size=0.3, random_state=42)

print('Iyer Data Results (DNN) with ' + str(best_iyer_layers) + ' layers')
main(iyer_trainX, iyer_trainY, iyer_testX, iyer_testY, best_iyer_layers)

print('Cho Data Results (DNN) with ' + str(best_cho_layers) + ' layers')
main(cho_trainX, cho_trainY, cho_testX, cho_testY, best_cho_layers)



# Cifar-10 Data 

# Used lecture 16 CNN code demo for help with loading Cifar data
# Download and normalize Cifar data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# Split cifar training data into training and validation sets
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, train_size=0.9)
# Reshaping data to be compatible with DNN
train_images = train_images.reshape(len(train_images), -1)
validation_images = validation_images.reshape(len(validation_images), -1)
test_images = test_images.reshape(len(test_images), -1)

def train_model(model, train_images, validation_images, train_labels, validation_labels):
    """_summary_

    Args:
       model (object): DNN model
       train_images (arraylike): training images
       validation_images (arraylike): validation images
       train_labels (arraylike): training labels
       validation_labels (arraylike): validation labels

    Returns:
        model (object): DNN model
        history (object): training history
    """
    # Train the model with cifar data
    # Used lecture 16 CNN code demo to help with the training of the model 
    history = model.fit(train_images, train_labels, batch_size=64, epochs=1,
                    validation_data=(validation_images, validation_labels))
    return model, history

def hyperparameter_tuning(layers_list, train_images, validation_images, train_labels, validation_labels):
    """_summary_

    Args:
       layers_list (list): list of number of hidden layers for experimentation
       train_images (arraylike): training images
       validation_images (arraylike): training images
       train_labels (arraylike): training labels
       validation_labels (arraylike): training labels

    Returns:
        best_num_layers (int): number of hidden layers that resulted in greatest validation accuracy
        best_model (object): DNN model with best performance
    """
    # Experimenting with the amount of hidden layer
    best_accuracy = 0
    best_num_layers = None
    best_model = None
    for num_layers in layers_list:
        model = DNN_model(train_images.shape[1], 10, num_layers)
        trained_model, history = train_model(model, train_images, validation_images, train_labels, validation_labels)
            
        accuracy = history.history['val_accuracy'][-1]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_layers = num_layers
            best_model = model
    return best_num_layers, best_model

# Executing hyperparameter tuning for the number hidden layers
best_num_layers, best_model = hyperparameter_tuning(layers_list, train_images, validation_images, train_labels, validation_labels)
print(f'(CIFAR-10) Best Number of Hidden Layers: {best_num_layers}')
# Getting evaluation  metrics for best model
train_accuracy, train_f1, train_auc = evaluate_model(best_model, validation_images, validation_labels)
test_accuracy, test_f1, test_auc = evaluate_model(best_model, test_images, test_labels)

print('CIFAR-10 Data Results (DNN)')
print(f'Training Accuracy: {train_accuracy}')
print(f'Training F1: {train_f1}')
print(f'Training ROC AUC: {train_auc}')
print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing F1: {test_f1}')
print(f'Testing ROC AUC: {test_auc}')
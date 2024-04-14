import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras import datasets

np.random.seed(42)

# preprocessing data
# removing rows with outliers (label = -1)
# iyer_data = pd.read_csv('data/iyer.txt', sep='\t', header=None)
# iyer_data = iyer_data[iyer_data[1] != -1]

# # split data into training and tes ting sets
# train_data = iyer_data.sample(frac=0.8)
# test_data = iyer_data.drop(train_data.index)

# cho data
cho_data = pd.read_csv('data/cho.txt', sep='\t', header=None)
cho_data = cho_data[cho_data[1] != -1]

# split data into training and testing sets
train_data = cho_data.sample(frac=0.7)
test_data = cho_data.drop(train_data.index)

# train logistic regression and k-fold cross validation on training data to find best hyperparameters
# solver = 'liblinear' is used for small datasets
# penalty = 'l2' is used to avoid overfitting ( just like in SVM )
# C = [0.01, 0.1, 1, 10, 100] is used to find the best regularization parameter

##############################################################################################
# Train a logistic regression mode
def train_LR(data, C):
    """_summary_

    Args:
       data (arraylike / matrixlike): train data or test data
        C (arraylike): regularization parameter

    Returns:
        model (object): logistic regression model
    """
    model = LR(solver='liblinear', penalty='l2', C=C, max_iter=1000, multi_class='ovr')
    model.fit(data.drop(columns=1), data[1])
    return model
# Hyperparameter tuning for Logistic Regression C parameter using 3-fold cross validation

def k_fold(data, C):
    """_summary_
    Args:
        data (arraylike / matrixlike): train data or test data
        C (arraylike): regularization parameter

    Returns:
        accuracy (double): accuracy of the mode
        
    Note: 
        DOES NOT RETURN AUC SCORE NOR F1 SCORE
    """
    train_data = data.sample(frac=0.8) # 80% training data randomly sampled
    X_train, X_val = train_test_split(train_data, test_size=0.2) # 80% training and 20% validation split training data into test and validation set for a k-fold cross validation
    
    model = train_LR(X_train, C)
    y_pred = model.predict(X_val.drop(columns=1))
    accuracy = accuracy_score(X_val[1], y_pred)
    
    return accuracy

def hyperparameter_tuning(data, C):
    """_summary_

    Args:
        data (arraylike / matrixlike): train data or test data
        C (arraylike): regularization parameter

    Returns:
        best_C (double): best regularization parameter
    """
    accuracies = []
    for c in C:
        accuracy = k_fold(data, c)
        accuracies.append(accuracy)
    best_C = C[np.argmax(accuracies)]
    return best_C

C = [0.01, 0.1, 1, 10, 100]
best_C = hyperparameter_tuning(train_data, C)
print(f'Best C: {best_C}')
##############################################################################################

# train the model with the best hyperparameter
n = 3
train_acc = []
train_roc_auc = []
for i in range(n):
    train_data_i = train_data.sample(frac=0.8) # 80% training data randomly sampled
    test_data_i = train_data.drop(train_data_i.index) # 20% test data
    model = train_LR(train_data_i, best_C)
    y_pred = model.predict(test_data_i.drop(columns=1))
    y_pred_prob = model.predict_proba(test_data_i.drop(columns=1))
    
    train_acc.append(accuracy_score(test_data_i[1], y_pred))
    train_roc_auc.append(roc_auc_score(test_data_i[1], y_pred_prob, multi_class='ovr'))
    
print(f'Training Accuracy: {np.mean(train_acc)}')
print(f'Standard Deviation: {np.std(train_acc)}')
print(f'Training ROC AUC: {np.mean(train_roc_auc)}')
print(f'Standard Deviation: {np.std(train_roc_auc)}')

# test the model on the test data
test_acc = []
test_roc_auc = []
for i in range(n):
    model = train_LR(train_data, best_C)
    test_data_i = test_data.sample(frac=0.8) # 80% test data randomly sampled
    y_pred = model.predict(test_data_i.drop(columns=1))
    y_pred_prob = model.predict_proba(test_data_i.drop(columns=1))
    
    test_acc.append(accuracy_score(test_data_i[1], y_pred))
    test_roc_auc.append(roc_auc_score(test_data_i[1], y_pred_prob, multi_class='ovr'))

print(f'Test Accuracy: {np.mean(test_acc)}')
print(f'Standard Deviation: {np.std(test_acc)}')
print(f'Test ROC AUC: {np.mean(test_roc_auc)}')
print(f'Standard Deviation: {np.std(test_roc_auc)}')

##############################################################################################


# cifar data 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# split cifar training data into training and validation sets
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, train_size=0.9)

# training model
def train_LR(train_images, train_labels, C):
    """_summary_

    Args:
       train_images (arraylike): training images
       train_labels (arraylike): training labels
        C (arraylike): regularization parameter

    Returns:
        model (object): logistic regression model
    """
    train_labels = train_labels.ravel()
    # this process is relatively time consuming due to the size of the dataset, which is why max iter is set to 100
    # i experimented with saga, sag, and newton-cg solvers and accuracies are very similar but newton-cg is more efficient
    model = LR(solver='newton-cg', penalty='l2', C=C, max_iter=100, multi_class='multinomial')
    train_images_reshaped = train_images.reshape(len(train_images), -1)
    model.fit(train_images_reshaped, train_labels)
    return model

# evaluating model
def evaluate_LR(model, validation_images, validation_labels):
    """_summary_
    Args:
        model: trained LR model
        validation_images (arraylike): validation images
        validation_labels (arraylike): validation labels

    Returns:
        accuracy (double): accuracy of the mode
 
    """
    validation_labels = validation_labels.ravel()
    validation_images_reshaped = validation_images.reshape(len(validation_images), -1)
    y_pred = model.predict(validation_images_reshaped)
    accuracy = accuracy_score(validation_labels, y_pred)

    f1 = f1_score(validation_labels, y_pred, average='weighted')

    y_pred_prob = model.predict_proba(validation_images_reshaped)
    auc = roc_auc_score(validation_labels, y_pred_prob, multi_class='ovr')

    return accuracy, f1, auc

def hyperparameter_tuning(train_images, validation_images, train_labels, validation_labels, C):
    """_summary_

    Args:
        train_images (arraylike): training images
        validation_images (arraylike): validation images
        train_labels (arraylike): training labels
        validation_labels (arraylike): validation labels
        C (arraylike): regularization parameter

    Returns:
        best_C (double): best regularization parameter
    """
    accuracies = []
    for c in C:
        model = train_LR(train_images, train_labels, c)
        accuracy, f1, roc = evaluate_LR(model, validation_images, validation_labels)
        accuracies.append(accuracy)
        #print(accuracy)
    best_C = C[np.argmax(accuracies)]
    return best_C

C = [0.01, 0.1, 1, 10, 100]
cifar_best_C = hyperparameter_tuning(train_images, validation_images, train_labels, validation_labels, C)
print(f'Best C: {cifar_best_C}')

best_model = train_LR(train_images, train_labels, cifar_best_C)
train_accuracy, train_f1, train_auc = evaluate_LR(best_model, validation_images, validation_labels)
test_accuracy, test_f1, test_auc = evaluate_LR(best_model, test_images, test_labels)

print(f'Training Accuracy: {train_accuracy}')
print(f'Training F1: {train_f1}')
print(f'Training ROC AUC: {train_auc}')
print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing F1: {test_f1}')
print(f'Testing ROC AUC: {test_auc}')
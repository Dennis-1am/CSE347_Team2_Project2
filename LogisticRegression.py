import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

np.random.seed(42)

# Load the Cho and Iyer Data
# preprocessing data
# removing rows with outliers (label = -1)

iyer_data = pd.read_csv('data/iyer.txt', sep='\t', header=None)
iyer_data = iyer_data[iyer_data[1] != -1]
iyer_labels = iyer_data[1]
iyer_data = iyer_data.drop(columns=[0,1])

# split data into training and testing sets
iyer_trainX, iyer_testX1, iyer_trainY, iyer_testY1 = train_test_split(
    iyer_data, iyer_labels, test_size=0.3, random_state=42
) 

iyer_train = pd.DataFrame(iyer_trainY).join(iyer_trainX)
iyer_test = pd.DataFrame(iyer_testY1).join(iyer_testX1)

# cho data
cho_data = pd.read_csv('data/cho.txt', sep='\t', header=None)
cho_data = cho_data[cho_data[1] != -1]
cho_labels = cho_data[1]
cho_data = cho_data.drop(columns=[0,1])

cho_trainX, cho_testX1, cho_trainY, cho_trainY1 = train_test_split(
    cho_data, cho_labels, test_size=0.3, random_state=42
)

cho_train = pd.DataFrame(cho_trainY).join(cho_trainX)
cho_test = pd.DataFrame(cho_trainY1).join(cho_testX1)

## K-Folds

def k_folds(data, C, kernel):
    model = SVC(C=C, kernel=kernel)
    
    # split data into K_folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    kf.get_n_splits(data)
    
    accuracy = []
    
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        model.fit(train_data.drop(columns=1), train_data[1])
        y_pred = model.predict(test_data.drop(columns=1))
        
        accuracy.append(accuracy_score(test_data[1], y_pred))
        
    return np.mean(accuracy)
       

## Hyper Parameter Tuning for SVM C and Kernel using 3-fold cross validation

def hyperparameter_tuning(data, C, kernel):
    """_summary_
    Args:
        data (arraylike / matrixlike): train data or test data
        C (arraylike): regularization parameter
        kernel (arraylike): kernel type
    Returns:
        accuracy (double): accuracy of the mode
    """
    best_C = 0
    best_kernel = ''
    accuracy = 0
    for c in C:
        for k in kernel:
            best_accuracy = k_folds(data, c, k)
            if best_accuracy > accuracy:
                accuracy = best_accuracy
                best_C = c
                best_kernel = k
                
    return best_C, best_kernel
            

C = [0.01, 0.1, 1, 10, 100]
kernel = ['rbf', 'linear', 'poly', 'sigmoid']

best_C, best_K = hyperparameter_tuning(iyer_train, C, kernel)

print(f'Best C: {best_C}', f'Best Kernel: {best_K}', sep='\n')

## Train the model with the best hyperparameter and simulate 3 times to get the average accuracy

def main(train_data, test_data, best_C=best_C, best_K=best_K):
    n = 3
    train_acc = []
    train_roc_auc = []
    train_f1 = []

    for _ in range(n):
        train_data_i, test_data_i = train_test_split(train_data, test_size=0.2, stratify=train_data[1])
        
        model = SVC(C=best_C, kernel=best_K)
        model1 = SVC(C=best_C, kernel=best_K, probability=True)
        model.fit(train_data_i.drop(columns=1), train_data_i[1])
        model1.fit(train_data_i.drop(columns=1), train_data_i[1])
        y_pred = model.predict(test_data_i.drop(columns=1))
        y_pred1 = model1.predict_proba(test_data_i.drop(columns=1))
        
        train_acc.append(accuracy_score(test_data_i[1], y_pred))
        train_roc_auc.append(roc_auc_score(test_data_i[1], y_pred1, multi_class='ovr'))
        train_f1.append(f1_score(test_data_i[1], y_pred, average='weighted'))
        
        
    print(f'Average Training Accuracy: {np.mean(train_acc)}')
    print(f'Standard Deviation: {np.std(train_acc)}')
    print(f'Average Training ROC AUC: {np.mean(train_roc_auc)}')
    print(f'Standard Deviation: {np.std(train_roc_auc)}')
    print(f'Average Training F1: {np.mean(train_f1)}')
    print(f'Standard Deviation: {np.std(train_f1)}')
    print('')

    ## test the model on the test data

    test_acc = []
    test_roc_auc = []
    test_f1 = []
    model = SVC(C=best_C, kernel=best_K)
    model1 = SVC(C=best_C, kernel=best_K, probability=True)
    model.fit(train_data.drop(columns=1), train_data[1])
    model1.fit(train_data.drop(columns=1), train_data[1])

    
    for _ in range(n):
        test_data_i = test_data.sample(frac=0.8) # 80% test data randomly sampled
        y_pred = model.predict(test_data_i.drop(columns=1))
        y_pred_prob = model1.predict_proba(test_data_i.drop(columns=1))
        
        test_acc.append(accuracy_score(test_data_i[1], y_pred))
        test_roc_auc.append(roc_auc_score(test_data_i[1], y_pred_prob, multi_class='ovr'))
        test_f1.append(f1_score(test_data_i[1], y_pred, average='weighted'))

    print(f'Test Accuracy: {np.mean(test_acc)}')
    print(f'Standard Deviation: {np.std(test_acc)}')
    print(f'Test ROC AUC: {np.mean(test_roc_auc)}')
    print(f'Standard Deviation: {np.std(test_roc_auc)}')
    print(f'Test F1: {np.mean(test_f1)}')
    print(f'Standard Deviation: {np.std(test_f1)}')
    print('')

print('Iyer Data Results (SVM)')
main(iyer_train, iyer_test, best_C, best_K)

print('Cho Data Results (SVM)')
main(cho_train, cho_test, best_C, best_K)
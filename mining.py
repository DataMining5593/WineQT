import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# declare all models
ridge_model = None


# Train all the models
def modelsTraining():
    global ridge_model

    loadTrainAndTest()


    ridge_model = ridge_fit(60) # lambda found in data visualise


    return None


# output the reult of all the models
def getModelsResult(data):
    global ridge_model


    result = []

    result.append(ridge_predict(data, ridge_model))

    return result


## Load the dataset
sample_train = None
sample_test = None
label_train = None
label_test = None

whole_train = None
whole_label = None


def loadTrainAndTest():
    global sample_train
    global sample_test
    global label_train
    global label_test
    global whole_train
    global whole_label

    data = np.loadtxt('mysite/winequality-all.csv', delimiter=';', skiprows=1)
    # Separate features and labels
    X = data[:, :-1]  # All columns except the last one (features)
    Y = data[:, -1]   # Last column (labels)

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random indices and shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Split indices into training and testing sets
    num_test = int(0.25 * len(X))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    # Create training and testing datasets
    sample_train = X[train_indices]
    sample_test = X[test_indices]
    label_train = Y[train_indices]
    label_test = Y[test_indices]

    whole_train = X.copy()
    whole_label = Y.copy()




## Ridge regression 
# train Ridge Regression
def ridge_fit(lamda):
    
    X = np.concatenate([np.ones((whole_train.shape[0], 1)), whole_train], axis=1)  # intercept bias
    XtX = np.dot(X.T, X)  # X^T * X
    XtY = np.dot(X.T, whole_label)  # X^T * y
    I = np.eye(XtX.shape[0])  # Identity matrix
    I[0, 0] = 0  # Exclude bias
    beta = np.linalg.solve(XtX + lamda * I, XtY)  # Solve for coefficients (beta)

    return beta

# make predictions
def ridge_predict(data, beta):

    X_test = np.concatenate([np.ones((data.shape[0], 1)), data], axis=1)
    label_test_pred = np.dot(X_test[0], beta)

    return int(round(label_test_pred, 0))


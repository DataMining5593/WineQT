import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import t


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
    global whole_train


    result = []


    datasetWithNew = np.concatenate((whole_train, data), axis=0)
    
    result.append(grubbs_test(datasetWithNew))


    result.append(ridge_predict(data, ridge_model))

    return result


################################## Load the dataset ######################""
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



########################## CLASSIFICATION ################
## Ridge regression 
# train Ridge Regression
def ridge_fit(lamda):
    global whole_train
    global whole_label
    
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



########################## ANOMALY DETECTION ################
def grubbs_test(data, alpha=0.15):
    n = len(data)
    last_outlier_found = True
    is_outlier = False

    while n > 2 and last_outlier_found and not is_outlier:

        centroid = np.mean(data, axis=0)  # Mean vector (centroid)
        distances = np.linalg.norm(data - centroid, axis=1)  # Euclidean distances to the centroid
        

        max_dist_index = np.argmax(distances)
        max_distance = distances[max_dist_index]
        std_dev = np.std(distances, ddof=1)  # Standard deviation of distances
        g_calculated = max_distance / std_dev  # Grubbs' test statistic

        t_crit = t.ppf(1 - alpha / (2 * n), df=n - 2)  # Two-tailed t-test critical value
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))


        if g_calculated > g_critical:

            if max_dist_index == n - 1:
                is_outlier = True
            else:
                data = np.delete(data, max_dist_index, axis=0)  # Remove the outlier
                n = len(data)
        else:
            last_outlier_found = False

    return is_outlier
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import t


# declare all models
ridge_model = None
logistic_model = None
logistic_bias = None
kmeans_centroid = None
decision_tree = None


# Train all the models
def modelsTraining():
    global ridge_model
    global logistic_model
    global logistic_bias
    global kmeans_centroid
    global decision_tree

    loadTrainAndTest()


    ridge_model = ridge_fit(60) # lambda found in data visualise
    logistic_model, logistic_bias = logistic_fit(50, 0.01)#50000, 0.01)

    kmeans_centroid = KMeans_fit(50, 1000, 1e-4)
    decision_tree = decision_tree_fit(None, 2)

    return None


# output the reult of all the models
def getModelsResult(data):
    global ridge_model
    global logistic_model
    global logistic_bias
    global whole_train
    global kmeans_centroid
    global decision_tree


    result = []


    datasetWithNew = np.concatenate((whole_train, data), axis=0)
    
    result.append(grubbs_test(datasetWithNew))


    result.append(ridge_predict(data, ridge_model))
    result.append(logistic_predict(data, logistic_model, logistic_bias))
    result.append(kmeans_predict(data, kmeans_centroid))
    result.append(decision_tree_predict_single(data[0], decision_tree))
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






## Logistic regression
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    y = y -min(y)
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot



def logistic_fit(num_ite, learning_rate):
    global whole_train
    global whole_label

    m, n = whole_train.shape
    num_classes = len(np.unique(whole_label))  # Determine the number of classes

    # Initialize weights with small random values
    weights = np.random.randn(n, num_classes) * 0.01
    bias = np.zeros(num_classes)

    # One-hot encode the labels
    y_one_hot = one_hot_encode(whole_label, num_classes)

    for i in range(num_ite):
        # Compute linear model
        z = np.dot(whole_train, weights) + bias

        # Apply softmax function
        probabilities = softmax(z)

        # Compute gradients
        dw = (1 / m) * np.dot(whole_train.T, (probabilities - y_one_hot))
        db = (1 / m) * np.sum(probabilities - y_one_hot, axis=0)

        # Optional: Clip gradients to avoid explosion
        max_grad_value = 10
        dw = np.clip(dw, -max_grad_value, max_grad_value)
        db = np.clip(db, -max_grad_value, max_grad_value)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # # Print loss every 100 iterations
        # if i % 100 == 0:
        #     loss = -np.mean(np.sum(y_one_hot * np.log(np.clip(probabilities, 1e-9, 1 - 1e-9)), axis=1))
        #     print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return weights, bias


def logistic_predict(data, beta, bias):
    z = np.dot(data, beta) + bias
    probabilities = softmax(z)
    return np.argmax(probabilities, axis=1)[0]



## KMEans
def count_items(iterable):
    counts = {}
    for item in iterable:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts


def most_common(counts, n=None):
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts if n is None else sorted_counts[:n]



def initialize_centroids(X, n_clusters):
        indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        return X[indices]

def compute_distances(X, centroids, n_cluster):
    distances = np.zeros((X.shape[0], n_cluster))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def update_centroids(X, labels, n_cluster):
    centroids = np.zeros((n_cluster, X.shape[1]))
    for i in range(n_cluster):
        points = X[labels == i]
        centroids[i] = points.mean(axis=0) if len(points) > 0 else np.zeros(X.shape[1])
    return centroids


def KMeans_fit(n_cluster, max_iter, tol):
    global whole_train
    global whole_label

    labels = None

    X = np.array(whole_train)
    centroids = initialize_centroids(whole_train, n_cluster)

    for _ in range(max_iter):
        distances = compute_distances(whole_train, centroids, n_cluster)
        new_labels = np.argmin(distances, axis=1)

        if labels is not None and np.all(labels == new_labels):
            break  # Stop if labels haven't changed significantly
        labels = new_labels

        new_centroids = update_centroids(whole_train, labels, n_cluster)
        if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tol):
            break  # Stop if centroids converge
        centroids = new_centroids
        
    # Determine majority label for each centroid
    centroid_labels = []
    for cluster_id in range(n_cluster):
        # Get all points assigned to the current centroid
        cluster_points = whole_label[labels == cluster_id]

        # If the cluster is empty, assign None
        if len(cluster_points) == 0:
            majority_label = None
        else:
            # Determine the majority label
            majority_label = most_common(count_items(cluster_points))

        centroid_labels.append([centroids[cluster_id], int(majority_label[0][0])])
    
    # print(centroid_labels[0])
    return centroid_labels

def kmeans_predict(Data, centroids):
        centers = [item[0] for item in centroids]
        distances = compute_distances(Data, centers, len(centroids))
        return centroids[np.argmin(distances, axis=1)[0]][1]

    

## Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Constructor for the Node class.
        - feature: Index of the feature used for splitting
        - threshold: Threshold value for splitting
        - left: Left subtree
        - right: Right subtree
        - value: Leaf value for prediction
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value




def entropy(y):
    "Calculate entropy for a label array."
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(parent_entropy, left_y, right_y):
    "Calculate Information Gain."
    n = len(left_y) + len(right_y)
    weight_left = len(left_y) / n
    weight_right = len(right_y) / n
    return parent_entropy - (weight_left * entropy(left_y) + weight_right * entropy(right_y))

def split(X_column, threshold):
    "Split the data into left and right groups based on the threshold."
    left_indices = np.where(X_column <= threshold)
    right_indices = np.where(X_column > threshold)
    return left_indices[0], right_indices[0]

def best_split(X, y):
    "Find the best split."
    best_gain = -1
    best_feature = None
    best_threshold = None
    parent_entropy = entropy(y)

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_indices, right_indices = split(X[:, feature_idx], threshold)
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            left_y, right_y = y[left_indices], y[right_indices]
            gain = information_gain(parent_entropy, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

def build_tree(X, y, max_depth, min_samples_split, depth=0):
    "Recursively build the decision tree."
    num_samples, num_features = X.shape
    num_labels = len(np.unique(y))

    if (num_samples < min_samples_split or
            (max_depth is not None and depth >= max_depth) or
            num_labels == 1):
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)

    best_feature, best_threshold, best_gain = best_split(X, y)

    if best_gain == 0:
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)

    left_indices, right_indices = split(X[:, best_feature], best_threshold)
    left_subtree = build_tree(X[left_indices], y[left_indices], max_depth, min_samples_split, depth + 1)
    right_subtree = build_tree(X[right_indices], y[right_indices], max_depth, min_samples_split, depth + 1)

    return Node(feature=best_feature, threshold=best_threshold,
                left=left_subtree, right=right_subtree)

def most_common_label(y):
    "Return the most common class label."
    return np.bincount(y).argmax()

def decision_tree_fit(max_depth, min_samples_split):
    "Train the decision tree."
    global whole_train
    global whole_label

    root = build_tree(whole_train, whole_label.astype(int), max_depth, min_samples_split)
    return root




def decision_tree_predict_single(data, tree):
    if tree.value is not None:
        return tree.value

    if data[tree.feature] <= tree.threshold:
        return decision_tree_predict_single(data, tree.left)
    return decision_tree_predict_single(data, tree.right)



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
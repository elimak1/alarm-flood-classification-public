import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from methods.utils import multivar_binary

# Implements paper
# Matthieu Lucke, Moncef Chioua, Chriss Grimholt, Martin Hollender, Nina F. Thornhill,
# Advances in alarm data analysis with a practical application to online alarm flood classification,
# Journal of Process Control,
# Volume 79,
# 2019,
# Pages 56-71,
# ISSN 0959-1524,
# https://doi.org/10.1016/j.jprocont.2019.04.010.

class SVM_Classifier:
    def __init__(self, vocabulary, class_probability_threshold=0.5, max_lag = 10):
        """
        Initialize the SVM_Classifier with a vocabulary and maximum lag.
        
        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        class_probability_threshold (float): The threshold for class probability.
        max_lag (int): The maximum lag for calculating the Jaccard similarity index in seconds.
        """

        self.vocabulary = vocabulary
        self.lags = list(range(max_lag+1))
        self.class_probability_threshold = class_probability_threshold

        self.svm_model = None

    def fit(self, X, Y):
        """
        Train the SVM_Classifier with the given data.

        This method converts the data to a binary representation, calculates the Alarm coactivation matrix (ACM) for each flood, 
        and then trains the SVM model with the lower triangle of the ACMs.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        X_binary = multivar_binary(X, self.vocabulary)
        # For each flood calculate Alarm coactivation matrix ACM
        ACMs = []
        for f in range(X_binary.shape[0]):
            M = X_binary.shape[1]
            acm = np.zeros((M,M))
            for i in range(M):
                for j in range(i, M):
                    similarity = 0
                    if i != j:
                        similarity = jaccard_similarity_index(X_binary[f,i], X_binary[f,j], self.lags)
                    elif np.any(X_binary[f,i]> 0):
                        similarity = 1
                    acm[i,j] = similarity
                    acm[j,i] = similarity
            ACMs.append(acm)
        flood_features = np.array([get_lower_triangle(acm) for acm in ACMs])


        # Find rbf kernel function parameters C and gamma
        parameters = {'C':[0.9, 1,1.5,2],'gamma':[1,10,20,30] }
        #parameters = {'C':[1],'gamma':[10] }
        model = svm.SVC(decision_function_shape='ovo', probability=True, kernel="rbf", random_state=1)
        clf = GridSearchCV(model, parameters)
        clf.fit(flood_features, Y)
        best_model_idx= np.argmax(clf.cv_results_["mean_test_score"])
        C = clf.cv_results_["params"][best_model_idx]["C"]
        gamma = clf.cv_results_["params"][best_model_idx]["gamma"]
        self.svm_model = svm.SVC(decision_function_shape='ovo', probability=True, kernel="rbf",
                C=C, gamma=gamma, random_state=1)
        self.svm_model.fit(flood_features, Y)

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method converts the data to a binary representation, calculates the Alarm coactivation matrix (ACM) for each flood, 
        and then uses the trained SVM model to predict the labels for the lower triangle of the ACMs.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        X_binary = multivar_binary(X, self.vocabulary)
        # For each flood calculate Alarm coactivation matrix ACM
        ACMs = []
        for f in range(X_binary.shape[0]):
            M = X_binary.shape[1]
            acm = np.zeros((M,M))
            for i in range(M):
                for j in range(i, M):
                    similarity = 0
                    if i != j:
                        similarity = jaccard_similarity_index(X_binary[f,i], X_binary[f,j], self.lags)
                    elif np.any(X_binary[f,i]> 0):
                        similarity = 1
                    acm[i,j] = similarity
                    acm[j,i] = similarity
            ACMs.append(acm)
        flood_features = np.array([get_lower_triangle(acm) for acm in ACMs])
        probabilites = self.svm_model.predict_proba(flood_features) 
        predictions = [np.argmax(p)if np.max(p) > self.class_probability_threhold else -1 for p in probabilites  ]
        return predictions


def jaccard_similarity_index(series1, series2, lags):
    """
    Calculate the Jaccard similarity index between two binary series.

    This function calculates the Jaccard similarity index between two binary series by shifting series2 by each lag in lags, 
    and then comparing the shifted series with series1. The maximum similarity across all lags is returned.

    Parameters:
    series1: The first binary series.
    series2: The second binary series.
    lags: A list of lags for shifting series2.

    Returns:
    max_similarity: The maximum Jaccard similarity index across all lags.
    """
    similarities = []
    for lag in lags:
        s1 = np.hstack((series1,np.zeros(lag)))
        s2_lag = np.hstack((np.zeros(lag),series2))
        a = np.sum((s1 == s2_lag) & (s1 == 1))
        if a == 0:
            similarities.append(0)
            continue

        b = np.sum((s1 != s2_lag) & (s1 == 1))
        c = np.sum((s1 != s2_lag) & (s2_lag == 1))
        similarities.append(a/(a+b+c))
    return np.max(similarities)

def get_lower_triangle(matrix):
    """
    Get the lower triangle of a matrix.

    This function gets the lower triangle of a matrix, including the diagonal, and returns it as a 1D array.

    Parameters:
    matrix: The matrix.

    Returns:
    lower_triangle: The lower triangle of the matrix as a 1D array.
    """
    feats = []
    N, M = matrix.shape
    for i in range(N):
        for j in range(i+1):
            feats.append(matrix[i,j])
    return np.array(feats)
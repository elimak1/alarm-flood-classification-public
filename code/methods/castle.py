import numpy as np
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath('../../MultiRocket'))
from multirocket.multirocket_multivariate import transform, fit
from collections import Counter
from methods.utils import multivar_binary


# Implements G. Manca, M. Dix and A. Fay, "Convolutional Kernel-Based Transformation and Clustering of Similar Industrial Alarm Floods,"
# 2022 IEEE Eighth International Conference on Big Data Computing Service and Applications (BigDataService),
# Newark, CA, USA, 2022, pp. 161-166, doi: 10.1109/BigDataService55688.2022.00033.

# Clone repo from https://github.com/ChangWeiTan/MultiRocket to adjacent folder

N_FEATURES = 10000 # Number of features created with Multirocket
FEATURES_PER_KERNEL = 9
JACCARD_DISTANCE_THRESHOLD = 0.4
N_CLUSTERING_SOLUTIONS = 8

class CASTLE_Classifier:
    def __init__(self, vocab, n_features=N_FEATURES, n_models=N_CLUSTERING_SOLUTIONS, features_per_kernel=FEATURES_PER_KERNEL, jd_threshold=JACCARD_DISTANCE_THRESHOLD):
        """
        Initializes a Castle object with the given parameters.
        
        Parameters:
        vocab (dict): A dictionary mapping alarm numbers to indices.
        n_features (int): Number of features created with Multirocket.
        n_models (int): Number of models to use for classification.
        features_per_kernel (int): Number of features for each kernel in Multirocket.
        jd_threshold (float): Jaccard distance threshold for clustering.
        """
        self.n_features = n_features
        self.n_models = n_models
        self.features_per_kernel = features_per_kernel
        self.jd_threshold = jd_threshold

        self.vocab = vocab
        self.train_X = None
        self.labels = []
        self.trained_features = []
        self.base_multirocket_parameters = []
        self.diff1_multirocket_parameters = []
        self.pcas = []
        self.scalers = []

    def fit(self, X, Y):
        """
        Train classifier with given data
        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        """
        
        X_binary = multivar_binary(X, self.vocab)
        # Reset previous model
        self.train_X = X_binary 
        self.labels = Y
        self.trained_features = []
        self.base_multirocket_parameters = []
        self.diff1_multirocket_parameters = []
        self.pcas = []
        self.scalers = []

        X_binary_diff1 = np.diff(X_binary, 1)
        for i in range(self.n_models):
            print("Start training model", i+1)
            self.base_multirocket_parameters.append(fit(X_binary, N_FEATURES/2))
            self.diff1_multirocket_parameters.append(fit(X_binary_diff1, N_FEATURES/2))
            X_features = transform(X_binary, X_binary_diff1, self.base_multirocket_parameters[i], self.diff1_multirocket_parameters[i],  self.features_per_kernel)
            pca = PCA(n_components=0.99, svd_solver='full')
            X_pca_features = pca.fit_transform(X_features)
            self.pcas.append(pca)

             # standardize featurs
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pca_features)
            self.scalers.append(scaler)
            self.trained_features.append(X_scaled)

    def predict(self, X):
        """
        Predicts the labels for the given input data.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        if len(self.trained_features) != self.n_models:
            raise Exception("Model is not trained")
        X_binary = multivar_binary(X, self.vocab)

        N = X_binary.shape[0]
        M = self.train_X.shape[0]

        # Jaccard-distance post processing applied to all models
        def jaccard_distance(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            return 1 - (intersection / union)
    
        jaccard_distances = np.zeros((N,M))
        for n in range(N):
            for m in range(M):
                n_non_zero_rows = np.any(X_binary[n] != 0, axis=1)
                n_indices = np.where(n_non_zero_rows)[0]

                m_non_zero_rows = np.any( self.train_X[m] != 0, axis=1)
                m_indices = np.where(m_non_zero_rows)[0]

                dist = jaccard_distance(set(n_indices), set(m_indices))
                jaccard_distances[n,m] = dist

        jaccard_mask = jaccard_distances > self.jd_threshold

        model_predictions = np.zeros((N, self.n_models))

        X_binary_diff1 = np.diff(X_binary, 1)
        for i in range(self.n_models):
            print("Predicting with model", i+1)
            features = transform(X_binary, X_binary_diff1, self.base_multirocket_parameters[i], self.diff1_multirocket_parameters[i],  self.features_per_kernel)
            pca_features = self.pcas[i].transform(features)
            scaled_features = self.scalers[i].transform(pca_features)

            # Eauclidean distances to hisorical floods
            trained_features = self.trained_features[i]
            N = scaled_features.shape[0]
            M = trained_features.shape[0]
            distances = np.zeros((N,M))
            for n in range(N):
                for m in range(M):
                    dist = np.linalg.norm(scaled_features[n] - trained_features[m])
                    distances[n,m] = dist
    
            max_dist = np.max(distances)

            distances[jaccard_mask] = max_dist
            # 1 NN, with equality voting and unclassified if all at max distance
            for n in range(N):
                min_indices = []
                min_dist = max_dist
                for m in range(M):
                    dist = distances[n, m]
                    if dist < min_dist:
                        min_indices = [m]
                    elif dist == min_dist and dist != max_dist:
                        min_indices.append(m)
                if len(min_indices) == 0:
                    model_predictions[n, i] = -1
                else:
                    nearest_idx = Counter(min_indices).most_common(1)[0][0]
                    model_predictions[n, i] =self.labels.iloc[nearest_idx]

        # Take prediction which are in > 50% of the models
        # Otherwise unclassified
        def most_common_element_above_threshold(lst):
            counter = Counter(lst)
            for element, count in counter.most_common():
                if count > 0.5 * self.n_models:
                    return element
            return -1
        prediction = [most_common_element_above_threshold(preds) for preds in model_predictions]
        return prediction
    

import pandas as pd
import numpy as np
from methods.utils import multivar_binary

# Implements papaer
# G. Manca, M. Dix and A. Fay, "Clustering of Similar Historical Alarm Subsequences in Industrial Control Systems
# Using Alarm Series and Characteristic Coactivations," in IEEE Access, vol. 9, pp. 154965-154974, 2021, doi: 10.1109/ACCESS.2021.3128695.

JACCARD_DISTANCE_THRESHOLD = 0.4

class ASSAM_Classifier:
    def __init__(self, vocab, jd_threshold = JACCARD_DISTANCE_THRESHOLD):
        """
        Initialize the Assam class.

        Parameters:
        vocab (dict): The vocabulary used for text representation.
        jd_threshold (float): The Jaccard distance threshold for similarity.
        """

        self.vocab = vocab
        self.jd_threshold = jd_threshold

        self.train_X = None
        self.labels = []


        self.frequencies = None
        self.coac_frequencies = None
        self.idf = None
        self.idf_coac = None

        self.tfidf_matrix = None

        self.tfidf_coac_matrix = None

    def fit(self, X, Y):
        """
        Train classifier with given data
        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        """
        self.train_X = X
        self.labels = Y


        X_binary = multivar_binary(X, self.vocab)

        # Calculated term-frequencies
        frequencies = np.sum(X_binary, axis=2)
        self.frequencies = frequencies
        tf = np.zeros(frequencies.shape)
        for flood in range(frequencies.shape[0]):
            for alarm in range(frequencies.shape[1]):
                max_alarm_freq =  np.max(frequencies[:, alarm])
                if max_alarm_freq > 0:
                    tf[flood, alarm] = frequencies[flood, alarm] / np.max(frequencies[:, alarm])
        idf = np.zeros(frequencies.shape[1])
        for alarm in range(frequencies.shape[1]):
            if np.count_nonzero(frequencies[:, alarm]) > 0:
                idf[alarm] = np.log(frequencies.shape[0] / np.count_nonzero(frequencies[:, alarm]))
        self.idf = idf

        self.tfidf_matrix= tf * idf

        # Calculate coactivations tfidf
        coac_frequencies = np.einsum('aik,ajk->aij', X_binary, X_binary)
        self.coac_frequencies = coac_frequencies
        tf_coac = np.zeros(coac_frequencies.shape)
        for flood in range(frequencies.shape[0]):
            for a1 in range(frequencies.shape[1]):
                for a2 in range(frequencies.shape[1]):
                    m = np.max(coac_frequencies[:, a1, a2])
                    if m > 0:
                        tf_coac[flood, a1, a2] = coac_frequencies[flood, a1, a2] / m
        idf_coac = np.zeros((coac_frequencies.shape[1], coac_frequencies.shape[2]))
        for a1 in range(coac_frequencies.shape[1]):
            for a2 in range(coac_frequencies.shape[2]):
                m = np.count_nonzero(coac_frequencies[:, a1, a2])
                if m > 0:
                    idf_coac[a1, a2] = np.log(coac_frequencies.shape[0] / m)
        self.idf_coac = idf_coac
        tfidf_coac = tf_coac * idf_coac
        self.tfidf_coac_matrix = tfidf_coac.reshape((tfidf_coac.shape[0], -1))

    def predict(self, X):
        """
        Predicts the labels for the given input data.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        
        # Jaccard-distance post processing applied to distances
        def jaccard_distance(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            return 1 - (intersection / union)

        train_alarms_by_flood = self.train_X.groupby('flood_id').apply(lambda x: x["alarmNumber"].map(str))
        pred_alarms_by_flood = X.groupby('flood_id').apply(lambda x: x["alarmNumber"].map(str))
        N = len(self.train_X.groupby('flood_id'))
        M = len(X.groupby('flood_id'))
        jaccard_distances = np.zeros((N,M))
        for i, n in enumerate(train_alarms_by_flood.index.get_level_values(0).unique()):
            for j, m in enumerate(pred_alarms_by_flood.index.get_level_values(0).unique()):
                dist = jaccard_distance(set(train_alarms_by_flood.loc[n,:]), set(pred_alarms_by_flood.loc[m,:]))
                jaccard_distances[i,j] = dist

        jaccard_mask = jaccard_distances > self.jd_threshold


        # Calculate features
        X_binary = multivar_binary(X, self.vocab)
        pred_frequencies = np.sum(X_binary, axis=2)
        pred_tf = np.zeros(pred_frequencies.shape)
        for flood in range(pred_frequencies.shape[0]):
            for alarm in range(pred_frequencies.shape[1]):
                if np.max(self.frequencies[:, alarm]) > 0:
                    pred_tf[flood, alarm] = pred_frequencies[flood, alarm] / np.max(self.frequencies[:, alarm])
        tfidf_matrix = pred_tf * self.idf

        # Calculate coactivations tfidf
        coac_frequencies = np.einsum('aik,ajk->aij', X_binary, X_binary)
        tf_coac = np.zeros(coac_frequencies.shape)
        for flood in range(coac_frequencies.shape[0]):
            for a1 in range(coac_frequencies.shape[1]):
                for a2 in range(coac_frequencies.shape[2]):
                    m = np.max(coac_frequencies[:, a1, a2])
                    if m > 0:
                        tf_coac[flood, a1, a2] = coac_frequencies[flood, a1, a2] / m
        tfidf_coac = tf_coac * self.idf_coac
        tfidf_coac_matrix = tfidf_coac.reshape((tfidf_coac.shape[0], -1))

        tfidf_distances = np.zeros((N,M))
        coac_distances = np.zeros((N,M))

        for n in range(N):
            for m in range(M):
                dist = np.linalg.norm(self.tfidf_matrix[n] - tfidf_matrix[m])
                tfidf_distances[n,m] = dist
                coac_dist = np.linalg.norm(self.tfidf_coac_matrix[n] - tfidf_coac_matrix[m])
                coac_distances[n,m] = coac_dist

        tfidf_distances[jaccard_mask] = np.max(tfidf_distances)
        coac_distances[jaccard_mask] = np.max(coac_distances)

        tfidf_nearest = np.argmin(tfidf_distances, axis=0)
        coac_nearst = np.argmin(coac_distances, axis=0)
        # If different labels are predicted, no preidction is made
        tfidf_pred = self.labels.iloc[tfidf_nearest].to_numpy()
        coac_pred = self.labels.iloc[coac_nearst].to_numpy()
        disagreement_mask = tfidf_pred != coac_pred
        tfidf_pred[disagreement_mask] = -1
        return tfidf_pred

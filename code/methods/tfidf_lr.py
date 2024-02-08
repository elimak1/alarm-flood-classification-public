import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from methods.assam import multivar_binary

# Implements paper:
# H. S. Alinezhad, J. Shang and T. Chen, "Open Set Online Classification of Industrial Alarm Floods With Alarm Ranking,"
# in IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-11, 2023, 
# Art no. 3500811, doi: 10.1109/TIM.2022.3232617.

class TFIDF_LR_Classifier():
    def __init__(self, vocab, use_confidence_thresholds=True):
            """
            Initialize the TFIDF_LR_Classifier.

            Parameters:
            vocab (dict): A dictionary mapping alarm strings to their corresponding indices.
            use_confidence_thresholds (bool, optional): A flag indicating whether to use confidence thresholds for classification.
                Defaults to True.
            """
            self.vocab = vocab
            self.reverse_vocab = {i: alarm for alarm, i in self.vocab.items()}
            self.frequencies = None
            self.idf = None
            self.t_max = None
            self.classifier = None
            self.enable_confidence_thresholds = use_confidence_thresholds
            self.confidence_thresholds = None
    
    def fit(self, X, Y):
        """
        Train the TFIDF_LR_Classifier with the given data.

        This method converts the alarm floods into a matrix of TF-IDF features, calculates a time-weighted matrix for the features, 
        and then trains the LR classifier with the product of the TF-IDF matrix and the time-weighted matrix.

        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        """
        X_binary = multivar_binary(X, self.vocab)

        # Calculate tfidf
        frequencies = np.sum(X_binary, axis=2)
        self.frequencies = frequencies
        tf = np.zeros(frequencies.shape)
        for flood in range(frequencies.shape[0]):
            for alarm in range(frequencies.shape[1]):
                if np.max(frequencies[:, alarm]) > 0:
                    tf[flood, alarm] = frequencies[flood, alarm] / np.max(frequencies[:, alarm])
        idf = np.zeros(frequencies.shape[1])
        for alarm in range(frequencies.shape[1]):
            if np.count_nonzero(frequencies[:, alarm]) > 0:
                idf[alarm] = np.log(frequencies.shape[0] / np.count_nonzero(frequencies[:, alarm]))
        self.idf = idf
        tfidf = tf * idf

        t_max = X["startTimestamp"].max()
        self.t_max = t_max
        tw_matrix = np.zeros(tfidf.shape)
        for i, flood in enumerate(X["flood_id"].unique()):
            for alarm in range(tfidf.shape[1]):
                alarm_activations = X[(X["alarmNumber"] == self.reverse_vocab[alarm]) & (X["flood_id"] == flood)]
                if len(alarm_activations) > 0:
                    tw_matrix[i, alarm] = np.log(t_max/(np.min(alarm_activations["startTimestamp"] + 1)))

        W = tfidf * tw_matrix
        self.classifier = LogisticRegression(C=5).fit(W, Y)

        # Get 95% confidence thresholds
        # By fitting gaussion distribution to the training class probabilites

        class_probabilities = self.classifier.predict_proba(W)
        if not self.enable_confidence_thresholds:
            self.confidence_thresholds = [0 for i in range(0, Y.max()+1)]
            return
        confidence_thresholds = []
        for i in range(0, Y.max()+1):
            mask = Y == i
            positive_probabilities = class_probabilities[mask, i]
            # Generated mirrored side of the distribution
            probs_mirrored = 2 - positive_probabilities
            probs = np.append(positive_probabilities,probs_mirrored)
            threshold = np.max([0.5 ,1 - 1.96*np.std(probs)])
            confidence_thresholds.append(threshold)
        self.confidence_thresholds = confidence_thresholds

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method converts the alarm floods into a matrix of TF-IDF features, calculates a time-weighted matrix for the features, 
        and then uses the LR classifier to predict the labels for the product of the TF-IDF matrix and the time-weighted matrix. 
        If confidence thresholds are enabled, it only predicts a label if the predicted probability for that label is above the threshold.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        pred_X = X.copy()
        X_pred_binary = multivar_binary(pred_X, self.vocab)

        frequencies_pred = np.sum(X_pred_binary, axis=2)
        tf_pred = np.zeros(frequencies_pred.shape)
        for flood in range(frequencies_pred.shape[0]):
            for alarm in range(frequencies_pred.shape[1]):
                if np.max(self.frequencies[:, alarm]) > 0:
                    tf_pred[flood, alarm] = frequencies_pred[flood, alarm] / np.max(self.frequencies[:, alarm])

        tfidf_pred = tf_pred * self.idf
        tw_matrix_pred = np.zeros(tfidf_pred.shape)
        for i, flood in enumerate(X["flood_id"].unique()):
            for alarm in range(tfidf_pred.shape[1]):
                alarm_activations = X[(X["alarmNumber"] == self.reverse_vocab[alarm]) & (X["flood_id"] == flood)]
                if len(alarm_activations) > 0:
                    tw_matrix_pred[i, alarm] = np.log(self.t_max/(np.min(alarm_activations["startTimestamp"] + 1)))
        W = tfidf_pred * tw_matrix_pred
        predicted_probs = self.classifier.predict_proba(W)
        predicted_labels = []
        for i in range(len(predicted_probs)):
            label = -1
            max_prob = 0

            for l in range(len(predicted_probs[i])):
                if predicted_probs[i, l] > self.confidence_thresholds[l] and predicted_probs[i, l] > max_prob:
                    max_prob = predicted_probs[i, l]
                    label = l
            predicted_labels.append(label)
        return predicted_labels
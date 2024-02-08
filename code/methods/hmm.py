import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os
import sys


## Implements paper
# Joshiba Ariamuthu Venkidasalapathy, Costas Kravaris,
# Hidden Markov model based fault diagnoser using binary alarm signals with an analysis on distinguishability,
# Computers & Chemical Engineering,
# Volume 160,
# 2022,
# 107689,
# ISSN 0098-1354,
# https://doi.org/10.1016/j.compchemeng.2022.107689.

H = 4 # number of hidden states

class HMM_Classifier:
    def __init__(self, vocabulary, n_hidden=H, log_prob_threshold=-1000):
        """
        Initialize the HMM_Classifier.

        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        n_hidden (int): The number of hidden states in the HMM.
        log_prob_threshold (float): If the log probability of a sample is lower than this threshold, the sample is classified as unknown.
        """

        self.vocabulary = vocabulary
        self.n_hidden = n_hidden
        self.log_prob_threshold = log_prob_threshold

        self.models = {}  # Own model for each fault

    def fit(self, X, Y):
        """
        Train the HMM_Classifier with the given data.

        This method encodes the alarm numbers in the data using the vocabulary, and then fits an HMM for each fault type 
        in the data. The HMMs are initialized multiple times, and the initialization that results in the highest accuracy 
        (measured using 5-fold cross-validation) is chosen.

        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        """

        X_encoded = X.groupby('flood_id')["alarmNumber"].apply(lambda alarms: [[self.vocabulary[a]] for a in alarms])

        lengths = X_encoded.map(len)
        labels = np.unique(Y)
        # Suppress error messages from hmmlearn when fitting with too few samples¨
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        for i in labels:
            indices = np.where(Y==i)[0]
            class_floods = np.concatenate(X_encoded.iloc[indices].to_list())
            class_lengths = lengths.iloc[indices].to_list()
            model = hmm.CategoricalHMM(n_components=self.n_hidden, n_iter=100)
            model.fit(class_floods, class_lengths)
            self.models[i] = model

        sys.stderr = stderr

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method encodes the alarm numbers in the data using the vocabulary, and then uses each HMM to score each group in the data. 
        The label of the HMM that gives the highest score is chosen as the prediction for the group.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.

        Raises:
        Exception: If the model has not been fitted.
        """
        if len(self.models) == 0:
            raise Exception("Model not fitted")

        X_encoded = X.groupby('flood_id')["alarmNumber"].apply(lambda alarms: [[self.vocabulary[a]] for a in alarms])
        predictions = []
        for i in range(len(X_encoded)):
            probs = {}
            for label, model in self.models.items():
                try:
                    log_prob = model.score(X_encoded.iloc[i])
                except:
                    # Sample contains alarms not used in training
                    log_prob = -np.inf
                probs[label] = log_prob
            if max(probs.values()) > self.log_prob_threshold:
                predictions.append(max(probs, key=probs.get))
            else:
                predictions.append(-1)
        return predictions


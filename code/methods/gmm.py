import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

## Implements paper
# CH. S. Alinezhad, J. Shang and T. Chen, "Early Classification of Industrial Alarm Floods Based on Semisupervised Learning,"
# in IEEE Transactions on Industrial Informatics, vol. 18, no. 3, pp. 1845-1853, March 2022, doi: 10.1109/TII.2021.3081417.
CLASSIFICATION_PROBABILITY_THRESHOLD = 0.5

class GMM_Classifier:
    def __init__(self, vocabulary, classification_probability_threshold=CLASSIFICATION_PROBABILITY_THRESHOLD):
            """
            Initialize the GMM_Classifier.

            Parameters:
            vocabulary (dict): A dictionary mapping alarm numbers to indices in the feature vector.
            classification_probability_threshold (float, optional): The threshold for classifying a data point as belonging to a class.
                Defaults to CLASSIFICATION_PROBABILITY_THRESHOLD.
            """
            
            self.vocabulary = vocabulary
            self.classification_probability_threshold = classification_probability_threshold

            self.ac = None

            self.labels = []
            self.gmm = None

    def fit(self, X, Y):
        """
        Train the GMM_Classifier with the given data.

        This method calculates the attenuation coefficient, creates feature vectors for each group in the data, 
        determines the optimal number of components for the Gaussian Mixture Model, and fits the GMM with the feature vectors.

        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        """
        # Determine attenuation coefficient
        # Get mean last trigger time of an alarm in historical alarm floods
        med_lifetime = X.groupby("flood_id")["startTimestamp"].max().median()
        self.ac = 1/med_lifetime

        flood_feature_vectors = X.groupby('flood_id').apply(lambda group: self.create_feature_vector(group))
        flood_feature_vectors = np.array([vector for vector in flood_feature_vectors])

        # Find optimal number of components for GMM
        max_components = 25
        s_scores = []
        for n_components in range(2, max_components+1):
            gm  = GaussianMixture(n_components=n_components, n_init=20)
            clusters = gm.fit_predict(flood_feature_vectors)
            s_score = silhouette_score(flood_feature_vectors, clusters, metric="euclidean")
            s_scores.append(s_score)

        c = np.argmax(s_scores) + 2
        print("Chosen number of components:", c)
        self.gmm = GaussianMixture(n_components=c, n_init=8)
        self.gmm.fit(flood_feature_vectors)

        # Solve cluster labels
        class_probabilites = self.gmm.predict_proba(flood_feature_vectors)
        # label using the most likely sample
        best_samples = [np.argmax(class_probabilites[:,i]) for i in range(class_probabilites.shape[1])]
        self.labels = Y.iloc[best_samples]

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method creates a feature vector for each group in the data, and then uses the Gaussian Mixture Model to predict 
        the probabilities of each group belonging to each class. If the maximum probability is above the classification 
        probability threshold, the group is classified as belonging to the class with the maximum probability.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        flood_feature_vectors = X.groupby('flood_id').apply(lambda group: self.create_feature_vector(group))
        flood_feature_vectors = np.array([vector for vector in flood_feature_vectors])
        prediction_probabilites = self.gmm.predict_proba(flood_feature_vectors)
        predicted_labels = [self.labels.iloc[np.argmax(probs)] if max(probs) > CLASSIFICATION_PROBABILITY_THRESHOLD \
                            else -1 for probs in prediction_probabilites ]
        return predicted_labels

    def create_feature_vector(self, group):
        """
        Creates a feature vector from the given group of events.

        This function iterates over each event in the group, and for each event, it calculates an exponential decay 
        based on the start timestamp of the event and an attenuation coefficient.

        Parameters:
        group: A DataFrame representing a group of events. Each event is expected to have an 'alarmNumber' and a 'startTimestamp'.

        Returns:
        features: The created feature vector. This is a 1D numpy array where each element corresponds to an alarm number in the vocabulary.
        """
        features = np.zeros((len(self.vocabulary)))
        for i, event in group.iterrows():
            idx = self.vocabulary[event["alarmNumber"]]
            features[idx] = max([np.exp(-self.ac*event["startTimestamp"]),features[idx]])
        return features
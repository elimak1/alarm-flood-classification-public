import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Implements paper
# Sylvie Charbonnier, Nabil Bouchair, Philippe Gayet,
# Fault template extraction to assist operators during industrial alarm floods,
# Engineering Applications of Artificial Intelligence,
# Volume 50,
# 2016,
# Pages 32-44,
# ISSN 0952-1976,
# https://doi.org/10.1016/j.engappai.2015.12.007.

# Default parameters from paper
MATCH = 1
MISMATCH = 1
GAP = 0

WEIGHT_IMPORTANCE = 5
FAULT_SIMILARITY_THRESHOLD = 0.1

class NW_Classifier:
    def __init__(self, vocabulary, match=MATCH, mismatch=MISMATCH, gap=GAP, weight_importance=WEIGHT_IMPORTANCE, fault_similarity_threshold=FAULT_SIMILARITY_THRESHOLD):
        """
        Initialize the NW_Classifier.

        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        match (int): The score for a match in the NW algorithm.
        mismatch (int): The score for a mismatch in the NW algorithm.
        gap (int): The score for a gap in the NW algorithm.
        weight_importance (float): The importance of the weighting vector in the similarity score.
        fault_similarity_threshold (float): The threshold for considering two faults as similar.
        """
        self.vocabulary = vocabulary

        self.fault_templates = []
        self.weighting_vector = None
        self.n_classes = None
        self.labels = None

        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.weight_importance = weight_importance
        self.fault_similarity_threshold = fault_similarity_threshold
        
    def fit(self, X, Y, truncate_samples=None):
        """
        Train the NW_Classifier with the given data.

        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        truncate_samples (int): The maximum number of samples to use for each class. If None, all samples are used.
        """
        X_sentences = X.groupby("flood_id").apply(lambda flood: flood["alarmNumber"].unique())
        self.n_classes = Y.nunique()


        for label in range(self.n_classes+1):
            class_indices = np.where(Y==label)[0]
            if truncate_samples != None and type(truncate_samples) == int:
                class_indices = class_indices[:truncate_samples] 
            N = len(class_indices)
            sequences = X_sentences.iloc[class_indices].to_list()
            clusters = {index: [index] for index in range(N)}
            similarity_scores = {}
            for i in range(N):
                for j in range(i+1, N):
                    a1, a2, a1_gaps, a2_gaps= nw(sequences[i], sequences[j], self.match, self.mismatch, self.gap)
                    similarity_scores[(i,j)] = get_similarity_score(a1, a2)
            while N > 1:
                i1, i2 = get_most_similar_sequences(similarity_scores)
                # Update sequecces
                a1, a2, a1_gaps, a2_gaps = nw(sequences[i1], sequences[i2], self.match, self.mismatch, self.gap)

                # Add gaps to the sequneces in clusters
                for seq_idx in clusters[i1]:
                    if seq_idx == i1:
                        sequences[seq_idx] = a1
                    else:
                        for gap_idx in a1_gaps:
                            sequences[seq_idx].insert(gap_idx, "-")
                for seq_idx in clusters[i2]:
                    if seq_idx == i2:
                        sequences[seq_idx] = a2
                    else:
                        for gap_idx in a2_gaps:
                            sequences[seq_idx].insert(gap_idx, "-")
                
                clusters[i1].extend(clusters[i2])
                new_cluster = clusters[i1]
                for k in new_cluster:
                    clusters[k] = new_cluster
                # All clusters are combined
                if len(clusters[i1]) == N:
                    break
                # Calculate new similarities and delete those which are in same cluster
                to_delete = []
                for key in similarity_scores:
                    if key[0] in clusters[i1] and key[1] in clusters[i1]:
                        to_delete.append(key)
                    elif key[0] in clusters[i1] or key[1] in clusters[i1]:
                        a1, a2, _, _ = nw(sequences[key[0]], sequences[key[1]], self.match, self.mismatch, self.gap)
                        similarity_scores[key] = get_similarity_score(a1,a2)
                for key in to_delete:
                    similarity_scores.pop(key)
            # Now all sequneces are aligned
            self.fault_templates.append(extract_fault_template(sequences))

        # Calculate weights for each alarm
        flood_feature_vectors = X.groupby('flood_id').apply(lambda g: create_flood_features(g, self.vocabulary))
        fault_features = []
        for i in range(self.n_classes+ 1):
            class_indices = np.where(Y==i)[0]
            k = len(class_indices)
            features = np.zeros(len(self.vocabulary), dtype=int)
            mask = np.sum(flood_feature_vectors.iloc[class_indices], axis=0) > k/2
            features[mask] = 1
            fault_features.append(features)
        fault_features = np.array(fault_features)
        weighting_vector = np.zeros((self.n_classes+ 1, len(self.vocabulary)))
        for i in range(self.n_classes):
            for j in range(len(self.vocabulary)):
                class_indices = np.where(Y==i)[0]
                other_indices = np.where(Y!=i)[0]
                k = len(class_indices)
                p = fault_features[i,j]
            
                vs = np.array(flood_feature_vectors.iloc[class_indices].to_list())[:,j]
                a = np.sum(vs == p)/k

                vs_other = np.array(flood_feature_vectors.iloc[other_indices].to_list())[:,j]
                b = np.sum(vs_other == p)/len(other_indices)
                w = (2*a -1)*(1 - b)
                weighting_vector[i,j] = w
        self.weighting_vector = weighting_vector
        self.labels = Y

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method groups the data by flood id, aligns each group with each fault template using the NW algorithm, 
        and then calculates the similarity score for each alignment. The label of the fault template with the highest 
        similarity score is chosen as the prediction for the group.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        X_sentences = X.groupby("flood_id").apply(lambda flood: flood["alarmNumber"].unique())

        predictions = []
        for group in X["flood_id"].unique():
            # align flood to each fault template
            flood_scores = []
            for i in range(self.n_classes+1):
                a1, a2, _, _ = nw(X_sentences[group], self.fault_templates[i])
                flood_scores.append(get_weighted_similarity_score(a1, a2,  self.weighting_vector[i], self.vocabulary, self.weight_importance))
            if np.max(flood_scores) < self.fault_similarity_threshold:
                predictions.append(-1)
            else:
                predictions.append(np.argmax(flood_scores))
        return predictions
    
    def non_significant_alarms(self, signficance_threshold=0.01):
        """
        Identify the non-significant alarms.

        This method calculates the maximum weight for each alarm in the vocabulary, and then identifies the alarms 
        whose maximum weight is less than the significance threshold.

        Parameters:
        significance_threshold: The threshold for considering an alarm as significant.

        Returns:
        not_significant: A list of non-significant alarms.
        """
        max_weights = np.max(self.weighting_vector,axis=0)
        not_significant = np.array(list(self.vocabulary))[np.where(max_weights < signficance_threshold)[0]]
        return not_significant
    
    def distinquish_faults(self, i,j):
        """
        Check if two faults can be distinguished.

        This method checks if there is at least one alarm that appears more than 50% of the time in one fault and never in the other fault. 
        If such an alarm exists, the two faults can be distinguished.

        Parameters:
        i: The index of the first fault.
        j: The index of the second fault.

        Returns:
        weights: A list of weights for each alarm in the vocabulary. The weight of an alarm is 0 if the alarm can be used to distinguish the two faults, and 1 otherwise.
        """
        weights = np.zeros(len(self.vocabulary))
        for vi in range(len(self.vocabulary)):
            class_indices = np.where(self.labels==i)[0]
            other_indices = np.where(self.labels==j)[0]
            k = len(class_indices)
            p = self.fault_features[i,vi]

            vs = np.array(self.flood_feature_vectors.iloc[class_indices].to_list())[:,vi]
            a = np.sum(vs == p)/k
            vs_other = np.array(self.flood_feature_vectors.iloc[other_indices].to_list())[:,vi]
            b = np.sum(vs_other == p)/len(other_indices)
            w = (2*a -1)*(1 - b)
            weights[vi] = w
        if np.max(weights) == 1:
            print(f"fault {i} can be distinquished from fault {j}")
            return True
        else:
            print(f"fault {i} can be confused with {j}, max weight is {np.max(weights)}")
            return False
        
    def plot_distinquishability(self):
        """
        Plot the distinguishability of the faults.

        This method plots the maximum weight for each fault. A smaller maximum weight indicates better distinguishability.

        Parameters:
        None

        Returns:
        None
        """
        print("Smaller maximal weight indicates better distinquishability")
        plt.xlabel('Fault number')
        plt.ylabel('Maximal weight')
        plt.grid(True)
        plt.scatter(range(self.n_classes+1),np.max(self.weighting_vector, axis=1), zorder=3)
        plt.show()

def create_flood_features(flood, vocab):
    """
    Create a binary representation of a flood.

    This function creates a binary representation of a flood, where each element indicates whether a particular alarm appears in the flood.

    Parameters:
    flood: A DataFrame representing a flood. Each row represents an event.
    vocab: A dictionary mapping alarm numbers to indices.

    Returns:
    features: A binary representation of the flood.
    """
    # Create a binary representation 1, indicating if alarm appears in a flood and 0 otherwise
    features = np.zeros((len(vocab)), dtype=int)
    for i, event in flood.iterrows():
        idx = vocab[event["alarmNumber"]]
        features[idx] = 1
    return features


def extract_fault_template(sequences):
    """
    Extract a fault template from a list of sequences.

    This function extracts a fault template from a list of sequences by choosing the most common event at each position.

    Parameters:
    sequences: A list of sequences.

    Returns:
    template: A fault template.
    """
    template = []
    df = pd.DataFrame(sequences)
    N, M = df.shape
    for i in range(M):
        counts = df.iloc[:,i].value_counts()
        if counts.index[counts.argmax()] != "-":
            template.append(counts.index[counts.argmax()])
    return template

def nw(x, y, match = 1, mismatch = 1, gap = 1):
    """
    Calculate the optimal alignment of two sequences using the Needleman-Wunsch (NW) algorithm.

    This function calculates the optimal alignment of two sequences using the NW algorithm. The alignment is scored based on matches, mismatches, and gaps.

    Parameters:
    x: The first sequence.
    y: The second sequence.
    match: The score for a match.
    mismatch: The score for a mismatch.
    gap: The score for a gap.

    Returns:
    tuple with the following elements:
    rx: A list representing the aligned sequence of x.
    ry: A list representing the aligned sequence of y.
    rx_new_gap_indices: A list representing the indices of new gaps in the aligned sequence of x.
    ry_new_gap_indices: A list representing the indices of new gaps in the aligned sequence of y.
    """
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j] and x[i] != "-":
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    rx_new_gap_indices = []
    ry_new_gap_indices = []
    counter = 0

    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            ry_new_gap_indices.append(counter)
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            rx_new_gap_indices.append(counter)
            j -= 1
        counter -=1

    x_gap_idx = np.array(rx_new_gap_indices) + (len(rx) - 1)
    y_gap_idx = np.array(ry_new_gap_indices) + (len(ry) - 1)
    # Reverse the strings.
    return rx[::-1], ry[::-1], sorted(x_gap_idx), sorted(y_gap_idx)

def get_similarity_score(seq1, seq2):
    """
    Calculate the similarity score between two sequences.

    This function calculates the similarity score between two sequences by comparing each element in the sequences. 
    The score is the ratio of matching elements to the total length of the sequences.

    Assumes that the two sequences have the same length.

    Parameters:
    seq1: The first sequence.
    seq2: The second sequence.

    Returns:
    score: The similarity score between the two sequences.
    """
    score = 0
    for i in range(len(seq1)):
        a = seq1[i]
        b = seq2[i]
        if a == b and a != "-":
            score+=1
    return score/len(seq1)

def get_most_similar_sequences(similarities):
    """
    This function finds the most similar sequences by finding the key corresponding to highest value in a dictionary.

    Parameters:
    similarities: A dictionary where the keys are pairs of sequences and the values are their similarity scores.

    Returns:
    most_similar: The pair of sequences with the highest similarity score.
    """
    return max(similarities, key=similarities.get)

def get_weighted_similarity_score(seq1, template, weights, vocab, a=WEIGHT_IMPORTANCE):
    """
    Calculate the weighted similarity score between a sequence and a template.

    This function calculates the weighted similarity score between a sequence and a template by comparing each element in the sequence 
    with the corresponding element in the template. The score is the ratio of the sum of the weights of the matching elements to the sum of the weights of all elements.

    Parameters:
    seq1: The sequence.
    template: The template.
    weights: A list of weights for each alarm in the vocabulary.
    vocab: A dictionary mapping alarm numbers to indices.
    a: The importance of the weights in the similarity score.

    Returns:
    score: The weighted similarity score between the sequence and the template.
    """
    score = 0
    for i in range(len(seq1)):
        alarm = seq1[i]
        template_alarm = template[i]
        if alarm == template_alarm and alarm != "-":
            score+= 1 + a*weights[vocab[alarm]]

    denominator = 0
    for i in range(len(seq1)):
        alarm = template[i]
        if alarm != "-":
            denominator+= 1 + a*weights[vocab[alarm]]
        else:
            denominator +=1
    return score/denominator
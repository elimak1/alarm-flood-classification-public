import pandas as pd
import numpy as np
from enum import IntEnum

# Implements paper:
# Cheng, Y. et al. (2013) Pattern matching of alarm flood sequences by a modified Smith–Waterman algorithm.
# Chemical engineering research & design. [Online] 91 (6), 1085–1094.

MU = -0.6
GAP_PENALTY = -0.4

class SW_Classifier():
    def __init__(self, vocabulary, similarity_threshold=0.5, gap_penalty=GAP_PENALTY, mu=MU):
        """
        Initialize the SW_Classifier with a vocabulary.

        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        similarity_threshold (float, optional): The similarity threshold for classifying alarms. Defaults to 0.5.
        gap_penalty (float, optional): The penalty for introducing a gap in the alignment. Defaults to GAP_PENALTY.
        mu (float, optional): The scaling factor for the similarity score. Defaults to MU.
        """
        self.vocabulary = vocabulary
        self.labels = None

        self.train_tw1 = None
        self.train_tw2 = None
        self.similarity_threshold = similarity_threshold
        self.gap_penalty = gap_penalty
        self.mu = mu

    def fit(self, X, Y):
        """
        Train the SW_Classifier with the given data.

        This method creates two time-weighted matrices for the data and stores them for later use in prediction.

        Parameters:
        X: A DataFrame with columns 'flood_id' and 'alarmNumber'. Each row represents an event.
        Y: The labels for the data. Each label represents a fault type.

        Returns:
        None
        """
        td_matrices = X.groupby("flood_id").apply(self.create_tw_matrix)
        self.train_tw1 = td_matrices.apply(lambda x: np.array([np.array([weighting_function_1(j) for j in i]) for i in x]))
        self.train_tw2 = td_matrices.apply(lambda x: np.array([np.array([weighting_function_2(j) for j in i]) for i in x]))
        self.labels = Y


    def predict(self, X):
        """
        Predict the labels for the given data.

        This method creates two time-weighted matrices for the data, calculates the similarity score between each matrix and the stored matrices from training, 
        and then chooses the label of the most similar matrix as the prediction.

        Parameters:
        X: A DataFrame with columns 'flood_id' and 'alarmNumber'. Each row represents an event.

        Returns:
        predictions: A list of predicted labels for the data. Each label represents a fault type.
        """
        td_matrices = X.groupby("flood_id").apply(self.create_tw_matrix)
        tw1 = td_matrices.apply(lambda x: np.array([np.array([weighting_function_1(j) for j in i]) for i in x]))
        tw2 = td_matrices.apply(lambda x: np.array([np.array([weighting_function_2(j) for j in i]) for i in x]))

        N = len(tw1)
        M = len(self.train_tw1)
        predictions = np.zeros(N)
        for i in range(N):
            similarity_scores = np.zeros(M)
            for j in range(M):
                score = flood_similarity_score(tw1.iloc[i], tw2.iloc[i], self.train_tw1.iloc[j], self.train_tw2.iloc[j], self.gap_penalty, self.mu)
                similarity_scores[j] = score
            nearest_neighbor = np.argmax(similarity_scores) if np.max(similarity_scores) > self.similarity_threshold else -1
            predictions[i] = self.labels.iloc[nearest_neighbor]
        return predictions
    
    def create_tw_matrix(self, flood):
        """
        Create a time-weighted matrix for a flood.

        This function creates a time-weighted matrix for a flood, where each element indicates the time of the corresponding alarm in the flood.

        Parameters:
        flood: A DataFrame representing a flood. Each row represents an event.

        Returns:
        tw_matrix: A time-weighted matrix for the flood.
        """
        tw_matrix = np.full((len(self.vocabulary), len(flood)), np.inf)
        
        for alarm_num, i in self.vocabulary.items():
            for j in range(len(flood)):
                flood_alarm_ts = flood.iloc[j]["startTimestamp"]
                matches = flood[flood["alarmNumber"] == alarm_num]["startTimestamp"]
                if len(matches) == 0:
                    continue
                time_dist = np.min(np.abs(matches - flood_alarm_ts))
                tw_matrix[i,j] = time_dist
        return tw_matrix


def weighting_function_1(x, sigma=0.5):
    """
    Calculate the weight of a time distance using a Gaussian function.

    This function calculates the weight of a time distance using a Gaussian function with standard deviation sigma.

    Parameters:
    x: The time distance.
    sigma: The standard deviation of the Gaussian function.

    Returns:
    weight: The weight of the time distance.
    """
    return np.exp(-x**2/2*sigma**2)

def weighting_function_2(x):
    """
    Calculate the weight of a time distance using a step function.

    This function calculates the weight of a time distance using a step function, where the weight is 1 if the time distance is 0, and 0 otherwise.

    Parameters:
    x: The time distance.

    Returns:
    weight: The weight of the time distance.
    """
    return 1 if x == 0 else 0

def similarity(wa, wb, mu = MU):
    """
    Calculate the similarity between two weights.

    This function calculates the similarity between two weights using a linear combination of their product and the minimum of the two weights.

    Parameters:
    wa: The first weight.
    wb: The second weight.
    mu: The weight of the minimum in the linear combination.

    Returns:
    similarity: The similarity between the two weights.
    """
    return max(wa*wb)*(1-mu) + mu

def smith_waterman(seq1, seq2, gap = GAP_PENALTY, mu = MU):
    """
    Calculate the optimal local alignment of two sequences using the Smith-Waterman (SW) algorithm.

    This function calculates the optimal local alignment of two sequences using the SW algorithm. The alignment is scored based on similarities and gaps. 
    The function returns the maximum score, the index of the maximum score in the alignment matrix, and the tracing matrix for backtracking.

    Parameters:
    seq1: The first sequence represented as a 2D numpy array.
    seq2: The second sequence represented as a 2D numpy array.
    gap: The penalty for a gap in the alignment.

    Returns:
    tuple containing:
    max_score: The maximum score of the local alignment.
    aligned_seq1: Aligned first sequence.
    aligned_seq2: Aligned second sequence.
    matrix: The alignment matrix.
    """
    # Generating the empty matrices for storing scores and tracing
    row = seq1.shape[1] + 1
    col = seq2.shape[1] + 1
    matrix = np.zeros(shape=(row, col))  
    tracing_matrix = np.zeros(shape=(row, col))  
    
    # Initialising the variables to find the highest scoring cell
    max_score = -1
    max_index = (-1, -1)

    class Trace(IntEnum):
        STOP = 0
        LEFT = 1 
        UP = 2
        DIAGONAL = 3
        
    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            match_value = similarity(seq1[:, i -1], seq2[:, j - 1], mu)
            diagonal_score = matrix[i - 1, j - 1] + match_value
            
            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] + gap
            
            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] + gap
            
            # Taking the highest score 
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)
            
            # Tracking where the cell's value is coming from    
            if matrix[i, j] == 0: 
                tracing_matrix[i, j] = Trace.STOP
                
            elif matrix[i, j] == horizontal_score: 
                tracing_matrix[i, j] = Trace.LEFT
                
            elif matrix[i, j] == vertical_score: 
                tracing_matrix[i, j] = Trace.UP
                
            elif matrix[i, j] == diagonal_score: 
                tracing_matrix[i, j] = Trace.DIAGONAL 
                
            # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i,j)
                max_score = matrix[i, j]
    # Initialising the variables for tracing, return indices of floods, -1 indicates a gap
    aligned_seq1 = []
    aligned_seq2 = []
    current_aligned_seq1 = -1
    current_aligned_seq2 = -1 
    (max_i, max_j) = max_index
    
    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = max_i - 1
            current_aligned_seq2 = max_j - 1
            max_i = max_i - 1
            max_j = max_j - 1
            
        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = max_i - 1
            current_aligned_seq2 = -1
            max_i = max_i - 1    
            
        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = -1
            current_aligned_seq2 = max_j - 1
            max_j = max_j - 1
            
        aligned_seq1.append(current_aligned_seq1) 
        aligned_seq2.append(current_aligned_seq2)
    
    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]
    
    return max_score, aligned_seq1, aligned_seq2, matrix


def flood_similarity_score(tw1_a, tw2_a, tw1_b, tw2_b, gap=GAP_PENALTY, mu=MU):
    """
    Calculate the similarity between two alarm floods using a modified Smith-Waterman algorithm.

    This function calculates the similarity between two alarm floods by comparing their time weight matrices. 
    It uses the Smith-Waterman algorithm to calculate the similarity scores between the matrices, 
    and then returns the maximum score normalized by the minimum length of the floods.

    Parameters:
    tw1_a: Time weight matrix 1 of alarm flood A.
    tw2_a: Time weight matrix 2 of alarm flood A.
    tw1_b: Time weight matrix 1 of alarm flood B.
    tw2_b: Time weight matrix 2 of alarm flood B.

    Returns:
    similarity_score: The similarity score between the alarm floods. Values are between 0 and 1, where 1 indicates high similarity.
    """
    similarity_score1, _ , _ , _ =  smith_waterman(tw1_a, tw2_b, gap, mu)
    similarity_score2, _ , _ , _ =  smith_waterman(tw1_b, tw2_a, gap, mu)
    return max(similarity_score1, similarity_score2)/ min(tw1_a.shape[1], tw1_a.shape[1])
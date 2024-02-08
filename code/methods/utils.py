import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def multivar_binary(X, vocab):
    """
    This function converts multivariate data into binary format.

    Parameters:
    X: The input data, expected to be a DataFrame with a 'flood_id' column.
    vocab: The vocabulary used for the conversion, expected to be a dictionary.

    Returns:
    X_binary: The converted data in binary format.
    """

    def alarm_group_to_binary(group, vocababulary):
        """
        This function converts a group of alarm events into a binary matrix.

        Parameters:
        group: The group of alarm events, expected to be a DataFrame.
        vocababulary: The vocabulary used for the conversion, expected to be a dictionary.

        Returns:
        matrix: The binary matrix representation of the alarm events.
        """

        # Limit lengths to two hours, every value represents one second
        vector_lengths = 60*60*2
        matrix = np.zeros((len(vocababulary), vector_lengths))

        for i, alarm_event in group.iterrows():
            # Convert the start and end timestamps to seconds
            start_idx = int(alarm_event["startTimestamp"]/1000)
            end_idx = int(alarm_event["endTimestamp"]/1000)

            # Set the corresponding indices in the matrix to 1
            matrix[vocababulary[alarm_event["alarmNumber"]], start_idx:end_idx+1] = 1
        return matrix

    X_binary = X.groupby("flood_id").apply(lambda flood: alarm_group_to_binary(flood, vocab))
    X_binary = np.array(X_binary.to_list())
    return X_binary


def plot_zipfs_distribution(vocabulary, X):
    """
    Plot Zipf's distribution for the alarm floods.

    This method  calculates the frequency of each alarm number, 
    and then plots the frequencies against their ranks in a log-log plot. 
    It also plots a line representing Zipf's law for comparison.

    Parameters:
    X: A DataFrame with columns 'flood_id' and 'alarmNumber'. Each row represents an event.

    Returns:
    None
    """
    documents = X.groupby('flood_id')["alarmNumber"].apply(lambda alarms: [vocabulary[alarm] for alarm in alarms])
    # Check if alarms follow Zipf's distribution
    flat_array = documents.explode().to_numpy()
    frequencies = Counter(flat_array)
    counts = np.array(list(frequencies.values()))
    tokens = np.array(list(frequencies.keys()))

    indices = np.argsort(-counts)
    counts = counts[indices]
    tokens = tokens[indices]

    ranks = np.arange(1, len(counts) + 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, counts, marker="o", label='Observed Frequencies')
    plt.loglog(ranks, [counts[0]/rank for rank in ranks], linestyle='--', color='r', label='Zipf\'s Law')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Checking Zipf\'s Law')
    plt.legend()
    plt.show()

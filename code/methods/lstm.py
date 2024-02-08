import pandas as pd
import keras
from keras.preprocessing import sequence
import numpy as np
import tensorflow as tf

# Default parameters from paper except batch size
# Dorgo, G. et al. (2018) Understanding the importance of process alarms based on the analysis of deep
# recurrent neural networks trained for fault isolation. Journal of chemometrics. [Online] 32 (4), .
N_EVENT = 4 # number of alarms 
SEQLEN = N_EVENT * 2 - 1 # number of events and temporal predicates fed to neural network
EMBEDDING_DIM = 4
LSTM_UNITS = 17
EPOCHS = 50
BATCH_SIZE = 32

class LSTM_Classifier:
    def __init__(self, vocabulary, n_classes, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, slen=SEQLEN, probability_threshold=0.5):
        """
        Initialize the LSTM_Classifier.

        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        n_classes (int): The number of classes.
        embedding_dim (int, optional): The dimension of the embedding layer. Defaults to EMBEDDING_DIM.
        lstm_units (int, optional): The number of units in the LSTM layers. Defaults to LSTM_UNITS.
        slen (int, optional): The length of the sequences. Defaults to SEQLEN.
        probability_threshold (float, optional): The threshold for the probability of a class. If the probability of a class is lower than this threshold, the class is classified as unknown. Defaults to 0.5.
        """
        self.vocabulary = {k: v for k, v in vocabulary.items()}
        self.n_classes = n_classes
        # Add temporal predicates to vocabulary
        self.vocabulary['E'] = len(self.vocabulary)
        self.vocabulary['B'] = len(self.vocabulary)
        self.vocabulary['I'] = len(self.vocabulary)
        self.vocabulary['O'] = len(self.vocabulary)
        self.probability_threshold = probability_threshold

        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(len(self.vocabulary)+1, embedding_dim))
        model.add(keras.layers.LSTM(lstm_units, return_sequences=True,
                input_shape=(slen, embedding_dim)))
        model.add(keras.layers.LSTM(lstm_units, return_sequences=True))
        model.add(keras.layers.LSTM(lstm_units))
        model.add(keras.layers.Dense(n_classes, activation='softmax'))


        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'],)
        self.model = model
        #print(model.summary())
        

    def fit(self, X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS):
        """
        Train the LSTM_Classifier with the given data.

        This method encodes the alarm numbers in the data using the vocabulary, pads the sequences to the maximum sequence length, 
        and then fits the LSTM model with the encoded data and labels.

        Parameters:
        X (DataFrame): pd DataFrame with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        batch_size (int): The number of samples per gradient update. Default is BATCH_SIZE.
        epochs (int): The number of epochs to train the model. Default is EPOCHS.

        Returns:
        history (History): A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs.
        """
        sequences = X.groupby("flood_id").apply(self.encode_floods)
        encoded_sequences = sequences.apply(lambda seq: [self.vocabulary[alarm] for alarm in seq])
        X_encoded = sequence.pad_sequences(encoded_sequences, maxlen=SEQLEN, dtype='int', padding='post', truncating='post', value=len(self.vocabulary))

        
        x_tensor = tf.convert_to_tensor(X_encoded)
        y_tensor = tf.keras.utils.to_categorical(Y, num_classes=self.n_classes)
        history = self.model.fit(x_tensor, y_tensor, batch_size=batch_size, epochs=epochs, verbose=0)
        return history
    
    def predict(self, X):
        """
        Predict the labels for the given data.

        This method encodes the alarm numbers in the data using the vocabulary, pads the sequences to the maximum sequence length, 
        and then uses the LSTM model to predict the labels for the encoded data.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """
        sequences = X.groupby("flood_id").apply(self.encode_floods)
        encoded_sequences = sequences.apply(lambda seq: [self.vocabulary[alarm] for alarm in seq])
        X_encoded = sequence.pad_sequences(encoded_sequences, maxlen=SEQLEN, dtype='int', padding='post', truncating='post', value=len(self.vocabulary))

        
        x_tensor = tf.convert_to_tensor(X_encoded)
        predictions = self.model.predict(x_tensor)
        # apply threshold
        return [np.argmax(prediction) if np.max(prediction) > self.probability_threshold else -1 for prediction in predictions]


    def temporal_relation(self, st1, et1, st2, et2):
        """
        Determine the temporal relation between two alarm events.

        This method compares the start and end times of two alarm events and determines their temporal relation. 
        The temporal relation can be one of the following:
        - 'E': The two events are equal (i.e., they have the same start and end times).
        - 'B': The first event is before the second event.
        - 'I': The first event contains the second event.
        - 'O': The two events overlap.

        Parameters:
        st1: The start time of the first event.
        et1: The end time of the first event.
        st2: The start time of the second event.
        et2: The end time of the second event.

        Returns:
        relation: A string representing the temporal relation between the two events.
        """

        if st1 == st2 and et1 == et2:
            return 'E'
        elif et1 <= st2:
            return 'B'
        elif (st1 < st2 and et2 <= et1) or (st1 == st2 and et2 < et1):
            return 'I'
        else:
            return 'O'
        
    def encode_floods(self, group):
        """
        Encode a group of alarm events into a sequence of alarm numbers and temporal relations.

        This method iterates over each event in the group, and for each event, it calculates the temporal relation 
        with the previous event and appends the relation and the alarm number of the event to the sequence.

        Parameters:
        group: A DataFrame representing a group of events. Each event is expected to have an 'alarmNumber', a 'startTimestamp', and an 'endTimestamp'.

        Returns:
        seq: The encoded sequence. This is a list of alarm numbers and temporal relations.
        """
        group = group.reset_index(drop=True)
        seq = [group.loc[0, "alarmNumber"]]
        for i in range(1, len(group) - 1):
            st1, et1 = group.loc[i - 1, 'startTimestamp'], group.loc[i - 1, 'endTimestamp']
            st2, et2 = group.loc[i, 'startTimestamp'], group.loc[i, 'endTimestamp']
            relation = self.temporal_relation(st1, et1, st2, et2)
            seq.extend([relation, group.iloc[i]["alarmNumber"]])
        return seq
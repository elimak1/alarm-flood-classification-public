import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from scipy.optimize import linprog

from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Implements paper:
# Wei, Lu & Wang, Liliang & Liu, Feng & Qian, Zheng. (2023).
# Clustering Analysis of Wind Turbine Alarm Sequences Based on Domain Knowledge-Fused Word2vec.
# Applied Sciences. 13. 10114. 10.3390/app131810114. 

EMBEDDING_DIM = 50
C = 4 # Window size
N_NEGATIVE_SAMPLES = 4 # Generated negatvie samples per positive sample
LAMBDA = 1e-6 # Weight of category loss

BATCH_SIZE = 1024
BUFFER_SIZE = 10000

AUTOTUNE = tf.data.AUTOTUNE

class Word2Vec_Classifier():
    def __init__(self, vocabulary, embedding_dim=EMBEDDING_DIM, c=C, n_negative_samples=N_NEGATIVE_SAMPLES, lambda_=LAMBDA, distance_threshold=1.0):
        """
        Initialize the Word2Vec_Classifier.

        Parameters:
        vocabulary (dict): A dictionary mapping alarm numbers to indices.
        embedding_dim (int): The dimension of the word embeddings.
        c (int): The window size for the Word2Vec algorithm.
        n_negative_samples (int): The number of negative samples to generate per positive sample.
        lambda_ (float): The weight of category loss.
        distance_threshold (float): The threshold for considering two embeddings as similar.

        Returns:
        None
        """
        self.vocabulary = vocabulary
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.embedding_dim = embedding_dim
        self.c = c
        self.n_negative_samples = n_negative_samples
        self.lambda_ = lambda_
        self.distance_threshold = distance_threshold

        self.final_embeddings = None
        self.labels = None
        self.train_documents = None


    def fit(self, X, Y, epochs, category_mappings=[], batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
        """
        Train the Word2Vec_Classifier with the given data.

        This method converts the alarm floods into documents of word indices, generates training data for the Word2Vec algorithm, 
        and then trains the Word2Vec model with the training data.

        Parameters:
        X (DataFrame):  pd Dataframe with columns flood_id, alarmNumber, startTimestamp, endTimestamp. Sorted by startTimestamp ascending
        Y (Series): pd series of labels, indexed by flood_id
        epochs (int): The number of epochs to train the Word2Vec model.
        category_mappings (list, optional): A list of category mappings for the alarms. Defaults to an empty list.
        batch_size (int, optional): The batch size for training the Word2Vec model. Defaults to BATCH_SIZE.
        buffer_size (int, optional): The buffer size for shuffling the training data. Defaults to BUFFER_SIZE.

        Returns:
        None
        """
        documents = X.groupby('flood_id')["alarmNumber"].apply(lambda alarms: [self.vocabulary[alarm] for alarm in alarms])
        self.train_documents = documents

        targets, contexts, labels = generate_training_data(documents, self.c, self.n_negative_samples, len(self.vocabulary))

        
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

        category_masks = [tf.equal(mapping[:, tf.newaxis], mapping[tf.newaxis, :]) for mapping in category_mappings]

        # Loss function uses alarm category information + typical softmax
        # Optimized_loss = (1 − λ) × Raw_loss + λ × Category_loss,

        def custom_loss(y_true, pair):
            dots, all_embeddings = pair
            y_true = tf.cast(y_true, 'float32')

            normalized_embeddings = tf.nn.l2_normalize(all_embeddings, axis=1)
            similarity_matrix = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

            # Calculate phi and gamma
            phi = tf.reduce_sum([tf.reduce_sum(tf.where(mask, 1 - similarity_matrix, 0)) for mask in category_masks])
            gamma = tf.reduce_sum([tf.reduce_sum(tf.where(~mask, similarity_matrix, 0)) for mask in category_masks])

            category_loss = phi + gamma
            raw_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dots, labels=y_true))

            return (1-self.lambda_) * raw_loss + self.lambda_ * category_loss
        
        model = SkipGram(len(self.vocabulary), self.embedding_dim)

        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
        

        for epoch in range(epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    # Forward pass: Compute predictions and loss
                    dots, all_embeddings = model(x_batch_train, training=True)
                    loss_value = custom_loss(y_batch_train, (dots, all_embeddings))

                # Backward pass: Compute gradients and update weights
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #if ( epoch % 10 == 0):
                #print(f"Epoch {epoch}, Loss: {float(loss_value)}")
        
        self.final_embeddings = model.predict(np.arange(len(self.vocabulary)))
        self.labels = Y

    def predict(self, X):
        """
        Predict the labels for the given data.

        This method converts the alarm floods into documents of word indices, calculates the average word embedding for each document, 
        and then uses a 1-nearest neighbors classifier to predict the labels for the documents based on their average word embeddings.

        Parameters:
        X (DataFrame): The input data to be classified.

        Returns:
        numpy.ndarray: The predicted labels for the input data.
        """

        embeds = np.array(self.final_embeddings)

        documents = X.groupby('flood_id')["alarmNumber"].apply(lambda alarms: [self.vocabulary[alarm] for alarm in alarms])

        predictions = []
        for j in range(len(documents)):

            distances = []
            for i in range(len(self.train_documents)):
                wmd, _= calculate_wmd(embeds[self.train_documents.iloc[i]], embeds[documents.iloc[j]])
                distances.append(wmd)
            if min(distances) > self.distance_threshold:
                predictions.append(-1)
                continue
            idx = np.argmin(distances)
            predictions.append(self.labels.iloc[idx])
        return predictions

    def plot_embeddings(self, subset_indices):
        """
        Plot the word embeddings in a 2D space.

        This method uses PCA to reduce the dimensionality of the word embeddings to 2, 
        and then plots the embeddings in a scatter plot. 
        It also annotates the points with their corresponding alarm numbers.

        Parameters:
        subset_indices: A list of indices for the embeddings to plot.

        Returns:
        None

        Raises:
        Exception: If the model is not trained.
        """
        if self.final_embeddings is None:
            raise Exception("Model is not trained")

        pca = PCA(n_components=2)
        pca.fit(self.final_embeddings)
        pca_features = pca.transform(self.final_embeddings)

        plt.figure(figsize=(10, 10))
        plt.scatter(pca_features[subset_indices, 0], pca_features[subset_indices, 1])
        for i in subset_indices:
            plt.annotate(self.inverse_vocabulary[i], (pca_features[i, 0], pca_features[i, 1]))
        plt.show()
        


class SkipGram(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.shared_embedding = Embedding(vocab_size, embedding_dim, input_length=1)

    def call(self, pair):
        """
        Forward pass of the SkipGram model.

        This method takes in a pair of target and context words, calculates their embeddings, 
        and then returns the dot product of the embeddings and all embeddings.

        Parameters:
        pair: A pair of target and context words.

        Returns:
        dots: The dot product of the target and context embeddings.
        all_embeddings: All embeddings of the model.
        """
        target, context = pair
        target_emb = self.shared_embedding(target)
        context_emb = self.shared_embedding(context)
        dots = tf.einsum('be,bce->bc', target_emb, context_emb, name="xd")
        vocab_indices = tf.range(self.vocab_size, dtype=tf.int32)
        all_embeddings = self.shared_embedding(vocab_indices)
        return dots, all_embeddings
    
    def predict(self, inputs):
        return self.shared_embedding(inputs)

def generate_training_data(sequences, window_size, num_ns, vocab_size):
    """
    Generate training data for the SkipGram model.

    This method generates positive and negative skip-gram pairs for each sequence in the dataset. 
    It appends the target word, context word, and label (1 for positive pairs, 0 for negative pairs) for each pair to the training data.

    Parameters:
    sequences: The sequences for which to generate training data.
    window_size: The window size for the SkipGram model.
    num_ns: The number of negative samples to generate per positive sample.
    vocab_size: The size of the vocabulary.

    Returns:
    targets, contexts, labels: The target words, context words, and labels for the training data.
    """
    targets, contexts, labels = [], [], []

    #sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    sampling_table = None

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in sequences:

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def calculate_wmd(embeddings1, embeddings2):
    """
    Calculate the Word Mover's Distance (WMD) between two sets of embeddings.

    This function calculates the WMD between two sets of embeddings by solving a linear programming problem. 
    The WMD is the minimum amount of work needed to transform one set of embeddings into the other, 
    where work is measured as the Euclidean distance between embeddings.

    Parameters:
    embeddings1: The first set of embeddings.
    embeddings2: The second set of embeddings.

    Returns:
    wmd: The Word Mover's Distance between the two sets of embeddings.
    res.x: The optimal solution to the linear programming problem.
    """
    # Number of words in each sequence
    n = len(embeddings1)
    m = len(embeddings2)

    # Cost matrix: Euclidean distance between each pair of embeddings
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(embeddings1[i] - embeddings2[j])

    # Linear programming problem setup
    # Objective: minimize the sum of C[i, j] * gamma[i, j]
    c = C.flatten()
    # Constraints: sum of gamma across rows and columns equals 1/n and 1/m respectively
    A_eq = []
    for i in range(n):
        row_constraint = np.zeros((n, m))
        row_constraint[i, :] = 1
        A_eq.append(row_constraint.flatten())
    for j in range(m):
        col_constraint = np.zeros((n, m))
        col_constraint[:, j] = 1
        A_eq.append(col_constraint.flatten())
    A_eq = np.array(A_eq)
    b_eq = np.array([1/n]*n + [1/m]*m)

    # Bounds for gamma[i, j]: 0 <= gamma[i, j]
    bounds = [(0, None) for _ in range(n*m)]
    # Solve linear programming problem
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # WMD is the sum of the element-wise product of C and the optimal gamma
    wmd = res.fun
    return wmd,res.x

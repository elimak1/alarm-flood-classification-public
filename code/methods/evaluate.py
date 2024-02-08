from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
import numpy as np
import pandas as pd

def print_metrics(preds, gt):
    accuracy = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average='weighted')
    ari = adjusted_rand_score(gt, preds)

    e0 = np.sum((gt == -1) & (preds != -1))
    e1 = np.sum((gt == -1) & (preds == -1))
    fdr = e0/(e0+e1 + 1e-10)

    a0 = np.sum((gt != -1) & (preds == -1))
    a1 = np.sum((gt != -1) & (preds != -1))
    mdr = a0/(a0+a1)
    print('Accuracy:', accuracy)
    print('F1 Score:', f1)
    print('Adjusted Rand Index:', ari)
    print('False Discovery Rate:', fdr)
    print('Missed Discovery Rate:', mdr)
    return accuracy, f1, ari, fdr, mdr

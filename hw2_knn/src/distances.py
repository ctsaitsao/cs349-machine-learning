import numpy as np
from math import sqrt


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    D = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i, j] = sqrt(np.sum((X[i, :] - Y[j, :])**2))

    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    D = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i, j] = np.sum(abs(X[i, :] - Y[j, :]))

    return D

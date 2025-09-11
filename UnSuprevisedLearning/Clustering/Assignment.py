import numpy as np
import matplotlib.pyplot as plt
from utils import *

def find_closest_centroids(X, centroids):

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i in range(X.shape[0]):
        # Array to hold distance between X[i] and each centroids[j]
        distance = [] 
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i]-centroids[j])
            distance.append(norm_ij)

        idx[i] = np.argmin(distance)
    ### END CODE HERE ###
    
    return idx

def find_closest_centroids_test(target):
    # With 2 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, -1],
                  [2, 2],[2.5, 2.5],[2, 2.5]])
    initial_centroids = np.array([[-1, -1], [2, 2]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [0, 0, 0, 1, 1, 1]), "Wrong values"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [2, 2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 0]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 1, 2]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"

X = load_data()

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)


# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    
    for k in range(K):
        points = X[idx == k]
        centroids[k] = points.mean(axis=0) if points.size else centroids[k]
    ### END CODE HERE ##

    return centroids
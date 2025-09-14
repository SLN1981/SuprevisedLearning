import numpy as np
import os

# Create synthetic data for clustering
np.random.seed(42)
# Create 3 clusters of points
cluster1 = np.random.randn(50, 2) * 0.75 + np.array([2, 2])
cluster2 = np.random.randn(50, 2) * 0.5 + np.array([6, 6])
cluster3 = np.random.randn(50, 2) * 0.5 + np.array([7, 2])

# Combine the clusters
X = np.vstack([cluster1, cluster2, cluster3])

# Save the data
np.save("ex7_X.npy", X)
print(f"Sample data saved to {os.path.abspath('ex7_X.npy')}")
print(f"Data shape: {X.shape}")

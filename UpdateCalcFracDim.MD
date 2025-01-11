# Local Fractal Dimension Calculation

## Overview

This algorithm calculates the local fractal dimension of data points using their k-nearest neighbors. It is based on the concept that fractal dimensions describe the local scaling behavior of data in high-dimensional spaces. The key steps involve computing pairwise distances, estimating local densities, and calculating the fractal dimension for each point in the dataset.

The method also incorporates a robust approach to handle variations in data density using clamping and normalization techniques to ensure stable results across points.

## Key Steps of the Algorithm

### 1. **Data Generation** (`generate_data_subtleoutliers`)
This function generates synthetic data using a specified number of clusters and transformations. It also generates outliers to simulate real-world data noise. The process involves the following steps:
- **Main Data Clusters**: Using `make_blobs`, the data is clustered into a specified number of centers.
- **Transformation**: Each cluster is transformed using dynamic elliptical matrices to create subtle variations.
- **Outliers**: Random outliers are introduced within the range of the data.

### 2. **Pairwise Distance Calculation** (`compute_pairwise_distances`)
The pairwise Euclidean distance matrix is computed for the data points. This is essential for determining the distances between points and identifying the k-nearest neighbors.

### 3. **Fractal Dimension Calculation** (`compute_fractal_dimension`)
This is the core function that computes the local fractal dimension of each point based on its nearest neighbors. The steps are as follows:

- **Pairwise Distance Matrix**: Compute the pairwise Euclidean distances for all points.
- **Sorting Distances**: Sort the distances and extract the distances corresponding to the k-nearest neighbors.
- **Local Density**: Calculate the local density for each point, which is defined as the inverse of the mean distance to its k-nearest neighbors.
- **Radius Calculation**: The average radius of a point is computed using the mean of the k-nearest neighbor distances, scaled by the local density.
- **Clamping and Normalization**: To ensure stable results, the radii are clamped to a reasonable range (via percentile-based normalization).
- **Fractal Dimension**: The fractal dimension is calculated using the formula:
  \[
  D = \frac{\log(k)}{\log(\frac{1}{r})}
  \]
  where \( r \) is the average radius for each point and \( k \) is the number of nearest neighbors.

### 4. **Additional Robustness**:
- **Clamping Radii**: To avoid numerical instabilities, the radii are clamped to values within a specified range. This is done by considering the 5th and 95th percentiles of non-zero radii values.
- **Epsilon**: A small constant is added to avoid division by zero and ensure the stability of logarithmic operations.

### 5. **Final Output**:
The fractal dimensions for each data point are computed and returned. These dimensions represent the scaling behavior of the data locally, which can be useful for anomaly detection, understanding data structures, and exploring the complexity of datasets.

---

## Example Python Code

```python
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def generate_data_subtleoutliers(n_samples, total_outliers, n_centers=3):
    np.random.seed(42)
    # Generate main data clusters with specified number of centers
    data, labels = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=1.0, random_state=42)
    # Generate transformation matrices dynamically for the specified number of centers
    transformation_matrices = [
        [[0.6 + 0.1 * i, -0.3 * i], [0.3 * i, 0.8 + 0.1 * i]] 
        for i in range(n_centers)
    ]
    transformed_data = []
    for i in range(n_centers):
        cluster_data = data[labels == i]
        transformed_cluster = np.dot(cluster_data, transformation_matrices[i % len(transformation_matrices)])
        transformed_data.append(transformed_cluster)
    transformed_data = np.vstack(transformed_data)
    
    # Generate outliers
    data_min = transformed_data.min(axis=0)
    data_max = transformed_data.max(axis=0)
    outliers = np.random.uniform(low=data_min, high=data_max, size=(total_outliers, 2))
    valid_pts = transformed_data
    all_data = np.vstack([transformed_data, outliers])
    return all_data, outliers, valid_pts

def compute_pairwise_distances(data):
    data = np.array(data)
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    return distances

def compute_fractal_dimension(data, k, epsilon=1e-3):
    data = np.array(data)
    n_samples = data.shape[0]
    if k < 1 or n_samples <= k:
        raise ValueError("Invalid k value: k must be between 1 and the number of points - 1.")
    
    distances = compute_pairwise_distances(data)
    sorted_distances = np.sort(distances, axis=1)
    sorted_distances = sorted_distances[:, 1:k+1]
    
    # Compute local densities
    local_densities = np.zeros(n_samples)
    for i in range(n_samples):
        mean_distance = np.mean(sorted_distances[i])
        local_densities[i] = 1.0 / (mean_distance + epsilon)
    
    radii = np.zeros(n_samples)
    for i in range(n_samples):
        mean_original_distance = np.mean(sorted_distances[i])
        radii[i] = np.maximum(np.log(mean_original_distance + epsilon) / local_densities[i], epsilon)
    
    # Robust clamping and normalization
    non_zero_radii = np.sort(radii[radii > 0])
    min_radius = non_zero_radii[max(int(0.05 * n_samples), 1)] 
    max_radius = non_zero_radii[min(int(0.95 * n_samples), n_samples - 1)]
    min_radius = np.maximum(min_radius, epsilon)
    
    for i in range(n_samples):
        radii[i] = (radii[i] - min_radius) / (max_radius - min_radius + epsilon)
        radii[i] = np.maximum(radii[i], 0.0001)
    
    # Calculate fractal dimensions
    fractal_dimensions = np.zeros(n_samples)
    log_k = np.log(k)
    for i in range(n_samples):
        radius_with_epsilon = np.maximum(radii[i] + epsilon, epsilon)
        log_argument = np.maximum(1.0 / radius_with_epsilon, .09)
        fractal_dimensions[i] = log_k / np.log(log_argument)
    
    return fractal_dimensions
```

---

## Is This Approach New?

The general idea of using nearest neighbors to estimate the local fractal dimension of data is not new. This approach is based on established techniques in fractal geometry and dimensionality estimation using local distances. However, this specific implementation introduces the following key points:

1. **Dynamic Transformation of Clusters**: The use of dynamically generated transformations for different clusters is a unique way to create subtle variations in the data distribution.
2. **Robust Clamping and Normalization**: The approach to clamp the radii based on robust percentiles is a more sophisticated method to handle variations in the data density, ensuring more stable fractal dimension calculations.
3. **Small `k` and `n_samples` Handling**: The handling of small datasets and low values for `k` can be considered an area of innovation, especially when combined with the robust normalization of radii.

This algorithm is a practical approach for calculating local fractal dimensions in datasets with subtle variations, but the underlying idea of estimating fractal dimensions using nearest neighbors is well-established in fractal analysis literature.
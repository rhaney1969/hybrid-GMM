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
        [[0.6 + 0.1 * i, -0.3 * i], [0.3 * i, 0.8 + 0.1 * i]]  # Dynamic elliptical/special transformations
        for i in range(n_centers)
    ]
    
    transformed_data = []
    for i in range(n_centers):
        cluster_data = data[labels == i]
        transformed_cluster = np.dot(cluster_data, transformation_matrices[i % len(transformation_matrices)])
        transformed_data.append(transformed_cluster)

    # Combine transformed clusters
    transformed_data = np.vstack(transformed_data)

    # Generate outliers within the range of the transformed data
    data_min = transformed_data.min(axis=0)
    data_max = transformed_data.max(axis=0)
    outliers = np.random.uniform(low=data_min, high=data_max, size=(total_outliers, 2))

    # Define valid points (main data points only)
    valid_pts = transformed_data

    # Combine main data with outliers
    all_data = np.vstack([transformed_data, outliers])
    return all_data, outliers, valid_pts

def compute_pairwise_distances(data):
    """Computes pairwise Euclidean distances between points in data."""
    data = np.array(data)
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    return distances
    
def compute_fractal_dimension(data, k, epsilon=1e-3):
    """
    Computes the fractal dimension of each point based on k-nearest neighbors.
    
    Parameters:
        data (list or np.ndarray): The input data points (n_samples x n_features).
        k (int): The number of nearest neighbors to consider.
        epsilon (float): Small constant to avoid division by zero or logarithmic singularities.
        
    Returns:
        fractal_dimensions (np.ndarray): Fractal dimensions for each data point.
    """
    data = np.array(data)
    n_samples = data.shape[0]
    
    if k < 1 or n_samples <= k:
        raise ValueError("Invalid k value: k must be between 1 and the number of points - 1.")
    
    # Step 1: Compute pairwise distances
    distances = compute_pairwise_distances(data)
    
    # Step 2: Sort distances to find k-nearest neighbors
    sorted_distances = np.sort(distances, axis=1)
    sorted_distances = sorted_distances[:, 1:k+1]  # Only take k-nearest neighbors (exclude self distance)
    
    # Step 3: Compute local densities
    local_densities = np.zeros(n_samples)
    for i in range(n_samples):
        mean_distance = np.mean(sorted_distances[i])
        local_densities[i] = 1.0 / (mean_distance + epsilon)
    
    # Step 4: Compute average radii with clamping
    radii = np.zeros(n_samples)
    for i in range(n_samples):
        mean_original_distance = np.mean(sorted_distances[i])
        radii[i] = np.maximum(np.log(mean_original_distance + epsilon) / local_densities[i], epsilon)
    
    # ***************************************************************** #
    #         ADDED THE FOLLOWING TO CLAMP RADII MORE ROBUSTLY          #
    # ***************************************************************** #
    
    # Compute robust percentiles for normalization
    non_zero_radii = np.sort(radii[radii > 0])  # Sort only non-zero radii
    min_radius = non_zero_radii[max(int(0.05 * n_samples), 1)]  # 5th percentile
    max_radius = non_zero_radii[min(int(0.95 * n_samples), n_samples - 1)]  # 95th percentile
    min_radius = np.maximum(min_radius, epsilon)  # Ensure it's not too small

    # Normalize radii with softer clamping
    for i in range(n_samples):
        radii[i] = (radii[i] - min_radius) / (max_radius - min_radius + epsilon)
        #radii[i] = np.maximum(radii[i], 0.1)  # Allow smaller values but avoid instability
        radii[i] = np.maximum(radii[i], 0.0001)  # Allow smaller values but avoid instability
    
    # ***************************************************************** #
    # ***************************************************************** #

    # Step 5: Calculate fractal dimensions
    fractal_dimensions = np.zeros(n_samples)
    log_k = np.log(k)
    for i in range(n_samples):
        radius_with_epsilon = np.maximum(radii[i] + epsilon, epsilon)
        #log_argument = np.maximum(1.0 / radius_with_epsilon, 2.0)  # Stricter lower bound (2.0)
        log_argument = np.maximum(1.0 / radius_with_epsilon, .09)  # Stricter lower bound (2.0)
        fractal_dimensions[i] = log_k / np.log(log_argument)
    
    return fractal_dimensions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data with a specified number of outliers.")
    parser.add_argument("--samples", type=int, default=11, help="Total number of samples to generate (default: 11)")
    parser.add_argument("--outliers", type=int, default=3, help="Total number of outliers to generate (default: 3)")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for GMM (default: 3)")
    parser.add_argument("--k", type=int, default=2, help="Number of nearest neighbors to consider for Fractal (default: 2)")
    args = parser.parse_args()

    # Generate synthetic data and outliers
    n_samples = args.samples
    total_outliers = args.outliers
    n_clusters = args.clusters
    k = args.k

    # Generate data
    data, defined_outliers, valid_pts = generate_data_subtleoutliers(n_samples, total_outliers, n_clusters)

    # Compute fractal dimensions
    frac_dims = compute_fractal_dimension(data, k)

    # Print the first 10 and last 5 fractal dimensions
    print("First 10 fractal dimensions:\n", frac_dims[:10])
    print("Last 5 fractal dimensions:\n", frac_dims[-5:])

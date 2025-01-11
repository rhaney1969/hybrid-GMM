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

def compute_fractal_dimension(data, k=5, epsilon=1e-10):
    # Step 1: 
    # - Compute pairwise distances between all points in the dataset
    distances = pairwise_distances(data)  # Shape: (n_samples, n_samples)

    # Step 2: 
    # - Sort distances for each point to find k-nearest neighbors
    # - Exclude the first column (self-distance = 0), so we only keep neighbors
    sorted_distances = np.sort(distances, axis=1)[:, 1:k+1]  # Shape: (n_samples, k)

    # Step 3: 
    # - Compute local densities for each point
    # - Density is inversely proportional to the mean distance to k neighbors
    local_densities = 1 / (np.mean(sorted_distances, axis=1) + epsilon) 

    # Step 4: 
    # - Scale the distances to k neighbors by the local density
    # - This normalization adjusts distances to account for the density of each point's neighborhood
    scaled_distances = sorted_distances * local_densities[:, np.newaxis] # Shape: (n_samples, k)

    # Step 5: 
    # - Compute the average radius for each point based on scaled distances
    # - The average of scaled distances provides the normalized radius for fractal dimension calculation
    radii = np.mean(scaled_distances, axis=1) # Shape: (n_samples,)

    # Step 6: 
    # - Calculate fractal dimension for each point
    # - The formula is based on log(k) and the inverse of the radius
    fractal_dimensions = np.log(k) / np.log(1 / (radii + epsilon))
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

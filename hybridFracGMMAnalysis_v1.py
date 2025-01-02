import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import argparse
import random
import math
import csv
import os

"""
Using fractal dimensional analysis pre-processing approach has FAR FEWER false-positive detections
of outliers/noise when data is then passed to Gaussian Mixture Model (GMM) with low number of k-nearest
neighbors to compute approximated fractal dimensions. 

For example: Shows Regular GMM detects more noise/outliers BUT to acquire that higher "accuracy" there
             are far more false-positives at 96.4% detected. Whereas Hybrid-GMM detects less noise/outliers,
             BUT has higher effective "accuracy". Hybrid-GMM has far fewer false-positives at 29%.

Total sample size = 3113
Acutal (true) outliers = 33
k = 5
- Regular GMM Accuracy = 0.94 (found outliers)
- Regular GMM Outliers detected = 873
- Regular GMM False-Positive outliers = 842
- Hybrid-GMM Accuracy = 0.67
- Hybrid-GMM Outliers detected = 31
- Hybrid-GMM False-Positive outliers = 9

"""

def load_points_from_csv(filename):
    """
    Loads points from a CSV file into valid points, outliers, and an all-points dataset.

    Args:
        filename (str): Path to the CSV file. The file must have a header row
                        with columns 'x', 'y', 'label', and 'clusters'. Points
                        with label -1 are considered outliers.

    Returns:
        tuple: A tuple containing three NumPy arrays:
               - valid_points: Points classified as valid (label != -1)
               - outliers: Points classified as outliers (label == -1)
               - all_points: Combined dataset of valid points and outliers
                 with their corresponding labels.
    """
    valid_points = []
    outliers = []
    all_points = []

    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                # Parse each row and handle possible errors
                x = float(row['x'])
                y = float(row['y'])
                label = int(row['label'])
                num_clusters = int(row['clusters'])  # Ensure clusters info is parsed
                all_points.append([x, y, label])  # Append to the combined dataset

                if label == -1:
                    outliers.append([x, y])  # Add to outliers if label is -1
                else:
                    valid_points.append([x, y])  # Otherwise, add to valid points

            except KeyError as e:
                raise ValueError(f"Missing expected column in CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Error parsing row {row}: {e}")

    return np.array(valid_points), np.array(outliers), np.array(all_points), num_clusters

def generate_data_gauss(n_samples, total_outliers):
    """
    Generates synthetic data with a specified number of outliers.
    - Clustered points fall within a Gaussian distribution in the range [0, 1].
    - Outliers are uniformly distributed in the range [-1.0, 2.0].

    CRITICAL: This generation causes the use of Fractal dimensional distribution to
              be less accurate with the default threshold. Lowering the threshold to
              enable the fractal filtering to pick up more of the potential outlier/noise
              points will improve accuracy of the hybrid-GMM approach
    """
    np.random.seed(42)

    # Generate cluster centers in the range [0, 1]
    cluster_centers = np.random.uniform(0, 1, size=(3, 2))

    # Generate clustered points around these centers with small spread
    data = []
    for center in cluster_centers:
        points = np.random.normal(loc=center, scale=0.1, size=(n_samples // 3, 2))
        data.append(points)
    data = np.vstack(data)

    # Generate outliers in the range [-1.0, 2.0]
    outliers = np.random.uniform(-1.0, 2.0, size=(total_outliers, 2))

    # Combine clustered data with outliers
    all_data = np.vstack([data, outliers])

    return all_data, outliers, data

def generate_data_subtleoutliers(n_samples, total_outliers, n_centers=3):
    """
    Generates synthetic data with a specified number of outliers and centers.
    - Generates non-spherical clustered data by introducing anisotropy.
    - Adds uniformly distributed noise as outliers within the range of the data clusters.
    - Supports a configurable number of centers for clusters.

    Introduce anisotropy (by applying linear transformations to clusters) and control for eccentricity in the 
    data distribution.

    Anisotropy
    - Applied linear transformations to each cluster using np.dot() with custom transformation matrices to create 
      elliptical and slanted clusters.
    """
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

def writeData(all_data, outliers, n_clusters, filename="output.csv"):
    """
    Writes the dataset with outliers to a CSV file with a header.
    Each row includes x, y, label, and the number of clusters.
    
    Args:
        all_data (numpy.ndarray): Combined dataset of clusters and outliers.
        outliers (numpy.ndarray): Outliers to be labeled with -1.
        n_clusters (int): Number of clusters in the dataset.
        filename (str): The name of the CSV file to write to.
    """
    # Create labels for the data
    labels = np.zeros(len(all_data), dtype=int)  # Default label for all data points is 0
    outlier_set = set(map(tuple, outliers))  # Convert outliers to a set of tuples for fast comparison
    for i, point in enumerate(all_data):
        if tuple(point) in outlier_set:
            labels[i] = -1  # Assign label -1 for outliers

    # Write to CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "label", "clusters"])  # Header
        for i in range(len(all_data)):
            writer.writerow([all_data[i, 0], all_data[i, 1], labels[i], n_clusters])

    print(f"Data successfully written to {filename}")

# Step 1: Generate Synthetic Data (with Specified Number of Outliers)
def generate_data(n_samples, n_centers, total_outliers):
    """
    Generates synthetic data with a specified number of outliers.
    - Generates clustered data using make_blobs.
    - Adds uniformly distributed noise as outliers.
    """
    np.random.seed(42)
    # Generate main data clusters
    data, _ = make_blobs(n_samples, centers=n_centers, cluster_std=1.0)
    
    # Generate outliers
    outliers = np.random.uniform(-10, 10, size=(total_outliers, 2))
    
    # Combine main data with outliers
    all_data = np.vstack([data, outliers])

    return all_data, outliers, data

# Step 2: Detect Outliers for Regular GMM (Using Mahalanobis Distance)
def detect_outliers_regular_gmm(data, labels, gmm, threshold=2.0):
    """
    Detect outliers based on the Mahalanobis distance from the cluster centers.
    Points whose Mahalanobis distance exceeds the threshold are marked as outliers.
    """
    outliers = []
    for cluster_idx in np.unique(labels):
        # Get the points assigned to this cluster
        cluster_data = data[labels == cluster_idx]
        
        # Get the mean and covariance matrix of the cluster
        mean = gmm.means_[cluster_idx]
        cov = gmm.covariances_[cluster_idx]
        
        # Compute the Mahalanobis distance for each point in the cluster
        mahalanobis_distances = cdist(cluster_data, [mean], metric='mahalanobis', VI=np.linalg.inv(cov))
        
        # Identify points with Mahalanobis distance greater than the threshold
        outliers.extend(cluster_data[mahalanobis_distances.flatten() > threshold])
    
    return np.array(outliers)

def optimal_k_with_silhouette(data, min_k=2, max_k=10):
    """
    Find the optimal value of k for k-means clustering using the silhouette score.

    Parameters:
        data (numpy.ndarray): Input dataset as an array of shape (n_samples, n_features).
        min_k (int): Minimum number of clusters to test.
        max_k (int): Maximum number of clusters to test.

    Returns:
        optimal_k (int): The optimal number of clusters (k).
        silhouette_scores (list): List of silhouette scores for each k.
    """
    silhouette_scores = []
    for k in range(min_k, max_k + 1):
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate silhouette score
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)
    
    # Find the optimal k (with the highest silhouette score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + min_k
    
    # Plot silhouette scores for each k
    plt.plot(range(min_k, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k')
    plt.show()
    
    return optimal_k, silhouette_scores

def optimal_k_with_elbow_method(data, min_k=2, max_k=50):
    """
    Find the optimal value of k for k-means clustering using the elbow method.
    
    Parameters:
        data (numpy.ndarray): Input dataset as an array of shape (n_samples, n_features).
        min_k (int): Minimum number of clusters to test.
        max_k (int): Maximum number of clusters to test.

    Returns:
        optimal_k (int): The optimal number of clusters (k).
        wcss_values (list): List of WCSS (Within-Cluster Sum of Squares) for each k.
    """
    # Step 1: Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Step 2: Calculate WCSS for each k in the range
    wcss_values = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        wcss = kmeans.inertia_  # WCSS (Within-Cluster Sum of Squares)
        wcss_values.append(wcss)
    
    # Step 3: Plot the WCSS values
    plt.plot(range(min_k, max_k + 1), wcss_values, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Step 4: Find the elbow (change in slope) and return the optimal k
    # The elbow is where the rate of decrease of WCSS slows down
    optimal_k = np.diff(wcss_values).argmin() + min_k
    
    return optimal_k, wcss_values

def find_optimal_k(data, k_min=3, k_max=None, epsilon=1e-10, step=1):
    """
    Determine the optimal value of k for computing fractal dimensions.

    Parameters:
        data (numpy.ndarray): Input dataset as an array of shape (n_samples, n_features).
        k_min (int): Minimum number of neighbors to consider for k.
        k_max (int): Maximum number of neighbors to consider for k.
                     If None, set it to 12.5% of the dataset size.
        epsilon (float): Small constant to avoid division by zero during computations.
        step (int): Step size for k iteration.

    POTENTIAL ISSUE:
        Could get stuck in a local minimum. Maybe use another method such as Simulated Annealing.

    Returns:
        optimal_k (int): The optimal value of k based on fractal dimension stability.
    """

    if k_max is None:
        k_max = math.ceil(0.125 * data.shape[0])  # Default to 12.5% of the dataset size

    k_values = range(k_min, k_max + 1, step)
    variances = []

    for k in k_values:
        # Compute fractal dimensions for the current k
        fractal_dimensions = compute_fractal_dimension(data, k=k, epsilon=epsilon)
        
        # Measure variance in fractal dimensions
        variances.append(np.var(fractal_dimensions))

    # Find the k with the smallest change in variance
    # Identify the "elbow" where variance stabilizes
    optimal_k_index = np.argmin(np.gradient(variances))
    optimal_k = k_values[optimal_k_index]

    print(f"Optimal k found: {optimal_k}")
    
    # Optionally, visualize the variances for debugging or analysis
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, variances, marker='o', label="Fractal Dimension Variance")
    plt.axvline(optimal_k, color='r', linestyle='--', label=f"Optimal k = {optimal_k}")
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Variance of Fractal Dimensions')
    plt.title('Optimal k Selection Based on Fractal Dimension Stability')
    plt.legend()
    plt.show()

    return optimal_k

# Step 3: Compute Fractal Dimensions
def compute_fractal_dimension(data, k=5, epsilon=1e-10):
    """
    Compute fractal dimension for each point in the dataset with density normalization.
    Fractal dimension is calculated using distances to k-nearest neighbors.

    IMPORTANT: This Python function implements the same logic as the CPP code for the class
               FractalAnalysis method `calculateFractalDimensions`.

    Parameters:
        data (numpy.ndarray): Input dataset as an array of shape (n_samples, n_features).
        k (int): Number of nearest neighbors to consider for fractal dimension calculation.
        epsilon (float): Small constant to avoid division by zero during computations.

    Returns:
        fractal_dimensions (numpy.ndarray): Array of fractal dimension values for each point.
    """

    # Step 0: 
    # - Calculate optimal k-nearest neighbors using 12.5% of total data size
    #total_points = data.shape[0]
    #k = math.ceil(0.125 * total_points)

    #print(f"Calculated k-neighbors = {k}")

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

# Step 4: Filter Points Based on Fractal Dimension
def filter_by_fractal_dimension(data, threshold=1.5, k=5):
    """
    Filter points based on their fractal dimension.
    Points with a fractal dimension below the threshold are considered outliers.
    """
    fractal_dimensions = compute_fractal_dimension(data, k)

    # Create a mask for points with fractal dimension above the threshold
    mask = fractal_dimensions > threshold

    # Separate filtered data (valid points) and outliers
    filtered_data = data[mask]
    outliers = data[~mask]
    return filtered_data, outliers

# Step 5: Cluster Data Using Gaussian Mixture Model (GMM)
def cluster_data(data, n_clusters=3):
    """
    Cluster the given data using Gaussian Mixture Model (GMM).
    The number of clusters is specified by n_clusters.
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data) # Fit the GMM to the data
    labels = gmm.predict(data) # Predict the cluster labels for the data
    return labels, gmm

# Step 6: Calculate Outlier Detection Accuracy
def calculate_accuracy(detected_outliers, defined_outliers):
    """
    Calculate the accuracy of outlier detection as the ratio of detected outliers to defined outliers.

    CRITICAL: Accuracy is defined not as detecting the same number of outliers, but that those outliers
              match the outliers that are actually outliers, i.e., same outliers defined in the input
              parameter `defined_outliers`
    """
    # Count how many detected outliers match the generated outliers
    detected_outliers_set = set(tuple(row) for row in detected_outliers)
    defined_outliers_set = set(tuple(row) for row in defined_outliers)
    
    # Find the intersection (matching outliers)
    matching_outliers = detected_outliers_set.intersection(defined_outliers_set)
    
    accuracy = len(matching_outliers) / len(defined_outliers)
    return accuracy

def calculate_accuracy_with_mismatches(detected_outliers, defined_outliers):
    """
    Calculate the accuracy of outlier detection as the ratio of detected outliers to defined outliers.
    Additionally, return the set of data points in `detected_outliers` that are not in `defined_outliers`.

    CRITICAL: Accuracy is defined not as detecting the same number of outliers, but that those outliers
              match the outliers that are actually outliers, i.e., same outliers defined in the input
              parameter `defined_outliers`.
    """
    # Convert inputs to sets of tuples for easy comparison
    detected_outliers_set = set(tuple(row) for row in detected_outliers)
    defined_outliers_set = set(tuple(row) for row in defined_outliers)
    
    # Find the intersection (matching outliers)
    matching_outliers = detected_outliers_set.intersection(defined_outliers_set)
    
    # Find the mismatches (detected but not actually outliers)
    mismatched_outliers = detected_outliers_set - defined_outliers_set
    
    # Calculate accuracy
    accuracy = len(matching_outliers) / len(defined_outliers)
    
    return accuracy, mismatched_outliers

def plotData(data, title='2D Data Points', pt_color='black'):
    """
    Plot 2-dimensional data points as a scatter-plot

    Args:
        data (numpy.ndarray): Dataset being plotted.
        title (str, optional): Title for dataset to plot. Defaults to '2D Data Points'.
        pt_color (str, optional): Color of dataset points to plot. Defaults to 'black'.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(data[:, 0], data[:, 1], color=pt_color, label=title, alpha=0.9)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def plotClusteredData(data_regular, labels_regular, outliers_regular, data_hybrid, labels_hybrid, outliers_hybrid, 
                      thresh_regular, thresh_hybrid, n_clusters):
    """
    Plot clustering results side by side the regular GMM and hybrid-GMM.

    Args:
        data_regular (numpy.ndarray): Original dataset for the regular clustering plot.
        labels_regular (numpy.ndarray): Cluster labels for the regular GMM clustering.
        outliers_regular (numpy.ndarray): Outliers detected by the regular GMM clustering.
        data_hybrid (numpy.ndarray): Dataset for hybrid GMM clustering.
        labels_hybrid (numpy.ndarray): Cluster labels for the hybrid GMM clustering.
        outliers_hybrid (numpy.ndarray): Outliers detected by the hybrid GMM clustering.
        thresh_regular (float): Threshold for regular GMM outlier detection.
        thresh_hybrid (float): Threshold for hybrid GMM outlier detection.
        n_clusters (int): Number of clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Plot regular GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = data_regular[labels_regular == cluster_idx]
        axes[0].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[0].scatter(outliers_regular[:, 0], outliers_regular[:, 1], color='black', label='Outliers', alpha=0.9)
    axes[0].set_title(f"Regular GMM Clustering\n(Outliers Detected by Mahalanobis Distance, threshold: {thresh_regular})")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].legend()

    # Plot hybrid GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = data_hybrid[labels_hybrid == cluster_idx]
        axes[1].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[1].scatter(outliers_hybrid[:, 0], outliers_hybrid[:, 1], color='black', label='Outliers', alpha=0.9)
    axes[1].set_title(f"Hybrid GMM with Fractal Filtering\n(Threshold: {thresh_hybrid})")
    axes[1].set_xlabel("X-axis")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plotClusteredDataWithSeparateMismatches(data_regular, labels_regular, outliers_regular, mismatched_regular, 
                                            data_hybrid, labels_hybrid, outliers_hybrid, mismatched_hybrid, 
                                            thresh_regular, thresh_hybrid, n_clusters):
    """
    Plot clustering results side by side for the regular GMM and hybrid-GMM, highlighting mismatched outliers
    separately for each clustering method.

    Args:
        data_regular (numpy.ndarray): Original dataset for the regular clustering plot.
        labels_regular (numpy.ndarray): Cluster labels for the regular GMM clustering.
        outliers_regular (numpy.ndarray): Outliers detected by the regular GMM clustering.
        mismatched_regular (set): Mismatched outliers detected by the regular GMM.
        data_hybrid (numpy.ndarray): Dataset for hybrid GMM clustering.
        labels_hybrid (numpy.ndarray): Cluster labels for the hybrid GMM clustering.
        outliers_hybrid (numpy.ndarray): Outliers detected by the hybrid GMM clustering.
        mismatched_hybrid (set): Mismatched outliers detected by the hybrid GMM.
        thresh_regular (float): Threshold for regular GMM outlier detection.
        thresh_hybrid (float): Threshold for hybrid GMM outlier detection.
        n_clusters (int): Number of clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Plot regular GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = data_regular[labels_regular == cluster_idx]
        axes[0].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[0].scatter(outliers_regular[:, 0], outliers_regular[:, 1], color='black', label='Outliers', alpha=0.9)
    if len(mismatched_regular) > 0:  # Ensure mismatched_regular is not empty
        mismatched_regular_array = np.array(list(mismatched_regular))  # Convert set to array for plotting
        axes[0].scatter(mismatched_regular_array[:, 0], mismatched_regular_array[:, 1], color='red', marker='x', label='Mismatched Outliers (Regular)', s=100)
    axes[0].set_title(f"Regular GMM Clustering\n(Outliers Detected by Mahalanobis Distance, threshold: {thresh_regular})")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].legend()

    # Plot hybrid GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = data_hybrid[labels_hybrid == cluster_idx]
        axes[1].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[1].scatter(outliers_hybrid[:, 0], outliers_hybrid[:, 1], color='black', label='Outliers', alpha=0.9)
    if len(mismatched_hybrid) > 0:  # Ensure mismatched_hybrid is not empty
        mismatched_hybrid_array = np.array(list(mismatched_hybrid))  # Convert set to array for plotting
        axes[1].scatter(mismatched_hybrid_array[:, 0], mismatched_hybrid_array[:, 1], color='blue', marker='x', label='Mismatched Outliers (Hybrid)', s=100)
    axes[1].set_title(f"Hybrid GMM with Fractal Filtering\n(Threshold: {thresh_hybrid})")
    axes[1].set_xlabel("X-axis")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_results_side_by_side(data, labels_regular, regular_outliers, hybrid_data, hybrid_labels, hybrid_outliers, 
                              hthresh, thresh, n_clusters, actual_outliers):
    """
    Plot clustering results side by side with an additional dataset of actual outliers.
    
    Parameters:
        data (numpy.ndarray): Original dataset for the regular clustering plot.
        labels_regular (numpy.ndarray): Cluster labels for the regular GMM clustering.
        regular_outliers (numpy.ndarray): Outliers detected by the regular GMM clustering.
        hybrid_data (numpy.ndarray): Dataset for hybrid GMM clustering.
        hybrid_labels (numpy.ndarray): Cluster labels for the hybrid GMM clustering.
        hybrid_outliers (numpy.ndarray): Outliers detected by the hybrid GMM clustering.
        hthresh (float): Threshold for hybrid GMM outlier detection.
        thresh (float): Threshold for regular GMM outlier detection.
        n_clusters (int): Number of clusters.
        actual_outliers (numpy.ndarray): Dataset containing actual outliers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Plot actual outliers
    axes[0].scatter(actual_outliers[:, 0], actual_outliers[:, 1], color='black', label='Actual Outliers', alpha=0.9)
    axes[0].set_title("Actual Outliers")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].legend()

    # Plot regular GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = data[labels_regular == cluster_idx]
        axes[1].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[1].scatter(regular_outliers[:, 0], regular_outliers[:, 1], color='black', label='Outliers', alpha=0.9)
    axes[1].set_title(f"Regular GMM Clustering\n(Outliers Detected by Mahalanobis Distance, threshold: {thresh})")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")
    axes[1].legend()

    # Plot hybrid GMM clustering
    for cluster_idx in range(n_clusters):
        cluster_data = hybrid_data[hybrid_labels == cluster_idx]
        axes[2].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))
    axes[2].scatter(hybrid_outliers[:, 0], hybrid_outliers[:, 1], color='black', label='Outliers', alpha=0.9)
    axes[2].set_title(f"Hybrid GMM with Fractal Filtering\n(Threshold: {hthresh})")
    axes[2].set_xlabel("X-axis")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def plot_datasets(valid_pts, outlier_pts):
    """
    Plots two datasets as a scatter plot.
    
    Parameters:
        valid_pts (ndarray): A 2D array representing valid points (blue).
        outlier_pts (ndarray): A 2D array representing outlier points (black).
    """
    # Ensure the inputs are 2D
    if valid_pts.shape[1] != 2 or outlier_pts.shape[1] != 2:
        raise ValueError("Both datasets must be 2D with shape (n_samples, 2).")
    
    # Plot the valid points in blue
    plt.scatter(valid_pts[:, 0], valid_pts[:, 1], color='blue', label='Valid Points')
    
    # Plot the outlier points in black
    plt.scatter(outlier_pts[:, 0], outlier_pts[:, 1], color='black', label='Outlier Points')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Valid and Outlier Points')
    
    # Display the plot
    plt.show()

def plotRegularGMMClustering(data_regular, labels_regular, outliers_regular, mismatched_regular, thresh_regular, n_clusters):
    """
    Plot clustering results for the regular GMM, highlighting mismatched outliers.

    Args:
        data_regular (numpy.ndarray): Original dataset for the regular clustering plot.
        labels_regular (numpy.ndarray): Cluster labels for the regular GMM clustering.
        outliers_regular (numpy.ndarray): Outliers detected by the regular GMM clustering.
        mismatched_regular (set): Mismatched outliers detected by the regular GMM.
        thresh_regular (float): Threshold for regular GMM outlier detection.
        n_clusters (int): Number of clusters.
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    for cluster_idx in range(n_clusters):
        cluster_data = data_regular[labels_regular == cluster_idx]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))

    plt.scatter(outliers_regular[:, 0], outliers_regular[:, 1], color='black', label='Outliers', alpha=0.9)

    if len(mismatched_regular) > 0:  # Ensure mismatched_regular is not empty
        mismatched_regular_array = np.array(list(mismatched_regular))  # Convert set to array for plotting
        plt.scatter(mismatched_regular_array[:, 0], mismatched_regular_array[:, 1], color='red', marker='x', label='Mismatched Outliers (Regular)', s=100)

    plt.title(f"Regular GMM Clustering\n(Outliers Detected by Mahalanobis Distance, threshold: {thresh_regular})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotHybridGMMClustering(data_hybrid, labels_hybrid, outliers_hybrid, mismatched_hybrid, thresh_hybrid, n_clusters):
    """
    Plot clustering results for the hybrid GMM, highlighting mismatched outliers.

    Args:
        data_hybrid (numpy.ndarray): Dataset for hybrid GMM clustering.
        labels_hybrid (numpy.ndarray): Cluster labels for the hybrid GMM clustering.
        outliers_hybrid (numpy.ndarray): Outliers detected by the hybrid GMM clustering.
        mismatched_hybrid (set): Mismatched outliers detected by the hybrid GMM.
        thresh_hybrid (float): Threshold for hybrid GMM outlier detection.
        n_clusters (int): Number of clusters.
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    for cluster_idx in range(n_clusters):
        cluster_data = data_hybrid[labels_hybrid == cluster_idx]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx}', alpha=0.7, color=colors(cluster_idx))

    plt.scatter(outliers_hybrid[:, 0], outliers_hybrid[:, 1], color='black', label='Outliers', alpha=0.9)

    if len(mismatched_hybrid) > 0:  # Ensure mismatched_hybrid is not empty
        mismatched_hybrid_array = np.array(list(mismatched_hybrid))  # Convert set to array for plotting
        plt.scatter(mismatched_hybrid_array[:, 0], mismatched_hybrid_array[:, 1], color='blue', marker='x', label='Mismatched Outliers (Hybrid)', s=100)

    plt.title(f"Hybrid GMM with Fractal Filtering\n(Threshold: {thresh_hybrid})")
    plt.xlabel("X-axis")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotClusteredDataWithSeparateMismatches(data_regular, labels_regular, outliers_regular, mismatched_regular, 
                                            data_hybrid, labels_hybrid, outliers_hybrid, mismatched_hybrid, 
                                            thresh_regular, thresh_hybrid, n_clusters):
    """
    Wrapper function to plot clustering results for both regular GMM and hybrid GMM separately.
    """
    plotRegularGMMClustering(data_regular, labels_regular, outliers_regular, mismatched_regular, thresh_regular, n_clusters)
    plotHybridGMMClustering(data_hybrid, labels_hybrid, outliers_hybrid, mismatched_hybrid, thresh_hybrid, n_clusters)


def write_clustering_data(valid_pts: int, outlier_pts: int, num_clusters: int, k: int, gmm_outliers: int, gmm_false_outliers: int, 
                          gmm_raw_accuracy: float, hgmm_outliers: int, hgmm_false_outliers: int, hgmm_raw_accuracy: float, 
                          output_filename: str):
    """
    Writes a CSV file with the provided parameters. Appends to the file if it exists,
    otherwise creates it with a header.

    Args:
        valid_pts (int): Number of valid points.
        outlier_pts (int): Number of outlier points.
        num_clusters (int): Number of clusters.
        k (int): Nearest-neighbor number employed by fractal dimension calculation.
        gmm_outliers (int): Number of outliers detected via GMM
        gmm_false_outliers (int): Number of false outliers detected via GMM
        gmm_raw_accuracy (float): Raw accuracy value for GMM.
        hgmm_outliers (int): Number of outliers detected via hybrid-GMM
        hgmm_false_outliers (int): Number of false outliers detected via hybrid-GMM
        hgmm_raw_accuracy (float): Raw accuracy value for hybrid-GMM
        output_filename (str): Name of the output CSV file.
    """
    file_exists = os.path.isfile(output_filename)  # Check if the file already exists

    with open(output_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(["valid_pts", "outlier_pts", "num_clusters", "k", "gmm_outliers", "gmm_false_outliers", "gmm_raw_accuracy", 
                             "hgmm_outliers", "hgmm_false_outliers", "hgmm_raw_accuracy"])

        # Write the data row
        writer.writerow([valid_pts, outlier_pts, num_clusters, k, gmm_outliers, gmm_false_outliers, gmm_raw_accuracy, hgmm_outliers, 
                         hgmm_false_outliers, hgmm_raw_accuracy])

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data with a specified number of outliers.")
    parser.add_argument("--file", type=str, default="data500.csv", help="Input data filename (default: \"data500.csv\")")
    parser.add_argument("--samples", type=int, default=500, help="Total number of samples to generate (default: 500)")
    parser.add_argument("--outliers", type=int, default=50, help="Total number of outliers to generate (default: 50)")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for GMM (default: 3)")
    parser.add_argument("--k", type=int, default=2, help="Number of nearest neighbors to consider for Fractal (default: 2)")
    parser.add_argument("--hthresh", type=float, default=1.5, help="Hybrid-GMM threshold (default: 1.5)")
    parser.add_argument("--thresh", type=float, default=1.5, help="GMM threshold (default: 1.5)")
    args = parser.parse_args()

    # Generate synthetic data and outliers
    n_samples = args.samples
    total_outliers = args.outliers
    n_clusters = args.clusters

    # Nearest eighbors
    k = args.k

    # Hybrid-GMM threshold
    hthresh = args.hthresh

    # GMM threshold
    thresh = args.thresh

    # Input data file
    infile = "./data/" + args.file

    #valid_pts, defined_outliers, data, n_clusters = load_points_from_csv(infile)
    #data, defined_outliers, valid_pts = generate_data(n_samples, n_clusters, total_outliers)
    #data, defined_outliers, valid_pts = generate_data_gauss(n_samples, total_outliers)
    data, defined_outliers, valid_pts = generate_data_subtleoutliers(n_samples, total_outliers, n_clusters)

    plot_datasets(valid_pts, defined_outliers)

    print(f"Actual (True) Outliers = {len(defined_outliers)}")

    # Write generated data to CSV
    #ofilename = "data" + str(n_samples) + ".csv"
    #writeData(data, defined_outliers, n_clusters, ofilename)

    # Find the optimal nearest-neighbor count based on the input dataset if NO
    # k was entered by the user
    #if k < 2:
    #    k = find_optimal_k(data, k_min=3, k_max=15)
    k = find_optimal_k(data, k_min=2, k_max=15)

    print(f"Total Clusters = {n_clusters}")
    print(f"Total Points = {len(data)}")
    print(f"Actual (True) Outliers = {defined_outliers.shape[0]}")
    
    # ****************************************************************************** #
    #           - REGULAR GMM (NO PRE-PROCESSING WITH FRACTAL ANALYSIS) -            #
    # ****************************************************************************** #
    
    labels_regular, gmm = cluster_data(data, n_clusters)
    regular_outliers = detect_outliers_regular_gmm(data, labels_regular, gmm, thresh)
    regular_accuracy, regular_mismatch = calculate_accuracy_with_mismatches(regular_outliers, defined_outliers)
    print(f"\nRegular GMM: Outliers detected = {len(regular_outliers)}")
    print(f"Regular GMM: False-Positive Outliers detected = {len(regular_mismatch)}")
    print(f"Regular GMM: Accuracy = {regular_accuracy:.2f}")

    # ****************************************************************************** #
    #             - HYBRID-GMM (PRE-PROCESSING WITH FRACTAL ANALYSIS) -              #
    # ****************************************************************************** #
    
    hybrid_data, hybrid_outliers = filter_by_fractal_dimension(data, hthresh, k)
    hybrid_labels, _ = cluster_data(hybrid_data, n_clusters)
    hybrid_accuracy, hybrid_mismatch = calculate_accuracy_with_mismatches(hybrid_outliers, defined_outliers)
    print(f"\nHybrid GMM: Outliers identified by fractal filtering = {len(hybrid_outliers)}")
    print(f"Hybrid GMM: False-Positive Outliers detected = {len(hybrid_mismatch)}")
    print(f"Hybrid GMM: Accuracy = {hybrid_accuracy:.2f}")

    ocluster_filename = "cluster_data.csv"
    write_clustering_data(len(valid_pts), len(defined_outliers), n_clusters, k, len(regular_outliers), len(regular_mismatch), 
                          regular_accuracy, len(hybrid_outliers), len(hybrid_mismatch), hybrid_accuracy, ocluster_filename)
    
    # Plot actual (true) outliers
    plotData(defined_outliers, "Actual (True) Outliers")

    plotHybridGMMClustering(hybrid_data, hybrid_labels, hybrid_outliers, hybrid_mismatch, hthresh, n_clusters)

    plotRegularGMMClustering(data, labels_regular, regular_outliers, regular_mismatch, thresh, n_clusters)
    
    #plotClusteredDataWithSeparateMismatches(data, labels_regular, regular_outliers,regular_mismatch,
    #                                        hybrid_data, hybrid_labels, hybrid_outliers, hybrid_mismatch, 
    #                                        thresh, hthresh, n_clusters)

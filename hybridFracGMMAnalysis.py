# -------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Richard Haney, Johns Hopkins University Applied Physics Laboratory (JHU/APL)
# All rights reserved.
#
# This software is provided "as is", without warranty of any kind, express or implied, including 
# but not limited to the warranties of merchantability, fitness for a particular purpose, and 
# noninfringement.
#
# Approved for public release: distribution is unlimited.
#
# You may use, modify, and distribute this software in accordance with the terms of the applicable 
# open-source license or your own licensing terms.
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import argparse
import csv
import os

"""
Using fractal dimensional analysis pre-processing approach has FAR FEWER false-positive detections
of outliers/noise when data is then passed to Gaussian Mixture Model (GMM) with low number of k-nearest
neighbors to compute approximated fractal dimensions. 

For example: Shows Regular GMM detects more noise/outliers but to acquire that higher "accuracy" there
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

# Step 1: Generate Synthetic Data (with Specified Number of Outliers)
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
        [[0.6 + 0.1 * i, -0.3 * i], [0.3 * i, 0.8 + 0.1 * i]]  # dynamic elliptical/special transformations
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

# Step 3: Compute Fractal Dimensions
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
        #radii[i] = np.maximum(radii[i], 0.1)  # allow smaller values but avoid instability
        radii[i] = np.maximum(radii[i], 0.0001)  # allow smaller values but avoid instability
    
    # ***************************************************************** #
    # ***************************************************************** #

    # Step 5: Calculate fractal dimensions
    fractal_dimensions = np.zeros(n_samples)
    log_k = np.log(k)
    for i in range(n_samples):
        radius_with_epsilon = np.maximum(radii[i] + epsilon, epsilon)
        log_argument = np.maximum(1.0 / radius_with_epsilon, .09)  # a stricter lower bound would be 2.0
        fractal_dimensions[i] = log_k / np.log(log_argument)
    
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

    # The random_state parameter ensures that the random number generator used internally by the GMM 
    # is initialized to a fixed seed. This guarantees that the same sequence of random numbers will 
    # be used each time the code runs, resulting in the same initialization of the GMM and, 
    # consequently, the same clustering results for the same input data.
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

def write_clustering_data(valid_pts: int, outlier_pts: int, num_clusters: int, k: int, gmm_outliers: int, gmm_false_outliers: int, 
                          gmm_raw_accuracy: float, hgmm_outliers: int, hgmm_false_outliers: int, hgmm_raw_accuracy: float, 
                          output_filename: str):
    """
    Writes a CSV file with the provided parameters. Appends to the file if it exists, otherwise creates it with a header.

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
    parser.add_argument("--samples", type=int, default=500, help="Total number of samples to generate (default: 500)")
    parser.add_argument("--outliers", type=int, default=50, help="Total number of outliers to generate (default: 50)")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for GMM (default: 3)")
    parser.add_argument("--k", type=int, default=2, help="Number of nearest neighbors to consider for Fractal (default: 2)")
    parser.add_argument("--hthresh", type=float, default=0.1, help="Hybrid-GMM threshold (default: 0.1)")
    parser.add_argument("--thresh", type=float, default=1.5, help="GMM threshold (default: 1.5)")
    args = parser.parse_args()

    # Generate synthetic data and outliers
    n_samples = args.samples
    total_outliers = args.outliers
    n_clusters = args.clusters

    # Nearest neighbors
    k = args.k

    # Hybrid-GMM threshold
    hthresh = args.hthresh

    # GMM threshold
    thresh = args.thresh

    # Generate data with outliers
    defined_data, defined_outliers, valid_pts = generate_data_subtleoutliers(n_samples, total_outliers, n_clusters)

    plot_datasets(valid_pts, defined_outliers)

    print(f"Actual (True) Outliers = {len(defined_outliers)}")

    print(f"Total Clusters = {n_clusters}")
    print(f"Total Points = {len(defined_data)}")
    print(f"Actual (True) Outliers = {defined_outliers.shape[0]}")
    
    # ****************************************************************************** #
    #           - REGULAR GMM (NO PRE-PROCESSING WITH FRACTAL ANALYSIS) -            #
    # ****************************************************************************** #
    
    labels_regular, gmm = cluster_data(defined_data, n_clusters)
    regular_outliers = detect_outliers_regular_gmm(defined_data, labels_regular, gmm, thresh)
    regular_accuracy, regular_mismatch = calculate_accuracy_with_mismatches(regular_outliers, defined_outliers)
    print(f"\nRegular GMM: Outliers detected = {len(regular_outliers)}")
    print(f"Regular GMM: False-Positive Outliers detected = {len(regular_mismatch)}")
    print(f"Regular GMM: Accuracy = {regular_accuracy:.2f}")

    # ****************************************************************************** #
    #             - HYBRID-GMM (PRE-PROCESSING WITH FRACTAL ANALYSIS) -              #
    # ****************************************************************************** #
    
    hybrid_data, hybrid_outliers = filter_by_fractal_dimension(defined_data, hthresh, k)
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

    plotRegularGMMClustering(defined_data, labels_regular, regular_outliers, regular_mismatch, thresh, n_clusters)

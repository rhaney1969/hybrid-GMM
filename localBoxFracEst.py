import numpy as np
import argparse
import csv
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def generate_koch_curve_points(target_points, length=1.0):
    """
    Function generates a Koch curve with approximately the value of the input target points.
    """

    # Initialize the points for a straight line segment
    points = np.array([[0, 0], [length, 0]])  # The initial line from (0, 0) to (length, 0)
    
    # Start with the number of points on the initial segment
    num_points = len(points)
    
    # Count iterations required to approach the target number of points
    iterations = 0
    while num_points < target_points:
        # Create new points by splitting each segment into 4 parts
        new_points = []
        for i in range(len(points) - 1):
            # Calculate the 4 new points for each segment
            p0 = points[i]
            p1 = points[i + 1]
            # Calculate distances
            segment = p1 - p0
            # The 1/3 and 2/3 points along the segment
            p2 = p0 + segment / 3
            p3 = p0 + 2 * segment / 3
            
            # Calculate the peak of the triangle (rotate the segment by 60 degrees)
            angle = np.pi / 3
            length = np.linalg.norm(segment) / 3
            peak = p2 + np.array([length * np.cos(angle), length * np.sin(angle)])
            
            # Add the points to the new points list
            new_points.extend([p0, p2, peak, p3])
        
        # Update the points list and the number of points
        points = np.array(new_points + [points[-1]])  # Append the last point
        num_points = len(points)
        iterations += 1
    
    # Return the points that form the Koch curve
    return points

def box_count_local(points, center_point, k_nearest, box_size):
    """
    Function estimates the LOCAL fractal dimension utilizing the box-counting method with a k-nearest
    neighborhood.
    """

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_nearest)
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors([center_point])
    
    # Get the points within the neighborhood
    local_points = points[indices[0]]
    
    # Compute the bounding box of the neighborhood
    min_x, max_x = np.min(local_points[:, 0]), np.max(local_points[:, 0])
    min_y, max_y = np.min(local_points[:, 1]), np.max(local_points[:, 1])
    
    # Create a grid of boxes for the local neighborhood
    boxes = {}
    for x, y in local_points:
        i = int((x - min_x) // box_size)
        j = int((y - min_y) // box_size)
        boxes[(i, j)] = True
    
    return len(boxes)

def estimate_local_fractal_dimension_box_count(points, k_nearest=10, min_box_size=0.05, max_box_size=0.5, num_steps=10):
    """
    Function estimates the local fractal using modified variant of box-counting method - for each point.
    """
    
    local_fractal_dimensions = np.zeros(len(points))
    
    box_sizes = np.logspace(np.log10(max_box_size), np.log10(min_box_size), num_steps)
    
    for i, point in enumerate(points):
        counts = []
        
        for box_size in box_sizes:
            count = box_count_local(points, point, k_nearest, box_size)
            counts.append(count)
        
        # Fit a line to the log-log scale to estimate the local fractal dimension
        log_counts = np.log(counts)
        log_box_sizes = np.log(box_sizes)
        slope, _ = np.polyfit(log_box_sizes, log_counts, 1)
        
        local_fractal_dimensions[i] = -slope  # The slope gives the local fractal dimension
    
    return local_fractal_dimensions

def compute_pairwise_distances(data):
    """
    Function computes pairwise Euclidean distances between all points defined within the data.
    """
    
    data = np.array(data)
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    return distances

def compute_fractal_dimension(data, k, epsilon=1e-3):
    """
    Function computes local fractal dimension for k-nearest neighbors.
    """
    
    data = np.array(data)
    n_samples = data.shape[0]
    
    if k < 1 or n_samples <= k:
        raise ValueError("Invalid k value: k must be between 1 and the number of points - 1.")
    
    distances = compute_pairwise_distances(data)
    sorted_distances = np.sort(distances, axis=1)[:, 1:k+1]  # k-nearest neighbors
    
    local_densities = np.zeros(n_samples)
    for i in range(n_samples):
        mean_distance = np.mean(sorted_distances[i])
        local_densities[i] = 1.0 / (mean_distance + epsilon)
    
    radii = np.zeros(n_samples)
    for i in range(n_samples):
        mean_distance = np.mean(sorted_distances[i])
        radii[i] = np.maximum(np.log(mean_distance + epsilon) / local_densities[i], epsilon)
    
    # Clamping and normalization
    non_zero_radii = np.sort(radii[radii > 0])
    min_radius = non_zero_radii[max(int(0.05 * n_samples), 1)]
    max_radius = non_zero_radii[min(int(0.95 * n_samples), n_samples - 1)]
    min_radius = np.maximum(min_radius, epsilon)
    
    for i in range(n_samples):
        radii[i] = (radii[i] - min_radius) / (max_radius - min_radius + epsilon)
        radii[i] = np.maximum(radii[i], 0.0001)
    
    fractal_dimensions = np.zeros(n_samples)
    log_k = np.log(k)
    for i in range(n_samples):
        radius_with_epsilon = np.maximum(radii[i] + epsilon, epsilon)
        log_argument = np.maximum(1.0 / radius_with_epsilon, .09)
        fractal_dimensions[i] = log_k / np.log(log_argument)
    
    return fractal_dimensions

def compute_levina_bickel_dimension(data, k, epsilon=1e-3):
    """
    Implements the full version of the Levina-Bickel estimator for local intrinsic dimension estimation.

    Parameters:
        data (array-like): Dataset of shape (n_samples, n_features)
        k (int): Number of nearest neighbors
        epsilon (float): Small constant to avoid log(0) or division by zero

    Returns:
        dimensions (numpy array): Estimated local fractal dimension for each point
    """
    data = np.array(data)
    n_samples = data.shape[0]

    if k < 2 or n_samples <= k:
        raise ValueError("Invalid k value: k must be at least 2 and less than the number of points.")

    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)  # Pairwise distance matrix
    sorted_distances = np.sort(distances, axis=1)[:, 1:k+1]  # Exclude self-distance

    dimensions = np.zeros(n_samples)
    for i in range(n_samples):
        r_k = sorted_distances[i, -1] + epsilon  # Distance to the k-th nearest neighbor
        
        # Ensure that distances are not zero by adding epsilon directly
        log_ratios = np.log(sorted_distances[i] + epsilon) - np.log(r_k)
        
        # Avoid negative or zero values in log-ratio by clamping them
        log_ratios = np.maximum(log_ratios, epsilon)  # Clamp the log-ratio to avoid instability

        # Sum the log-ratios and calculate the dimension
        sum_log_ratios = np.sum(log_ratios) / (k - 1)
        
        # The dimension is the negative inverse of the average log-ratio
        dimensions[i] = -1.0 / sum_log_ratios

    return dimensions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare local fractal dimension estimation with local box-counting using Koch fractal curve.")
    parser.add_argument("--points", type=int, default=500, help="Total number of points for Koch fractal curve (default: 500)")
    parser.add_argument("--k", type=int, default=10, help="K-nearest neighbors (default: 10)")
    args = parser.parse_args()

    # Example usage: generate a Koch curve with a desired number of points
    target_points = args.points  # Set the target number of points
    koch_points = generate_koch_curve_points(target_points, length=1.0)
    
    # Now you can use this `koch_points` to estimate the local fractal dimensions

    # Estimate fractal dimension using the compute_fractal_dimension function
    k = args.k # Number of nearest neighbors
    local_fractal_dims = compute_fractal_dimension(koch_points, k)
    
    # Estimate local fractal dimension using the local box-counting method
    local_box_count_dims = estimate_local_fractal_dimension_box_count(koch_points, k_nearest=k)

    # Compute the absolute error between the two methods
    absolute_error = np.abs(local_fractal_dims - local_box_count_dims)
    
    """
    # Estimate local fractal dimension using pure Levina-Bickel estimator
    local_levina_bickel_dims = compute_levina_bickel_dimension(koch_points, k)
    
    # Compute the absolute error between the two methods
    absolute_error_box_count = np.abs(local_fractal_dims - local_box_count_dims)
    absolute_error_levina_bickel = np.abs(local_fractal_dims - local_levina_bickel_dims)

    # Compute the absolute error between the two methods
    absolute_error = np.abs(local_fractal_dims - local_box_count_dims)
    absolute_error_levina_bickel = np.abs(local_fractal_dims - local_levina_bickel_dims)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=local_fractal_dims, cmap='viridis', label='Local Fractal Dimension (compute_fractal_dimension)', alpha=0.5)
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=local_box_count_dims, cmap='plasma', label='Local Fractal Dimension (Box Counting)', alpha=0.5)
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=local_levina_bickel_dims, cmap='inferno', label='Local Fractal Dimension (Levina-Bickel)', alpha=0.5)
    plt.title("Point Cloud with Estimated Local Fractal Dimensions (All Methods)")
    plt.colorbar(label="Fractal Dimension")
    plt.legend()
    plt.show()

    # Plot the absolute error between the methods
    plt.figure(figsize=(10, 6))
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=absolute_error_box_count, cmap='coolwarm', label='Absolute Error (Box Counting vs compute_fractal_dimension)', alpha=0.7)
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=absolute_error_levina_bickel, cmap='twilight', label='Absolute Error (Levina-Bickel vs compute_fractal_dimension)', alpha=0.7)
    plt.title("Absolute Error Between Local Fractal Dimensions (Box Counting, Levina-Bickel vs compute_fractal_dimension)")
    plt.colorbar(label="Absolute Error")
    plt.legend()
    plt.show()

    # Print a comparison
    print("Mean Absolute Error (Box Counting vs compute_fractal_dimension): ", np.mean(absolute_error_box_count))
    print("Mean Absolute Error (Levina-Bickel vs compute_fractal_dimension): ", np.mean(absolute_error_levina_bickel))

    # Print a comparison
    print("Mean Local Fractal Dimension (compute_fractal_dimension): ", np.mean(local_fractal_dims))
    print("Mean Local Fractal Dimension (Box Counting): ", np.mean(local_box_count_dims))
    print("Mean Local Fractal Dimension (Levina-Bickel): ", np.mean(local_levina_bickel_dims))
    """

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=local_fractal_dims, cmap='viridis', label='Local Fractal Dimension (compute_fractal_dimension)', alpha=0.5)
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=local_box_count_dims, cmap='plasma', label='Local Fractal Dimension (Box Counting)', alpha=0.5)
    plt.title("Point Cloud with Estimated Local Fractal Dimensions (Both Methods)")
    plt.colorbar(label="Fractal Dimension")
    plt.legend()
    plt.show()

    # Plot the absolute error between the two methods
    plt.figure(figsize=(10, 6))
    plt.scatter(koch_points[:, 0], koch_points[:, 1], c=absolute_error, cmap='coolwarm', label='Absolute Error', alpha=0.7)
    plt.title("Absolute Error Between Local Fractal Dimensions (Box Counting vs compute_fractal_dimension)")
    plt.colorbar(label="Absolute Error")
    plt.show()

    # Print a comparison
    print("Mean Absolute Error: ", np.mean(absolute_error))
    print("Median Absolute Error: ", np.median(absolute_error))

    # Print a comparison
    print("Mean Local Fractal Dimension (compute_fractal_dimension): ", np.mean(local_fractal_dims))
    print("Mean Local Fractal Dimension (Box Counting): ", np.mean(local_box_count_dims))

    output_filename = "local_fracs.csv"
    file_exists = os.path.isfile(output_filename)  # Check if the file already exists
    with open(output_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(["target_pts", "k", "mean_abs_error", "median_abs_error", "mean_local_frac", "mean_local_box"])

        # Write the data row
        writer.writerow([len(koch_points), k, np.mean(absolute_error), np.median(absolute_error), np.mean(local_fractal_dims), np.mean(local_box_count_dims)])
# Estimating Local Fractal Dimensions: Accuracy Check Between `compute_fractal_dimension` and Box Counting Methods

## Overview

This document explains the Python code used to compare two methods of estimating the **local fractal dimension** for a point cloud: 

1. The **`compute_fractal_dimension`** method.
2. The **local box-counting method** (`estimate_local_fractal_dimension_box_count`).

We use the **Koch curve** as a test case to evaluate the accuracy of the methods. The Koch curve is a well-known fractal that provides a reliable benchmark for validating local fractal dimension estimation methods.

---

## Concepts and Algorithms

### Fractal Dimension

The **fractal dimension** is a measure of complexity in geometric structures, specifically how a fractal pattern scales with respect to its size. For a given set of data points or a geometric object, the fractal dimension quantifies the object's **self-similarity** at different scales.

In the context of point clouds or geometric figures, local fractal dimensions estimate how the density of points scales in the vicinity of each point.

### Local Fractal Dimension

Local fractal dimension estimation refers to the method of calculating the fractal dimension at each individual point in a point cloud. There are two methods implemented here:

- **`compute_fractal_dimension`**: This method uses pairwise distances to estimate local fractal dimensions.
- **Box-Counting Method**: This method estimates the fractal dimension by counting the number of boxes of a certain size that are needed to cover a subset of the data points.

### Box Counting Method

The **box-counting method** is a classic approach for estimating the fractal dimension of a geometric object. Here's how it works:
- A grid of boxes (of various sizes) is overlaid on the data points.
- The number of boxes required to cover the points is counted.
- The scaling behavior of the number of boxes with respect to the box size is used to estimate the fractal dimension.

Mathematically, the box-counting method follows:
\[
D = \lim_{\epsilon \to 0} \frac{\log(N(\epsilon))}{\log(1/\epsilon)}
\]
Where:
- \( N(\epsilon) \) is the number of boxes of size \( \epsilon \) required to cover the data points.
- \( D \) is the estimated fractal dimension.

In the local box-counting method, this procedure is applied to a neighborhood around each data point, yielding a **local fractal dimension** for each point.

---

## The Koch Curve

The **Koch curve** is a well-known fractal that starts with a straight line segment and recursively replaces each line segment with four new segments arranged in a specific pattern. This process increases the total number of points exponentially, and the shape becomes increasingly jagged with each iteration.

### Properties of the Koch Curve:
- It is a **self-similar fractal**, meaning that its shape looks similar at any scale.
- Its **fractal dimension** is approximately **1.2619**, which is greater than the dimension of a straight line (1), but less than the dimension of a plane (2).

The Koch curve serves as a good **benchmark** for fractal dimension estimation because:
- Its known fractal dimension allows us to compare the accuracy of the estimation methods.
- It is relatively simple to generate with a known recursive structure.

---

## The Python Code

The Python code provided estimates local fractal dimensions for a point cloud generated from a **Koch curve** and compares the results between the two methods: `compute_fractal_dimension` and the local box-counting method.

### Code Overview

1. **Generating the Koch Curve**: 
   - The Koch curve is generated iteratively. Initially, a line segment is drawn, and then each segment is recursively divided into four smaller segments (creating the characteristic "bump").
   - The number of points in the curve grows exponentially with each iteration.

2. **`compute_fractal_dimension`**: 
   - This function estimates the local fractal dimension based on pairwise distances between points and their nearest neighbors. The dimensions are calculated by considering the mean distance between a point and its \(k\)-nearest neighbors and using this information to estimate local density and radii.
   - The function applies a formula to compute the fractal dimension based on logarithmic relationships.

3. **Local Box-Counting Method**: 
   - This method estimates the local fractal dimension by applying a box-counting approach around each point. The number of boxes of different sizes needed to cover a set of nearest neighbors is calculated, and the logarithmic scaling is used to estimate the local fractal dimension.

4. **Comparison**: 
   - The local fractal dimensions estimated by the two methods are compared, and the **absolute error** is computed.
   - The results are visualized by plotting the **fractal dimensions** and **absolute errors** between the two methods.

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def box_count_local(points, center_point, k_nearest, box_size):
    """Estimate the local fractal dimension using the box-counting method within a k-nearest neighborhood."""
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
    """Estimate the local fractal dimensions using the box-counting method for each point."""
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

def generate_koch_curve_points(target_points, length=1.0):
    """Generates a Koch curve with approximately 'target_points' points."""
    points = np.array([[0, 0], [length, 0]])  # The initial line from (0, 0) to (length, 0)
    num_points = len(points)
    iterations = 0
    while num_points < target_points:
        new_points = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            segment = p1 - p0
            p2 = p0 + segment / 3
            p3 = p0 + 2 * segment / 3
            angle = np.pi / 3
            length = np.linalg.norm(segment) / 3
            peak = p2 + np.array([length * np.cos(angle), length * np.sin(angle)])
            new_points.extend([p0, p2, peak, p3])
        points = np.array(new_points + [points[-1]])
        num_points = len(points)
        iterations += 1
    return points
```

---

## Key Results

1. **Koch Curve as a Benchmark**:
   - The Koch curve is an excellent benchmark for testing fractal dimension estimation methods due to its known fractal dimension (\( \approx 1.2619 \)) and self-similarity.

2. **Comparing Estimation Methods**:
   - The comparison between the **`compute_fractal_dimension`** and the **local box-counting method** is carried out using **absolute error**.
   - Results show that both methods estimate similar fractal dimensions for the Koch curve, with the error being minimal.

---

## Summary

The **Koch curve** provides a controlled and predictable structure for testing fractal dimension estimation methods. The **local box-counting** and **pairwise distance-based methods** give comparable results when estimating the fractal dimensions of the Koch curve. The Python code we employed allows for computation and comparison of these methods, and the accuracy of the methods can be assessed by analyzing the error between the two approaches.

This work highlights the effectiveness of the **box-counting method** and the **pairwise distance method** in estimating the **local fractal dimension** and provides a foundation for further exploration in analyzing more complex fractal shapes or point clouds.

---

## Relationship to `Levina-Bickel estimator`

The algorithm implemented by the Python function `compute_fractal_dimension` is *closer* in relation to a version of the `Levina-Bickel estimator` rather than the traditional `localized box-counting` method. This is because of the following:

### **Localized Box-Counting Method**

The **localized box-counting method** is a straightforward fractal dimension estimation method. It works by:
1. Defining a neighborhood around each point (often using the \(k\)-nearest neighbors).
2. Counting how many boxes of various sizes are needed to cover the points within that neighborhood.
3. Plotting the logarithmic relationship between the box size and the number of boxes needed to cover the points.
4. Fitting a straight line to this log-log plot to estimate the local fractal dimension.

In the function `compute_fractal_dimension`:
- Does *explicitly* involve **box counting**.
- Instead, the method computes distances between points (or their neighbors) and uses those distances in a **logarithmic** relationship to compute a local fractal dimension. 
- The relationship in the function focuses on **local density** estimates (using pairwise distances and their mean), *not* on a grid of boxes covering the points.

### **Levina-Bickel Estimator (or Local Fractal Dimension Estimation via Pairwise Distances)**

The **Levina-Bickel estimator** is a method for estimating the **local fractal dimension** of a point cloud that focuses on pairwise distances between points (often, the \(k\)-nearest neighbors). The estimator works as follows:
1. For each point, compute the pairwise distances to its \(k\)-nearest neighbors.
2. Use the **mean distance** to estimate the **local density**.
3. The method leverages the relationship between the density and the **fractal dimension** by considering how the distances between points scale.

The function `compute_fractal_dimension`:
- Computes **pairwise distances** between points using `np.linalg.norm`.
- Estimates **local density** based on the mean distance to the \(k\)-nearest neighbors and uses a **logarithmic** relationship to estimate the fractal dimension.
- This is similar to the **Levina-Bickel estimator**, where the **local density** is related to how the distances between neighbors behave.

In particular, the method uses:
- The **mean distance** to the \(k\)-nearest neighbors to calculate the local density.
- The relationship between the **local radii** (based on distances) and the **logarithmic scaling** of those distances to estimate fractal dimensions, which closely mirrors the **Levina-Bickel approach**.

### Key Differences and Similarities:

1. **Box-Counting vs. Pairwise Distance**:
   - **Box-counting** *explicitly* involves creating a grid of boxes and counting how many are needed to cover a set of points. The function `compute_fractal_dimension` does not involve such a grid-based procedure.
   - `compute_fractal_dimension` works based on **pairwise distances**, which is more in line with methods like **Levina-Bickel**, rather than localized box counting.

2. **Logarithmic Scaling**:
   - Both methods involve **logarithmic scaling**, but the box-counting method does it in terms of box sizes and counts, while the **Levina-Bickel estimator** scales in terms of distances and local densities.

3. **Levina-Bickel Estimator**:
   - The **Levina-Bickel** method estimates the local fractal dimension using the **logarithmic relationship** between distances in a point cloud, which is exactly what your function is doing.
   - The method involves an estimate of **local density**, where the scaling of point separations reflects the fractal nature of the point cloud.

---

### Conclusion:

The `compute_fractal_dimension` function is **closer to the Levina-Bickel estimator** because it:
- Relies on **pairwise distances** between points (or their \(k\)-nearest neighbors).
- Uses a **logarithmic relationship** between distances and local density.
- Estimates the fractal dimension based on the scaling behavior of distances within local neighborhoods.

On the other hand, the **localized box-counting method** would involve covering the neighborhood with boxes of various sizes and counting how many are required to cover the points in the neighborhood, which is not part of your method.

Therefore, the function `compute_fractal_dimension` seems to implement a form of **local fractal dimension estimation using pairwise distances**, akin to the **Levina-Bickel estimator**, rather than the **localized box-counting method**.

---

## The Novelity of `compute_fractal_dimension`: Point Clouds

Based on testing and comparison with well-established fractal dimension estimators (such as the **Levina-Bickel estimator**) on known fractals like the **Sierpinski triangle** and **Sierpinski carpet**, it's quite possible that the `compute_fractal_dimension` function is better suited for **point clouds** or **irregular datasets** rather than **well-defined mathematical fractals**. 

Key points to consider:

### 1. **Nature of the Method**
   The `compute_fractal_dimension` method primarily relies on **local distances** between points, as well as **density estimation** through nearest neighbors. This approach works well for *point clouds* where the fractal structure is **irregular** and **not defined by a mathematical formula** like the Sierpinski triangle or carpet. These kinds of data sets are less regular and more **noisy** in nature.

   - **Point clouds** tend to have noisy, fragmented, or continuous structures, which makes them better suited to methods that use **local spatial relationships** and **density estimates** (which is what `compute_fractal_dimension` does).
   - **Mathematical fractals**, such as the Sierpinski triangle or carpet, are typically **self-similar** and **precisely defined**. Methods like the **Levina-Bickel estimator** and **box-counting** are more suited for these types of fractals because they can capture the scaling behavior inherent in these self-similar structures.

### 2. **Local vs Global Fractal Dimension**
   The `compute_fractal_dimension` function seems to estimate **local fractal dimension** by considering the density and neighborhood relationships around each point. However, **local fractal dimensions** may not align well with the **global fractal dimension** of highly structured fractals like the Sierpinski triangle and carpet, which are typically used to measure the overall **global fractal dimension**.

   - **Sierpinski triangle/carpet** are known to have a fixed global fractal dimension, and methods that focus on local estimation (like `compute_fractal_dimension`) can miss this because their focus is on local variations rather than the global scaling properties.
   - **Levina-Bickel and box-counting methods** capture more of this **global self-similarity**, which is why they perform well on these classic fractals.

### 3. **Point Cloud Data and Irregular Fractals**
   The `compute_fractal_dimension` method could potentially be more useful for datasets that are **irregular or noisy** (such as point clouds, natural images, or biological data) where there might not be a clearly defined global fractal dimension. For such data:

   - Local fractal dimension estimation can provide insights into the **local complexity** and **density variations** of the dataset. This could be useful in fields like **pattern recognition**, **neural networks**, **machine learning**, or **spatial analysis**, where point clouds and irregular structures are common.
   - It may also be beneficial when trying to analyze fractal-like objects that don’t have explicit **self-similarity** but still exhibit **local scaling behavior** in certain regions.

### 4. **Possible Improvements or Revisions**
   - **Hybrid Approaches**: It’s possible that `compute_fractal_dimension` could be adapted or extended to work better with self-similar fractals. For example, adding a more **global scaling analysis** alongside the local density estimation could help it match more closely with established methods when dealing with well-defined fractals.
   - **Specific Tuning for Irregular Data**: Given its current performance, this function might be best for **irregular or non-ideal fractals**, and not for mathematical fractals. Thus, it could be tailored to point cloud data, where local scaling properties are more critical than global self-similarity.

### 5. **Domain-Specific Use Case**
   If you are dealing with **point cloud data** in **3D scanning**, **computer vision**, or **geospatial data**, `compute_fractal_dimension` might be more useful because these applications often deal with objects or structures that don’t necessarily adhere to simple, well-known fractals like the Sierpinski series. In these cases, **local fractal dimension estimation** is more important for capturing variations in spatial complexity.

### Conclusion
The `compute_fractal_dimension` function is likely **not ideal for mathematical fractals like the Sierpinski triangle or carpet**, primarily because these fractals exhibit well-defined global scaling properties, which are better captured by methods like **Levina-Bickel** and **box-counting**.

However, for **point cloud data** or **irregular, noisy fractal-like data**, this method may be quite valuable, as it is designed to capture **local variations in fractality** and **density-based scaling**. 

Therefore, the function `compute_fractal_dimension` could be considered a **novel method** for estimating **local fractal dimensions** in real-world, irregular, or noisy datasets rather than an alternative to global fractal dimension estimators like those based on box-counting or Levina-Bickel.
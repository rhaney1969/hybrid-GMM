# README.md

## Usage

The latest, most reliable Python code is the file `hybridFracGMMAnalysis_v3.py` as the calculation of fractal dimension is less extreme. While the
values of fractal dimension are not always what one would expect in a purely geometric data layout, they are sufficiently fractal-like to be employed
as a filter that defines higher **F1-Score** and lower **False Positive Rate** when the problem space becomes sufficiently complex.

As the different methods employ different approaches, the hybrid-GMM uses fractal dimension which is geometric in nature, and the GMM with Mahalanobis
distance is probablistic, different filtering thresholds are defined. The probablistic approach, when given a lower threshold results in an unfair 
advantage for hybrid-GMM with regards to being overly conservative and high false postive results. Therefore in an attempt for more fair comparison,
different thresholds were passed to the GMM with Mahalanobis distance and the hybrid-GMM. Using empirical results, a threshold of `1.5` was set
for the probablistic method and a threshold of `0.1` was chosen for the geometric method.

## Background

### Gaussian Mixture Models (GMM)
A **Gaussian Mixture Model (GMM)** is a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean, variance, and weight. GMMs are widely used in clustering, density estimation, and anomaly detection tasks. The algorithm fits the data by using the **Expectation-Maximization (EM)** algorithm, which iteratively estimates the parameters of the Gaussian components and assigns data points to the most likely cluster. This model is particularly useful for data that exhibits multimodal distributions.

### Fractal Analysis
**Fractal analysis** involves the study of structures or patterns that exhibit self-similarity at various scales. These structures are characterized by a **fractal dimension**, which quantifies the complexity or roughness of a pattern. The **Box-counting method** is one of the most common techniques to estimate the fractal dimension. It works by counting how many boxes of different sizes are required to cover the data points, and the relationship between the box size and the count is used to compute the fractal dimension.

In the context of data analysis, fractal dimension can reveal intricate structures in data that may be overlooked by traditional methods. It can be particularly helpful for identifying clusters or noise in datasets that exhibit complex, non-linear relationships.

### Combining GMM and Fractal Analysis
The combination of **Gaussian Mixture Models** and **Fractal Analysis** has the potential to provide enhanced clustering and noise filtering capabilities. While GMM is effective for identifying clusters in data, it may struggle with noise and non-linear patterns. Fractal analysis, on the other hand, can be used to evaluate the complexity of neighborhoods around each data point, allowing us to filter out noise and enhance the clustering process.

By incorporating fractal features into the data before applying GMM, we can improve the model's ability to distinguish meaningful clusters from noise. This approach helps capture complex patterns that may not be easily detected using traditional clustering techniques alone.

## Code Explanation

This Python code demonstrates how to integrate **Fractal Analysis** with **Gaussian Mixture Models** (GMM) for clustering and noise filtering in synthetic data. Below is a step-by-step explanation of the process.

### Step 1: Generate Synthetic Data (with Noise)
Synthetic data is generated using the `make_blobs` function, which creates clusters of data points. Noise is then added to the data by generating random points that do not belong to any cluster.

### Step 2: Fractal Dimension Calculation (Local Density Method)

The **Local Density method** is used to compute the fractal dimension of each point in the dataset by analyzing its local neighborhood. Instead of using a global measure like the box-counting method, this approach calculates the fractal dimension for each point based on its k-nearest neighbors. The process involves the following key steps:

1. **Pairwise Distances**: The algorithm computes the pairwise Euclidean distances between all points in the dataset.
2. **k-Nearest Neighbors**: For each point, the k-nearest neighbors are identified based on the sorted pairwise distances.
3. **Local Density Calculation**: The local density of each point is estimated by averaging the distances to its k-nearest neighbors. The local density is then defined as the inverse of the mean distance, which gives a measure of how concentrated the points are in the local region.
4. **Average Radius**: For each point, an average radius is computed by scaling the mean of the k-nearest neighbor distances with the local density. This radius is clamped and normalized to ensure numerical stability across the dataset.
5. **Fractal Dimension**: The fractal dimension is calculated for each point using the formula:
   \[
   D = \frac{\log(k)}{\log(\frac{1}{r})}
   \]
   where \( r \) is the calculated radius for each point, and \( k \) is the number of nearest neighbors. This local calculation reflects how the density and spatial arrangement of neighboring points influence the fractal structure of the data.

This method allows for a more localized and nuanced estimation of fractal dimensions, as opposed to the global box-counting approach, which looks at the overall structure of the dataset without considering the local distribution of points.

### Step 3: Filter Noise Based on Fractal Dimension
Noise filtering is performed based on the fractal dimension. For each data point, the fractal dimension of its neighborhood is calculated. If the fractal dimension exceeds a threshold, the point is retained. Otherwise, it is considered noise and filtered out.

### Step 4: Combine Fractal Features and Apply Gaussian Mixture Model (GMM)
Fractal features are computed for each data point, and these features are appended to the original data. The modified data is then passed to a GMM for clustering. The GMM is used to model the data as a mixture of Gaussian distributions, and each data point is assigned to a cluster based on its likelihood.

### Step 5: GMM Without Fractal Analysis
For comparison, a second GMM is applied to the original data (without fractal features). This allows us to evaluate the impact of fractal analysis on clustering performance.

### Step 6: Visualization
The clustering results are visualized in three subplots:
1. The original data with noise.
2. The GMM clustering results after incorporating fractal features.
3. The GMM clustering results without fractal features.

## Potential Benefits of Combining GMM and Fractal Analysis
1. **Enhanced Clustering Accuracy:** By incorporating fractal features, the model can better handle complex, non-linear patterns in the data.
2. **Noise Filtering:** Fractal dimension can help distinguish between meaningful data points and noise, improving the quality of clustering.
3. **Multiscale Insight:** Fractal analysis captures patterns at multiple scales, making it especially useful for identifying clusters in data with varying levels of granularity.

## Requirements

- **Python 3.x**
- `numpy`
- `matplotlib`
- `sklearn`
- `scipy`

You can install the required libraries using the following command:

```bash
pip install numpy matplotlib scikit-learn scipy
```

## Example Usage

To run the code and see the clustering results with and without fractal analysis, simply execute the following Python script.

```bash
python3 ./hybridFracGMMAnalysis_v3.py
```

This will display the original data, GMM results with fractal analysis, and GMM results without fractal analysis in a single visualization. 

## Conclusion

By integrating Gaussian Mixture Models with Fractal Analysis, this approach enhances the ability to cluster complex, noisy datasets and extract more meaningful patterns. The fractal dimension provides a novel feature that helps to distinguish between structure and noise, leading to more robust and accurate clustering results.

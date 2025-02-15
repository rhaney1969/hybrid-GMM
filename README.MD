# Hybrid-GMM Analysis

This file briefly explains the background of the **Hybrid-GMM** research, that is, the application of fractal dimensional analysis to Gaussian Mixture Model (**GMM**). The **Hybrid-GMM** clustering performance is compared to the ground state, defined by the **GMM** with applied **Mahalanobis** (**M-GMM**).

The directory, `data_results`, contains all generated data analysis results from the execution of our method, **Hybrid-GMM**, in comparison to the ground state **M-GMM**. These results are what is placed in our paper and are stored here in compressed format `data_results.tar.gz`

## Usage

The Python file `hybridFracGMMAnalysis.py` is the main file that defines Hybrid Gaussian Mixture Model (**Hybrid-GMM**) method and the **Mahalanobis** with Gaussian Mixture Model (**M-GMM**) comparison approach, i.e., the *ground state*. While the values of fractal dimension are not always what one would expect in a *purely* geometric data layout, they are sufficiently *fractal-like* to be employed as a filter that defines higher **F1-Score** and lower **False Positive Rate** when the problem space becomes sufficiently complex.

As the different methods employ different approaches, the **Hybrid-GMM** uses fractal dimension which is geometric in nature, and the **M-GMM** is probablistic, different filtering thresholds are defined. The probablistic approach, when given a lower threshold results in an unfair advantage for **Hybrid-GMM** with regards to being overly conservative and high false positive results. Therefore in an attempt for more fair comparison, different thresholds were passed to the **M-GMM** and the **Hybrid-GMM**. Using empirical results, a threshold of `1.5` was set for the probablistic method and a threshold of `0.1` was chosen for the geometric method.

## Background

### Gaussian Mixture Models (GMM)
A **Gaussian Mixture Model (GMM)** is a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean, variance, and weight. **GMM**s are widely used in clustering, density estimation, and anomaly detection tasks. The algorithm fits the data by using the **Expectation-Maximization (EM)** algorithm, which iteratively estimates the parameters of the Gaussian components and assigns data points to the most likely cluster. This model is particularly useful for data that exhibits multimodal distributions.

### Fractal Analysis
**Fractal analysis** involves the study of structures or patterns that exhibit self-similarity at various scales. These structures are characterized by a **fractal dimension**, which quantifies the complexity or roughness of a pattern. The **Box-counting method** is one of the most common techniques to estimate the fractal dimension. It works by counting how many boxes of different sizes are required to cover the data points, and the relationship between the box size and the count is used to compute the fractal dimension.

In the context of data analysis, fractal dimension can reveal intricate structures in data that may be overlooked by traditional methods. It can be particularly helpful for identifying clusters or noise in datasets that exhibit complex, non-linear relationships.

### Combining GMM and Fractal Analysis
The combination of **Gaussian Mixture Models** and **Fractal Analysis** can provide enhanced clustering and noise filtering capabilities. While **GMM**s are effective for identifying clusters in data, they can struggle with noise and non-linear patterns. The Fractal analysis, when properly applied, could be utilized to evaluate the complexity of neighborhoods around each data point, enabling the filtering out of noise and enhancing the clustering process.

By incorporating fractal features into the data *before* applying **GMM**s, we have the potential to improve the model's ability to distinguish *meaningful* clusters from noise. This approach helps capture complex patterns that may not be easily detected using the traditional clustering technique alone, such as **GMM**s.

## Code Explanation

The Python file `hybridFracGMMAnalysis.py` demonstrates the integration of **Fractal Analysis** with **Gaussian Mixture Models** (**Hybrid-GMM**) for clustering and noise filtering in 2-dimensional synthetic data. 

The following provides a brief discussion of this process.

### Step 1: Generate Synthetic Data (with Noise)
The 2-dimensional synthetic data is generated using the `make_blobs` function, which creates clusters of data points. Noise is then added to the data by generating random points that do not belong to any cluster.

### Step 2: Fractal Dimension Calculation (Local Density Method)

The **Local Density method** is used to compute the fractal dimension of each point in the dataset by analyzing its *local* neighborhood. Instead of using a *global* measure like the **box-counting** method, this approach calculates the fractal dimension for *each* point based on its k-nearest neighbors. The process of calculating the local fractal dimension is described in the following set of steps:

Given a dataset \(X = \{x_1, x_2, \dots, x_n\}\) with \(n\) samples in a feature space of dimension \(d\), we compute the fractal dimension using the following steps:

1. **Pairwise Distances**: The algorithm computes the pairwise **Euclidean** distances between all points in the dataset.
\[
    d_{ij} = \|x_i - x_j\|
\]
where \(\| \cdot \|\) denotes the **Euclidean** norm.

2. **Sorting Distances to Find k-Nearest Neighbors**: For each point \(x_i\), we sort the distances and extract the \(k\) smallest distances (excluding self-distance).

3. **Local Density Estimation**: The local density around each point is *estimated* as:
\[
    \rho_i = \frac{1}{\bar{d}_i + \varepsilon}
\]
where \(\bar{d}_i\) is the mean distance to the \(k\) nearest neighbors, and \(\varepsilon\) is a small constant to prevent division by zero.

4. **Computation of Average Radii**: The radius around each point is given by:
\[
    r_i = \max\left(\frac{\log(\bar{d}_i + \varepsilon)}{\rho_i}, \varepsilon\right)
\]

   - To improve robustness, the radii are clamped based on the 5th and 95th percentiles:
\[
    r_i = \frac{r_i - r_{\min}}{r_{\max} - r_{\min} + \varepsilon}
\]
where \(r_{\min}\) and \(r_{\max}\) are computed from the nonzero radii values in the dataset and define the 5th and 95th percentiles respectively.

5. **Fractal Dimension Calculation**: Lastly, the fractal dimension for each point is computed as:
\[
    D_i = \frac{\log k}{\log \max\left(\frac{1}{r_i + \varepsilon}, \delta\right)}
\]
where a lower bound of $\delta$, `0.09`, is enforced for numerical stability.

This method allows for a more *localized* and nuanced estimation of fractal dimensions, as opposed to the *global* **box-counting** approach, which looks at the overall structure of the dataset without considering the local distribution of points. As mentioned in the paper, this is method is not invented by us but was gleaned from the set of references declared in the paper.

### Step 3: Filter Noise Based on Fractal Dimension
Noise filtering is performed based on the fractal dimension. For each data point, the fractal dimension of its neighborhood is calculated. If the fractal dimension exceeds a threshold, the point is retained. Otherwise, it is considered noise and filtered out.

### Step 4: Combine Fractal Features and Apply Gaussian Mixture Model (GMM)
Fractal features are computed for each data point, and these features are appended to the original data. The modified data is then passed to a **GMM** for clustering. The **GMM** is used to model the data as a mixture of Gaussian distributions, and each data point is assigned to a cluster based on its likelihood.

### Step 5: GMM Without Fractal Analysis
For comparison, a second **GMM** is applied to the original data (*without* fractal features applied), whereby the **Mahalanobis** distance from each cluster is employed as the threshold for definition of noise or outlier data points. This allows us to evaluate the impact of fractal analysis on clustering performance using a *ground state*.

### Step 6: Visualization
The clustering results are visualized in three subplots:
1. The original data with noise.
2. The **GMM** clustering results after incorporating fractal features, e.g., **Hybrid-GMM**.
3. The **GMM** clustering results without fractal features but employing the **Mahalanobis** distance threshold, e.g., **M-GMM**.

## Some Potential Benefits of Hybrid-GMM
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

To run the code and see the clustering results with *and* without fractal analysis, simply execute the following Python script.

```bash
python3 ./hybridFracGMMAnalysis.py
```

This will display the original data, **Hybrid-GMM** results, and **M-GMM** results as visualized scatter plots and generate an ASCII CSV file with critical data for further analysis via the other Python files in this code base. 

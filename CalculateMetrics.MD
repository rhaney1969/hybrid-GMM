# Calculating Metrics

This file briefly explains how the performance metrics for Hybrid-GMM and M-GMM are calculated from the resulting ASCII CSV file generated by the execution of the Python `hybridFracGMMAnalysis.py` file.

## Generate ASCII CSV Input File

The first step is to execute the Python script `hybridFracGMMAnalysis.py` with the input parameters, e.g., number of clusters, points, k-nearest neighbors, etc., with the following example that executes with 5 clusters, 3000 points, 117 outliers and 23 k-nearest neighbors:

```bash
python ./hybridGMMAnalysis.py --clusters 5 --outliers 117 --samples 3000 --k 23
```

After execution, the above line will generate scatter plots, that can be saved for visual records, and append to an external ASCII CSV file (if it exists) called `cluster_data.csv`. 

To generate useful metrics the `hybridGMMAnalysis.py` code should be executed with differing inputs, such as increasing numbers of outliers to build entries to the `cluster_data.csv` file.

## Ingest ASCII CSV File

Once you have a sufficient number of entries defined in the ASCII CSV file `cluster_data.csv`, pass this file as input to the Python script `plot_cluster_metrics_with_k.py`, as follows:

```bash
python3 ./plot_cluster_metrics_with_k.py
```

The above script will prompt you for the input CSV file (e.g., `cluster_data.csv`), the number of `valid sample points`, the number of `clusters`, and the value of the k-nearest neighbors, e.g., `k`, as shown in the following example:

```text
Enter the input CSV file, e.g., cluster_data.csv: cluster_data.csv
Enter the valid number of points: 3000
Enter the number of clusters: 5
Enter the value of k: 23
```

The `plot_cluster_metrics_with_k.py` script will iterate through the input ASCII CSV file grouping the input entries (based on the inputs `valid points`, `clusters`, and `k` (k-nearest neighbors)) generating plots for F1-Score, False Positive Rate, and Raw Accuracy. These plots can be saved as a visual record. Along with the plots, the script will create two ASCII CSV files, `f1score.csv` and `fpr.csv`, that contain the hardcopy output of data that was employed to generate the visual plots.
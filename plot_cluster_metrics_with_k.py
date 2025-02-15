"""
This file provides functionality for validating input data and generating metrics plots (F1-score, False Positive Rate, and raw accuracy) 
based on clustering analysis for Gaussian Mixture Models with Mahalanobis distance for thresholds (M-GMM) and the Gaussian Mixture Models
with applied fractal dimensional analysis for filtering as a hybrid method (Hybrid-GMM).

Functions:
    validate_input(csv_filename: str, valid_pts: int, num_clusters: int, k: int) -> bool
    generate_metrics_plot_f1(file_path: str, valid_pts: int, num_clusters: int, k: int, output_csv_path: str) -> None
    generate_metrics_plot_fpr(file_path: str, valid_pts: int, num_clusters: int, k: int, output_csv_path: str) -> None
    generate_metrics_plot_raw_accuracy(file_path: str, valid_pts: int, num_clusters: int, k: int) -> None
"""

import csv
import matplotlib.pyplot as plt
import os
import sys

def validate_input(csv_filename: str, valid_pts: int, num_clusters: int, k: int) -> bool:
    """
    Validates the existence of a CSV file and checks for matching records.

    Args:
        csv_filename (str): Path to the CSV file.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        k (int): Additional parameter to match.

    Returns:
        bool: True if the file exists and contains matching data, False otherwise.
    """
    if not os.path.isfile(csv_filename):
        print(f"WARNING: The file `{csv_filename}` does not exist.")
        return False
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (int(row['valid_pts']) == valid_pts and 
                int(row['num_clusters']) == num_clusters and 
                int(row['k']) == k):
                return True

    print("WARNING: No matching data found for the provided input arguments.")
    return False

def generate_metrics_plot_f1(file_path, valid_pts, num_clusters, k, output_csv_path):
    """
    Generates and displays an F1-score comparison plot for M-GMM and Hybrid-GMM models,
    and saves the data to a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        k (int): Additional parameter to match.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    f1_data = {'gmm_f1': [], 'hgmm_f1': []}
    x_values = []
    csv_output_data = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            if (int(row['valid_pts']) == valid_pts and 
                int(row['num_clusters']) == num_clusters and 
                int(row['k']) == k):
                
                # True Positives for M-GMM
                gmm_tp = int(row['gmm_outliers']) - int(row['gmm_false_outliers'])
                # False Positives for M-GMM
                gmm_fp = int(row['gmm_false_outliers'])
                # True Positives for Hybrid-GMM
                hgmm_tp = int(row['hgmm_outliers']) - int(row['hgmm_false_outliers'])
                # False Positives for Hybrid-GMM
                hgmm_fp = int(row['hgmm_false_outliers'])

                # False Negatives for M-GMM
                gmm_fn = int(row['outlier_pts']) - gmm_tp
                # False Negatives for Hybrid-GMM
                hgmm_fn = int(row['outlier_pts']) - hgmm_tp

                # Precision for M-GMM
                gmm_precision = float(gmm_tp / (gmm_tp + gmm_fp))
                # Precision for Hybrid-GMM
                hgmm_precision = float(hgmm_tp / (hgmm_tp + hgmm_fp))

                # Recall for M-GMM
                gmm_recall = float(gmm_tp / (gmm_tp + gmm_fn))
                # Recall for Hybrid-GMM
                hgmm_recall = float(hgmm_tp / (hgmm_tp + hgmm_fn))

                # Final F1-Score for M-GMM
                gmm_f1 = 2 * gmm_precision * gmm_recall / (gmm_precision + gmm_recall)
                # Final F1-Score for Hybrid-GMM
                hgmm_f1 = 2 * hgmm_precision * hgmm_recall / (hgmm_precision + hgmm_recall)

                f1_data['gmm_f1'].append(gmm_f1)
                f1_data['hgmm_f1'].append(hgmm_f1)

                outlier_pts = int(row['outlier_pts'])
                x_values.append(outlier_pts)

                csv_output_data.append({
                    'valid_pts': valid_pts,
                    'num_clusters': num_clusters,
                    'k': k,
                    'outlier_pts': outlier_pts,
                    'gmm_f1': gmm_f1,
                    'hgmm_f1': hgmm_f1
                })

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['valid_pts', 'num_clusters', 'k', 'outlier_pts', 'gmm_f1', 'hgmm_f1']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(csv_output_data)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f1_data['gmm_f1'], label='GMM F1-Score', marker='o')
    plt.plot(x_values, f1_data['hgmm_f1'], label='Hybrid-GMM F1-Score', marker='x')
    plt.xlabel('True Outliers')
    plt.ylabel('F1-Score')
    plt.title(f'F1-Score Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}, k={k}')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_metrics_plot_fpr(file_path, valid_pts, num_clusters, k, output_csv_path):
    """
    Generates and displays a False Positive Rate (FPR) comparison plot for M-GMM and Hybrid-GMM models.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        k (int): Additional parameter to match.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    fpr_data = {'gmm_fpr': [], 'hgmm_fpr': []}
    x_values = []
    csv_output_data = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            if (int(row['valid_pts']) == valid_pts and 
                int(row['num_clusters']) == num_clusters and 
                int(row['k']) == k):
                
                # False Positive Rate for M-GMM
                gmm_fpr = float(row['gmm_false_outliers']) / float(row['valid_pts'])
                # False Positive Rate for Hybrid-GMM
                hgmm_fpr = float(row['hgmm_false_outliers']) / float(row['valid_pts'])

                fpr_data['gmm_fpr'].append(gmm_fpr)
                fpr_data['hgmm_fpr'].append(hgmm_fpr)

                outlier_pts = int(row['outlier_pts'])
                x_values.append(outlier_pts)

                csv_output_data.append({
                    'valid_pts': valid_pts,
                    'num_clusters': num_clusters,
                    'k': k,
                    'outlier_pts': outlier_pts,
                    'gmm_fpr': gmm_fpr,
                    'hgmm_fpr': hgmm_fpr
                })

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['valid_pts', 'num_clusters', 'k', 'outlier_pts', 'gmm_fpr', 'hgmm_fpr']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(csv_output_data)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, fpr_data['gmm_fpr'], label='GMM False Positive Rate', marker='o')
    plt.plot(x_values, fpr_data['hgmm_fpr'], label='Hybrid-GMM False Positive Rate', marker='x')
    plt.xlabel('True Outliers')
    plt.ylabel('False Positive Rate')
    plt.title(f'False Positive Rate Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}, k={k}')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_metrics_plot_raw_accuracy(file_path, valid_pts, num_clusters, k):
    """
    Generates and displays a raw accuracy comparison plot for M-GMM and Hybrid-GMM models.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        k (int): Additional parameter to match.

    Returns:
        None
    """
    accuracy_data = {'gmm_accuracy': [], 'hgmm_accuracy': []}
    x_values = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            if (int(row['valid_pts']) == valid_pts and 
                int(row['num_clusters']) == num_clusters and 
                int(row['k']) == k):
                
                # Raw Accuracy for M-GMM
                gmm_accuracy = float(row['gmm_raw_accuracy'])
                # Raw Accuracy for Hybrid-GMM
                hgmm_accuracy = float(row['hgmm_raw_accuracy'])

                accuracy_data['gmm_accuracy'].append(gmm_accuracy)
                accuracy_data['hgmm_accuracy'].append(hgmm_accuracy)

                x_values.append(int(row['outlier_pts']))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, accuracy_data['gmm_accuracy'], label='GMM Raw Accuracy', marker='o')
    plt.plot(x_values, accuracy_data['hgmm_accuracy'], label='Hybrid-GMM Raw Accuracy', marker='x')
    plt.xlabel('True Outliers')
    plt.ylabel('Raw Accuracy')
    plt.title(f'Raw Accuracy Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}, k={k}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file = str(input("Enter the input CSV file, e.g., cluster_data.csv: "))
    valid_pts = int(input("Enter the valid number of points: "))
    num_clusters = int(input("Enter the number of clusters: "))
    k = int(input("Enter the value of k: "))

    if not validate_input(file, valid_pts, num_clusters, k):
        sys.exit(1)

    f1score_fn = "f1score.csv"
    fpr_fn = "fpr.csv"

    generate_metrics_plot_f1(file, valid_pts, num_clusters, k, f1score_fn)
    generate_metrics_plot_fpr(file, valid_pts, num_clusters, k, fpr_fn)
    generate_metrics_plot_raw_accuracy(file, valid_pts, num_clusters, k)

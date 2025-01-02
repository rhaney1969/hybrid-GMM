"""
This module provides functionality for validating input data and generating metrics plots
(F1-score and False Positive Rate) based on clustering analysis.

Functions:
    validate_input(csv_filename: str, valid_pts: int, num_clusters: int) -> bool
    generate_metrics_plot_f1(file_path: str, valid_pts: int, num_clusters: int) -> None
    generate_metrics_plot_fpr(file_path: str, valid_pts: int, num_clusters: int) -> None
"""

import csv
import matplotlib.pyplot as plt
import os
import sys

def validate_input(csv_filename: str, valid_pts: int, num_clusters: int) -> bool:
    """
    Validates the existence of a CSV file and checks for matching records.

    Args:
        csv_filename (str): Path to the CSV file.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.

    Returns:
        bool: True if the file exists and contains matching data, False otherwise.
    """
    if not os.path.isfile(csv_filename):
        print(f"WARNING: The file `{csv_filename}` does not exist.")
        return False
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (int(row['valid_pts']) == valid_pts and int(row['num_clusters']) == num_clusters):
                return True

    print("WARNING: No matching data found for the provided input arguments.")
    return False

def generate_metrics_plot_f1(file_path, valid_pts, num_clusters, output_csv_path):
    """
    Generates and displays an F1-score comparison plot for GMM and Hybrid-GMM models,
    and saves the data to a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    # Dictionary to store accuracy values
    f1_data = {'gmm_f1': [], 'hgmm_f1': []}
    x_values = []  # To store the number of outliers for each row
    csv_output_data = []  # To store rows for the output CSV file

    # Open and read the CSV file
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        # Iterate through rows
        for row in csvreader:
            # Check if valid_pts and num_clusters match
            if int(row['valid_pts']) == valid_pts and int(row['num_clusters']) == num_clusters:
                # Calculate True and False Positives
                gmm_tp = int(row['gmm_outliers']) - int(row['gmm_false_outliers'])
                gmm_fp = int(row['gmm_false_outliers'])
                hgmm_tp = int(row['hgmm_outliers']) - int(row['hgmm_false_outliers'])
                hgmm_fp = int(row['hgmm_false_outliers'])

                # Calculate False negatives
                gmm_fn = int(row['outlier_pts']) - gmm_tp
                hgmm_fn = int(row['outlier_pts']) - hgmm_tp

                # Calculate precision
                gmm_precision = float(gmm_tp / (gmm_tp + gmm_fp))
                hgmm_precision = float(hgmm_tp / (hgmm_tp + hgmm_fp))

                # Calculate recall
                gmm_recall = float(gmm_tp / (gmm_tp + gmm_fn))
                hgmm_recall = float(hgmm_tp / (hgmm_tp + hgmm_fn))

                # Store gmm f1 and hgmm f1
                gmm_f1 = 2 * gmm_precision * gmm_recall / (gmm_precision + gmm_recall)
                hgmm_f1 = 2 * hgmm_precision * hgmm_recall / (hgmm_precision + hgmm_recall)

                f1_data['gmm_f1'].append(gmm_f1)
                f1_data['hgmm_f1'].append(hgmm_f1)

                # Store the number of outliers for the x-axis
                outlier_pts = int(row['outlier_pts'])
                x_values.append(outlier_pts)

                # Append row data for CSV output
                csv_output_data.append({
                    'valid_pts': valid_pts,
                    'num_clusters': num_clusters,
                    'outlier_pts': outlier_pts,
                    'gmm_f1': gmm_f1,
                    'hgmm_f1': hgmm_f1
                })

    # Write the data to the output CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['valid_pts', 'num_clusters', 'outlier_pts', 'gmm_f1', 'hgmm_f1']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        csvwriter.writeheader()
        # Write the rows
        csvwriter.writerows(csv_output_data)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f1_data['gmm_f1'], label='GMM F1-Score', marker='o')
    plt.plot(x_values, f1_data['hgmm_f1'], label='Hybrid-GMM F1-Score', marker='x')

    # Add labels, title, and legend
    plt.xlabel('True Outliers')
    plt.ylabel('F1-Score')
    plt.title(f'F1-Score Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

#def generate_metrics_plot_f1(file_path, valid_pts, num_clusters, output_csv_path):
#    """
#    Generates and displays an F1-score comparison plot for GMM and Hybrid-GMM models.
#
#    Args:
#        file_path (str): Path to the CSV file containing the data.
#        valid_pts (int): The valid number of points.
#        num_clusters (int): The number of clusters.
#        output_csv_path (str): Path to save the output CSV file.
#
#    Returns:
#        None
#    """
#    # Dictionary to store accuracy values
#    f1_data = {'gmm_f1': [], 'hgmm_f1': []}
#    x_values = [] # To store the number of outliers for each row
#
#    # Open and read the CSV file
#    with open(file_path, 'r') as csvfile:
#        csvreader = csv.DictReader(csvfile)
#        csv_output_data = [] # To store rows for the output CSV
#
#        # Iterate through rows
#        for row in csvreader:
#            # Check if valid_pts and num_clusters match
#            if int(row['valid_pts']) == valid_pts and int(row['num_clusters']) == num_clusters:
#                # Calculate True and False Postives
#                gmm_tp = int(row['gmm_outliers']) - int(row['gmm_false_outliers'])
#                gmm_fp = int(row['gmm_false_outliers'])
#                hgmm_tp = int(row['hgmm_outliers']) - int(row['hgmm_false_outliers'])
#                hgmm_fp = int(row['hgmm_false_outliers'])
#                # Calculate False negatives
#                gmm_fn = int(row['outlier_pts']) - gmm_tp
#                hgmm_fn = int(row['outlier_pts']) - hgmm_tp
#                # Calculate precision
#                gmm_precision = float(gmm_tp / (gmm_tp + gmm_fp))
#                hgmm_precision = float(hgmm_tp / (hgmm_tp + hgmm_fp))
#                # Calculate recall
#                gmm_recall = float(gmm_tp / (gmm_tp + gmm_fn))
#                hgmm_recall = float(hgmm_tp / (hgmm_tp + hgmm_fn))
#
#                gmm_f1 = 2 * gmm_precision * gmm_recall / (gmm_precision + gmm_recall)
#                hgmm_f1 = 2 * hgmm_precision * hgmm_recall / (hgmm_precision + hgmm_recall)
#
#                f1_data['gmm_f1'].append(gmm_f1)
#                f1_data['hgmm_f1'].append(hgmm_f1)
#
#                # Store gmm f1 and hgmm f1
#                #f1_data['gmm_f1'].append(2 * gmm_precision * gmm_recall / (gmm_precision + gmm_recall))
#                #f1_data['hgmm_f1'].append(2 * hgmm_precision * hgmm_recall / (hgmm_precision + hgmm_recall))
#
#                # Store the number of outliers for the x-axis
#                outlier_pts = int(row['outlier_pts'])
#                x_values.append(outlier_pts)
#
#                # Append row data for CSV output
#                csv_output_data.append({
#                    'valid_pts': valid_pts,
#                    'num_clusters': num_clusters,
#                    'gmm_f1': gmm_f1,
#                    'hgmm_f1': hgmm_f1
#                    })
#    
#    # Write the data to the output CSV file
#    with open(output_csv_path, 'w', newline='') as csvfile:
#        fieldnames = ['valid_pts','num_clusters','outlier_pts','gmm_f1','hgmm_f1']
#        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#        # Write the header
#        csvwriter.writeheader()
#        # Write the rows
#        csvwriter.writerows(csv_output_data)
#
#    # Plot the data
#    plt.figure(figsize=(10, 6))
#    plt.plot(x_values, f1_data['gmm_f1'], label='GMM F1-Score', marker='o')
#    plt.plot(x_values, f1_data['hgmm_f1'], label='Hybrid-GMM F1-Score', marker='x')
#
#    # Add labels, title, and legend
#    plt.xlabel('True Outliers')
#    plt.ylabel('F1-Score')
#    plt.title(f'F1-Score Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}')
#    plt.legend()
#    plt.grid(True)
#
#    # Display the plot
#    plt.show()

def generate_metrics_plot_fpr(file_path, valid_pts, num_clusters, output_csv_path):
    """
    Generates and displays a False Positive Rate (FPR) comparison plot for GMM and Hybrid-GMM models.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    # Dictionary to store accuracy values
    fpr_data = {'gmm_fpr': [], 'hgmm_fpr': []}
    x_values = [] # To store the number of outliers for each row
    csv_output_data = []  # To store rows for the output CSV file

    # Open and read the CSV file
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        # Iterate through rows
        for row in csvreader:
            # Check if valid_pts and num_clusters match
            if int(row['valid_pts']) == valid_pts and int(row['num_clusters']) == num_clusters:
                # Calculate False Postive Rate (FPR)
                gmm_fpr = float(row['gmm_false_outliers']) / float(row['valid_pts'])
                hgmm_fpr = float(row['hgmm_false_outliers']) / float(row['valid_pts'])

                # Store gmm fpr and hgmm fpr
                fpr_data['gmm_fpr'].append(gmm_fpr)
                fpr_data['hgmm_fpr'].append(hgmm_fpr)

                # Store the number of outliers for the x-axis
                outlier_pts = int(row['outlier_pts'])
                x_values.append(outlier_pts)

                # Append row data for CSV output
                csv_output_data.append({
                    'valid_pts': valid_pts,
                    'num_clusters': num_clusters,
                    'outlier_pts': outlier_pts,
                    'gmm_fpr': gmm_fpr,
                    'hgmm_fpr': hgmm_fpr
                })

    # Write the data to the output CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['valid_pts', 'num_clusters', 'outlier_pts', 'gmm_fpr', 'hgmm_fpr']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        csvwriter.writeheader()
        # Write the rows
        csvwriter.writerows(csv_output_data)


    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, fpr_data['gmm_fpr'], label='GMM False Postive Rate', marker='o')
    plt.plot(x_values, fpr_data['hgmm_fpr'], label='Hybrid-GMM False Positive Rate', marker='x')

    # Add labels, title, and legend
    plt.xlabel('True Outliers')
    plt.ylabel('False Positive Rate')
    plt.title(f'False Positive Rate Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def generate_metrics_plot_raw_accuracy(file_path, valid_pts, num_clusters):
    """
    Generates and displays a raw accuracy comparison plot for GMM and Hybrid-GMM models, that is,
    the absoulte ratio of outliers identified by the two different methods.

    Args:
        file_path (str): Path to the CSV file containing the data.
        valid_pts (int): The valid number of points.
        num_clusters (int): The number of clusters.

    Returns:
        None
    """
    # Dictionary to store accuracy values
    accuracy_data = {'gmm_accuracy': [], 'hgmm_accuracy': []}
    x_values = [] # To store the number of outliers for each row

    # Open and read the CSV file
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        # Iterate through rows
        for row in csvreader:
            # Check if valid_pts and num_clusters match
            if int(row['valid_pts']) == valid_pts and int(row['num_clusters']) == num_clusters:
                # Calculate False Postive Rate (FPR)
                gmm_accuracy = float(row['gmm_raw_accuracy'])
                hgmm_accuracy = float(row['hgmm_raw_accuracy'])

                # Store gmm fpr and hgmm fpr
                accuracy_data['gmm_accuracy'].append(gmm_accuracy)
                accuracy_data['hgmm_accuracy'].append(hgmm_accuracy)

                # Store the number of outliers for the x-axis
                x_values.append(int(row['outlier_pts']))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, accuracy_data['gmm_accuracy'], label='GMM Raw Accuracy', marker='o')
    plt.plot(x_values, accuracy_data['hgmm_accuracy'], label='Hybrid-GMM Raw Accuracy', marker='x')

    # Add labels, title, and legend
    plt.xlabel('True Outliers')
    plt.ylabel('Raw Accuracy')
    plt.title(f'Raw Accuracy Comparison for valid_pts={valid_pts}, num_clusters={num_clusters}')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Prompt user for input
    file = str(input("Enter the input CSV file, e.g., cluster_data.csv: "))
    valid_pts = int(input("Enter the valid number of points: "))
    num_clusters = int(input("Enter the number of clusters: "))

    # Validate input and proceed if valid
    if not validate_input(file, valid_pts, num_clusters):
        sys.exit(1)

    # Output CSV ASCII file
    f1score_fn = "f1score.csv"
    fpr_fn = "fpr.csv"

    # Pass input to function to generate metrics and display result
    generate_metrics_plot_f1(file, valid_pts, num_clusters, f1score_fn)
    generate_metrics_plot_fpr(file, valid_pts, num_clusters, fpr_fn)
    generate_metrics_plot_raw_accuracy(file, valid_pts, num_clusters)

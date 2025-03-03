import csv
import os
import matplotlib.pyplot as plt

def plot_mean_abs_error(csv_filename, target_pts):
    k_values = []
    mean_abs_errors = []
    
    # Check if the CSV file exists
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        return
    
    # Read the CSV file
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if int(row["target_pts"]) == target_pts:
                k_values.append(int(row["k"]))
                mean_abs_errors.append(float(row["mean_abs_error"]))
    
    # Check if we have data to plot
    if not k_values:
        print(f"No data found for target_pts = {target_pts}")
        return
    
    # Sort the values by k to ensure correct plotting
    sorted_pairs = sorted(zip(k_values, mean_abs_errors))
    k_values, mean_abs_errors = zip(*sorted_pairs)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mean_abs_errors, marker='o', linestyle='-')
    plt.xlabel("k")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"Mean Absolute Error vs k for target pts = {target_pts}")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    csv_filename = "local_fracs.csv"
    target_pts = int(input("Enter the value for target_pts: "))
    plot_mean_abs_error(csv_filename, target_pts)